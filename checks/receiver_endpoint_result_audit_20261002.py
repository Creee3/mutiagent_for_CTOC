"""Independent stdlib audit of one frozen endpoint run; never loads a model."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import string
import struct

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"no_msg", "token_full", "kv_full"}
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def ids_hash(ids):
    return sha(json.dumps(ids, separators=(",", ":")).encode())


def tensor_ids_hash(ids):
    return sha(struct.pack("<" + str(len(ids)) + "q", *ids))


def normalize(text):
    text = "".join(c for c in str(text).lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def token_f1(prediction, gold):
    prediction, gold = normalize(prediction), normalize(gold)
    if any((prediction == s or gold == s) and prediction != gold
           for s in ("yes", "no", "noanswer")):
        return 0.0
    p, g = prediction.split(), gold.split()
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


class Audit:
    def __init__(self):
        self.errors, self.checks = [], 0

    def check(self, condition, label, detail=None):
        self.checks += 1
        if not condition:
            self.errors.append({"check": label, "detail": detail})


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate header key: " + key)
        result[key] = value
    return result


def audit_packet(path, sender, settings, audit):
    packet = path.read_bytes()
    if len(packet) < 4:
        raise ValueError("short packet")
    header_size = struct.unpack("<I", packet[:4])[0]
    if not 0 < header_size <= len(packet) - 4:
        raise ValueError("invalid header length")
    header_bytes = packet[4:4 + header_size]
    h = json.loads(header_bytes.decode("utf-8"), object_pairs_hook=no_duplicate_keys)
    body = memoryview(packet)[4 + header_size:]
    qid, a_ids = sender["question_id"], sender["ids"]
    n = len(a_ids)
    label = path.name
    audit.check(canonical(h) == header_bytes, label + ": canonical header")
    audit.check(h["format"] == "owned-prefix-wire-v1" and h["mode"] == "full"
                and h["owner"] is None, label + ": complete independent packet")
    audit.check(h["qid"] == qid and h["revision"] == settings["model_revision"],
                label + ": source identity")
    audit.check(h["length"] == h["position_extent"] == n and h["position_base"] == 0,
                label + ": position extent")
    audit.check(len(body) == h["body_nbytes"], label + ": complete body length")
    tok = h["tokens"]
    audit.check(tok == {"dtype": "int32_le", "count": n, "offset": 0, "nbytes": n * 4},
                label + ": token descriptor")
    packet_ids = list(struct.unpack("<" + str(n) + "i", body[:n * 4]))
    audit.check(packet_ids == a_ids, label + ": actual A token bytes")
    audit.check(h["history_sha256"] == tensor_ids_hash(a_ids), label + ": history hash")
    segments = h["segments"]
    expected_segment = {"id": sender["source_state_id"], "node": "A", "kind": "prompt",
                        "start": 0, "end": n, "tokens_sha256": tensor_ids_hash(a_ids)}
    audit.check(segments == [expected_segment], label + ": source ledger")
    audit.check(h["ledger_sha256"] == sha(canonical(segments)), label + ": ledger hash")
    audit.check(h["origin"] == sha(canonical({"revision": settings["model_revision"],
                "qid": qid, "root_event": segments[0]})), label + ": origin hash")
    cursor, cache_bytes = n * 4, 0
    if h["kind"] == "kv":
        layers = h["cache"]
        audit.check(len(layers) == settings["full_layers"], label + ": complete layers")
        descriptors, blocks, key_blocks, value_blocks = [], [], [], []
        for i, entries in enumerate(layers):
            descriptor = {}
            audit.check(set(entries) == {"key", "value"}, label + f": layer {i} keys")
            for name in ("key", "value"):
                entry = entries[name]
                shape = entry["shape"]
                audit.check(len(shape) == 4 and shape[0] == 1 and shape[2] == n
                            and all(type(v) is int and v > 0 for v in shape),
                            label + f": {i}/{name} shape")
                size = math.prod(shape) * DTYPE_BYTES[entry["dtype"]]
                audit.check(entry["dtype"] == "bfloat16" and entry["payload_shape"] == shape
                            and entry["nbytes"] == entry["payload_nbytes"] == size
                            and entry["offset"] == cursor,
                            label + f": {i}/{name} complete tensor interval")
                block = body[cursor:cursor + size]
                audit.check(len(block) == size, label + f": {i}/{name} complete bytes")
                blocks.append(block)
                (key_blocks if name == "key" else value_blocks).append(block)
                descriptor[name] = {k: entry[k] for k in ("shape", "dtype", "nbytes")}
                cursor += size
                cache_bytes += size
            audit.check(entries["key"]["shape"] == entries["value"]["shape"],
                        label + f": layer {i} matching K/V shape")
            descriptors.append(descriptor)
        descriptor_bytes = canonical(descriptors)
        raw_digest, enhanced = hashlib.sha256(), hashlib.sha256(descriptor_bytes + b"\x00")
        chain = hashlib.sha256(str([n] * len(layers)).encode())
        for block in blocks:
            raw_digest.update(block)
            enhanced.update(block)
        for block in key_blocks + value_blocks:
            chain.update(block)
        audit.check(h["cache_descriptor_sha256"] == sha(descriptor_bytes), label + ": descriptors hash")
        audit.check(h["cache_byte_sha256"] == raw_digest.hexdigest(), label + ": actual cache bytes hash")
        audit.check(h["cache_sha256"] == enhanced.hexdigest(), label + ": enhanced cache hash")
        audit.check(h["chain_cache_sha256"] == chain.hexdigest(), label + ": source cache hash")
    else:
        audit.check(h["kind"] == "text" and "cache" not in h, label + ": text kind")
    audit.check(cursor == len(body), label + ": no trailing/missing bytes")
    info = {"payload_bytes": len(packet), "payload_content_bytes": len(body),
            "payload_header_bytes": header_size + 4, "token_content_bytes": n * 4,
            "kv_tensor_content_bytes": cache_bytes, "packet_sha256": sha(packet),
            "encoding": "owned-prefix-wire-v1", "position_base": 0,
            "cache_descriptors": h.get("cache"), "network_transfer_measured": False}
    return h, info


def byte_level_decoder(tokenizer_path):
    """Decode the verified Qwen ByteLevel vocabulary with stdlib, no tokenization."""
    tokenizer = read(tokenizer_path)
    if tokenizer["decoder"]["type"] != "ByteLevel":
        raise ValueError("unsupported independent decoder")
    inverse_vocab = {index: token for token, index in tokenizer["model"]["vocab"].items()}
    specials = set()
    for token in tokenizer["added_tokens"]:
        inverse_vocab[token["id"]] = token["content"]
        if token["special"]:
            specials.add(token["id"])
    alphabet = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    glyphs = list(alphabet)
    extra = 0
    for value in range(256):
        if value not in alphabet:
            alphabet.append(value)
            glyphs.append(256 + extra)
            extra += 1
    inverse_bytes = {chr(glyph): value for glyph, value in zip(glyphs, alphabet)}

    def decode(ids, skip_special=True):
        characters = "".join(inverse_vocab[i] for i in ids if not (skip_special and i in specials))
        return bytes(inverse_bytes[c] for c in characters).decode("utf-8", errors="replace")
    return decode


def main(args):
    audit = Audit()
    run = Path(args.run_dir).resolve()
    conditions_path = ROOT / "receiver_endpoint_conditions.json"
    settings, manifest, summary = read(conditions_path), read(run / "manifest.json"), read(run / "summary.json")
    complete = read(run / "COMPLETE.json")
    inputs_path = ROOT / settings["input_file"]
    data = read(inputs_path)
    senders = {s["question_id"]: s for s in data["senders"]}
    cases = {r["key"]: r for r in data["receivers"]}
    audit.check(len(senders) == len(data["senders"]) == 20, "20 unique frozen A inputs")
    audit.check(len(cases) == len(data["receivers"]) == 72, "72 unique frozen receiver inputs")
    audit.check(manifest["conditions_sha256"] == file_sha(conditions_path), "conditions SHA")
    audit.check(settings["input_sha256"] == file_sha(inputs_path), "frozen input SHA")
    for key, value in settings.items():
        audit.check(manifest.get(key) == value, "manifest frozen field " + key)
    for name, digest in settings["source_hashes"].items():
        audit.check(file_sha(run / "code" / name) == digest, "run code snapshot " + name)
    for name, digest in settings["input_source_hashes"].items():
        audit.check(file_sha(ROOT / name) == digest, "frozen source SHA " + name)
    audit.check(manifest["network_transfer_measured"] is False, "no network measurement claim")
    generation = manifest["generation_kwargs"]
    audit.check(generation["max_new_tokens"] == 48 and generation["do_sample"] is False
                and generation["num_beams"] == 1 and generation["repetition_penalty"] == 1.0,
                "effective greedy decode settings")
    audit.check(manifest["identity_layer_map"] == {str(i): i for i in range(settings["full_layers"])},
                "identity full layer map")
    eos = generation["eos_token_id"]
    eos = set(eos if isinstance(eos, list) else [eos])
    decoder = byte_level_decoder(Path(args.tokenizer))
    verified = read(ROOT / "checks/model7b_manifest.json")
    audit.check(file_sha(args.tokenizer) == verified["files"]["tokenizer.json"]["sha256"],
                "independent decoder tokenizer SHA")
    safe_path = ROOT / "data/receiver_runtime_views_20261002/runtime_payloads.json"
    safe = {r["lookup_id"]: r["payload"] for r in read(safe_path)}
    for qid, sender in senders.items():
        audit.check(ids_hash(sender["ids"]) == sender["ids_sha256"], "frozen A IDs hash " + qid)
        decoded_a = decoder(sender["ids"], skip_special=False)
        audit.check(sender["prompt"] in decoded_a and decoded_a.endswith("<|im_end|>\n"),
                    "closed-user A prompt " + qid)
    for key, case in cases.items():
        a_ids = senders[case["question_id"]]["ids"]
        audit.check(ids_hash(case["ids"]) == case["ids_sha256"]
                    and ids_hash(a_ids + case["ids"]) == case["joint_ids_sha256"],
                    "frozen B/joint hash " + key)
        p = safe[case["lookup_id"]]
        audit.check(set(p) == {"question", "sender_material", "receiver_visible_text"},
                    "selector payload whitelist " + key)
        expected_prompt = settings["new_role"] + "\nQuestion: " + p["receiver_visible_text"].split("\nQuestion: ", 1)[1]
        audit.check(case["prompt"] == expected_prompt, "B visible material/held retained " + key)
        decoded_b = decoder(case["ids"], skip_special=False)
        audit.check(case["prompt"] in decoded_b and decoded_b.endswith("<|im_start|>assistant\n"),
                    "B actual decoded input " + key)
    pairs = [(key, key + ":other_document_control") for key, r in cases.items()
             if r["condition"] in ("partial", "full")]
    audit.check(len(pairs) == 26, "26 control pairs")
    audit.check(all(len(cases[a]["ids"]) == len(cases[b]["ids"]) for a, b in pairs),
                "26 newly derived complete B length matches")
    partial_qids = {r["question_id"] for r in cases.values() if r["condition"] == "partial"}
    audit.check(len(partial_qids) == 6 and all(q + ":missing" in cases and q + ":full" in cases
                for q in partial_qids), "same six three-state questions")
    events = [json.loads(line) for line in (run / "calls.jsonl").read_text(encoding="utf-8").splitlines() if line]
    event_keys = [(e["phase"], e["kind"], e["tag"]) for e in events]
    event_map = dict(zip(event_keys, events))
    audit.check(len(event_map) == len(events), "unique phase/kind/tag calls")
    expected_events = {("main", "sender_prefill", q + ":A") for q in senders}
    warm_key = settings["warmup_key"]
    warm_qid = cases[warm_key]["question_id"]
    expected_events.add(("warmup", "sender_prefill", warm_qid + ":A"))
    expected_events |= {("main", "generate", key + ":" + method) for key in cases for method in METHODS}
    expected_events |= {("warmup", "generate", warm_key + ":" + method) for method in METHODS}
    audit.check(set(event_keys) == expected_events, "exact expected call coverage")
    audit.check(all(e["status"] == "ok" for e in events), "all calls succeeded")
    phase_counts = dict(Counter(e["phase"] + "_" + e["kind"] for e in events))
    counts = dict(Counter(e["kind"] for e in events))
    audit.check(phase_counts == settings["expected_calls"], "216/3 and 20/1 phase counts")
    audit.check(summary["phase_counts"] == phase_counts and summary["counts"] == counts,
                "independent counts equal summary")
    audit.check(complete["counts"] == phase_counts and summary["status"] == "completed", "COMPLETE phase counts")
    audit.check(not (run / "FAILED.json").exists(), "no FAILED artifact")
    expected_sources = [("main", q) for q in senders] + [("warmup", warm_qid)]
    source_infos, packet_table = {}, []
    audit.check({p.name for p in (run / "sources").glob("*.json")} ==
                {f"{phase}_{q}.json" for phase, q in expected_sources}, "21 exact source files")
    audit.check({p.name for p in (run / "packets").glob("*.bin")} ==
                {f"{phase}_{q}_{m}.bin" for phase, q in expected_sources for m in ("token_full", "kv_full")},
                "42 exact packet files")
    for phase, qid in expected_sources:
        sender = senders[qid]
        source = read(run / "sources" / f"{phase}_{qid}.json")
        audit.check(source["A_ids"] == sender["ids"] and source["question_id"] == qid
                    and source["phase"] == phase, "source frozen A " + phase + ":" + qid)
        call = event_map[(phase, "sender_prefill", qid + ":A")]
        audit.check(source["sender_prefill"] == call and call["model_forward_calls"] == 1
                    and call["effective_position_ids"] == [[list(range(len(sender["ids"]))) ]],
                    "source actual prefill positions " + phase + ":" + qid)
        info_by_method = {}
        for method, meta_key in (("token_full", "text_packet"), ("kv_full", "kv_packet")):
            path = run / "packets" / f"{phase}_{qid}_{method}.bin"
            header, info = audit_packet(path, sender, settings, audit)
            for key, value in info.items():
                audit.check(source[meta_key].get(key) == value, "source packet metric " + path.name + ":" + key)
            if method == "token_full":
                audit.check(info == sender["prepared_text_packet_info"], "frozen text packet exact match " + path.name)
            else:
                audit.check(header["chain_cache_sha256"] == source["cache_source_hash"], "source real cache hash " + path.name)
                audit.check(info["kv_tensor_content_bytes"] == source["cache_resident_tensor_bytes"], "resident cache bytes " + path.name)
            info_by_method[method] = info
            packet_table.append({"phase": phase, "question_id": qid, "method": method,
                                 **{k: v for k, v in info.items() if k != "cache_descriptors"}})
        source_infos[(phase, qid)] = (source, info_by_method)
    item_paths = list((run / "items").glob("*.json"))
    rows = [read(p) for p in item_paths]
    row_map = {(r["phase"], r["key"]): r for r in rows}
    audit.check(len(rows) == len(row_map) == 73, "72 main plus one warmup result rows")
    audit.check(set(row_map) == {("main", key) for key in cases} | {("warmup", warm_key)}, "exact row coverage")
    diagnostic = defaultdict(list)
    exact_pairs, eos_main, truncated_main = 0, 0, 0
    output_pair_differences = []
    gold = {r["question_id"]: r["answer"] for r in read(ROOT / "data/receiver_evidence_20261002/scoring_labels.json")}
    example_outputs = []
    for (phase, key), row in row_map.items():
        case = cases[key]
        qid = case["question_id"]
        source, packets = source_infos[(phase, qid)]
        for field, value in case.items():
            audit.check(row.get(field) == value, "result frozen B field " + phase + ":" + key + ":" + field)
        audit.check(row["A_ids_sha256"] == senders[qid]["ids_sha256"]
                    and row["cache_source_hash"] == source["cache_source_hash"]
                    and row["source_cache_unchanged"] is True, "result source cache binding " + phase + ":" + key)
        audit.check(set(row["methods"]) == METHODS and set(row["method_order"]) == METHODS
                    and len(row["method_order"]) == 3, "three endpoint methods " + phase + ":" + key)
        n_a, n_b = len(senders[qid]["ids"]), len(case["ids"])
        if phase == "main":
            text_result, kv_result = row["methods"]["token_full"], row["methods"]["kv_full"]
            same = text_result["output_ids"] == kv_result["output_ids"]
            exact_pairs += same
            if not same:
                first_difference = next((i for i, (a, b) in enumerate(zip(text_result["output_ids"], kv_result["output_ids"])) if a != b),
                                        min(len(text_result["output_ids"]), len(kv_result["output_ids"])))
                output_pair_differences.append({"key": key, "first_difference_index": first_difference,
                    "token_full_response": text_result["response"], "kv_full_response": kv_result["response"]})
        for method, result in row["methods"].items():
            tag = key + ":" + method
            call = event_map[(phase, "generate", tag)]
            ids = result["output_ids"]
            length = len(ids)
            audit.check(all(type(i) is int and i >= 0 for i in ids) and 0 < length <= 48,
                        "actual output ID length " + phase + ":" + tag)
            prefix = n_a if method == "kv_full" else 0
            prompt_len = n_b if method == "no_msg" else n_a + n_b
            positions = [[list(range(prefix, prompt_len))]] + [[[prompt_len + i - 1]] for i in range(1, length)]
            audit.check(call["effective_position_ids"] == positions and call["model_forward_calls"] == length
                        and result["model_forward_calls"] == length, "actual receiver positions/forwards " + phase + ":" + tag)
            ended, truncated = ids[-1] in eos, length >= 48 and ids[-1] not in eos
            audit.check(not eos.intersection(ids[:-1]), "no earlier EOS before stopping " + phase + ":" + tag)
            audit.check(result["generated_tokens"] == length and result["eos_stopped"] == ended
                        and result["actually_truncated"] == truncated, "EOS and truncation " + phase + ":" + tag)
            audit.check(decoder(ids) == result["response"], "independent ByteLevel output decode " + phase + ":" + tag)
            if method == "no_msg":
                audit.check(all(result[k] == 0 for k in ("payload_bytes", "payload_content_bytes", "payload_header_bytes",
                            "token_content_bytes", "kv_tensor_content_bytes")), "no_msg no A payload " + phase + ":" + tag)
            else:
                for field, value in packets[method].items():
                    audit.check(result.get(field) == value, "result real packet " + phase + ":" + tag + ":" + field)
            audit.check(result["network_transfer_measured"] is False, "result has no network measure " + phase + ":" + tag)
            if phase == "main":
                eos_main += ended
                truncated_main += truncated
                prediction, answer = normalize(result["response"]), normalize(gold[qid])
                diagnostic[(case["condition"], method)].append({"em": prediction == answer,
                    "f1": token_f1(result["response"], gold[qid]),
                    "gold_surface_contained": (" " + answer + " ") in (" " + prediction + " ")})
                if case["condition"] == "full":
                    example_outputs.append({"question_id": qid, "method": method,
                                            "response": result["response"], "gold": gold[qid]})
    underlying = sum(e["model_forward_calls"] for e in events)
    audit.check(summary["underlying_model_forward_calls"] == underlying, "independent underlying forwards")
    audit.check(summary["EOS_main_answers"] == eos_main and summary["truncated_main_answers"] == truncated_main,
                "independent main EOS/truncation totals")
    quality = [{"condition": condition, "method": method, "n": len(values),
                "em_count": sum(v["em"] for v in values), "token_f1_mean": sum(v["f1"] for v in values) / len(values),
                "gold_surface_containment_count_not_semantic_accuracy": sum(v["gold_surface_contained"] for v in values)}
               for (condition, method), values in sorted(diagnostic.items())]
    report = {"passed": not audit.errors, "checks": audit.checks, "errors": audit.errors,
              "run_dir": str(run), "audit_script_sha256": file_sha(__file__), "model_calls": 0,
              "counts": counts, "phase_counts": phase_counts, "underlying_model_forward_calls": underlying,
              "main_rows": 72, "warmup_rows": 1, "packet_files_audited": len(packet_table),
              "full_prompt_control_pairs": len(pairs), "partial_unique_questions": len(partial_qids),
              "token_full_kv_full_exact_output_pairs": exact_pairs, "main_EOS": eos_main,
              "main_truncated": truncated_main, "independent_output_decode": "verified Qwen ByteLevel vocabulary; stdlib only",
              "token_full_kv_full_output_differences": output_pair_differences,
              "decoder_tokenizer_sha256": file_sha(args.tokenizer), "packet_table": packet_table,
              "diagnostic_metrics_not_benchmark": quality, "full_condition_actual_outputs": example_outputs,
              "limits": ["No model replay, SSH, network measurement, or old-run audit was performed.",
                         "Source-cache immutability relies on the frozen runner's successful live hash assertions.",
                         "Recorded input identities are bound by frozen IDs and runner assertions; hidden live tensors were not replayed.",
                         "Strict EM, token F1 and gold surface containment are diagnostic metrics; containment is not semantic correctness.",
                         "No semantic-message-value conclusion is made; complete-prefix answer formatting must be checked first."]}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "checks", "errors", "phase_counts", "underlying_model_forward_calls",
          "packet_files_audited", "token_full_kv_full_exact_output_pairs", "main_EOS", "main_truncated")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(ROOT / "runs/receiver_endpoint_20261001_204127"))
    parser.add_argument("--report", default=str(ROOT / "checks/receiver_endpoint_result_audit_20261002.json"))
    parser.add_argument("--tokenizer", default=str(ROOT.parent / "c2c_selective_acceptance/assets/sharer/tokenizer.json"))
    main(parser.parse_args())
