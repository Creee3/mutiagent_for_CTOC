"""Independent stdlib audit of the CPU-only same-six answer-format freeze."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"no_msg", "token_full", "kv_full", "material_text"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def ids_hash(ids):
    return sha(json.dumps(ids, separators=(",", ":")).encode())


def render(messages, generation):
    value = "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n"
                    for m in messages)
    return value + ("<|im_start|>assistant\n" if generation else "")


def independent_token_packet(sender, revision):
    ids, qid = sender["ids"], sender["question_id"]
    n = len(ids)
    history_hash = sha(struct.pack("<" + str(n) + "q", *ids))
    segment = {"id": qid + ":answer_format_source:A:prompt", "node": "A", "kind": "prompt",
               "start": 0, "end": n, "tokens_sha256": history_hash}
    parts = [segment]
    header = {"format": "owned-prefix-wire-v1", "kind": "text", "mode": "full",
              "revision": revision, "qid": qid, "length": n, "position_base": 0,
              "position_extent": n, "history_sha256": history_hash, "segments": parts,
              "ledger_sha256": sha(canonical(parts)),
              "origin": sha(canonical({"revision": revision, "qid": qid, "root_event": segment})),
              "owner": None, "tokens": {"dtype": "int32_le", "count": n, "offset": 0, "nbytes": n * 4},
              "body_nbytes": n * 4}
    raw_header = canonical(header)
    packet = struct.pack("<I", len(raw_header)) + raw_header + struct.pack("<" + str(n) + "i", *ids)
    info = {"payload_bytes": len(packet), "payload_content_bytes": n * 4,
            "payload_header_bytes": len(raw_header) + 4, "token_content_bytes": n * 4,
            "kv_tensor_content_bytes": 0, "packet_sha256": sha(packet),
            "encoding": "owned-prefix-wire-v1", "position_base": 0,
            "cache_descriptors": None, "network_transfer_measured": False}
    return packet, header, info


def main(args):
    errors, checks = [], 0

    def check(ok, label, detail=None):
        nonlocal checks
        checks += 1
        if not ok:
            errors.append({"check": label, "detail": detail})

    conditions_path = ROOT / "receiver_answer_format_conditions.json"
    settings = read(conditions_path)
    inputs_path = ROOT / settings["input_file"]
    data, cpu = read(inputs_path), read(ROOT / "checks/receiver_answer_format_cpu.json")
    for name, digest in settings["source_hashes"].items():
        check(file_sha(ROOT / name) == digest, "frozen dependency SHA " + name)
    for name, digest in settings["input_source_hashes"].items():
        check(file_sha(ROOT / name) == digest, "input source SHA " + name)
    check(settings["input_sha256"] == file_sha(inputs_path) == cpu["inputs_sha256"], "frozen input SHA")
    check(cpu["conditions_sha256"] == file_sha(conditions_path), "conditions SHA")
    check(cpu["script_sha256"] == file_sha(ROOT / "prepare_receiver_answer_format.py"), "builder SHA")
    check(settings["model_calls_executed"] == cpu["model_calls"] == 0
          and settings["inference_runner_ready"] is False and cpu["model_weights_loaded"] is False
          and cpu["new_KV_generated"] is False and cpu["gpu_initialized"] is False,
          "CPU preparation only, no executed model budget")
    check(settings["greedy"] is True and settings["num_beams"] == 1
          and settings["answer_max_new_tokens"] == 48 and settings["repetition_penalty"] == 1.0
          and settings["shift_back"] is False, "candidate decode settings")
    check(settings["proposed_expected_calls"] == {"main_generate": 72, "warmup_generate": 4,
          "main_sender_prefill": 6, "warmup_sender_prefill": 1}, "candidate 76/7 budget is separate from execution")
    helper = ROOT / "checks/receiver_endpoint_result_audit_20261002.py"
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("_stdlib_endpoint_decoder", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer_path = Path(args.tokenizer)
    decode = module.byte_level_decoder(tokenizer_path)
    check(file_sha(tokenizer_path) == settings["model_tokenizer_hashes"]["tokenizer.json"], "decoder tokenizer SHA")
    tokenizer_config_path = tokenizer_path.with_name("tokenizer_config.json")
    check(file_sha(tokenizer_config_path) == settings["model_tokenizer_hashes"]["tokenizer_config.json"], "tokenizer config SHA")
    original = read(ROOT / "data/receiver_evidence_20261002/selector_input_views.json")
    schedule = read(ROOT / "data/receiver_runtime_views_20261002/scheduler_metadata.json")
    safe = read(ROOT / "data/receiver_runtime_views_20261002/runtime_payloads.json")
    partial_ids = {r["question_id"] for r in schedule if r["condition"] == "partial"}
    selected_original = [s for s in original["senders"] if s["question_id"] in partial_ids]
    expected_qids = [s["question_id"] for s in selected_original]
    source = {s["question_id"]: s for s in selected_original}
    senders = {s["question_id"]: s for s in data["senders"]}
    check(len(partial_ids) == len(senders) == len(data["senders"]) == 6, "six original partial-eligible senders")
    check([s["question_id"] for s in data["senders"]] == settings["selected_question_ids"] == expected_qids,
          "all original partial-eligible IDs, original order, no outcome filter")
    meta = {r["lookup_id"]: r for r in schedule}
    chosen = [r for r in safe if meta[r["lookup_id"]]["question_id"] in partial_ids
              and meta[r["lookup_id"]]["condition"] in ("missing", "partial", "full")]
    expected_keys = [meta[r["lookup_id"]]["source_key"] for r in chosen]
    safe_by_id = {r["lookup_id"]: r["payload"] for r in chosen}
    rows = data["receivers"]
    check(len(rows) == len(chosen) == 18 and len({r["key"] for r in rows}) == 18, "18 unique receiver inputs")
    check([r["key"] for r in rows] == expected_keys, "original selected receiver order")
    check(dict(Counter(r["condition"] for r in rows)) == settings["condition_counts"] ==
          {"missing": 6, "partial": 6, "full": 6}, "same six missing/partial/full pairing")
    system = {"role": "system", "content": settings["system_rule"]}
    check("return unknown" in settings["system_rule"] and "complete sentence" in settings["system_rule"],
          "shared short-answer and abstention rule recorded")
    source_packet_table, length_table = [], []
    for qid, sender in senders.items():
        old = source[qid]
        check(sender["sender_material"] == old["sender_material"] and sender["question"] == old["question"],
              "A public material and question unchanged " + qid)
        source_user = "Source A evidence for this question.\nQuestion: " + old["question"] + "\nMaterial:\n" + old["sender_material"]
        messages = [system, {"role": "user", "content": source_user}]
        check(sender["messages"] == messages and sender["rendered"] == render(messages, False), "A data-only messages " + qid)
        check(decode(sender["ids"], False) == sender["rendered"] and ids_hash(sender["ids"]) == sender["ids_sha256"],
              "A IDs independently decoded and hashed " + qid)
        check(sender["cache_status"] == "not_generated_new_prefix", "new A cache not generated " + qid)
        token_packet, header, info = independent_token_packet(sender, settings["model_revision"])
        check(sender["source_state_id"] == header["segments"][0]["id"], "new A token source ledger " + qid)
        check(sender["prepared_token_packet_info"] == info, "independently reconstructed actual tokenwire SHA/bytes " + qid)
        header_size = struct.unpack("<I", token_packet[:4])[0]
        check(list(struct.unpack("<" + str(len(sender["ids"])) + "i", token_packet[4 + header_size:])) == sender["ids"],
              "independent tokenwire numeric decode " + qid)
        packet_info = sender["prepared_material_packet"]
        actual = packet_info["packet_text"].encode("utf-8")
        body = json.loads(actual.decode("utf-8"))
        expected_body = {"format": "source-material-json-v1", "source": "A", "qid": qid,
                         "revision": settings["model_revision"], "material": old["sender_material"]}
        expected_bytes = json.dumps(expected_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        check(body == expected_body and actual == expected_bytes, "actual UTF8 JSON source/material byte identity " + qid)
        material_size = len(old["sender_material"].encode("utf-8"))
        check(packet_info["packet_sha256"] == sha(actual) and packet_info["payload_bytes"] == len(actual)
              and packet_info["material_utf8_bytes"] == material_size
              and packet_info["json_metadata_and_escaping_bytes"] == len(actual) - material_size
              and packet_info["network_transfer_measured"] is False, "UTF8 actual SHA and full overhead bytes " + qid)
        source_packet_table.append({"question_id": qid, "A_prefix_tokens": len(sender["ids"]),
                                   "tokenwire_bytes": len(token_packet), "tokenwire_sha256": sha(token_packet),
                                   "material_json_bytes": len(actual), "material_json_sha256": sha(actual)})
    maximum, plan_count = 0, 0
    for row in rows:
        qid, key = row["question_id"], row["key"]
        sender, p = senders[qid], safe_by_id[row["lookup_id"]]
        schedule_row = meta[row["lookup_id"]]
        check(row["runtime_payload"] == p and set(p) == {"question", "sender_material", "receiver_visible_text"},
              "unchanged runtime whitelist " + key)
        check(row["condition"] == schedule_row["condition"] and key == schedule_row["source_key"], "scheduling identity " + key)
        remainder = p["receiver_visible_text"].split("\nQuestion: ", 1)[1]
        b_user = settings["B_role"] + "\nQuestion: " + remainder
        b_message = {"role": "user", "content": b_user}
        canonical_messages = sender["messages"] + [b_message]
        canonical_rendered = render(canonical_messages, True)
        suffix_text = canonical_rendered[len(sender["rendered"]):]
        inputs = row["method_inputs"]
        check(row["B_user"] == b_user and row["canonical_messages"] == canonical_messages
              and row["canonical_rendered"] == canonical_rendered, "canonical unchanged B material/held " + key)
        check(canonical_rendered.count("<|im_start|>system\n") == 1
              and "<|im_start|>system\n" not in suffix_text
              and row["B_suffix_text"] == suffix_text, "one common system, suffix contains no system " + key)
        check(sender["ids"] + row["B_suffix_ids"] == inputs["token_full"]["ids"], "canonical exact A IDs plus suffix " + key)
        check(decode(inputs["token_full"]["ids"], False) == canonical_rendered
              and decode(row["B_suffix_ids"], False) == suffix_text, "independent canonical/suffix decode " + key)
        check(ids_hash(row["B_suffix_ids"]) == row["B_suffix_ids_sha256"]
              and row["A_ids_sha256"] == sender["ids_sha256"], "suffix and A binding " + key)
        no_rendered = render([system, b_message], True)
        strong_user = b_user + "\nReceived material from A:\n" + sender["sender_material"]
        strong_rendered = render([system, {"role": "user", "content": strong_user}], True)
        check(row["strong_text_user"] == strong_user and strong_user.startswith(b_user)
              and strong_user.endswith(sender["sender_material"]), "strong text keeps B and complete A " + key)
        check(decode(inputs["no_msg"]["ids"], False) == no_rendered
              and decode(inputs["material_text"]["ids"], False) == strong_rendered,
              "no_msg and material_text actual decoded inputs preserve B " + key)
        check(set(inputs) == METHODS, "four method plans " + key)
        for method, method_input in inputs.items():
            check(ids_hash(method_input["ids"]) == method_input["ids_sha256"], "method input hash " + key + ":" + method)
            plan_count += 1
        check(inputs["kv_full"]["ids"] == row["B_suffix_ids"]
              and inputs["kv_full"]["prefix_tokens"] == len(sender["ids"])
              and inputs["kv_full"]["joint_ids_sha256"] == inputs["token_full"]["ids_sha256"],
              "KV suffix and full joint identity " + key)
        prompt_lengths = {m: len(v["ids"]) + (len(sender["ids"]) if m == "kv_full" else 0)
                          for m, v in inputs.items()}
        budgeted = max(prompt_lengths.values()) + 48
        maximum = max(maximum, budgeted)
        check(row["maximum_prompt_tokens_plus_48"] == budgeted, "prompt capacity arithmetic " + key)
        length_table.append({"key": key, "prompt_lengths": prompt_lengths, "plus_answer_budget": budgeted})
    check(plan_count == cpu["method_input_plans"] == 72, "72 plans, not calls")
    check(maximum == cpu["maximum_prompt_tokens_plus_48"] == settings["maximum_prompt_tokens_plus_answer_budget"] == 422,
          "independent max prompt plus budget")
    expected_encodes = len(senders) + 4 * len(rows)
    check(expected_encodes == cpu["tokenizer_encode_calls"] == 78, "78 builder encoding calls, not model calls")
    check(cpu["canonical_prefix_suffix_equal_pairs"] == cpu["single_system_full_prompts"] ==
          cpu["held_material_preserved_rows"] == 18, "CPU report canonical/preservation counts")
    check(cpu["new_source_token_packet_roundtrips"] == cpu["utf8_material_packet_roundtrips"] == 6,
          "CPU source packet roundtrip counts")
    check(settings["warmup_key"] == rows[0]["key"] and settings["warmup_key"] in expected_keys,
          "warmup reuses an existing same-six view")
    builder = (ROOT / "prepare_receiver_answer_format.py").read_text(encoding="utf-8")
    check(all(word not in builder for word in ("scoring_labels.json", "data/calibration.json", "diagnostic_annotations.json",
                                               "summary.json", "output_ids", "AutoModel", ".generate(")),
          "builder does not read scores/gold/model outputs or generate a model")
    check(settings["answer_evaluation"]["gold_aware_extraction"] is False
          and any("changed together" in limit for limit in settings["limits"]),
          "raw scoring and non-single-factor scope recorded")
    report = {"passed": not errors, "checks": checks, "errors": errors, "model_calls": 0,
              "audit_script_sha256": file_sha(__file__), "decoder_helper_sha256": file_sha(helper),
              "conditions_sha256": file_sha(conditions_path), "inputs_sha256": file_sha(inputs_path),
              "selected_question_ids": expected_qids, "sender_count": 6, "receiver_count": 18,
              "method_input_plans": plan_count, "builder_tokenizer_encode_calls": expected_encodes,
              "maximum_prompt_tokens_plus_answer_budget": maximum,
              "proposed_only_generate": 76, "proposed_only_sender_prefill": 7,
              "source_packet_table": source_packet_table, "method_prompt_lengths": length_table,
              "runtime_constraint": "Use frozen method IDs and A IDs only; do not re-encode suffix_text/rendered strings at runtime.",
              "limits": ["Stdlib only; no SSH, model replay, or prior run re-audit.",
                         "UTF8 material packet bytes are present as packet_text; tokenwire bytes were independently reconstructed and match the recorded actual-packet SHA.",
                         "The six questions were already-used public support-sentence calibration inputs, not new held-out evidence.",
                         "The shared unknown rule, A data representation, and chat layout change together; this is format calibration, not a single-factor cause experiment.",
                         "Explicitly held duplicate text does not prove semantic knowledge or permission to delete corresponding KV.",
                         "No new KV or model answer exists in this preparation; 76/7 is a candidate future budget."]}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "checks", "errors", "sender_count", "receiver_count",
          "method_input_plans", "builder_tokenizer_encode_calls", "maximum_prompt_tokens_plus_answer_budget", "model_calls")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(ROOT / "checks/audit_receiver_answer_format_20261002.json"))
    parser.add_argument("--tokenizer", default=str(ROOT.parent / "c2c_selective_acceptance/assets/sharer/tokenizer.json"))
    main(parser.parse_args())
