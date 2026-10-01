"""Standard-library analysis for the six-question answer-format check.

No model imports, automatic answer extraction, or automatic answer-only gate.
Raw EM/F1 reuse the original scorer through analyze_receiver_endpoints.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from analyze_receiver_endpoints import average, existing_scorer, read, sha, write

ROOT = Path(__file__).resolve().parent
METHODS = ("no_msg", "token_full", "kv_full", "material_text")
CONDITIONS = ("missing", "partial", "full")
EXPECTED_PHASE_COUNTS = {"main_generate": 72, "warmup_generate": 4,
                         "main_sender_prefill": 6, "warmup_sender_prefill": 1}
COST_FIELDS = ("receiver_model_seconds", "receiver_prefill_gpu_seconds",
               "receiver_decode_gpu_seconds", "sender_prefill_seconds",
               "packet_encode_seconds", "packet_restore_seconds",
               "from_scratch_stage_sum_seconds", "preexisting_cache_stage_sum_seconds")


def summary_for(rows, scores):
    assert len(rows) == len({row["question_id"] for row in rows}) == 6
    result = {"paired_question_count": 6, "methods": {}, "message_vs_no_message": {}}
    for method in METHODS:
        values = [row["methods"][method] for row in rows]
        scored = [scores[(row["question_id"], row["condition"], method)] for row in rows]
        result["methods"][method] = {
            "raw_exact_match_rate": average(value["exact_match"] for value in scored),
            "raw_mean_token_f1": average(value["token_f1"] for value in scored),
            "normalized_unknown_abstention_count": sum(value["normalized_unknown"] for value in scored),
            "eos_stopped_count": sum(value["eos_stopped"] for value in values),
            "actually_truncated_count": sum(value["actually_truncated"] for value in values),
            "mean_generated_tokens": average(value["generated_tokens"] for value in values),
            "mean_payload_bytes": average(value["payload_bytes"] for value in values),
            "mean_payload_content_bytes": average(value["payload_content_bytes"] for value in values),
            "mean_payload_header_bytes": average(value["payload_header_bytes"] for value in values),
            "single_pass_stage_means_seconds": {
                field: average(value[field] for value in values) for field in COST_FIELDS},
            "header_interpretation": ("JSON metadata plus escaping overhead above raw material UTF-8 bytes"
                                      if method == "material_text" else "encoded protocol header/framing"),
        }
    for method in METHODS[1:]:
        pairs = []
        for row in rows:
            prefix = (row["question_id"], row["condition"])
            message, absent = scores[(*prefix, method)], scores[(*prefix, "no_msg")]
            pairs.append({"question_id": row["question_id"],
                "raw_token_f1_delta": message["token_f1"] - absent["token_f1"],
                "raw_exact_match_delta": int(message["exact_match"]) - int(absent["exact_match"])})
        result["message_vs_no_message"][method] = {
            "paired_question_count": 6,
            "mean_raw_token_f1_delta": average(pair["raw_token_f1_delta"] for pair in pairs),
            "mean_raw_exact_match_delta": average(pair["raw_exact_match_delta"] for pair in pairs),
            "question_pairs": pairs,
            "semantic_message_value_interpretation_requires_manual_format_review": True,
        }
    return result


def analyze(folder):
    folder = folder.resolve()
    assert folder.is_relative_to((ROOT / "runs").resolve()), "run must remain inside this experiment"
    manifest_path, summary_path = folder / "manifest.json", folder / "summary.json"
    manifest, summary = read(manifest_path), read(summary_path)
    assert manifest["expected_counts"] == {"generate": 76, "sender_prefill": 7}
    assert manifest["expected_phase_counts"] == EXPECTED_PHASE_COUNTS
    assert summary["counts"] == manifest["expected_counts"]
    assert summary["phase_counts"] == EXPECTED_PHASE_COUNTS
    assert tuple(manifest["methods"]) == METHODS
    budget = manifest["answer_max_new_tokens"]
    assert budget == 48
    qids = manifest["selected_question_ids"]
    assert len(qids) == len(set(qids)) == 6
    input_path = (ROOT / manifest["input_file"]).resolve()
    assert input_path.is_relative_to(ROOT.resolve())
    assert sha(input_path) == manifest["input_sha256"]
    frozen = read(input_path)
    assert {sender["question_id"] for sender in frozen["senders"]} == set(qids)
    assert len(frozen["senders"]) == 6 and len(frozen["receivers"]) == 18
    paths = sorted((folder / "items").glob("*.json"))
    rows = [read(path) for path in paths]
    main = [row for row in rows if row["phase"] == "main"]
    warmup = [row for row in rows if row["phase"] == "warmup"]
    assert len(rows) == len(main) + len(warmup) == 19
    assert len(main) == 18 and len(warmup) == 1
    assert dict(Counter(row["condition"] for row in main)) == {condition: 6 for condition in CONDITIONS}
    assert len({row["key"] for row in main}) == 18
    index = {(row["question_id"], row["condition"]): row for row in main}
    assert set(index) == {(qid, condition) for qid in qids for condition in CONDITIONS}
    assert {row["key"] for row in main} == {row["key"] for row in frozen["receivers"]}
    assert warmup[0]["key"] == manifest["warmup_key"]

    source_paths = sorted((folder / "sources").glob("*.json"))
    sources = [read(path) for path in source_paths]
    assert len(sources) == 7
    source_index = {(source["phase"], source["question_id"]): source for source in sources}
    expected_sources = {("main", qid) for qid in qids} | {("warmup", warmup[0]["question_id"])}
    assert len(source_index) == 7 and set(source_index) == expected_sources
    for source in sources:
        event = source["sender_prefill"]
        assert event["phase"] == source["phase"] and event["kind"] == "sender_prefill"
        assert event["model_forward_calls"] == 1 and event["seconds"] >= 0
        assert event.get("status", "ok") == "ok"
        for packet in ("text_packet", "kv_packet", "material_packet"):
            value = source[packet]
            assert value["payload_bytes"] == value["payload_content_bytes"] + value["payload_header_bytes"]
            assert value["encode_seconds"] >= 0

    events_path = folder / "calls.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert all(event.get("status", "ok") == "ok" for event in events)
    assert dict(Counter(event["kind"] for event in events)) == manifest["expected_counts"]
    phase_counts = dict(Counter(event["phase"] + "_" + event["kind"] for event in events))
    assert phase_counts == EXPECTED_PHASE_COUNTS

    labels_path = ROOT / "data/receiver_evidence_20261002/scoring_labels.json"
    labels = read(labels_path)
    gold = {label["question_id"]: label["answer"] for label in labels}
    assert len(labels) == len(gold) == 20 and set(qids).issubset(gold)
    normalize, token_f1, scorer = existing_scorer()
    scores, forward_calls, agreements = {}, 0, []
    eos = manifest.get("eos_token_id", manifest.get("generation_kwargs", {}).get("eos_token_id"))
    assert eos is not None, "EOS IDs must be recorded"
    eos = eos if isinstance(eos, list) else [eos]
    packet_name = {"token_full": "text_packet", "kv_full": "kv_packet", "material_text": "material_packet"}
    for row in rows:
        assert row["source_cache_unchanged"] is True
        source = source_index[(row["phase"], row["question_id"])]
        assert row["cache_source_hash"] == source["cache_source_hash"]
        assert set(row["methods"]) == set(METHODS)
        for method, value in row["methods"].items():
            ids = value["output_ids"]
            assert ids and all(type(token) is int for token in ids)
            assert value["generated_tokens"] == len(ids) <= budget
            assert value["model_forward_calls"] == len(ids)
            assert type(value["eos_stopped"]) is bool and type(value["actually_truncated"]) is bool
            assert value["eos_stopped"] == (ids[-1] in eos)
            assert value["actually_truncated"] == (len(ids) >= budget and ids[-1] not in eos)
            assert isinstance(value["response"], str)
            forward_calls += value["model_forward_calls"]
            for field in COST_FIELDS:
                assert type(value[field]) in (int, float) and math.isfinite(value[field]) and value[field] >= 0
            for field in ("payload_bytes", "payload_content_bytes", "payload_header_bytes"):
                assert type(value[field]) is int and value[field] >= 0
            assert value["payload_bytes"] == value["payload_content_bytes"] + value["payload_header_bytes"]
            if method == "no_msg":
                assert value["payload_bytes"] == value["packet_encode_seconds"] == value["packet_restore_seconds"] == 0
            else:
                packet = source[packet_name[method]]
                for field in ("payload_bytes", "payload_content_bytes", "payload_header_bytes"):
                    assert value[field] == packet[field]
                assert value["packet_encode_seconds"] == packet["encode_seconds"]
            sender_seconds = source["sender_prefill"]["seconds"] if method == "kv_full" else 0
            assert value["sender_prefill_seconds"] == sender_seconds, "text/no-message must not pay A forward"
            preexisting = value["receiver_model_seconds"] + value["packet_encode_seconds"] + value["packet_restore_seconds"]
            standalone = preexisting + sender_seconds
            assert math.isclose(value["preexisting_cache_stage_sum_seconds"], preexisting, rel_tol=1e-7, abs_tol=1e-9)
            assert math.isclose(value["from_scratch_stage_sum_seconds"], standalone, rel_tol=1e-7, abs_tol=1e-9)
            if row["phase"] == "main":
                scores[(row["question_id"], row["condition"], method)] = {
                    "exact_match": normalize(value["response"]) == normalize(gold[row["question_id"]]),
                    "token_f1": token_f1(value["response"], gold[row["question_id"]]),
                    "normalized_unknown": normalize(value["response"]) == "unknown",
                }
        if row["phase"] == "main":
            agreements.append({"question_id": row["question_id"], "condition": row["condition"],
                               "matched": row["methods"]["token_full"]["output_ids"] == row["methods"]["kv_full"]["output_ids"]})
    assert forward_calls + 7 == sum(event["model_forward_calls"] for event in events)
    grouped = {condition: summary_for([index[(qid, condition)] for qid in qids], scores) for condition in CONDITIONS}
    per_question = [{"question_id": qid, "conditions": {
        condition: {method: {"response": index[(qid, condition)]["methods"][method]["response"],
                            **scores[(qid, condition, method)]} for method in METHODS}
        for condition in CONDITIONS}} for qid in qids]
    actual_main_prefill = sum(source_index[("main", qid)]["sender_prefill"]["seconds"] for qid in qids)
    result = {
        "run_name": folder.name, "structural_and_call_audit_passed": True,
        "existing_scorer": scorer, "question_ids": qids, "independent_questions": 6,
        "main_receiver_views": 18, "main_answers": 72, "warmup_answers_excluded_from_quality": 4,
        "call_counts": summary["counts"], "phase_counts": phase_counts,
        "same_six_question_paired_conditions": grouped, "per_question_raw_responses_and_scores": per_question,
        "manual_format_review": {"completed_by_this_script": False,
            "required_before_semantic_message_value_conclusion": True,
            "no_token_length_or_gold_extraction_gate": True,
            "review_categories": ["answer phrase", "whole sentence or explanation", "abstention", "other"]},
        "token_kv_output_token_agreement": {"matched": sum(pair["matched"] for pair in agreements),
            "compared_views": 18, "independent_questions": 6, "diagnostic_only_not_a_gate": True,
            "pairs": agreements},
        "source_prefill_allocation": {"actual_main_sources": 6,
            "actual_main_source_seconds": actual_main_prefill,
            "counterfactual_independent_kv_requests": 18,
            "counterfactual_recomputed_source_stage_sum_seconds": sum(row["methods"]["kv_full"]["sender_prefill_seconds"] for row in main),
            "not_an_independently_observed_wall_time": True},
        "source_hashes": {"manifest": sha(manifest_path), "summary": sha(summary_path),
            "inputs": sha(input_path), "scoring_labels": sha(labels_path), "calls": sha(events_path),
            "analysis_code": sha(Path(__file__)),
            "item_files": {path.name: sha(path) for path in paths},
            "source_files": {path.name: sha(path) for path in source_paths}},
        "limits": [
            "six previously used public support-sentence calibration questions, not a confirmatory benchmark",
            "raw response EM/F1 use the existing scorer; no gold-span extraction or replacement scores",
            "normalized unknown is a descriptive abstention count, not proof of calibrated uncertainty or factual correctness",
            "manual per-item answer-format review is required; this script does not automatically infer answer-only compliance",
            "system rule, A representation and chat layout changed together; not causal isolation of the previous formatting issue",
            "A only prefills; original generated node-output text communication is untested",
            "canonical token/KV share exact input arrays; material_text has a different strong text workflow",
            "whole A prefix includes system/question/framing, not just previously missing facts",
            "held material stays visible; full source material is not oracle-pruned",
            "material_text payload_header_bytes includes JSON metadata and escaping overhead above raw UTF-8 material",
            "timings start from prepared tokens, exclude prior tokenization uniformly, and include lightweight recording",
            "single-pass stage sums and preexisting-cache scenarios do not establish speedup, network acceleration or compression",
        ],
    }
    write(folder / "analysis.json", result)
    report = ["# 六题短答案检查", "",
        "这轮只检查原同组6道校准题。缺少、部分、全部持有各6份输入，四方法共72次主回答；4次预热不计质量。A主prefill6次，预热1次。", "",
        "所有分数直接用原始回答和既有EM/F1函数，没有按金标抽答案。unknown只统计为弃答，不等于判断模型不可能知道答案。", "",
        "| 持有状态 | 方法 | raw F1 | raw EM | 相对无消息F1差 | unknown | EOS/触顶 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    names = {"missing": "缺少", "partial": "部分", "full": "全部"}
    methods = {"no_msg": "无消息", "token_full": "精确token全文", "kv_full": "完整KV", "material_text": "完整材料文本"}
    for condition, group in grouped.items():
        for method in METHODS:
            value = group["methods"][method]
            gain = "—" if method == "no_msg" else f'{100*group["message_vs_no_message"][method]["mean_raw_token_f1_delta"]:+.2f}'
            report.append(f'| {names[condition]} | {methods[method]} | {100*value["raw_mean_token_f1"]:.2f} | '
                          f'{100*value["raw_exact_match_rate"]:.2f} | {gain} | {value["normalized_unknown_abstention_count"]} | '
                          f'{value["eos_stopped_count"]}/{value["actually_truncated_count"]} |')
    agreement = result["token_kv_output_token_agreement"]
    report += ["", f'精确token/KV回答逐token相同{agreement["matched"]}/18份，仅作描述，不能当质量或逐字门禁。', "",
        "**还需要人工逐题复核格式。** 本脚本不根据长度或gold-span宣布答案短语合格。若不同方法仍系统性整句/短语不同，不能据主分数推断语义消息价值。逐题回答已放analysis.json，人工结论应另存。", "",
        "完整材料文本是一个正常B输入的强工作流对照；它与canonical token/KV的token布局不同。A没有生成分析，前缀仍含system/问题/来源标记，不能把全部差异归因于缺失事实。", "",
        "## 成本边界", "",
        "文本和无消息不做A模型prefill。KV实际复用6份新来源cache服务18个接收视图；analysis同时列18个独立请求重做A的阶段估算。已有cache场景另列，不让来源cache创建成本消失。", "",
        "消息包使用实际编码字节；完整材料文本的payload_header_bytes是JSON元数据及转义增加的开销，不是裸前缀协议头。token/KV则按各自真实内容和头记录；KV包仍含来源token IDs。", "",
        "计时从预先冻结token数组开始，所有此前tokenization统一排除。单次轻量记录和阶段相加不等于独立端到端运行；没有真实网络，不宣称速度、网络加速或压缩收益。", "",
        "这只是六题短支持句校准。新规则、A表示和chat布局一起变化，不能证明此前格式问题的单一原因；也未完成生成消息、接收方选择或去重算法评测。", ""]
    (folder / "report.md").write_text("\n".join(report), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    result = analyze(ROOT / "runs" / args.run_name)
    print(json.dumps({"run_name": result["run_name"], "audit_passed": True,
                      "questions": 6, "main_views": 18, "call_counts": result["call_counts"],
                      "manual_format_review_still_required": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
