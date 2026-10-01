"""Analyze the frozen receiver-material endpoints without loading a model.

Scores call the project's existing normalize/hotpot_token_f1 definitions. Their
AST nodes are loaded unchanged so the scorer's unrelated upstream utils import
does not make this standard-library analysis import Torch or NLTK.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import string

ROOT = Path(__file__).resolve().parent
METHODS = ("no_msg", "token_full", "kv_full")
CONDITIONS = ("missing", "partial", "full")
CONTROL_SUFFIX = "_other_document_control"
EXPECTED_CONDITIONS = {
    "missing": 20, "partial": 6, "full": 20,
    "partial_other_document_control": 6, "full_other_document_control": 20,
}
COST_FIELDS = (
    "receiver_model_seconds", "receiver_prefill_gpu_seconds",
    "receiver_decode_gpu_seconds", "sender_prefill_seconds",
    "packet_encode_seconds", "packet_restore_seconds",
    "from_scratch_stage_sum_seconds", "preexisting_cache_stage_sum_seconds",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def existing_scorer():
    source = ROOT / "score_repro_outputs.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names = ("normalize", "hotpot_token_f1")
    functions = [node for node in parsed.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    assert len(functions) == len(names), "ambiguous existing scorer definitions"
    namespace = {"Counter": Counter, "re": re, "string": string}
    # Execute exactly the existing function definitions, without its unused
    # top-level utils import. No scoring formula is reimplemented here.
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["normalize"], namespace["hotpot_token_f1"], {
        "path": str(source), "sha256": sha(source),
        "functions": list(names), "source_definitions_executed_unchanged": True,
    }


def average(values):
    values = list(values)
    return statistics.mean(values) if values else None


def check_counts(counts, origin):
    assert isinstance(counts, dict), f"{origin}: counts must be a mapping"
    source_count = sum(counts.get(name, 0) for name in ("source_prefill", "sender_prefill"))
    assert counts.get("generate") == 219, f"{origin}: expected 219 generate calls, got {counts}"
    assert source_count == 21, f"{origin}: expected 21 source prefill calls, got {counts}"
    return {"generate": counts["generate"], "source_prefill": source_count,
            "raw_counts": counts}


def group_summary(rows, score_lookup):
    assert len(rows) == len({row["question_id"] for row in rows})
    result = {"independent_question_count": len(rows), "methods": {}, "message_vs_no_message": {}}
    for method in METHODS:
        values = [row["methods"][method] for row in rows]
        scores = [score_lookup[(row["question_id"], row["condition"], method)] for row in rows]
        result["methods"][method] = {
            "mean_token_f1": average(score["token_f1"] for score in scores),
            "exact_match_rate": average(score["exact_match"] for score in scores),
            "eos_stopped_count": sum(value["eos_stopped"] for value in values),
            "actually_truncated_count": sum(value["actually_truncated"] for value in values),
            "mean_generated_tokens": average(value["generated_tokens"] for value in values),
            "mean_payload_bytes": average(value["payload_bytes"] for value in values),
            "mean_payload_content_bytes": average(value["payload_content_bytes"] for value in values),
            "mean_payload_header_bytes": average(value["payload_header_bytes"] for value in values),
            "single_pass_cost_means_seconds": {
                field: average(value[field] for value in values) for field in COST_FIELDS},
        }
    for method in METHODS[1:]:
        pairs = []
        for row in rows:
            key = (row["question_id"], row["condition"])
            message, no_message = score_lookup[(*key, method)], score_lookup[(*key, "no_msg")]
            pairs.append({"question_id": row["question_id"],
                "token_f1_delta": message["token_f1"] - no_message["token_f1"],
                "exact_match_delta": int(message["exact_match"]) - int(no_message["exact_match"])})
        result["message_vs_no_message"][method] = {
            "paired_question_count": len(pairs),
            "mean_token_f1_delta": average(pair["token_f1_delta"] for pair in pairs),
            "mean_exact_match_delta": average(pair["exact_match_delta"] for pair in pairs),
            "question_pairs": pairs,
        }
    result["text_kv_output_token_agreement"] = {
        "matched": sum(row["methods"]["token_full"]["output_ids"] ==
                       row["methods"]["kv_full"]["output_ids"] for row in rows),
        "compared": len(rows), "diagnostic_only_not_a_gate": True,
    }
    return result


def analyze(folder):
    folder = folder.resolve()
    assert folder.is_relative_to((ROOT / "runs").resolve()), "run must be inside this experiment's runs"
    manifest_path, summary_path = folder / "manifest.json", folder / "summary.json"
    manifest, summary = read(manifest_path), read(summary_path)
    paths = sorted((folder / "items").glob("*.json"))
    assert paths, "no item files"
    rows = [read(path) for path in paths]
    main = [row for row in rows if row["phase"] == "main"]
    warm = [row for row in rows if row["phase"] == "warmup"]
    assert len(rows) == len(main) + len(warm) == 73, "expected 72 main rows and one warmup row"
    assert len(main) == 72 and len(warm) == 1
    assert dict(Counter(row["condition"] for row in main)) == EXPECTED_CONDITIONS
    index = {(row["question_id"], row["condition"]): row for row in main}
    assert len(index) == len(main), "duplicate question/condition row"
    labels_path = ROOT / "data/receiver_evidence_20261002/scoring_labels.json"
    labels = read(labels_path)
    gold = {row["question_id"]: row["answer"] for row in labels}
    assert len(gold) == len(labels) == 20
    main_ids = {row["question_id"] for row in main}
    assert main_ids == set(gold), "main rows must use exactly the 20 frozen calibration questions"
    for condition in ("missing", "full"):
        assert {row["question_id"] for row in main if row["condition"] == condition} == main_ids
    partial_ids = sorted(row["question_id"] for row in main if row["condition"] == "partial")
    assert len(partial_ids) == 6 and set(partial_ids).issubset(main_ids)

    normalize, token_f1, scorer = existing_scorer()
    scores = {}
    cost_errors = []
    for row in rows:
        assert set(row["methods"]) == set(METHODS)
        for method, value in row["methods"].items():
            ids = value["output_ids"]
            assert ids and all(type(token) is int for token in ids)
            assert value["generated_tokens"] == len(ids) <= 48
            assert type(value["eos_stopped"]) is bool and type(value["actually_truncated"]) is bool
            assert value["actually_truncated"] == (len(ids) == 48 and not value["eos_stopped"])
            assert isinstance(value["response"], str)
            for field in COST_FIELDS:
                assert isinstance(value[field], (int, float)) and math.isfinite(value[field]) and value[field] >= 0, field
            for field in ("payload_bytes", "payload_content_bytes", "payload_header_bytes"):
                assert type(value[field]) is int and value[field] >= 0, field
            assert value["payload_bytes"] == value["payload_content_bytes"] + value["payload_header_bytes"]
            if method == "no_msg":
                assert value["payload_bytes"] == 0
            if method != "kv_full":
                assert value["sender_prefill_seconds"] == 0, "text/no-message must not pay unnecessary A prefill"
            preexisting = (value["receiver_model_seconds"] + value["packet_encode_seconds"]
                           + value["packet_restore_seconds"])
            standalone = preexisting + value["sender_prefill_seconds"]
            for field, expected in (("preexisting_cache_stage_sum_seconds", preexisting),
                                    ("from_scratch_stage_sum_seconds", standalone)):
                if not math.isclose(value[field], expected, rel_tol=1e-7, abs_tol=1e-9):
                    cost_errors.append({"question_id": row["question_id"], "condition": row["condition"],
                        "phase": row["phase"], "method": method, "field": field,
                        "reported": value[field], "sum_from_recorded_stages": expected})
            if row["phase"] == "main":
                key = (row["question_id"], row["condition"], method)
                scores[key] = {"token_f1": token_f1(value["response"], gold[row["question_id"]]),
                               "exact_match": normalize(value["response"]) == normalize(gold[row["question_id"]])}
    assert not cost_errors, f"stage-sum mismatch: {cost_errors[:3]}"

    counts = check_counts(summary["counts"], "summary")
    declared = manifest.get("expected_counts")
    if declared is not None:
        check_counts(declared, "manifest expected_counts")
    ledger = {"available": False, "independently_checked": False,
              "limitation": "summary counts alone are not an independent call ledger"}
    ledger_path = folder / "calls.jsonl"
    if ledger_path.exists():
        events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert all(event.get("status", "ok") == "ok" for event in events), "failed ledger event"
        ledger_counts = dict(Counter(event["kind"] for event in events))
        verified = check_counts(ledger_counts, "calls.jsonl")
        assert verified["generate"] == counts["generate"] and verified["source_prefill"] == counts["source_prefill"]
        ledger = {"available": True, "independently_checked": True, "sha256": sha(ledger_path), **verified}

    primary = {condition: group_summary([row for row in main if row["condition"] == condition], scores)
               for condition in CONDITIONS}
    paired_three = {condition: group_summary([index[(qid, condition)] for qid in partial_ids], scores)
                    for condition in CONDITIONS}
    three_details = []
    for qid in partial_ids:
        values = {condition: {method: scores[(qid, condition, method)] for method in METHODS}
                  for condition in CONDITIONS}
        three_details.append({"question_id": qid, "conditions": values})

    controls = {}
    for condition in ("partial", "full"):
        control_condition = condition + CONTROL_SUFFIX
        control_rows = [row for row in main if row["condition"] == control_condition]
        qids = sorted(row["question_id"] for row in control_rows)
        assert qids == sorted(row["question_id"] for row in main if row["condition"] == condition)
        pairs = []
        for qid in qids:
            per_method = {}
            for method in METHODS:
                held, other = scores[(qid, condition, method)], scores[(qid, control_condition, method)]
                entry = {"held_minus_other_token_f1": held["token_f1"] - other["token_f1"],
                         "held_minus_other_exact_match": int(held["exact_match"]) - int(other["exact_match"])}
                if method != "no_msg":
                    held_gain = held["token_f1"] - scores[(qid, condition, "no_msg")]["token_f1"]
                    other_gain = other["token_f1"] - scores[(qid, control_condition, "no_msg")]["token_f1"]
                    entry["held_minus_other_message_gain_token_f1"] = held_gain - other_gain
                per_method[method] = entry
            pairs.append({"question_id": qid, "methods": per_method})
        controls[condition] = {
            "paired_question_count": len(qids),
            "other_material_summary": group_summary(control_rows, scores),
            "held_minus_other_means": {
                method: {field: average(pair["methods"][method][field] for pair in pairs)
                         for field in pairs[0]["methods"][method]}
                for method in METHODS}, "question_pairs": pairs,
            "semantic_irrelevance_verified": False,
        }

    # The 72 receiver states reuse the same 20 sources. This stage allocation
    # is a counterfactual per-request estimate, not the actual run wall time.
    shared_source = []
    for qid in sorted(main_ids):
        entries = [row["methods"]["kv_full"]["sender_prefill_seconds"] for row in main if row["question_id"] == qid]
        assert all(math.isclose(value, entries[0], rel_tol=1e-7, abs_tol=1e-9) for value in entries), "fixed source timing differs across states"
        shared_source.append(entries[0])
    agreements = [{"question_id": row["question_id"], "condition": row["condition"],
                   "matched": row["methods"]["token_full"]["output_ids"] == row["methods"]["kv_full"]["output_ids"]}
                  for row in main]
    result = {
        "run_name": folder.name, "input_audit_passed": True,
        "source_hashes": {"manifest": sha(manifest_path), "summary": sha(summary_path),
                          "scoring_labels": sha(labels_path), "analysis_code": sha(Path(__file__)),
                          "items": {path.name: sha(path) for path in paths}},
        "existing_scorer": scorer, "main_questions": 20, "main_receiver_views": 72,
        "main_generate_calls": 216, "warmup_rows_excluded_from_quality": 1,
        "reported_call_counts": counts, "call_ledger_audit": ledger,
        "primary_by_receiver_condition": primary,
        "paired_three_state_same_six": {"question_ids": partial_ids, "conditions": paired_three,
                                        "per_question": three_details},
        "length_matched_other_material_controls": controls,
        "text_kv_output_token_agreement": {"matched": sum(value["matched"] for value in agreements),
            "compared_receiver_views": 72, "independent_questions": 20,
            "diagnostic_only_not_a_gate": True, "per_receiver_view": agreements},
        "main_source_prefill_allocation": {
            "actually_computed_unique_sources": 20,
            "sum_unique_source_prefill_seconds": sum(shared_source),
            "counterfactual_per_request_source_prefill_sum_seconds":
                sum(row["methods"]["kv_full"]["sender_prefill_seconds"] for row in main),
            "independent_kv_requests": 72,
            "not_the_measured_actual_run_wall_time": True,
        },
        "per_receiver_scores": [{"question_id": row["question_id"], "condition": row["condition"],
            "methods": {method: scores[(row["question_id"], row["condition"], method)] for method in METHODS}}
            for row in main],
        "answer_format": {
            "scored_response": "entire decoded response",
            "gold_aware_answer_extraction_used": False,
            "response_format_review_required_before_semantic_utility_claim": True,
        },
        "limits": [
            "raw EM/F1 scores the entire answer; sentence packaging versus answer-only output can confound semantic utility",
            "20 public calibration questions with support sentences; not a confirmatory or long-context benchmark",
            "three receiver states are paired on the same six questions; 72 views are not independent questions",
            "message is the whole A prefix containing role, question and material, not only missing facts",
            "A only prefills; no sender-generated message or original node-output text baseline was tested",
            "receiver-visible material is not proof of correct understanding or independently cached private state",
            "other-question materials are length controls, not certified semantically irrelevant evidence",
            "single-pass instrumented timings are descriptive; no robust speedup or network latency claim",
            "encoded payload bytes include headers; no message selection/compression or network transfer was tested",
            "from-prompt-ready stage sums exclude earlier tokenization/model loading and are not full wall latency",
            "preexisting-cache scenario defers source production cost; it does not make cache creation free",
            "text/KV token agreement is descriptive and not a per-token/floating-point pass gate",
        ],
    }
    write(folder / "analysis.json", result)
    report = ["# 接收方材料通信：本轮结果", "",
              "只用了20道公开校准题。72份接收输入产生216次主回答；另有3次预热回答和21次发送方prefill（主实验20次、预热1次）。预热不计质量。", "",
              "这轮先看完整A前缀有什么作用。A没有生成分析；完整消息包含角色、题目和材料，分数变化不能全部归因于新增事实。", "",
              "下面是整段回答的原始EM/F1，没有用金标提取答案。整句作答与只输出答案短语会影响分数；先查逐题回答格式，再判断信息是否有用。本轮格式问题的例子见项目根目录《完整消息价值对照_结果_20261002.md》。", "",
              "## 缺少、部分持有、全部持有", "",
              "表中F1/EM为百分数，增量是同题完整消息减无额外消息。部分持有只有6题，不能和20题均值当作同组比较。", "",
              "| 接收状态 | 题数 | 方法 | F1 | EM | F1增量 | EOS/触顶 |", "| --- | ---: | --- | ---: | ---: | ---: | --- |"]
    names = {"missing": "缺少", "partial": "部分", "full": "全部"}
    method_names = {"no_msg": "无额外消息", "token_full": "完整token文本", "kv_full": "完整KV"}
    for condition in CONDITIONS:
        group = primary[condition]
        for method in METHODS:
            value = group["methods"][method]
            gain = "—" if method == "no_msg" else f'{100 * group["message_vs_no_message"][method]["mean_token_f1_delta"]:+.2f}'
            report.append(f'| {names[condition]} | {group["independent_question_count"]} | {method_names[method]} | '
                          f'{100 * value["mean_token_f1"]:.2f} | {100 * value["exact_match_rate"]:.2f} | {gain} | '
                          f'{value["eos_stopped_count"]}/{value["actually_truncated_count"]} |')
    report += ["", "三状态严格配对同一组6题：", "",
               "| 接收状态 | 无消息F1 | 完整token文本F1 | 完整KV F1 |", "| --- | ---: | ---: | ---: |"]
    for condition in CONDITIONS:
        values = paired_three[condition]["methods"]
        report.append(f'| {names[condition]} | ' + " | ".join(f'{100 * values[method]["mean_token_f1"]:.2f}' for method in METHODS) + " |")
    report += ["", "## 等长其他题材料控制", "",
               "26对控制单独报告，不额外算成26道独立题。其他题材料可能含相关信息，也可能在半句处截断。", "",
               "| 配对 | 题数 | 方法 | 持有证据减其他材料：F1差 | 消息增益之差 |", "| --- | ---: | --- | ---: | ---: |"]
    for condition, group in controls.items():
        for method in METHODS:
            value = group["held_minus_other_means"][method]
            gain = value.get("held_minus_other_message_gain_token_f1")
            report.append(f'| {names[condition]} | {group["paired_question_count"]} | {method_names[method]} | '
                          f'{100 * value["held_minus_other_token_f1"]:+.2f} | ' +
                          ("—" if gain is None else f"{100 * gain:+.2f}") + " |")
    agreement = result["text_kv_output_token_agreement"]
    report += ["", f'文本/KV逐token相同：{agreement["matched"]}/72份输入，仅作描述，不作为运行门禁。', "",
               "## 成本怎么读", "",
               "文本不用额外A forward。KV主实验只实际生成20份固定来源cache，服务72个反事实接收视图；analysis.json同时列出每个独立请求重新A prefill的成本估算，两者不能混用。", "",
               "分项包含接收推理、包编码/恢复和A prefill；已有cache场景另列。阶段相加从prompt就绪起算，未包括此前tokenization和模型加载；也不是整个运行的wall time。单次轻量记录的GPU耗时不支持稳健加速结论。", "",
               "payload_bytes是实际编码包长度，包含包头。没有测网络，也没有运行删减选择，因此不宣称网络加速或压缩收益。无A生成意味着原版“仅传节点输出”的文本对照仍待补齐。", "",
               "逐题分数、配对差、EOS/触顶、实际包字节、成本均在同目录analysis.json；原始回答仍在items。", ""]
    (folder / "report.md").write_text("\n".join(report), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    result = analyze(ROOT / "runs" / args.run_name)
    print(json.dumps({"run_name": result["run_name"], "main_questions": result["main_questions"],
        "main_receiver_views": result["main_receiver_views"], "input_audit_passed": True,
        "reported_call_counts": result["reported_call_counts"],
        "call_ledger_independently_checked": result["call_ledger_audit"]["independently_checked"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
