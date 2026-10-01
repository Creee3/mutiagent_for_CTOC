"""Plot verified main7b results locally; never load a model or contact the server.

Usage: python plot_results.py --run-name MAIN_RUN_NAME
Outputs go to runs/MAIN_RUN_NAME/figures. Existing outputs require --force.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
METHODS = (
    "receiver_only", "raw_context", "nld_original", "summary_oneway",
    "sampled_context", "kv_full", "kv_top50",
)
LABELS = {
    "receiver_only": "无通信",
    "raw_context": "文字：完整材料",
    "nld_original": "文字：原版 NLD",
    "summary_oneway": "文字：单向摘要",
    "sampled_context": "文字：原文采样",
    "kv_full": "KV：全部层",
    "kv_top50": "KV：校准选层",
}
EXPECTED_STAGES = {
    "calibration:generate": 20, "calibration:sender_prefill": 20,
    "warmup:generate": 10, "warmup:sender_prefill": 2,
    "evaluation:generate": 1200, "evaluation:sender_prefill": 240,
}
OUTPUT_NAMES = (
    "discussion_3panel.png", "discussion_3panel.pdf", "means_and_ci.csv",
    "all_evaluation_rows.csv", "provenance.json",
)
N_BOOTSTRAP = 10000
SEED = 42


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    require(path.is_file(), f"Missing required source: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value, field: str, *, lower=0, upper=None) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"Non-numeric {field}")
    result = float(value)
    require(math.isfinite(result) and result >= lower and
            (upper is None or result <= upper), f"Invalid {field}: {value}")
    return result


def load_verified(run_name: str):
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name)) and
            run_name not in {".", ".."}, "--run-name must be a single directory name")
    run = ROOT / "runs" / run_name
    require(run.is_dir() and not run.is_symlink(), "Run directory missing or symlinked")
    sources = [run / "COMPLETE.json", run / "manifest.json", run / "summary.json",
               ROOT / "data" / "evaluation.json"]
    complete, manifest, summary, evaluation = [read_json(p) for p in sources]
    require(complete.get("phase") == "main7b" and manifest.get("phase") == "main7b",
            "Only completed main7b runs may be plotted")
    require(summary.get("run_name") == run_name and summary.get("complete_marker") is True,
            "Summary does not identify this completed run")
    coverage = summary.get("coverage", {})
    require(coverage.get("expected_evaluation_rows") == 840 and coverage.get("found") == 840
            and coverage.get("duplicates") == 0 and coverage.get("missing") == []
            and coverage.get("unexpected") == [], "Coverage audit must be exactly 840/840")
    audit = summary.get("stage_count_audit", {})
    require(audit.get("passed") is True and audit.get("expected") == EXPECTED_STAGES
            and audit.get("recorded") == EXPECTED_STAGES,
            "Stage-count audit is missing, failed, or inconsistent")
    cfg = manifest["config"]
    require(tuple(cfg.get("methods", ())) == METHODS and cfg.get("timing_extra_passes") == 2
            and cfg.get("timing_repeat_questions") == 10, "Unrecognized experiment conditions")
    require(manifest.get("revision") == cfg.get("model_7b_revision")
            and bool(manifest.get("revision")), "Missing/mismatched 7B model revision")
    ids = [r["id"] for r in evaluation]
    require(len(ids) == 100 and len(set(ids)) == 100, "Need exactly 100 unique frozen questions")
    hashes = {r["id"]: r["question_hash"] for r in evaluation}
    require(len(set(hashes.values())) == 100, "Frozen questions contain duplicate text hashes")
    expected = {(q, method, repeat) for repeat in range(3)
                for q in (ids if repeat == 0 else ids[:10]) for method in METHODS}
    paths = sorted((run / "items").glob("*.json"))
    require(bool(paths), "No item records")
    source_rows = [(p, read_json(p)) for p in paths]
    rows = [(p, r) for p, r in source_rows if r.get("phase") == "evaluation"]
    require(len(rows) == 840, "Need exactly 840 evaluation item files")
    keys = [(r.get("id"), r.get("method"), r.get("repeat")) for _, r in rows]
    require(len(set(keys)) == 840 and set(keys) == expected,
            "Actual item coverage does not match the fixed question/method/repeat design")
    bykey = {}
    for path, row in rows:
        require(row.get("status") == "ok" and row.get("deployment") == "shared",
                f"Invalid/failed evaluation item: {path.name}")
        require(row.get("model_revision") == manifest["revision"] and
                row.get("question_hash") == hashes[row["id"]],
                f"Question/model mismatch: {path.name}")
        number(row.get("standard_token_f1"), f"F1 in {path.name}", upper=1)
        number(row.get("seconds"), f"seconds in {path.name}")
        number(row.get("payload", {}).get("logical_total_bytes"), f"bytes in {path.name}")
        require(row["payload"].get("network_transfer_measured", False) is False,
                "Unexpected network measurement; this plot describes same-GPU logical payload")
        if row["method"] == "receiver_only":
            require(row["payload"]["logical_total_bytes"] == 0,
                    "No-communication condition must have exactly zero payload")
        else:
            require(row["payload"]["logical_total_bytes"] > 0,
                    "Communicating conditions must have positive payload")
        bykey[(row["id"], row["method"], row["repeat"])] = row
    # The summary must agree with the records, rather than merely passing its flags.
    for method in METHODS:
        group = [bykey[(q, method, 0)] for q in ids]
        reported = summary.get("methods", {}).get(method, {})
        require(reported.get("questions") == 100 and reported.get("completed") == 100,
                f"Incomplete first-pass summary: {method}")
        for summary_key, row_key in (("standard_token_f1", "standard_token_f1"),
                                     ("mean_total_seconds", "seconds")):
            mean = sum(r[row_key] for r in group) / 100
            require(math.isclose(float(reported.get(summary_key, math.nan)), mean,
                                 rel_tol=1e-9, abs_tol=1e-10),
                    f"Summary/record mismatch for {method}/{summary_key}")
        payload_mean = sum(r["payload"]["logical_total_bytes"] for r in group) / 100
        require(math.isclose(float(reported.get("mean_payload_bytes", math.nan)), payload_mean,
                             rel_tol=1e-9, abs_tol=1e-10),
                f"Summary/record mismatch for {method}/payload")
    return run, manifest, summary, ids, bykey, sources + paths


def csv_text(fieldnames, rows) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def atomic_text(path: Path, content: str) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(content, encoding="utf-8", newline="")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--force", action="store_true", help="Replace previous figure outputs only")
    args = parser.parse_args()
    run, manifest, summary, ids, bykey, sources = load_verified(args.run_name)
    out = run / "figures"
    require(not out.is_symlink(), "Output directory must not be a symlink")
    if not args.force:
        require(not any((out / name).exists() for name in OUTPUT_NAMES),
                "Figure outputs exist; use --force to replace them")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    import numpy as np

    font_candidates = [Path("C:/Windows/Fonts/msyh.ttc"),
                       Path("C:/Windows/Fonts/simhei.ttf"),
                       Path("C:/Windows/Fonts/msyh.ttf")]
    font_path = next((p for p in font_candidates if p.is_file()), None)
    require(font_path is not None, "No supported Chinese font found; nothing plotted")
    fontManager.addfont(str(font_path))
    font_name = FontProperties(fname=str(font_path)).get_name()
    rng = np.random.default_rng(SEED)
    # The same sampled question IDs are used for every method and both metrics.
    indices = rng.integers(0, len(ids), size=(N_BOOTSTRAP, len(ids)))
    stats = []
    for method in METHODS:
        group = [bykey[(q, method, 0)] for q in ids]
        f1 = np.asarray([r["standard_token_f1"] for r in group], dtype=float)
        seconds = np.asarray([r["seconds"] for r in group], dtype=float)
        f1_lo, f1_hi = np.percentile(f1[indices].mean(axis=1), [2.5, 97.5])
        t_lo, t_hi = np.percentile(seconds[indices].mean(axis=1), [2.5, 97.5])
        stats.append({
            "method": method, "label_zh": LABELS[method], "n_unique_questions": 100,
            "standard_f1_mean": float(f1.mean()), "standard_f1_ci95_low": float(f1_lo),
            "standard_f1_ci95_high": float(f1_hi), "total_seconds_mean": float(seconds.mean()),
            "total_seconds_ci95_low": float(t_lo), "total_seconds_ci95_high": float(t_hi),
            "logical_payload_bytes_mean": float(np.mean([r["payload"]["logical_total_bytes"] for r in group])),
            "logical_payload_KiB_mean": float(np.mean([r["payload"]["logical_total_bytes"] for r in group]) / 1024),
        })
    colors = {"none": "#666666", "text": "#0072B2", "kv": "#B55D00"}
    shapes = {"none": "D", "text": "o", "kv": "s"}
    kind = lambda method: "none" if method == "receiver_only" else "kv" if method.startswith("kv_") else "text"
    style = {"font.family": font_name, "font.size": 11, "axes.titlesize": 13,
             "axes.labelsize": 11, "text.color": "black", "axes.labelcolor": "black",
             "xtick.color": "black", "ytick.color": "black", "axes.unicode_minus": False,
             "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
             "axes.facecolor": "white", "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(16, 6.7), layout="constrained")
        grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
        quality_ax = fig.add_subplot(grid[0, 0])
        time_ax = fig.add_subplot(grid[0, 1], sharey=quality_ax)
        message_grid = grid[0, 2].subgridspec(1, 2, width_ratios=[0.14, 1], wspace=0.03)
        zero_ax = fig.add_subplot(message_grid[0, 0], sharey=quality_ax)
        message_ax = fig.add_subplot(message_grid[0, 1], sharey=quality_ax)
        axes = (quality_ax, time_ax, zero_ax, message_ax)
        for ax in axes:
            ax.set_ylim(6.6, -0.6)
            ax.set_yticks(range(7))
            ax.tick_params(axis="y", length=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.grid(axis="y", color="#EEEEEE", linewidth=0.8)
            ax.set_axisbelow(True)
        quality_ax.set_yticklabels([LABELS[m] for m in METHODS])
        for ax in (time_ax, zero_ax, message_ax):
            ax.tick_params(axis="y", labelleft=False)
        quality_ax.set(title="(a) 答案质量", xlabel="标准 token F1（0—1）", xlim=(0, 1))
        quality_ax.set_xticks(np.linspace(0, 1, 6))
        time_ax.set(title="(b) 完整计算耗时", xlabel="秒 / 题", xlim=(0, max(s["total_seconds_ci95_high"] for s in stats) * 1.12))
        message_ax.set(title="(c) 逻辑消息大小", xlabel="KiB / 题（正值为 log10 轴）", xscale="log")
        positive = [s["logical_payload_KiB_mean"] for s in stats if s["logical_payload_KiB_mean"] > 0]
        message_ax.set_xlim(min(positive) / 1.8, max(positive) * 1.8)
        message_ax.xaxis.set_major_locator(LogLocator(base=10))
        message_ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        message_ax.minorticks_off()
        zero_ax.set_xlim(-0.5, 0.5)
        zero_ax.set_xticks([0], ["0"])
        zero_ax.set_xlabel("独立零值列", fontsize=9)
        zero_ax.spines["right"].set_visible(True)
        zero_ax.spines["right"].set_linestyle(":")
        zero_ax.spines["right"].set_color("#888888")
        for i, s in enumerate(stats):
            category = kind(s["method"])
            color, marker = colors[category], shapes[category]
            for ax, prefix in ((quality_ax, "standard_f1"), (time_ax, "total_seconds")):
                ax.hlines(i, s[f"{prefix}_ci95_low"], s[f"{prefix}_ci95_high"],
                          color=color, linewidth=1.8)
                ax.scatter(s[f"{prefix}_mean"], i, s=67, marker=marker,
                           color=color, edgecolors="black", linewidths=0.6, zorder=3)
            target = zero_ax if s["logical_payload_KiB_mean"] == 0 else message_ax
            target.scatter(s["logical_payload_KiB_mean"], i, s=67, marker=marker,
                           color=color, edgecolors="black", linewidths=0.6, zorder=3)
        quality_ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
        time_ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
        message_ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
        fig.suptitle("HotpotQA · Qwen2.5-7B-Instruct · 两节点通信对照（内部讨论）", fontsize=15)
        legend = [Line2D([], [], marker=shapes[k], color=colors[k], linestyle="none",
                         markeredgecolor="black", markersize=7, label=label)
                  for k, label in (("none", "无通信"), ("text", "文字通信"), ("kv", "KV 通信"))]
        # Direct method labels and redundant marker shapes identify all groups; leave the footer clear.
        fig.supxlabel("首遍 100 道相同题；点 = 均值；线 = 95% 题目 bootstrap 区间\n"
                      "额外重复计时未并入均值；消息含必要元数据，1 KiB = 1024 bytes；同卡计算，未测网络传输。",
                      fontsize=9)
        out.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            destination = out / f"discussion_3panel.{extension}"
            temporary = out / f".discussion_3panel.{os.getpid()}.{extension}"
            fig.savefig(temporary, format=extension, dpi=300, transparent=False)
            os.replace(temporary, destination)
        plt.close(fig)

    atomic_text(out / "means_and_ci.csv", csv_text(list(stats[0]), stats))
    raw_rows = []
    for repeat in range(3):
        for q in (ids if repeat == 0 else ids[:10]):
            for method in METHODS:
                r = bykey[(q, method, repeat)]
                raw_rows.append({"id": q, "question_hash": r["question_hash"], "method": method,
                                 "repeat": repeat, "included_in_main_means_and_ci": repeat == 0,
                                 "status": r["status"], "standard_token_f1": r["standard_token_f1"],
                                 "seconds": r["seconds"],
                                 "logical_payload_bytes": r["payload"]["logical_total_bytes"]})
    atomic_text(out / "all_evaluation_rows.csv", csv_text(list(raw_rows[0]), raw_rows))
    sources.append(Path(__file__).resolve())
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "run_name": args.run_name,
        "purpose": "internal exploratory discussion; not a publication-compliance claim",
        "source_files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "source_commit": manifest["config"].get("source_commit"),
        "model_revision": manifest["revision"], "run_manifest": manifest,
        "verification": {"phase": "main7b", "coverage": summary["coverage"],
                         "stage_count_audit": summary["stage_count_audit"],
                         "all_840_evaluation_statuses_ok": True},
        "software": {"python": sys.version, "platform": platform.platform(),
                     "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "font": {"path": str(font_path), "name": font_name, "sha256": sha256(font_path)},
        "statistics": {"estimand": "arithmetic mean over the first pass of 100 unique questions",
                       "methods": list(METHODS), "question_id_order": ids,
                       "quality_and_time_n_per_method": 100, "first_pass_rows": 700,
                       "bootstrap": {"resamples": N_BOOTSTRAP, "seed": SEED,
                                     "unit": "question", "interval": "95% percentile CI [2.5, 97.5]",
                                     "shared_indices_across_methods_and_metrics": True,
                                     "indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()},
                       "repeat_policy": "140 extra rows retained in CSV but excluded from all plotted means/CI",
                       "message_interval": "none; plotted message size is the first-pass arithmetic mean",
                       "ci_limit": "question-sampling uncertainty; not hardware-repeat variability or a significance test"},
        "units_and_semantics": {
            "quality": "standard HotpotQA token F1 in [0, 1], not author threshold pass rate",
            "time": "complete pipeline seconds from raw C/Q; separately logged instrumentation overhead removed",
            "time_exclusions": "model load and calibration excluded; both sender and receiver computation included",
            "message": "logical payload includes tensors or UTF-8 text plus necessary metadata; KiB=bytes/1024",
            "network_transfer_measured": False,
            "deployment_limit": "one shared frozen weight copy on one GPU; not original two-model memory or network speed",
        },
        "encoding": {"layout": "three aligned horizontal dot panels; message panel contains a separate zero column",
                     "quality_limits": [0, 1], "time_starts_at_zero": True,
                     "message_positive_scale": "log10", "zero_policy": "no artificial positive substitute; separate zero-only axis",
                     "colors": colors, "marker_shapes": shapes, "method_labels_zh": LABELS,
                     "text_color": "black", "dimensions_inches": [16, 6.7], "png_dpi": 300,
                     "background": "opaque white", "pdf_fonttype": 42},
        "integrity": {"raw_items_unchanged": True, "missing_values_excluded": False,
                      "visual_review_pending": True, "publisher_requirements_verified": False},
        "alt_text_zh": "三面板横向点图，比较七种两节点通信方法的标准F1、完整计算秒数与逻辑消息KiB。"
                       "前两项给出首遍100题的均值和95% bootstrap区间；消息正值用log10轴，无通信零值单独列出。"
                       "结果仅用于内部讨论，未测跨机器网络传输。",
        "output_files_sha256": {name: sha256(out / name) for name in OUTPUT_NAMES if name != "provenance.json"},
    }
    atomic_text(out / "provenance.json", json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_dir": str(out), "main_unique_questions": 100,
                      "main_rows": 700, "timing_only_extra_rows": 140,
                      "visual_review_pending": True}, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise SystemExit(f"Plot refused: {exc}") from exc
