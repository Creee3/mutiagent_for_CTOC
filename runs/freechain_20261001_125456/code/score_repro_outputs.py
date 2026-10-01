import argparse
from collections import Counter, defaultdict
import json
import re
import string

from utils import f1_match


def normalize(text):
    text = str(text).lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def hotpot_token_f1(prediction, gold):
    prediction = normalize(prediction)
    gold = normalize(gold)
    for special in ("yes", "no", "noanswer"):
        if (prediction == special or gold == special) and prediction != gold:
            return 0.0
    prediction_tokens, gold_tokens = prediction.split(), gold.split()
    common = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        row["standard_token_f1"] = hotpot_token_f1(row["response"], row["gold"])
        row["upstream_f1_gt_half_pass"] = bool(f1_match(row["gold"], row["response"]))
        row["exact_match"] = normalize(row["response"]) == normalize(row["gold"])
        grouped[(row["phase"], row["model_label"])].append(row)
    summary = {}
    for (phase, model_label), group in grouped.items():
        summary[f"{phase}/{model_label}"] = {
            "count": len(group),
            "mean_standard_token_f1": sum(row["standard_token_f1"] for row in group) / len(group),
            "upstream_f1_gt_half_pass_rate": sum(row["upstream_f1_gt_half_pass"] for row in group) / len(group),
            "exact_match_rate": sum(row["exact_match"] for row in group) / len(group),
            "inference_seconds": sum(row["seconds"] for row in group),
        }
    result = {"summary": summary, "per_item": rows}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
