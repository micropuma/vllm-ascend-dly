#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def read_predictions(path: Path):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            example_id = str(row["example_id"])
            if example_id in rows:
                raise ValueError(f"duplicate example_id {example_id!r} in {path}:{line_number}")
            for key in ("prediction", "extracted_answer", "gold_answer", "correct"):
                if key not in row:
                    raise ValueError(f"missing {key!r} in {path}:{line_number}")
            if not isinstance(row["correct"], bool):
                raise ValueError(f"correct must be boolean in {path}:{line_number}")
            rows[example_id] = row
    return rows


def exact_binomial_two_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    log_probs = [
        math.lgamma(trials + 1) - math.lgamma(k + 1) - math.lgamma(trials - k + 1) - trials * math.log(2.0)
        for k in range(tail + 1)
    ]
    maximum = max(log_probs)
    probability = 2.0 * math.exp(maximum) * sum(math.exp(value - maximum) for value in log_probs)
    return min(1.0, probability)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of normalized GSM8K prediction JSONL files.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--details", type=Path)
    parser.add_argument("--expect-count", type=int, default=1319)
    parser.add_argument("--noninferiority-margin-pp", type=float, default=-1.0)
    parser.add_argument("--require-token-ids", action="store_true")
    args = parser.parse_args()

    baseline = read_predictions(args.baseline)
    candidate = read_predictions(args.candidate)
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())
        extra = sorted(candidate.keys() - baseline.keys())
        raise SystemExit(f"sample IDs differ: missing candidate={missing[:10]}, extra candidate={extra[:10]}")
    if len(baseline) != args.expect_count:
        raise SystemExit(f"expected {args.expect_count} paired examples, found {len(baseline)}")

    b0_correct = sum(bool(row["correct"]) for row in baseline.values())
    cand_correct = sum(bool(row["correct"]) for row in candidate.values())
    b_only = c_only = text_equal = token_equal = token_comparable = 0
    details = []
    for example_id in baseline:
        b, c = baseline[example_id], candidate[example_id]
        if b["gold_answer"] != c["gold_answer"]:
            raise SystemExit(f"gold answer differs for sample {example_id}")
        b_ok, c_ok = bool(b["correct"]), bool(c["correct"])
        b_only += b_ok and not c_ok
        c_only += c_ok and not b_ok
        same_text = b["prediction"] == c["prediction"]
        text_equal += same_text
        has_tokens = "token_ids" in b and "token_ids" in c
        token_comparable += has_tokens
        b_tokens = b.get("token_ids", [])
        c_tokens = c.get("token_ids", [])
        same_tokens = has_tokens and b_tokens == c_tokens
        token_equal += same_tokens
        first_token_difference = None
        if has_tokens and not same_tokens:
            first_token_difference = next(
                (i for i, (left, right) in enumerate(zip(b_tokens, c_tokens)) if left != right),
                min(len(b_tokens), len(c_tokens)),
            )
        details.append({
            "example_id": example_id,
            "baseline_correct": b_ok,
            "candidate_correct": c_ok,
            "prediction_equal": same_text,
            "token_ids_comparable": has_tokens,
            "token_ids_equal": same_tokens if has_tokens else None,
            "first_token_difference": first_token_difference,
            "baseline_token_count": len(b_tokens) if has_tokens else None,
            "candidate_token_count": len(c_tokens) if has_tokens else None,
            "baseline_text_chars": len(b["prediction"]),
            "candidate_text_chars": len(c["prediction"]),
            "extracted_answer_equal": b["extracted_answer"] == c["extracted_answer"],
            "baseline_extracted_answer": b["extracted_answer"],
            "candidate_extracted_answer": c["extracted_answer"],
            "baseline_finish_reason": b.get("finish_reason"),
            "candidate_finish_reason": c.get("finish_reason"),
        })
    if args.require_token_ids and token_comparable != len(baseline):
        raise SystemExit(f"token IDs missing for {len(baseline) - token_comparable} paired samples")

    count = len(baseline)
    summary = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "paired_examples": count,
        "baseline_correct": b0_correct,
        "candidate_correct": cand_correct,
        "baseline_accuracy": b0_correct / count,
        "candidate_accuracy": cand_correct / count,
        "accuracy_delta_percentage_points": 100.0 * (cand_correct - b0_correct) / count,
        "noninferiority_margin_percentage_points": args.noninferiority_margin_pp,
        "score_gate_pass": 100.0 * (cand_correct - b0_correct) / count >= args.noninferiority_margin_pp,
        "baseline_only_correct": b_only,
        "candidate_only_correct": c_only,
        "mcnemar_exact_two_sided_p": exact_binomial_two_sided(c_only, b_only + c_only),
        "prediction_text_equal_count": text_equal,
        "token_ids_comparable_count": token_comparable,
        "token_ids_equal_count": token_equal,
    }
    print(json.dumps(summary, indent=2))
    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        with args.details.open("x", encoding="utf-8") as stream:
            for row in details:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
