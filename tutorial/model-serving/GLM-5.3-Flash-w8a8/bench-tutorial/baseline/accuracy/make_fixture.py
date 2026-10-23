#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the first GSM8K examples in prepared dataset order.")
    parser.add_argument("--input", required=True, type=Path, help="prepared GSM8K test.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--expected-total", type=int, default=1319)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite fixture: {args.output}")

    rows = []
    total_rows = 0
    with args.input.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            total_rows += 1
            if index >= args.count:
                continue
            example = json.loads(line)
            rows.append({
                "example_id": f"gsm8k-main-test-{index:04d}",
                "dataset_row": index,
                "question": example["question"],
                "gold_answer": example["answer"],
            })
    if len(rows) != args.count:
        raise SystemExit(f"expected {args.count} rows, found {len(rows)}")
    if total_rows != args.expected_total:
        raise SystemExit(f"expected {args.expected_total} dataset rows, found {total_rows}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "fixture": str(args.output),
        "count": len(rows),
        "dataset_total_rows": total_rows,
        "selection": "first rows in prepared GSM8K main/test order",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "fixture_sha256": sha256(args.output),
        "prompt_and_token_ids": "must be added by the tokenizer/model-runner capture adapter",
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
