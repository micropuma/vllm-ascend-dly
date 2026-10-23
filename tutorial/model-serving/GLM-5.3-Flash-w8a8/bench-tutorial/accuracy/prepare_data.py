#!/usr/bin/env python3
import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


REPO = Path("/workspace/vllm-ascend-dly-git")
DATASETS = REPO / "tutorial/model-serving/datasets"
ROOT = REPO / "tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/accuracy"
PREPARED = ROOT / "data-prepared"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_ceval():
    source = DATASETS / "ceval-exam"
    target = PREPARED / "ceval"
    fields = ["question", "A", "B", "C", "D", "answer", "explanation"]
    manifest = {"source": str(source), "splits": {}, "files": []}
    subjects = sorted(path.name for path in source.iterdir() if path.is_dir())
    for split in ("dev", "val", "test"):
        split_dir = target / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_rows = 0
        split_labeled = 0
        for subject in subjects:
            candidates = sorted((source / subject).glob(f"{split}-*.parquet"))
            rows = []
            for parquet in candidates:
                rows.extend(pq.read_table(parquet).to_pylist())
            out = split_dir / f"{subject}_{split}.csv"
            with out.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    normalized = {key: row.get(key, "") or "" for key in fields}
                    writer.writerow(normalized)
                    split_labeled += bool(str(normalized["answer"]).strip())
            split_rows += len(rows)
            manifest["files"].append({
                "source": [str(path) for path in candidates],
                "output": str(out),
                "rows": len(rows),
                "sha256": sha256(out),
            })
        manifest["splits"][split] = {"rows": split_rows, "labeled": split_labeled}
    manifest["subjects"] = subjects
    (target / "conversion-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def convert_gsm8k():
    source = DATASETS / "gsm8k/main"
    target = PREPARED / "gsm8k"
    target.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(source), "splits": {}}
    for split in ("train", "test"):
        files = sorted(source.glob(f"{split}-*.parquet"))
        rows = []
        for parquet in files:
            rows.extend(pq.read_table(parquet, columns=["question", "answer"]).to_pylist())
        out = target / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "rows": len(rows),
            "files": [{"source": str(path), "sha256": sha256(path)} for path in files],
            "output": str(out),
            "output_sha256": sha256(out),
        }
    (target / "conversion-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    convert_ceval()
    convert_gsm8k()
    print(f"prepared local datasets under {PREPARED}")
