#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


root = Path(sys.argv[1])
results = []
for dataset in ("ceval", "gsm8k"):
    base = root / dataset
    json_files = sorted(base.rglob("*.json"))
    summaries = []
    for path in json_files:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and ("accuracy" in value or "ceval" in str(path).lower()):
            summaries.append({"file": str(path.relative_to(root)), "data": value})
    text_files = sorted(base.rglob("*.txt"))
    score_lines = []
    for path in text_files:
        text = path.read_text(errors="replace")
        for line in text.splitlines():
            if re.search(r"accuracy|score|ceval|gsm8k", line, re.I):
                score_lines.append(line.strip())
    results.append({
        "dataset": dataset,
        "exit_code": (base / "exit_code").read_text().strip() if (base / "exit_code").exists() else None,
        "json_summaries": summaries,
        "score_lines": score_lines[-30:],
    })

out = {"run_dir": str(root), "datasets": results}
(root / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
for result in results:
    print(json.dumps(result, ensure_ascii=False))
