#!/usr/bin/env python3
"""Locate the 2 stuck in-flight requests in tracelab_full.jsonl and dump their metadata."""
import json, sys

path = "/workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/out/agent-20261018T140121Z/tracelab_full/tracelab_full.jsonl"

recs = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            recs.append(json.loads(line))

print(f"total records: {len(recs)}")

def is_stuck(r):
    m = r.get("metadata", {})
    if m.get("was_cancelled"):
        return True
    if m.get("request_end_ns") in (None, 0, ""):
        return True
    # metrics missing the two core fields => never completed
    mt = r.get("metrics", {})
    if mt.get("time_to_first_token") is None or mt.get("request_latency") is None:
        return True
    return False

stuck = [r for r in recs if is_stuck(r)]
print(f"stuck count: {len(stuck)}")

KEYS = [
    "session_num", "conversation_id", "turn_index", "source_kind",
    "agent_depth", "root_correlation_id", "x_correlation_id",
    "source_trace_id", "source_outer_idx", "was_cancelled",
    "context_overflow_skip", "request_start_ns", "request_ack_ns",
    "request_end_ns", "benchmark_phase", "phase_name",
]

for i, r in enumerate(stuck):
    m = r.get("metadata", {})
    print(f"\n===== stuck #{i} =====")
    for k in KEYS:
        print(f"  {k} = {m.get(k)!r}")
    print("  metrics keys present:", sorted(r.get("metrics", {}).keys()))
    print("  metrics values:", {k: v for k, v in r.get("metrics", {}).items() if v is not None})

# Also summarize how many total were cancelled vs complete, by source_kind
from collections import Counter
c = Counter(r["metadata"].get("source_kind") for r in recs)
print("\nsource_kind histogram:", dict(c))
dep = Counter(r["metadata"].get("agent_depth") for r in recs)
print("agent_depth histogram:", dict(dep))
