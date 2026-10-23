#!/usr/bin/env python3
"""Precisely classify every record's terminal state in tracelab_full.jsonl.

Categories:
  - in-flight: request_end_ns is None/0/missing (never got an end time)
  - error:     metrics has 'error_isl' (input-sequence-length rejection)
  - cancelled: metadata.was_cancelled is True
  - normal:    has time_to_first_token + request_latency
"""
import json
from collections import Counter

path = "/workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/out/agent-20261018T140121Z/tracelab_full/tracelab_full.jsonl"

recs = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            recs.append(json.loads(line))

print(f"total records: {len(recs)}")

inflight, err, cancelled, normal, other = [], [], [], [], []
for r in recs:
    m = r.get("metadata", {})
    mt = r.get("metrics", {})
    end = m.get("request_end_ns")
    if m.get("was_cancelled"):
        cancelled.append(r)
    elif "error_isl" in mt:
        err.append(r)
    elif end in (None, 0, ""):
        inflight.append(r)
    elif mt.get("time_to_first_token") is not None and mt.get("request_latency") is not None:
        normal.append(r)
    else:
        other.append(r)

print(f"\nin-flight (no end_ns): {len(inflight)}")
print(f"error_isl rejections:   {len(err)}")
print(f"cancelled:              {len(cancelled)}")
print(f"normal completed:       {len(normal)}")
print(f"other:                  {len(other)}")
print(f"sum: {len(inflight)+len(err)+len(cancelled)+len(normal)+len(other)}")

# dump in-flight records (the truly stuck ones)
for i, r in enumerate(inflight):
    m = r.get("metadata", {})
    print(f"\n### IN-FLIGHT #{i}")
    for k in ("session_num","conversation_id","turn_index","source_kind","agent_depth",
              "root_correlation_id","source_outer_idx","request_start_ns","request_ack_ns",
              "request_end_ns","benchmark_phase","phase_name"):
        print(f"    {k} = {m.get(k)!r}")
    print("    metrics:", {k: v for k, v in r.get("metrics", {}).items()})

# error_isl value distribution vs 133120
print("\n### error_isl values (input-seq-length at rejection):")
vals = sorted(r["metrics"]["error_isl"]["value"] for r in err)
print("  min/median/max:", vals[0], vals[len(vals)//2], vals[-1])
below = [v for v in vals if v <= 133120]
above = [v for v in vals if v > 133120]
print(f"  <=133120 (max-model-len): {len(below)}  >133120: {len(above)}")
print("  all values:", vals)

# what source_kind / agent_depth do errors hit
print("\nerror source_kind:", dict(Counter(r["metadata"]["source_kind"] for r in err)))
print("error agent_depth:", dict(Counter(r["metadata"]["agent_depth"] for r in err)))
