#!/usr/bin/env python3
"""Aggregate metrics from tracelab_full.jsonl (all 3138 records).

1. Discover the full metric schema (keys + units).
2. Aggregate latency/throughput metrics over the 3109 normal records.
3. Report the 29 context-overflow (error_isl) rejections separately.
"""
import json
from collections import Counter, defaultdict

path = "/workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/out/agent-20261018T140121Z/tracelab_full/tracelab_full.jsonl"

recs = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            recs.append(json.loads(line))

# 1) schema discovery
key_units = defaultdict(set)
for r in recs:
    for k, v in r.get("metrics", {}).items():
        if isinstance(v, dict) and "value" in v:
            key_units[k].add(v.get("unit"))
        else:
            key_units[k].add(type(v).__name__)
print("=== metric schema (key -> units) ===")
for k in sorted(key_units):
    print(f"  {k}: {sorted(key_units[k])}")

# metadata keys
meta_keys = set()
for r in recs:
    meta_keys.update(r.get("metadata", {}).keys())
print("\n=== metadata keys ===")
print("  ", sorted(meta_keys))

def num(v):
    if isinstance(v, dict):
        return v.get("value")
    return v

def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, int(p / 100.0 * len(xs)))
    return xs[i]

# 2) split normal vs error
normal = [r for r in recs if "error_isl" not in r.get("metrics", {}) and r["metrics"].get("time_to_first_token") is not None]
err = [r for r in recs if "error_isl" in r.get("metrics", {})]
print(f"\n=== counts: total={len(recs)} normal={len(normal)} error_isl={len(err)} ===")

def agg(records, field):
    vals = [num(r["metrics"].get(field)) for r in records if num(r["metrics"].get(field)) is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "p50": pct(vals, 50),
        "p90": pct(vals, 90),
        "p99": pct(vals, 99),
        "max": max(vals),
    }

# derive TPOT (ms/token) = decode_duration / output_token_count
tpot_vals = []
itl_vals = []
for r in normal:
    m = r["metrics"]
    dd = num(m.get("decode_duration"))
    ot = num(m.get("output_token_count"))
    if dd is not None and ot:
        tpot_vals.append(dd / ot)
    itl = num(m.get("inter_token_latency")) or num(m.get("time_per_output_token"))
    if itl is not None:
        itl_vals.append(itl)

print("\n=== aggregate over normal records ===")
for field in ("time_to_first_token", "request_latency", "decode_duration",
              "input_sequence_length", "output_token_count"):
    a = agg(normal, field)
    if a:
        print(f"  {field}: n={a['n']} mean={a['mean']:.2f} p50={a['p50']:.2f} p90={a['p90']:.2f} p99={a['p99']:.2f} max={a['max']:.2f}")

if tpot_vals:
    print(f"  TPOT(derived decode_dur/out_tok): n={len(tpot_vals)} mean={sum(tpot_vals)/len(tpot_vals):.2f} p50={pct(tpot_vals,50):.2f} p90={pct(tpot_vals,90):.2f} p99={pct(tpot_vals,99):.2f} max={max(tpot_vals):.2f}")
if itl_vals:
    print(f"  ITL(present): n={len(itl_vals)} mean={sum(itl_vals)/len(itl_vals):.2f} p50={pct(itl_vals,50):.2f} p90={pct(itl_vals,90):.2f} p99={pct(itl_vals,99):.2f}")

# total wall + throughput
starts = [num(r["metadata"]["request_start_ns"]) for r in normal if num(r["metadata"].get("request_start_ns"))]
ends = [num(r["metadata"]["request_end_ns"]) for r in normal if num(r["metadata"].get("request_end_ns"))]
if starts and ends:
    wall_s = (max(ends) - min(starts)) / 1e9
    tot_in = sum(num(r["metrics"].get("input_sequence_length")) or 0 for r in normal)
    tot_out = sum(num(r["metrics"].get("output_token_count")) or 0 for r in normal)
    print(f"\n=== throughput (normal records) ===")
    print(f"  wall(s)={wall_s:.1f}  total_in_tok={tot_in}  total_out_tok={tot_out}")
    print(f"  req/s={len(normal)/wall_s:.3f}  out_tok/s={tot_out/wall_s:.2f}  in_tok/s={tot_in/wall_s:.2f}")

# prefix cache hit (if present)
for field in ("prefix_cache_hit_rate", "theoretical_prefix_cache_hit", "prompt_tokens_details_cache_hit", "cache_hit"):
    a = agg(normal, field)
    if a:
        print(f"  {field}: mean={a['mean']:.4f} p50={a['p50']:.4f}")

# 3) error_isl detail
print("\n=== error_isl (context overflow) ===")
vals = sorted(num(r["metrics"]["error_isl"]) for r in err)
print(f"  n={len(err)}  min={vals[0]}  median={vals[len(vals)//2]}  max={vals[-1]}")
print(f"  <=133120: {sum(1 for v in vals if v <= 133120)}   >133120: {sum(1 for v in vals if v > 133120)}")
print(f"  source_kind: {dict(Counter(r['metadata']['source_kind'] for r in err))}")
print(f"  agent_depth: {dict(Counter(r['metadata']['agent_depth'] for r in err))}")
