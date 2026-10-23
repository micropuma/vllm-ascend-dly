#!/usr/bin/env python3
"""
analyze.py —— 结果 JSON → μ/ρ/达成率/三门/Little's Law 对账 → summary.md

读 AIPerf 生成的 summary JSON（GenAI-Perf 兼容结构），计算：
  - 闭环 μ(L) 曲线（μ = request_throughput.avg, 单位 requests/s），**分输入档各一条**
  - μ_max(isl) = 该档曲线最高点（并判定是否收敛/需续扫 64/96）
  - SLO 可用并发（TTFT P95 ≤ SLO 的最高并发档）
  - 开环逐点：ρ = λ/μ_max(isl)（**分档分母**）、达成率、三门
  - 可持续 λ = 全过三门的最高 λ（分档）
  - Little's Law 对账：L = 完成RPS × (TTFT_P50 + 512×ITL_P50) 回查本档闭环曲线，偏差 <5%

用法：
  analyze.py --run-id RUN_ID [--emit-rates] [--output summary.md]
  --emit-rates 产出 out/<RUN_ID>/rates.txt（每行 "isl rate1 rate2 ..."，供 run_open_loop.sh 分档读取）
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ---- 常量（与 config.env 对齐）----
OUTPUT_LEN = 512
INPUT_LENS = (8192, 32768)

# 开环布点比例（对齐 950-analysis-03 §2.2）：轻载 / 可用区下沿 / 可用区上沿 / 过饱和
RATE_FRACTIONS = (0.45, 0.70, 0.85, 1.20)

# SLO（v0）：TTFT P95 按输入档分层；ITL P99 关注项；达成率 ≥95%
SLO_TTFT_P95 = {8192: 3.0, 16384: 5.0, 32768: 8.0}
SLO_ITL_P99 = 1.0
SLO_ACHIEVE = 0.95
RHO_GATE = 0.85


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_metric(d: dict, name: str, stat: str = "avg"):
    m = d.get(name)
    if not isinstance(m, dict):
        return None
    return m.get(stat)


def get_count(d: dict, name: str = "request_count"):
    m = d.get(name)
    if not isinstance(m, dict):
        return None
    if m.get("count") is not None:
        return m["count"]
    return m.get("avg")


def find_closed_loop(root: Path, isl: int) -> dict[int, dict]:
    """返回 {concurrency: summary_dict}。"""
    out = {}
    isl_dir = root / "concurrency" / f"isl{isl}"
    if not isl_dir.is_dir():
        return out
    for cdir in sorted(isl_dir.iterdir()):
        if not cdir.is_dir() or not cdir.name.startswith("c"):
            continue
        try:
            c = int(cdir.name[1:])
        except ValueError:
            continue
        jf = cdir / f"{cdir.name}.json"
        if jf.is_file():
            out[c] = load_summary(jf)
    return out


def find_open_loop(root: Path, isl: int) -> dict[float, dict]:
    """返回 {rate: summary_dict}。"""
    out = {}
    isl_dir = root / "rps" / f"isl{isl}"
    if not isl_dir.is_dir():
        return out
    for rdir in sorted(isl_dir.iterdir()):
        if not rdir.is_dir() or not rdir.name.startswith("rate-"):
            continue
        rate_str = rdir.name[len("rate-"):].replace("_", ".")
        try:
            rate = float(rate_str)
        except ValueError:
            continue
        jf = rdir / f"{rdir.name}.json"
        if jf.is_file():
            out[rate] = load_summary(jf)
    return out


def mu_of(summary: dict) -> float:
    """产能 μ = request_throughput（requests/s），缺省退回 output_token_throughput/512。"""
    rt = get_metric(summary, "request_throughput", "avg")
    if rt is not None:
        return float(rt)
    ott = get_metric(summary, "output_token_throughput", "avg")
    if ott is not None:
        return float(ott) / OUTPUT_LEN
    return 0.0


def ttft_p95_s(summary: dict) -> float:
    v = get_metric(summary, "time_to_first_token", "p95")
    return float(v) / 1000.0 if v is not None else float("nan")


def ttft_p50_s(summary: dict) -> float:
    v = get_metric(summary, "time_to_first_token", "p50")
    return float(v) / 1000.0 if v is not None else float("nan")


def itl_p50_s(summary: dict) -> float:
    v = get_metric(summary, "inter_token_latency", "p50")
    return float(v) / 1000.0 if v is not None else float("nan")


def itl_p99_s(summary: dict) -> float:
    v = get_metric(summary, "inter_token_latency", "p99")
    return float(v) / 1000.0 if v is not None else float("nan")


def closed_loop_curve(closed: dict[int, dict]) -> list[tuple[int, float]]:
    return [(c, mu_of(closed[c])) for c in sorted(closed)]


def interpolate_closed(curve: list[tuple[int, float]], target_mu: float):
    """在闭环曲线上按 μ 反查并发（线性插值）。返回 (concurrency, ok)。"""
    if not curve:
        return float("nan"), False
    if target_mu <= curve[0][1]:
        return float(curve[0][0]), True
    if target_mu >= curve[-1][1]:
        return float(curve[-1][0]), False  # 超出曲线右端（饱和），标记外推
    for (c1, m1), (c2, m2) in zip(curve, curve[1:]):
        if m1 <= target_mu <= m2:
            if m2 == m1:
                return float(c1), True
            frac = (target_mu - m1) / (m2 - m1)
            return c1 + frac * (c2 - c1), True
    return float("nan"), False


def _growth(mus):
    if len(mus) < 2 or mus[-2] <= 0:
        return 0.0
    return (mus[-1] - mus[-2]) / mus[-2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--root", default=None, help="bench-tutorial 根目录（默认脚本上级）")
    ap.add_argument("--emit-rates", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent
    run_dir = root / "out" / args.run_id

    # ---- 闭环：分输入档各算 μ_max 与 μ(L) 曲线 ----
    all_curves = {}
    mu_max_by_isl = {}
    for isl in INPUT_LENS:
        closed = find_closed_loop(run_dir, isl)
        if not closed:
            continue
        curve = closed_loop_curve(closed)
        all_curves[isl] = (closed, curve)
        mu_max_by_isl[isl] = max(m for _, m in curve) if curve else 0.0

    if args.emit_rates:
        # 分档反推开环布点，写 rates.txt（每行 "isl rate1 rate2 ..."）
        lines = []
        for isl in INPUT_LENS:
            mu = mu_max_by_isl.get(isl, 0.0)
            if mu <= 0:
                print(f"ERROR: 未找到 isl{isl} 闭环结果，无法反推 RATES", file=sys.stderr)
                continue
            rates = " ".join(f"{mu * r:.3f}" for r in RATE_FRACTIONS)
            lines.append(f"{isl} {rates}")
        if not lines:
            print("ERROR: 未找到任何闭环结果，无法反推 RATES", file=sys.stderr)
            sys.exit(1)
        rates_path = run_dir / "rates.txt"
        rates_path.parent.mkdir(parents=True, exist_ok=True)
        rates_path.write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        print(f"(已写入 {rates_path})", file=sys.stderr)
        return

    # ---- 生成 summary.md ----
    lines = []
    lines.append(f"# bench-tutorial 压测汇总（run {args.run_id}）")
    lines.append("")
    lines.append(f"生成时间：{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("")
    lines.append("## 0. 口径")
    lines.append("")
    lines.append(f"- μ = request_throughput（requests/s，固定输出 {OUTPUT_LEN}）")
    lines.append("- μ_max(isl) = 该输入档闭环曲线最高点（分档，8K/32K 各一个）")
    lines.append("- ρ = λ/μ_max(isl)（**分档分母**，不跨档取全局 μ_max）")
    lines.append("- 达成率 = 完成请求/目标请求")
    lines.append("- 三门：① ρ<0.85 ② 达成率>95% ③ TTFT P95≤SLO（8K≤3s/32K≤8s）")
    lines.append("- Little's Law：L = 完成RPS × (TTFT_P50 + 512×ITL_P50) 回查本档闭环曲线，偏差<5%判自洽")
    lines.append("")
    lines.append("## 1. 闭环 μ(L) 曲线")
    lines.append("")
    lines.append("| 输入档 | 并发 | μ (req/s) | TTFT P95 (s) | ITL P99 (s) | n | SLO 可用? |")
    lines.append("|---|---|---|---|---|---|---|")
    for isl in INPUT_LENS:
        if isl not in all_curves:
            continue
        closed, curve = all_curves[isl]
        slo_ttft = SLO_TTFT_P95.get(isl, 8.0)
        for c in sorted(closed):
            s = closed[c]
            mu = mu_of(s)
            t95 = ttft_p95_s(s)
            i99 = itl_p99_s(s)
            n = get_count(s) or 0
            ok = "✓" if (t95 <= slo_ttft and not (t95 != t95)) else "✗"
            lines.append(f"| {isl} | {c} | {mu:.3f} | {t95:.2f} | {i99:.2f} | {n} | {ok} |")
        mus = [m for _, m in curve]
        lines.append("")
        lines.append(f"**{isl} 收敛判定**：μ 曲线 = {['%.3f' % m for m in mus]}；"
                     f"最高档相邻增幅 = {_growth(mus):.1%}（<5% 判收敛，否则续扫 64/96）。")
        lines.append("")

    for isl in INPUT_LENS:
        if isl in mu_max_by_isl:
            lines.append(f"**μ_max({isl}) = {mu_max_by_isl[isl]:.3f} req/s**")
    lines.append("")

    # ---- 开环 ----
    lines.append("## 2. 开环（按到达率 λ）")
    lines.append("")
    lines.append("| 输入档 | λ (req/s) | 完成 RPS | ρ=λ/μ_max | 达成率 | TTFT P95 (s) | 门①②③ | Little L | 对账偏差 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for isl in INPUT_LENS:
        open_pts = find_open_loop(run_dir, isl)
        if not open_pts:
            continue
        slo_ttft = SLO_TTFT_P95.get(isl, 8.0)
        curve = all_curves.get(isl, ({}, []))[1]
        mu_max_isl = mu_max_by_isl.get(isl, 0.0)
        for rate in sorted(open_pts):
            s = open_pts[rate]
            rho = rate / mu_max_isl if mu_max_isl > 0 else float("inf")
            achieved = mu_of(s)
            total = get_metric(s, "request_count", "avg") or 0
            done = get_metric(s, "completed_request_count", "avg")
            if done is None:
                done = total
            achieve = (done / total) if total else float("nan")
            t95 = ttft_p95_s(s)
            gate1 = rho < RHO_GATE
            gate2 = achieve > SLO_ACHIEVE
            gate3 = t95 <= slo_ttft
            gates = "".join("✓" if g else "✗" for g in (gate1, gate2, gate3))
            L = achieved * (ttft_p50_s(s) + OUTPUT_LEN * itl_p50_s(s))
            c_implied, in_range = interpolate_closed(curve, achieved)
            dev = (L - c_implied) / c_implied if c_implied and c_implied > 0 else float("nan")
            lines.append(
                f"| {isl} | {rate:.3f} | {achieved:.3f} | {rho:.3f} | {achieve:.1%} "
                f"| {t95:.2f} | {gates} | {L:.2f} | {dev:.1%} |"
            )
        lines.append("")

    lines.append("")
    lines.append("## 3. 交付物")
    lines.append("")
    for isl in INPUT_LENS:
        if isl in mu_max_by_isl:
            lines.append(f"- μ_max({isl}) = {mu_max_by_isl[isl]:.3f} req/s")
    lines.append("- 可持续 λ = 全过三门的最高 λ（分档，见上表门①②③全 ✓ 的最高 λ）")
    lines.append("- 推荐工作点 = SLO 可用并发 / 可持续 λ（见闭环表 ✓ 列）")
    lines.append("- 瓶颈定位 = 断崖劣化点对到具体配置参数（见闭环表 TTFT P95 增幅远超 μ 增幅的档位）")
    lines.append("")

    text = "\n".join(lines)
    out_path = Path(args.output) if args.output else run_dir / "summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"summary 已写入 {out_path}")
    print(text)


if __name__ == "__main__":
    main()
