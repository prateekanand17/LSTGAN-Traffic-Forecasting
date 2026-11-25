"""Summarize a MASTRA run (plan Section 5 statistics).

Unit of analysis = scenario: repeated runs are averaged within (scenario, variant) first.
Per variant x metric: mean, 95% bootstrap CI of the mean (10,000 resamples), n.
Paired comparisons vs --ref: mean difference, paired bootstrap 95% CI, Wilcoxon signed-rank p, matched-pairs
rank-biserial r, Holm-adjusted p across all (variant, metric) comparisons in the call.

Usage: python evaluation/summarize_mastra.py <run_id> [--ref V_full] [--metrics wzpa30,wzpa60,crr]
Writes evaluation/results/<run_id>/summary.json and summary.md. Dry-run records are refused unless --allow-dry-run.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_METRICS = ["wzpa30", "wzpa60", "zpa30_exact", "zpa60_exact", "det_wzpa30", "valid_rate", "false_alarm_rate30",
                   "hallucinated_route_rate", "numeric_faithfulness", "cdr", "crr", "n_conflicts_initial",
                   "n_conflicts_final", "conflict_free", "reroute_target_freeflow_rate", "implausible_route_rate",
                   "rounds_used", "messages", "link_density", "fallback_used", "mean_attempts",
                   "llm_calls", "llm_prompt_tokens", "llm_completion_tokens", "llm_latency_s"]
B = 10000


def boot_ci(x, rng):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [None, None]
    m = rng.choice(x, size=(B, len(x)), replace=True).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj.tolist()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--allow-dry-run", action="store_true")
    args = ap.parse_args()
    d = os.path.join(ROOT, "evaluation", "results", args.run_id)
    recs = [json.loads(l) for l in open(os.path.join(d, "records.jsonl"), encoding="utf-8")]
    if any(r.get("dry_run") for r in recs) and not args.allow_dry_run:
        sys.exit("records contain dry-run output; refuse to summarize (pass --allow-dry-run for pipeline tests only)")
    metrics = args.metrics.split(",")
    rng = np.random.default_rng(0)

    # (variant, sample) -> metric -> list over runs
    cell = defaultdict(lambda: defaultdict(list))
    regime = {}
    status = defaultdict(lambda: defaultdict(int))
    for r in recs:
        status[r["variant"]][r["metrics"].get("status")] += 1
        regime[r["sample_idx"]] = r["regime"]
        for m in metrics:
            v = r["metrics"].get(m)
            if isinstance(v, (int, float)):
                cell[(r["variant"], r["sample_idx"])][m].append(float(v))
    per = defaultdict(dict)  # variant -> metric -> {sample: mean over runs}
    for (v, s), mm in cell.items():
        for m, vals in mm.items():
            per[v].setdefault(m, {})[s] = float(np.mean(vals))

    summary = {"run_id": args.run_id, "n_records": len(recs), "status_counts": status, "variants": {}, "by_regime": {}}
    for v in sorted(per):
        summary["variants"][v] = {}
        for m in metrics:
            x = list(per[v].get(m, {}).values())
            if x:
                summary["variants"][v][m] = {"mean": float(np.mean(x)), "ci95": boot_ci(x, rng), "n": len(x)}
        for reg in sorted(set(regime.values())):
            for m in ("wzpa30", "wzpa60", "crr"):
                x = [val for s, val in per[v].get(m, {}).items() if regime[s] == reg]
                if x:
                    summary["by_regime"].setdefault(v, {}).setdefault(reg, {})[m] = {"mean": float(np.mean(x)), "n": len(x)}

    comps = []
    if args.ref and args.ref in per:
        for v in sorted(per):
            if v == args.ref:
                continue
            for m in metrics:
                a, b = per[args.ref].get(m, {}), per[v].get(m, {})
                common = sorted(set(a) & set(b))
                if len(common) < 5:
                    continue
                diff = np.array([b[s] - a[s] for s in common])
                if np.allclose(diff, 0):
                    p, rbc = 1.0, 0.0
                else:
                    p = float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
                    nz = diff[diff != 0]
                    ranks = stats.rankdata(np.abs(nz))
                    rbc = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
                comps.append({"variant": v, "ref": args.ref, "metric": m, "n_pairs": len(common),
                              "mean_diff": float(diff.mean()), "ci95": boot_ci(diff, rng),
                              "wilcoxon_p": p, "rank_biserial": rbc})
        if comps:
            for c, padj in zip(comps, holm([c["wilcoxon_p"] for c in comps])):
                c["p_holm"] = padj
    summary["paired_vs_ref"] = comps

    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    lines = [f"# Summary of run `{args.run_id}`", "", f"Records: {len(recs)}. Unit: scenario (runs averaged first). "
             "CI: 95% bootstrap of the mean.", "", "| variant | metric | mean | 95% CI | n |", "|---|---|---|---|---|"]
    for v, mm in summary["variants"].items():
        for m, s in mm.items():
            ci = "–" if s["ci95"][0] is None else f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"
            lines.append(f"| {v} | {m} | {s['mean']:.3f} | {ci} | {s['n']} |")
    if comps:
        lines += ["", f"## Paired differences vs {args.ref} (variant − ref)", "",
                  "| variant | metric | n | mean diff | 95% CI | Wilcoxon p | Holm p | r_rb |", "|---|---|---|---|---|---|---|---|"]
        for c in comps:
            ci = "–" if c["ci95"][0] is None else f"[{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"
            lines.append(f"| {c['variant']} | {c['metric']} | {c['n_pairs']} | {c['mean_diff']:+.3f} | {ci} | "
                         f"{c['wilcoxon_p']:.3g} | {c['p_holm']:.3g} | {c['rank_biserial']:+.2f} |")
    with open(os.path.join(d, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
