"""W-C analysis: rule-agent cells across forecaster seeds (plan §10.0 hypothesis, task N10).

Hypothesis (fixed 2026-10-13): negotiating over the learned topology leaves fewer conflicts than random and
distance topologies, with equal or better WZPA, across 3+ forecaster seeds.

Unit of analysis: scenario. For each scenario a metric is averaged over the forecaster seeds in which the scenario
has a result (seed-level values are kept for the per-seed table). The five random rewirings are averaged per
scenario into one control (R_neg_random_mean, R_neg_random_consensus_mean).
Conflict metrics are compared on "contested" scenario-seeds (>= 1 conflict among the round-0 proposals); the
round-0 proposals of rule agents do not depend on the topology, so every cell starts from the same conflicts.
Paired tests: mean difference (cell - reference), paired bootstrap 95% CI, Wilcoxon signed-rank, matched-pairs
rank-biserial r, Holm across the planned family (internal plan of 2026-10-13; kNN-100 was added after the first
results, see papers/MASTRA/MASTRA_Status.md M38). Family (review R-21, 2026-10-18): conflict metrics only (severities
are not negotiated, so WZPA tests are identical by construction) and no duplicate cells (R_neg_none = R_none,
R_neg_learned_dense = R_neg_full); each comparison uses its own bootstrap seed, so identical cells get identical CIs.

Usage: python evaluation/analyze_rule_cells.py rule_cells_s42 rule_cells_s123 rule_cells_s456 [--out rule_cells_summary]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from evaluation.summarize_mastra import boot_ci, holm  # noqa: E402

METRICS = ["n_conflicts_initial", "n_conflicts_final", "conflict_free", "crr", "reroute_target_freeflow_rate",
           "n_reroutes_final", "rounds_used", "messages", "link_density", "fallback_used", "wzpa30", "wzpa60",
           "zpa30_exact"]
CONFLICT_METRICS = {"n_conflicts_final", "conflict_free", "crr", "rounds_used", "messages", "fallback_used", "link_density"}
REF = "R_neg"
FAMILY = [  # (cell, reference) - planned comparisons (kNN-100 added after the first results)
    ("R_neg_random_mean", REF), ("R_neg_knn100", REF), ("R_neg_distance", REF), ("R_neg_knn", REF),
    ("R_neg_full", REF),
    ("R_neg_consensus", REF), ("R_neg_consensus", "R_neg_random_consensus_mean"),
    ("R_none", REF), ("R_central", REF), ("R_neg_fb", REF), ("R_neg_K1", REF), ("R_neg_K2", REF),
]
PRIMARY = ["n_conflicts_final", "conflict_free"]


def load(run_ids):
    """-> data[variant][(seed, sample)] = metrics dict (SUCCESS records only), regime[sample]"""
    data, regime = defaultdict(dict), {}
    for rid in run_ids:
        for line in open(os.path.join(ROOT, "evaluation", "results", rid, "records.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r.get("dry_run"):
                sys.exit(f"{rid}: dry-run record found; refusing")
            regime[r["sample_idx"]] = r["regime"]
            if r["metrics"].get("status") == "SUCCESS":
                data[r["variant"]][(r["forecast_seed"], r["sample_idx"])] = r["metrics"]
    for base, n in (("R_neg_random", 5), ("R_neg_random_consensus", 5)):
        names = [f"{base}{i}" for i in range(n)]
        if all(x in data for x in names):
            keys = set.intersection(*(set(data[x]) for x in names))
            data[f"{base}_mean"] = {k: {m: float(np.mean([data[x][k][m] for x in names]))
                                        for m in METRICS if all(isinstance(data[x][k].get(m), (int, float)) for x in names)}
                                    for k in keys}
    return data, regime


def values(data, variant, metric, contested_only):
    """{sample: mean over seeds}"""
    acc = defaultdict(list)
    ref = data.get(REF, {})
    for (seed, s), m in data.get(variant, {}).items():
        if contested_only and not (ref.get((seed, s), m).get("n_conflicts_initial", 0) > 0):
            continue
        v = m.get(metric)
        if isinstance(v, (int, float)) and v == v:
            acc[s].append(float(v))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def paired(a, b, rng):
    common = sorted(set(a) & set(b))
    if len(common) < 5:
        return None
    diff = np.array([b[s] - a[s] for s in common])
    if np.allclose(diff, 0):
        p, rbc = 1.0, 0.0
    else:
        p = float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
        nz = diff[diff != 0]
        ranks = stats.rankdata(np.abs(nz))
        rbc = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    return {"n_pairs": len(common), "n_nonzero": int((diff != 0).sum()), "mean_diff": float(diff.mean()), "ci95": boot_ci(diff, rng), "wilcoxon_p": p,
            "rank_biserial": rbc}


def fmt_ci(ci):
    return "–" if ci[0] is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--out", default="rule_cells_summary")
    args = ap.parse_args()
    data, regime = load(args.run_ids)
    rng = np.random.default_rng(0)
    seeds = sorted({k[0] for v in data.values() for k in v})
    out = {"run_ids": args.run_ids, "seeds": seeds, "cells": {}, "family": [], "per_seed": {}, "by_regime": {}}

    # 1. cell means
    for v in sorted(data):
        out["cells"][v] = {}
        for m in METRICS:
            x = list(values(data, v, m, m in CONFLICT_METRICS).values())
            if x:
                out["cells"][v][m] = {"mean": float(np.mean(x)), "ci95": boot_ci(x, rng), "n": len(x)}

    # 2. planned family (conflict metrics only; one bootstrap seed per comparison)
    fam = []
    for cell, ref in FAMILY:
        if cell not in data or ref not in data:
            continue
        for m in PRIMARY:
            r = paired(values(data, ref, m, m in CONFLICT_METRICS), values(data, cell, m, m in CONFLICT_METRICS),
                       np.random.default_rng(1))
            if r:
                fam.append({"cell": cell, "ref": ref, "metric": m, **r})
    for c, padj in zip(fam, holm([c["wilcoxon_p"] for c in fam]) if fam else []):
        c["p_holm"] = padj
    out["family"] = fam

    # 3. per seed: learned vs controls on the primary metric
    for seed in seeds:
        sub = {v: {k: m for k, m in d.items() if k[0] == seed} for v, d in data.items()}
        row = {}
        for v in ("R_none", REF, "R_neg_random_mean", "R_neg_knn100", "R_neg_distance", "R_neg_none", "R_neg_full",
                  "R_neg_consensus", "R_central"):
            x = list(values(sub, v, "n_conflicts_final", True).values())
            if x:
                row[v] = {"mean": float(np.mean(x)), "n": len(x)}
        out["per_seed"][seed] = row

    # 4. by regime
    for v in ("R_none", REF, "R_neg_random_mean", "R_neg_knn100", "R_neg_distance", "R_neg_none", "R_central"):
        for reg in ("HIGH", "MODERATE", "CASCADE", "NORMAL"):
            x = [val for s, val in values(data, v, "n_conflicts_final", True).items() if regime[s] == reg]
            if x:
                out["by_regime"].setdefault(v, {})[reg] = {"mean": float(np.mean(x)), "n": len(x)}

    d = os.path.join(ROOT, "evaluation", "results", args.out)
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)

    L = [f"# Rule-agent cells (W-C): runs {', '.join(args.run_ids)}", "",
         f"Forecaster seeds: {seeds}. Unit: scenario (seeds averaged). Conflict metrics on contested scenario-seeds "
         "(>= 1 round-0 conflict); WZPA on all scenarios with zones. CI: 95% bootstrap.", "",
         "## 1. Cell means", "",
         "| cell | final conflicts | conflict-free | CRR | rounds | messages | link density | fallback | WZPA@30 | WZPA@60 | n (contested) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["B_rules", "B_rules_h", "R_none", "R_central", REF, "R_neg_fb", "R_neg_K1", "R_neg_K2", "R_neg_consensus",
             "R_neg_consensus3", "R_neg_learned_dense", "R_neg_distance", "R_neg_knn", "R_neg_knn40", "R_neg_knn60",
             "R_neg_knn80", "R_neg_knn100", "R_neg_full", "R_neg_random_mean",
             "R_neg_random_consensus_mean", "R_neg_none"]
    for v in order:
        c = out["cells"].get(v)
        if not c:
            continue
        g = lambda m: f"{c[m]['mean']:.3f}" if m in c else "–"
        L.append(f"| {v} | {g('n_conflicts_final')} | {g('conflict_free')} | {g('crr')} | {g('rounds_used')} | "
                 f"{g('messages')} | {g('link_density')} | {g('fallback_used')} | {g('wzpa30')} | {g('wzpa60')} | "
                 f"{c.get('n_conflicts_final', {}).get('n', '–')} |")
    L += ["", "## 2. Planned comparisons (cell − reference; Holm across the table)", "",
          "| cell | reference | metric | n | n non-zero | mean diff | 95% CI | Wilcoxon p | Holm p | r_rb |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in fam:
        L.append(f"| {c['cell']} | {c['ref']} | {c['metric']} | {c['n_pairs']} | {c['n_nonzero']} | {c['mean_diff']:+.3f} | {fmt_ci(c['ci95'])} | "
                 f"{c['wilcoxon_p']:.3g} | {c['p_holm']:.3g} | {c['rank_biserial']:+.2f} |")
    L += ["", "## 3. Final conflicts per forecaster seed (contested scenarios)", "",
          "| seed | " + " | ".join(out["per_seed"][seeds[0]].keys()) + " |",
          "|---|" + "---|" * len(out["per_seed"][seeds[0]])]
    for seed in seeds:
        r = out["per_seed"][seed]
        L.append(f"| {seed} | " + " | ".join(f"{r[k]['mean']:.3f} (n={r[k]['n']})" for k in r) + " |")
    L += ["", "## 4. Final conflicts by regime (contested scenarios)", "",
          "| cell | HIGH | MODERATE | CASCADE | NORMAL |", "|---|---|---|---|---|"]
    for v, rr in out["by_regime"].items():
        L.append(f"| {v} | " + " | ".join(f"{rr[g]['mean']:.3f} (n={rr[g]['n']})" if g in rr else "–"
                                          for g in ("HIGH", "MODERATE", "CASCADE", "NORMAL")) + " |")
    open(os.path.join(d, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
