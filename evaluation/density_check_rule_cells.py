"""W-C follow-up: is the learned-vs-random gap in final conflicts explained by zone-level link density? (finding M35)

Within each contested scenario-seed (>= 1 round-0 conflict) we have the learned topology and five degree-preserving
random rewirings. Model with scenario-seed fixed effects (within transformation):
    final_conflicts = a_scenario + b * link_density + c * is_learned
c is the learned-graph effect at equal zone-level link density. CI from a cluster bootstrap over scenarios
(2,000 resamples). Also reported: the same model for the consensus graph vs its rewirings.
Usage: python evaluation/density_check_rule_cells.py rule_cells_s42 rule_cells_s123 rule_cells_s456
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(run_ids):
    rows = defaultdict(dict)   # (seed, sample) -> variant -> (conflicts, density)
    for rid in run_ids:
        for line in open(os.path.join(ROOT, "evaluation", "results", rid, "records.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            m = r["metrics"]
            if m.get("status") == "SUCCESS" and m.get("n_conflicts_initial", 0) > 0:
                rows[(r["forecast_seed"], r["sample_idx"])][r["variant"]] = (m["n_conflicts_final"], m.get("link_density", np.nan))
    return rows


def fit(rows, learned, randoms):
    blocks = {}   # (seed, sample) -> (demeaned X, demeaned y, raw X)
    for k, v in rows.items():
        if learned in v and all(x in v for x in randoms):
            names = [learned] + randoms
            X = np.array([[v[n][1], float(n == learned)] for n in names], float)
            y = np.array([v[n][0] for n in names], float)
            blocks[k] = (X - X.mean(0), y - y.mean(), X)
    groups = sorted(blocks)

    def est(sel):
        Xw = np.vstack([blocks[gr][0] for gr in sel])
        yw = np.concatenate([blocks[gr][1] for gr in sel])
        return np.linalg.lstsq(Xw, yw, rcond=None)[0]

    b = est(groups)
    rng = np.random.default_rng(0)
    by_sample = defaultdict(list)
    for gr in groups:
        by_sample[gr[1]].append(gr)
    samples = sorted(by_sample)
    boot = np.array([est([gr for s in rng.choice(samples, size=len(samples), replace=True) for gr in by_sample[s]])
                     for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    raw = np.vstack([blocks[gr][2] for gr in groups])
    return {"n_scenario_seeds": len(groups), "n_scenarios": len(samples),
            "density_learned": float(raw[raw[:, 1] == 1, 0].mean()), "density_random": float(raw[raw[:, 1] == 0, 0].mean()),
            "b_density": float(b[0]), "b_density_ci": [float(lo[0]), float(hi[0])],
            "c_learned": float(b[1]), "c_learned_ci": [float(lo[1]), float(hi[1])]}


def main():
    rows = load(sys.argv[1:])
    out = {"learned_vs_random": fit(rows, "R_neg", [f"R_neg_random{i}" for i in range(5)]),
           "consensus_vs_random_consensus": fit(rows, "R_neg_consensus", [f"R_neg_random_consensus{i}" for i in range(5)])}
    d = os.path.join(ROOT, "evaluation", "results", "rule_cells_summary")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "density_check.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
