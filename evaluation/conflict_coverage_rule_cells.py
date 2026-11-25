"""Rule-agent study: conflict coverage of each topology (no API).

coverage = share of round-0 conflicting zone pairs that are neighbours in the topology. Rule agents resolve every
conflict they can see (all residual conflicts after negotiation are between non-neighbours), so coverage is the
mechanism by which a topology helps. Also reported: zone-level link density, so coverage can be compared at equal
density (coverage / density = how much more often the graph links conflict-prone pairs than a random pair).
Contested scenario-seeds, seeds pooled, scenario-level means with 95% bootstrap CI.
Usage: python evaluation/conflict_coverage_rule_cells.py rule_cells_s42 rule_cells_s123 rule_cells_s456
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from evaluation.summarize_mastra import boot_ci  # noqa: E402
from pipeline.coordination.common import detect_conflicts  # noqa: E402

KNN = ["R_neg_knn40", "R_neg_knn60", "R_neg_knn80", "R_neg_knn100"]
CELLS = ["R_neg", "R_neg_consensus", "R_neg_learned_dense", "R_neg_distance", "R_neg_knn", "R_neg_consensus3"] + KNN + \
        [f"R_neg_random{i}" for i in range(5)] + [f"R_neg_random_consensus{i}" for i in range(5)]


def main():
    rng = np.random.default_rng(0)
    cov = defaultdict(lambda: defaultdict(list))
    dens = defaultdict(lambda: defaultdict(list))
    for rid in sys.argv[1:]:
        for line in open(os.path.join(ROOT, "evaluation", "results", rid, "records.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["variant"] not in CELLS or r["metrics"].get("status") != "SUCCESS":
                continue
            res = r["result"]
            zs = [z for z in res["zones"] if z["advisory"].get("valid") and not z["advisory"].get("suppressed")]
            init = {z["zone_id"]: z["initial_reroutes"] for z in zs}
            roads = {z["zone_id"]: z["roads"] for z in res["zones"]}
            pairs = {(c[2], c[3]) for c in detect_conflicts(init, roads)}
            if not pairs:
                continue
            nb = {z["zone_id"]: set(z["neighbors"]) for z in res["zones"]}
            cov[r["variant"]][r["sample_idx"]].append(np.mean([b in nb[a] for a, b in pairs]))
            dens[r["variant"]][r["sample_idx"]].append(r["metrics"].get("link_density", np.nan))
    out = {}

    def summ(name, variants):
        c = defaultdict(list); d = defaultdict(list)
        for v in variants:
            for s, x in cov[v].items():
                c[s] += x
            for s, x in dens[v].items():
                d[s] += x
        cs = [float(np.mean(x)) for x in c.values()]
        ds = [float(np.nanmean(x)) for x in d.values()]
        out[name] = {"coverage": float(np.mean(cs)), "coverage_ci95": boot_ci(cs, rng), "link_density": float(np.mean(ds)),
                     "coverage_over_density": float(np.mean(cs) / np.mean(ds)) if np.mean(ds) > 0 else None,
                     "n_scenarios": len(cs)}

    for v in ["R_neg", "R_neg_consensus", "R_neg_learned_dense", "R_neg_distance", "R_neg_knn", "R_neg_consensus3"] + KNN:
        summ(v, [v])
    summ("R_neg_random_mean", [f"R_neg_random{i}" for i in range(5)])
    summ("R_neg_random_consensus_mean", [f"R_neg_random_consensus{i}" for i in range(5)])
    d = os.path.join(ROOT, "evaluation", "results", "rule_cells_summary")
    json.dump(out, open(os.path.join(d, "conflict_coverage.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"{k:30s} coverage {v['coverage']:.3f} [{v['coverage_ci95'][0]:.3f}, {v['coverage_ci95'][1]:.3f}]  "
              f"density {v['link_density']:.3f}  ratio {v['coverage_over_density']}  n={v['n_scenarios']}")


if __name__ == "__main__":
    main()
