"""Rule-agent study, follow-up analyses (no API):
  1. residual conflicts after negotiation: between topology neighbours (visible, not resolved) or between zones that
     are not neighbours (invisible to both agents); stop reasons of the loop;
  2. reroute quality per cell: reroutes per scenario and share of targets free-flowing in ground truth at +30 min;
  3. runtime per scenario (CPU, pipeline only; topology construction cached and excluded).
Contested scenario-seeds only (>= 1 round-0 conflict), seeds pooled. Scenario-level bootstrap CI (10,000).
Usage: python evaluation/residual_conflicts_rule_cells.py rule_cells_s42 rule_cells_s123 rule_cells_s456
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from evaluation.summarize_mastra import boot_ci  # noqa: E402
from pipeline.coordination.common import detect_conflicts  # noqa: E402

CELLS = ["R_none", "R_neg", "R_neg_distance", "R_neg_random0", "R_neg_full", "R_central", "R_neg_fb"]


def main():
    rng = np.random.default_rng(0)
    contested = set()
    recs = defaultdict(dict)
    for rid in sys.argv[1:]:
        for line in open(os.path.join(ROOT, "evaluation", "results", rid, "records.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            key = (r["forecast_seed"], r["sample_idx"])
            if r["variant"] in CELLS + ["B_rules_h"]:
                recs[r["variant"]][key] = r
            if r["variant"] == "R_none" and r["metrics"].get("n_conflicts_initial", 0) > 0:
                contested.add(key)
    out = {"n_contested_scenario_seeds": len(contested)}

    # 1. residual conflicts of R_neg
    vis = invis = 0
    per_sample = defaultdict(list)
    stops = Counter()
    for key in contested:
        r = recs["R_neg"][key]
        res = r["result"]
        stops[res["coordination"]["stop_reason"]] += 1
        zs = {z["zone_id"]: z for z in res["zones"] if z["advisory"].get("valid") and not z["advisory"].get("suppressed")}
        final = {zid: z["advisory"].get("actionable_reroutes") or [] for zid, z in zs.items()}
        nb = {z["zone_id"]: set(z["neighbors"]) for z in res["zones"]}
        roads = {z["zone_id"]: z["roads"] for z in res["zones"]}
        v = i = 0
        for c in detect_conflicts(final, roads):
            if c[3] in nb[c[2]]:
                v += 1
            else:
                i += 1
        vis += v
        invis += i
        if v + i:
            per_sample[key[1]].append(i / (v + i))
    share = [float(np.mean(x)) for x in per_sample.values()]
    out["residual"] = {"visible": vis, "invisible": invis, "share_invisible": invis / max(1, vis + invis),
                       "share_invisible_per_scenario_mean": float(np.mean(share)) if share else None,
                       "share_invisible_ci95": boot_ci(share, rng) if share else None, "n_scenarios": len(share),
                       "stop_reasons": dict(stops)}

    # 2. reroute quality and 3. runtime
    out["cells"] = {}
    for cell in CELLS:
        rr = defaultdict(list); ff = defaultdict(list); rt = defaultdict(list)
        for key in contested:
            r = recs[cell].get(key)
            if not r:
                continue
            m = r["metrics"]
            rr[key[1]].append(m.get("n_reroutes_final", 0))
            if m.get("reroute_target_freeflow_rate") is not None:
                ff[key[1]].append(m["reroute_target_freeflow_rate"])
            rt[key[1]].append(r["elapsed_s"])
        agg = lambda d: [float(np.mean(v)) for v in d.values()]
        out["cells"][cell] = {
            "reroutes_per_scenario": {"mean": float(np.mean(agg(rr))), "ci95": boot_ci(agg(rr), rng), "n": len(rr)},
            "target_freeflow_rate": {"mean": float(np.mean(agg(ff))), "ci95": boot_ci(agg(ff), rng), "n": len(ff)},
            "runtime_s": {"mean": float(np.mean(agg(rt))), "median": float(np.median(agg(rt))), "max": float(np.max(agg(rt)))},
        }
    d = os.path.join(ROOT, "evaluation", "results", "rule_cells_summary")
    json.dump(out, open(os.path.join(d, "residual_and_quality.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
