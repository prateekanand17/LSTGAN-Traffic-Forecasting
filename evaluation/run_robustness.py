"""Robustness runs for the rule-agent study (plan §13 F2; review findings M39, M42, M43, M46). No API calls.

Every run uses rule (per-horizon) agents, K = 3 rounds, no fallback (the R_neg configuration) unless the setting
changes it, on the frozen 120-scenario list of rule_cells_s42 (same sha1 for every run).

Settings (Variant switches in pipeline/orchestrator.py; the defaults are the main configuration):
    base    main configuration (reproduces rule_cells_s*; also adds the corridor-overlap topology)
    r3, r8  candidate corridors within 3 km / 8 km of the zone centre (main: 5 km)              F2b
    seg     segment-level conflicts: footprints on the road must share a sensor (main: whole road) F2c
    eps010, eps020  DBSCAN eps 0.010 / 0.020 deg (main: 0.015)                                    F2e
    K5, K10 up to 5 / 10 negotiation rounds (main: 3)                                              F2e
    thr1, thr5  neighbour threshold 0.001 / 0.005 on the zone link (main: 0)                       F2e
    prio    zone priorities (GMP) always computed on the learned graph, so the topology changes only who
            talks to whom (review R-17)
    static  zones and severities from the static LSTGAN forecasts (no learned graph; seeds 42/456/789/1024), with
            the Adaptive-LSTGAN learned graph of the same seed as the topology (status M58, B3)
Extra learned graph (status M58, B2): alpha1 = graph of Adaptive-LSTGAN trained with alpha = 1 (5 seeds), with five
degree-preserving rewirings. (B1, the Graph WaveNet graph, was dropped: its embeddings collapsed, status M59.)
Experiments (cells x settings x seeds) are listed in EXPERIMENTS; `rewire` runs random rewirings 5..99 of the
learned graph (0..4 are in `base`), giving 100 draws per seed (F2a).

Writes evaluation/results/robustness_f2/records_<experiment>_s<seed>.jsonl (one compact line per scenario run)
and run_config_<time>.json. Resumable. Analysis: python evaluation/analyze_robustness.py

Usage: python evaluation/run_robustness.py --experiment base --seed 42
       python evaluation/run_robustness.py --list
"""
import argparse
import copy
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pipeline import config as C  # noqa: E402
from pipeline import data_source as D  # noqa: E402
from pipeline.coordination.common import detect_conflicts  # noqa: E402
from pipeline.orchestrator import Variant, run_scenario  # noqa: E402
from evaluation.metrics import scenario_metrics  # noqa: E402
from evaluation.run_mastra import file_hashes  # noqa: E402

OUT = os.path.join(C.RESULTS_DIR, "robustness_f2")
SCENARIOS = os.path.join(C.RESULTS_DIR, "rule_cells_s42", "scenarios.json")

SETTINGS = {
    "base": {}, "r3": {"radius_km": 3.0}, "r8": {"radius_km": 8.0}, "seg": {"conflict_level": "segment"},
    "eps010": {"dbscan_eps": 0.010}, "eps020": {"dbscan_eps": 0.020}, "K5": {"rounds": 5}, "K10": {"rounds": 10},
    "thr1": {"neighbor_threshold": 0.001}, "thr5": {"neighbor_threshold": 0.005}, "prio": {}, "static": {},
}
GMP_ON_LEARNED = {"prio"}
FORECASTER = {"static": "static"}   # setting -> forecaster (default adaptive)

# cell -> (agent coordination, topology, neighbour rule)
CELLS = {
    "R_none": ("none", "learned", "topology"),
    "R_central": ("central_rule", "learned", "topology"),
    "R_neg": ("negotiate", "learned", "topology"),
    "R_neg_corridor": ("negotiate", "learned", "corridor"),
    "R_neg_knn100": ("negotiate", "knn:100", "topology"),
    "R_neg_knn80": ("negotiate", "knn:80", "topology"),
    "R_neg_knn": ("negotiate", "knn", "topology"),
    "R_neg_distance": ("negotiate", "distance", "topology"),
    "R_neg_full": ("negotiate", "full", "topology"),
    "R_neg_none": ("negotiate", "none", "topology"),
    "R_neg_consensus": ("negotiate", "consensus", "topology"),
}
for _i in range(100):
    CELLS[f"R_neg_random{_i}"] = ("negotiate", f"random:{_i}", "topology")
for _g in ("alpha1",):
    CELLS[f"R_neg_{_g}"] = ("negotiate", _g, "topology")
    for _i in range(5):
        CELLS[f"R_neg_{_g}_random{_i}"] = ("negotiate", f"random_{_g}:{_i}", "topology")

RAND5 = [f"R_neg_random{i}" for i in range(5)]
MAIN = ["R_none", "R_central", "R_neg", "R_neg_corridor", "R_neg_knn100", "R_neg_knn80", "R_neg_knn", "R_neg_distance",
        "R_neg_full", "R_neg_none"] + RAND5
EXPERIMENTS = {  # experiment -> (setting, cells)
    "base": ("base", MAIN + ["R_neg_consensus"]),
    "rewire": ("base", [f"R_neg_random{i}" for i in range(5, 100)]),
    "r3": ("r3", MAIN), "r8": ("r8", MAIN), "seg": ("seg", MAIN),
    "eps010": ("eps010", MAIN), "eps020": ("eps020", MAIN),
    "K5": ("K5", ["R_neg", "R_neg_knn100", "R_neg_knn80"] + RAND5),
    "K10": ("K10", ["R_neg", "R_neg_knn100", "R_neg_knn80"] + RAND5),
    "thr1": ("thr1", ["R_neg", "R_neg_knn100", "R_neg_knn80"] + RAND5),
    "thr5": ("thr5", ["R_neg", "R_neg_knn100", "R_neg_knn80"] + RAND5),
    "prio": ("prio", ["R_neg", "R_neg_knn100", "R_neg_knn80", "R_neg_knn", "R_neg_distance", "R_neg_none", "R_neg_full",
                      "R_neg_corridor"] + RAND5),
    "alpha1": ("base", ["R_neg_alpha1"] + [f"R_neg_alpha1_random{i}" for i in range(5)]),
    "static": ("static", MAIN),
}
SEEDS = {"static": (42, 123, 456, 789, 1024)}   # static seed 123: forecasts from Kaggle_MASTRA_F2/check_v3 (M63)

_topo_cache = {}


def comm_adj(name: str, seed: int):
    key = (name, seed)
    if key not in _topo_cache:
        if name.split(":")[0] in ("random", "random_alpha1"):
            _topo_cache[key] = D.topology(name.split(":")[0], seed=seed, rand_seed=int(name.split(":")[1]))
        else:
            _topo_cache[key] = D.topology(name, seed=seed)
    return _topo_cache[key]


def variant_for(cell: str, setting: str) -> Variant:
    coord, topo, rule = CELLS[cell]
    return Variant(f"{cell}@{setting}", agent="rules_h", coordination=coord, topology=topo, neighbor_rule=rule,
                   **SETTINGS[setting])


def coverage(res: dict):
    """(round-0 conflicting zone pairs, of which neighbours) under the run's conflict definition."""
    zs = [z for z in res["zones"] if z["advisory"].get("valid") and not z["advisory"].get("suppressed")]
    fp = res.get("footprints")
    fp = {z: {r: frozenset(v) for r, v in f.items()} for z, f in fp.items()} if fp else None
    pairs = {(c[2], c[3]) for c in detect_conflicts({z["zone_id"]: z["initial_reroutes"] for z in zs},
                                                    {z["zone_id"]: z["roads"] for z in res["zones"]}, fp)}
    nb = {z["zone_id"]: set(z["neighbors"]) for z in res["zones"]}
    return len(pairs), sum(b in nb[a] for a, b in pairs)


KEEP = ("status", "n_zones", "n_conflicts_initial", "n_conflicts_final", "conflict_free", "crr", "n_reroutes_final",
        "rounds_used", "messages", "n_neighbor_pairs", "link_density", "stop_reason", "wzpa30", "wzpa60")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or not args.experiment:
        for e, (s, cells) in EXPERIMENTS.items():
            print(f"{e:8s} setting={s:7s} {SETTINGS[s]} cells={len(cells)}")
        return
    setting, cells = EXPERIMENTS[args.experiment]
    if args.experiment in SEEDS and args.seed not in SEEDS[args.experiment]:
        sys.exit(f"{args.experiment}: no input files for seed {args.seed}; available {SEEDS[args.experiment]}")
    os.makedirs(OUT, exist_ok=True)
    scen = json.load(open(SCENARIOS))
    json.dump({"args": vars(args), "setting": setting, "switches": SETTINGS[setting], "cells": cells,
               "forecaster": FORECASTER.get(setting, "adaptive"),
               "scenarios_sha1": scen["sha1"], "file_sha1": file_hashes(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(os.path.join(OUT, f"run_config_{args.experiment}_s{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"),
              indent=1)
    path = os.path.join(OUT, f"records_{args.experiment}_s{args.seed}.jsonl")
    done = set()
    if os.path.exists(path):
        good = []
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:   # a line cut off when a run was stopped: drop it and rerun that scenario
                continue
            if (r["cell"], r["sample_idx"]) not in done:
                done.add((r["cell"], r["sample_idx"]))
                good.append(line if line.endswith("\n") else line + "\n")
        open(path, "w", encoding="utf-8").writelines(good)
    fc = FORECASTER.get(setting, "adaptive")
    base_inp = {s["sample_idx"]: D.build_input(s["sample_idx"], args.seed, forecaster=fc) for s in scen["scenarios"]}
    learned = D.topology("learned", seed=args.seed)
    t_start, n = time.time(), 0
    with open(path, "a", encoding="utf-8") as f:
        for cell in cells:
            v = variant_for(cell, setting)
            A = comm_adj(v.topology, args.seed)
            for s in scen["scenarios"]:
                if (cell, s["sample_idx"]) in done:
                    continue
                inp = copy.copy(base_inp[s["sample_idx"]])
                inp.comm_adj = A
                inp.gmp_adj = learned if setting in GMP_ON_LEARNED else None
                res = run_scenario(inp, v, meta={"run_id": "robustness_f2", "sample_idx": s["sample_idx"]})
                met = scenario_metrics(res, inp.actual_speed, inp.roads)
                rec = {"experiment": args.experiment, "setting": setting, "cell": cell, "seed": args.seed,
                       "sample_idx": s["sample_idx"], "regime": s["regime"],
                       "metrics": {k: met[k] for k in KEEP if k in met}}
                if res["status"] == "SUCCESS":
                    rec["conflict_pairs0"], rec["covered_pairs0"] = coverage(res)
                    rec["priority_order"] = [z["zone_id"] for z in sorted(
                        res["zones"], key=lambda z: (z["detector_severity"] == "HIGH", z["detector_severity"] == "MEDIUM",
                                                     round(z["updated_cascade_risk"], 6), z["zone_id"]), reverse=True)]
                f.write(json.dumps(rec, default=float) + "\n")
                n += 1
            f.flush()
            print(f"{args.experiment} s{args.seed} {cell} done ({n} runs, {time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
