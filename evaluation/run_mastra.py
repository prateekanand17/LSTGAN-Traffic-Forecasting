"""MASTRA evaluation runner (task P1.6). Replaces run_ablation.py / run_ablation_fast.py (kept as archives only).

Usage examples:
    python evaluation/run_mastra.py --run-id smoke --variants R_neg,L_neg --per-regime 2 --dry-run
    python evaluation/run_mastra.py --run-id rule_cells --variants @rule_cells --per-regime 30
    python evaluation/run_mastra.py --list-variants

Variant names (v2, plan §9.3; the v1 names V_full / T_* / A_* / F_* were retired on 2026-10-17):
    B_*            deterministic and single-LLM baselines, no coordination
    {R,L}_none     rule (per-horizon) / LLM agents, proposals used as they are
    {R,L}_central  central coordinator with a global view (rule = greedy by priority, LLM = supervisor)
    {R,L}_neg      negotiation over the learned topology, K = NEGOTIATION_ROUNDS, no fallback
    {R,L}_neg_fb   same + central rule fallback if conflicts remain (main system)
    {R,L}_neg_<topology>, {R,L}_neg_K1/K2, L_neg_nogates, L_neg_nogmp, L_neg_reactive, L_neg_oracle
Groups: @rule_cells (all rule-agent cells, no API), @llm_core (2 x 2 LLM cells), @llm_topology.

Writes evaluation/results/<run_id>/:
    run_config.json   CLI args, pipeline.config values, sha1 of every pipeline/ and evaluation/ source file
    scenarios.json    frozen scenario list (+ sha1)
    records.jsonl     one line per (scenario, variant, run): full scenario result + metrics
    calls.jsonl       every LLM call (written by pipeline.llm_client)
Resumable: (scenario, variant, run) triples already in records.jsonl are skipped.
Summaries: python evaluation/summarize_mastra.py <run_id>
"""
import argparse
import functools
import glob
import hashlib
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pipeline import config as C  # noqa: E402
from pipeline import data_source as D  # noqa: E402
from pipeline.llm_client import LLMClient, LLMError  # noqa: E402
from pipeline.orchestrator import Variant, run_scenario  # noqa: E402
from evaluation.metrics import scenario_metrics  # noqa: E402

TOPOLOGIES = ["learned_dense", "distance", "knn", "knn:40", "knn:60", "knn:80", "knn:100", "full", "consensus",
              "consensus:3", "none"] + [f"random:{i}" for i in range(5)] + [f"random_consensus:{i}" for i in range(5)]


def _tname(t: str) -> str:
    return t.replace(":", "")


def build_variants() -> dict:
    V = {
        "B_rules": Variant("B_rules", agent="rules", coordination="none"),
        "B_rules_h": Variant("B_rules_h", agent="rules_h", coordination="none"),
        "B_rules_reactive": Variant("B_rules_reactive", agent="rules", coordination="none", forecast_mode="reactive"),
        "B_rules_oracle": Variant("B_rules_oracle", agent="rules", coordination="none", forecast_mode="oracle"),
        "B_rules_h_reactive": Variant("B_rules_h_reactive", agent="rules_h", coordination="none", forecast_mode="reactive"),
        "B_single": Variant("B_single", agent="single", coordination="none", gmp=False),
    }
    for p, agent in (("R", "rules_h"), ("L", "llm")):
        central = "central_rule" if p == "R" else "central_llm"
        V[f"{p}_none"] = Variant(f"{p}_none", agent=agent, coordination="none")
        V[f"{p}_central"] = Variant(f"{p}_central", agent=agent, coordination=central)
        V[f"{p}_neg"] = Variant(f"{p}_neg", agent=agent, coordination="negotiate")
        V[f"{p}_neg_fb"] = Variant(f"{p}_neg_fb", agent=agent, coordination="negotiate", fallback="rule")
        for t in TOPOLOGIES:
            n = f"{p}_neg_{_tname(t)}"
            V[n] = Variant(n, agent=agent, coordination="negotiate", topology=t)
        for k in (1, 2):
            V[f"{p}_neg_K{k}"] = Variant(f"{p}_neg_K{k}", agent=agent, coordination="negotiate", rounds=k)
    V["L_neg_nogates"] = Variant("L_neg_nogates", agent="llm", coordination="negotiate", gates=False)
    V["L_neg_nogmp"] = Variant("L_neg_nogmp", agent="llm", coordination="negotiate", gmp=False)
    V["L_neg_reactive"] = Variant("L_neg_reactive", agent="llm", coordination="negotiate", forecast_mode="reactive")
    V["L_neg_oracle"] = Variant("L_neg_oracle", agent="llm", coordination="negotiate", forecast_mode="oracle")
    return V


VARIANTS = build_variants()
GROUPS = {
    "rule_cells": [n for n, v in VARIANTS.items() if not v.needs_llm],
    "llm_core": ["L_none", "L_central", "L_neg", "L_neg_fb", "B_single"],
    "llm_topology": ["L_neg"] + [f"L_neg_{_tname(t)}" for t in TOPOLOGIES],
}


def comm_adj_for(variant: Variant, forecast_seed: int):
    return _comm_adj(variant.topology, forecast_seed)


@functools.lru_cache(maxsize=64)
def _comm_adj(name: str, forecast_seed: int):
    """Topologies are deterministic per (name, forecaster seed); cached so random rewirings are built once."""
    for prefix in ("random_consensus:", "random:"):
        if name.startswith(prefix):
            return D.topology(prefix[:-1], seed=forecast_seed, rand_seed=int(name.split(":")[1]))
    return D.topology(name, seed=forecast_seed)


def file_hashes():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "pipeline", "**", "*.py"), recursive=True) +
                    glob.glob(os.path.join(ROOT, "evaluation", "*.py"))):
        with open(p, "rb") as f:
            out[os.path.relpath(p, ROOT).replace("\\", "/")] = hashlib.sha1(f.read()).hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--variants", default="R_neg", help="comma list of names and/or @groups")
    ap.add_argument("--list-variants", action="store_true")
    ap.add_argument("--per-regime", type=int, default=2, help="scenarios per regime (HIGH, MODERATE, CASCADE, NORMAL)")
    ap.add_argument("--runs", type=int, default=1, help="repeated LLM runs per scenario")
    ap.add_argument("--forecast-seed", type=int, default=C.DEFAULT_FORECAST_SEED)
    ap.add_argument("--scenario-seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="no API calls; rule-based stand-in, labelled in all logs")
    ap.add_argument("--max-scenarios", type=int, default=None)
    args = ap.parse_args()

    if args.list_variants:
        for n, v in VARIANTS.items():
            print(f"{n:28s} agent={v.agent:8s} coord={v.coordination:13s} topology={v.topology:20s} K={v.rounds} "
                  f"fallback={v.fallback} llm={v.needs_llm}")
        print("groups:", {g: len(v) for g, v in GROUPS.items()})
        return
    if not args.run_id:
        sys.exit("--run-id is required")
    names = []
    for v in (x.strip() for x in args.variants.split(",") if x.strip()):
        names += GROUPS[v[1:]] if v.startswith("@") and v[1:] in GROUPS else [v]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        sys.exit(f"unknown variants {unknown}; choose from {sorted(VARIANTS)}")

    out_dir = os.path.join(C.RESULTS_DIR, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    scen_path = os.path.join(out_dir, "scenarios.json")
    if os.path.exists(scen_path):
        scenarios = json.load(open(scen_path))["scenarios"]
    else:
        sc = D.select_scenarios({r: args.per_regime for r in ("HIGH", "MODERATE", "CASCADE", "NORMAL")},
                                seed=args.scenario_seed)
        scenarios = [s.__dict__ for s in sc]
        blob = json.dumps(scenarios, sort_keys=True)
        json.dump({"sha1": hashlib.sha1(blob.encode()).hexdigest(), "scenarios": scenarios}, open(scen_path, "w"), indent=1)
    if args.max_scenarios:
        scenarios = scenarios[:args.max_scenarios]

    cfg_vals = {k: v for k, v in vars(C).items() if k.isupper() and isinstance(v, (int, float, str, tuple, dict, list))}
    json.dump({"args": vars(args), "config": cfg_vals, "file_sha1": file_hashes(),
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(os.path.join(out_dir, f"run_config_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1, default=str)

    rec_path = os.path.join(out_dir, "records.jsonl")
    done = set()
    if os.path.exists(rec_path):
        for line in open(rec_path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["sample_idx"], r["variant"], r["run"]))

    need_llm = any(VARIANTS[n].needs_llm for n in names)
    llm = LLMClient(log_dir=out_dir, dry_run=args.dry_run) if need_llm else None
    roads = D.build_input(0).roads
    total = len(scenarios) * len(names) * args.runs
    k = 0
    for run in range(args.runs):
        for s in scenarios:
            for vname in names:
                k += 1
                if (s["sample_idx"], vname, run) in done:
                    continue
                v = VARIANTS[vname]
                inp = D.build_input(s["sample_idx"], args.forecast_seed, v.forecast_mode, comm_adj_for(v, args.forecast_seed))
                meta = {"run_id": args.run_id, "sample_idx": s["sample_idx"], "regime": s["regime"], "run": run}
                t0 = time.time()
                try:
                    res = run_scenario(inp, v, llm=llm if v.needs_llm else None, meta=meta, dry_run=args.dry_run)
                    met = scenario_metrics(res, inp.actual_speed, inp.roads)
                    err = None
                except LLMError as e:
                    res, met, err = None, {"status": "LLM_ERROR"}, str(e)
                rec = {"sample_idx": s["sample_idx"], "regime": s["regime"], "timestamp": s["timestamp"],
                       "variant": vname, "run": run, "forecast_seed": args.forecast_seed,
                       "dry_run": bool(args.dry_run and v.needs_llm),
                       "elapsed_s": round(time.time() - t0, 2), "metrics": met, "error": err, "result": res}
                with open(rec_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=float) + "\n")
                print(f"[{k}/{total}] {vname} sample={s['sample_idx']} {s['regime']} -> {met.get('status')} "
                      f"wzpa30={met.get('wzpa30')} ({rec['elapsed_s']}s){' ERROR ' + err if err else ''}", flush=True)


if __name__ == "__main__":
    main()
