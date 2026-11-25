"""MASTRA orchestrator (v2, negotiating zone agents; plan §9): the single code path used by the evaluation runner
and the dashboard.

Stages:
  1 parse -> 2 forecast gate (recorded) -> 3 zone detection (+ cascade BFS on the distance graph) ->
  4 numeric context: GMP over the communication topology (inp.comm_adj) ->
  5 candidate corridors per zone, blocked roads (zones forecast HIGH), neighbour sets N(i) over the same topology ->
  6 round 0: every zone agent proposes (rule or LLM; LLM calls run the retry gates) ->
  7 coordination: none | negotiate (<= K revise rounds over N(i), optional central fallback) | central_rule |
    central_llm ->
  8 override gates on the final advisories -> result dict (zones, initial and final reroutes, transcript, usage).
The DQ agent is not in the main path (plan §9.2); Variant.dq runs it on the final advisories and only records it.
Every component is a real switch in `Variant`.
"""
import copy
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

from pipeline import config as C
from pipeline.coordination.agents import RuleAgent, ZoneContext, make_agent
from pipeline.coordination.common import (candidate_corridors, corridor_neighbors, link_density, neighbors,
                                          zone_footprints)
from pipeline.coordination.coordinators import coordinate
from pipeline.coordination.negotiation import negotiate, plan_conflicts
from pipeline.generate.advisory_dq_agent import AdvisoryQualityAgent
from pipeline.generate.det_guards import DeterministicGuards, congested_roads
from pipeline.generate.schema import structural_errors
from pipeline.generate.zone_worker import SYSTEM as WORKER_SYSTEM, build_prompt as worker_prompt
from pipeline.graph.message_passing import GraphMessagePassingLayer, no_communication
from pipeline.inputs import PipelineInput
from pipeline.llm_client import LLMClient
from pipeline.parse import Parser
from pipeline.validate.forecast_gate import ForecastIntegrityGate
from pipeline.zone_detection import ZoneDetector

AGENTS = ("rules", "rules_h", "llm", "single")
COORDINATIONS = ("none", "negotiate", "central_rule", "central_llm")


@dataclass
class Variant:
    name: str
    agent: str = "llm"                  # rules | rules_h | llm | single (one LLM call for all zones)
    coordination: str = "negotiate"     # none | negotiate | central_rule | central_llm
    topology: str = "learned"           # pipeline.data_source.topology name; also defines N(i)
    gmp: bool = True                    # numeric neighbour context in the zone state / prompt
    rounds: int = C.NEGOTIATION_ROUNDS  # K, negotiate only
    fallback: Optional[str] = None      # None | "rule" | "llm": central coordinator if conflicts remain after K rounds
    gates: bool = True                  # retry gates (LLM calls) + override gates
    dq: bool = False                    # record an LLM DQ verdict on the final advisories (evaluation only)
    forecast_mode: str = "forecast"     # forecast | reactive | oracle
    gmp_weight_agg: str = C.GMP_WEIGHT_AGG
    gmp_pressure: str = C.GMP_PRESSURE
    # robustness switches (plan §13 F2); the defaults are the main configuration
    radius_km: float = C.REROUTE_RADIUS_KM       # candidate corridors within this radius of the zone centre
    dbscan_eps: float = C.DBSCAN_EPS_DEG
    neighbor_threshold: float = C.NEIGHBOR_THRESHOLD
    conflict_level: str = "road"                 # road (whole road-direction) | segment (footprints overlap)
    neighbor_rule: str = "topology"              # topology (N(i) from comm_adj) | corridor (corridor-overlap)

    def __post_init__(self):
        if self.agent not in AGENTS or self.coordination not in COORDINATIONS:
            raise ValueError(f"bad variant {self}")
        if self.agent == "single" and self.coordination != "none":
            raise ValueError("the single-LLM baseline has no coordination stage")
        if self.fallback not in (None, "rule", "llm"):
            raise ValueError(f"bad fallback {self.fallback}")
        if self.conflict_level not in ("road", "segment") or self.neighbor_rule not in ("topology", "corridor"):
            raise ValueError(f"bad robustness switch {self}")

    @property
    def needs_llm(self) -> bool:
        return self.agent in ("llm", "single") or self.coordination == "central_llm" or self.fallback == "llm" or self.dq


def _usage_delta(llm, before):
    if llm is None or before is None:
        return None
    now = llm.snapshot()
    return {k: (round(now[k] - before[k], 3) if isinstance(now[k], float) else now[k] - before[k]) for k in now}


def _reroutes(p: dict) -> list:
    return list(p.get("actionable_reroutes") or []) if p.get("valid") and not p.get("suppressed") else []


def run_scenario(inp: PipelineInput, variant: Variant, llm: Optional[LLMClient] = None, meta: Optional[dict] = None,
                 dry_run: bool = False, log_dir: Optional[str] = None) -> dict:
    meta = dict(meta or {}, variant=variant.name)
    parsed = Parser().parse(inp)
    gate = ForecastIntegrityGate().check(inp, parsed)
    zones = ZoneDetector(eps=variant.dbscan_eps).detect(inp, parsed)
    base = {"variant": asdict(variant), "timestamp": getattr(inp, "timestamp", None), "t0": inp.timestamp_idx,
            "forecast_gate": {"passed": gate.passed, "reason": gate.reason}, "is_rush_hour": parsed.is_rush_hour,
            "n_zones": len(zones)}
    if not zones:
        return {**base, "status": "NO_CONGESTION", "zones": [], "coordination": None, "llm_usage": None}

    A = getattr(inp, "comm_adj", None)
    gmp_inp = inp
    if getattr(inp, "gmp_adj", None) is not None:   # F2: priorities from a fixed graph, independent of the topology
        gmp_inp = copy.copy(inp)
        gmp_inp.comm_adj = inp.gmp_adj
    if variant.gmp:
        states = GraphMessagePassingLayer(variant.gmp_weight_agg, variant.gmp_pressure).run(zones, gmp_inp)
    else:
        states = no_communication(zones)
    by_id = {s.zone.zone_id: s for s in states}
    blocked = congested_roads(states)
    zone_roads = {zid: list(s.zone.roads) for zid, s in by_id.items()}

    ctxs: Dict[str, ZoneContext] = {}
    for zid, s in by_id.items():
        cand = candidate_corridors(s, inp, variant.radius_km)
        ctxs[zid] = ZoneContext(state=s, inp=inp, is_rush=parsed.is_rush_hour, candidates=[r for r, _ in cand],
                                candidate_speeds=dict(cand), blocked=set(blocked), include_neighbors=variant.gmp,
                                meta=meta)
    cands = {zid: c.candidates for zid, c in ctxs.items()}
    fp = zone_footprints(states, inp, cands, variant.radius_km) if variant.conflict_level == "segment" else None
    if variant.neighbor_rule == "corridor":
        nb = corridor_neighbors(states, cands, fp)
    else:
        nb = neighbors(states, A, variant.neighbor_threshold)

    if variant.needs_llm and llm is None:
        llm = LLMClient(log_dir=log_dir, dry_run=dry_run)
    usage0 = llm.snapshot() if (llm is not None and variant.needs_llm) else None

    # --- round 0: proposals ------------------------------------------------------------------------------
    if variant.agent == "single":
        proposals = _single_agent(llm, ctxs, meta, variant.gates)
        agents = {}
    else:
        agents = {zid: make_agent(variant.agent, llm, use_retry=variant.gates) for zid in ctxs}
        proposals = {zid: agents[zid].propose(ctxs[zid]) for zid in ctxs}
    initial = copy.deepcopy(proposals)

    # --- coordination --------------------------------------------------------------------------------------
    coord = {"mode": variant.coordination, "topology": variant.topology,
             "n_neighbor_pairs": sum(len(v) for v in nb.values()) // 2, "link_density": link_density(nb),
             "rounds_used": 0, "stop_reason": None, "messages": 0, "transcript": [],
             "fallback_used": False, "coordinator": None}
    final_rr = {zid: _reroutes(p) for zid, p in proposals.items()}
    if variant.coordination == "negotiate":
        neg = negotiate(agents, ctxs, proposals, nb, zone_roads, variant.rounds, fp)
        proposals = neg["proposals"]
        coord.update(rounds_used=neg["rounds_used"], stop_reason=neg["stop_reason"], messages=neg["messages"],
                     transcript=neg["transcript"])
        final_rr = {zid: _reroutes(p) for zid, p in proposals.items()}
        if variant.fallback and plan_conflicts(proposals, zone_roads, fp):
            c = coordinate(variant.fallback, llm, proposals, ctxs, zone_roads, meta)
            final_rr.update(c["final_reroutes"])
            coord.update(fallback_used=True, coordinator=c)
    elif variant.coordination in ("central_rule", "central_llm"):
        c = coordinate(variant.coordination.split("_")[1], llm, proposals, ctxs, zone_roads, meta)
        final_rr.update(c["final_reroutes"])
        coord["coordinator"] = c

    # --- final advisories, override gates, optional DQ record -------------------------------------------------
    guards = DeterministicGuards()
    dq_agent = AdvisoryQualityAgent(llm) if variant.dq else None
    zone_records = []
    for zid, s in by_id.items():
        adv = dict(proposals[zid])
        if adv.get("valid") and not adv.get("suppressed"):
            adv["actionable_reroutes"] = final_rr.get(zid, [])
        if variant.gates and adv.get("valid"):
            adv = guards.apply(adv, s, parsed)
        dq = dq_agent.check(adv, s, meta) if (dq_agent and adv.get("valid")) else None
        zone_records.append({
            "zone_id": zid, "sensor_indices": s.zone.sensor_indices, "roads": s.zone.roads,
            "detector_severity": s.zone.severity, "avg_speed_60min": s.zone.avg_speed_60min,
            "forecast_30min": float(np.mean(inp.pred_speed[5, s.zone.sensor_indices])),
            "forecast_60min": float(np.mean(inp.pred_speed[11, s.zone.sensor_indices])),
            "trend": s.zone.trend, "cascade_risk_raw": s.zone.cascade_risk_raw,
            "updated_cascade_risk": s.updated_cascade_risk, "network_pressure": s.network_pressure,
            "n_gmp_neighbors": len(s.neighbor_messages), "neighbors": sorted(nb[zid]), "n_faulty": s.zone.n_faulty,
            "candidates": ctxs[zid].candidates,
            "initial_reroutes": _reroutes(initial[zid]),
            "advisory": {k: v for k, v in adv.items() if k not in ("gate_errors",)},
            "gate_errors": adv.get("gate_errors", []), "dq": dq,
        })
    out = {**base, "status": "SUCCESS", "zones": zone_records, "coordination": coord,
           "llm_usage": _usage_delta(llm, usage0)}
    if fp is not None:
        out["footprints"] = {z: {r: sorted(v) for r, v in f.items()} for z, f in fp.items()}
    return out


def _single_agent(llm, ctxs: Dict[str, ZoneContext], meta, use_retry):
    """Single-LLM baseline: one call sees every zone (no neighbour messages) and returns all advisories."""
    blocks = []
    for ctx in ctxs.values():
        c = copy.copy(ctx)
        c.include_neighbors = False
        blocks.append(worker_prompt(c))
    prompt = ("Write one advisory for EACH zone below. Return JSON {\"advisories\": [ ...one object per zone... ]}.\n\n"
              + "\n\n=====\n\n".join(blocks))
    dry = None
    if getattr(llm, "dry_run", False):
        ra = RuleAgent(per_horizon=True)
        dry = {"advisories": [{k: v for k, v in ra.propose(ctx).items() if k not in ("attempts", "valid", "gate_errors")}
                              for ctx in ctxs.values()]}
    attempts = 1 + (C.MAX_RETRIES if use_retry else 0)
    got, errs = {}, []
    for attempt in range(attempts):
        user = prompt if attempt == 0 else prompt + "\n\nPrevious output rejected:\n- " + "\n- ".join(errs) + \
            "\nReturn a corrected JSON object."
        out = llm.chat_json("single", C.WORKER_MODEL, WORKER_SYSTEM, user, C.WORKER_TEMPERATURE,
                            C.MAX_TOKENS["worker"] * max(1, len(ctxs)), {**meta, "attempt": attempt}, dry_value=dry)
        got = {a.get("zone_id"): a for a in out.get("advisories", []) if isinstance(a, dict)} \
            if isinstance(out.get("advisories"), list) else {}
        errs = []
        for zid, ctx in ctxs.items():
            if zid not in got:
                errs.append(f"missing advisory for {zid}")
            else:
                errs += [f"{zid}: {e}" for e in structural_errors(got[zid], zid, ctx.state.zone.roads, ctx.candidates)]
        if not errs:
            break
    res = {}
    for zid, ctx in ctxs.items():
        a = got.get(zid, {"_parse_error": "missing"})
        e = structural_errors(a, zid, ctx.state.zone.roads, ctx.candidates)
        res[zid] = {**a, "attempts": attempt + 1, "valid": not e, "gate_errors": [e] if e else []}
    return res


class MASTraPipeline:
    """Compatibility wrapper for src/app.py (dashboard). Uses run_scenario with the main system (LLM agents,
    negotiation over the learned topology, rule fallback) and a live LLM; raises if no API key is configured."""

    def run(self, inp: PipelineInput, eval_mode: bool = False):
        from types import SimpleNamespace
        from pipeline import data_source as D
        if getattr(inp, "comm_adj", None) is None:
            inp.comm_adj = D.learned_adj(C.DEFAULT_FORECAST_SEED)
        if getattr(inp, "roads", None) is None:
            st = D.load_stations().set_index("station_id")
            inp.roads = [f"{st.loc[s, 'road_name']}-{st.loc[s, 'direction']}" for s in inp.station_ids]
        res = run_scenario(inp, Variant("dashboard", agent="llm", coordination="negotiate",
                                        fallback="rule" if C.COORDINATOR_FALLBACK else None))
        return SimpleNamespace(status=res["status"], integrated_plan=res, outputs=res)
