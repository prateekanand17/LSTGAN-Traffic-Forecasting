"""MASTRA metrics computed from one scenario result (pipeline.orchestrator.run_scenario) and ground truth.

Severity metrics score the AGENT'S OWN predictions (advisory.severity_30min / severity_60min) against the
ground-truth zone severity at t+T (config scale on the mean observed speed of the zone's sensors at step T,
ignoring missing readings). Zones whose advisory failed the structural gate after all retries, or was
suppressed, count as wrong in the intent-to-treat (ITT) score and are excluded from the *_valid score.

Conflicts use the single shared detector (pipeline.coordination.common.detect_conflicts, task N7):
  - two or more zones reroute onto the same to_road, or
  - a zone reroutes onto a road covered by another detected zone.
They are counted on the round-0 proposals (initial) and on the FINAL plan (after negotiation / coordinator).
conflict_free = 1 if the final plan has no conflict. CRR = share of the initial conflicts that are absent from the
final plan AND whose final destination roads are free-flow (>= SEV_MEDIUM_BELOW mph) in ground truth at +30 min.
Coordination metrics come from result["coordination"]: rounds used, messages exchanged, neighbour-pair density of
the topology, fallback use. implausible_route_rate = share of final reroute targets outside the zone's candidate
corridors. LLM usage (calls, tokens, latency) comes from result["llm_usage"].
"""
import re
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np

from pipeline import config as C
from pipeline.coordination.common import detect_conflicts
from pipeline.generate.schema import hallucinated_routes

ORD = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def gt_severity(actual: np.ndarray, idx: List[int], step: int) -> Optional[str]:
    v = actual[step - 1, idx]
    v = v[v > C.SENSOR_FAULT_SPEED]
    return C.speed_to_severity(float(v.mean())) if v.size else None


def wscore(pred: Optional[str], gt: str) -> float:
    if pred not in ORD:
        return 0.0
    d = abs(ORD[pred] - ORD[gt])
    return 1.0 if d == 0 else (0.5 if d == 1 else 0.0)


def road_speed(actual: np.ndarray, roads: List[str], road: str, step: int) -> Optional[float]:
    idx = [i for i, r in enumerate(roads) if r == road]
    v = actual[step - 1, idx]
    v = v[v > C.SENSOR_FAULT_SPEED]
    return float(v.mean()) if v.size else None


MPH = re.compile(r"(\d+(?:\.\d+)?)\s*mph", re.I)


def scenario_metrics(res: dict, actual: np.ndarray, roads: List[str]) -> dict:
    m = {"status": res["status"], "n_zones": res["n_zones"]}
    if res["status"] != "SUCCESS":
        return m
    zs = res["zones"]
    s30 = s30v = s60 = s60v = e30 = e60 = det30 = det60 = 0.0
    nvalid = n = 0
    false_alarm = gt_low = 0
    halluc_bad = halluc_tot = 0
    faithful = mph_total = 0
    attempts = []
    gates = {}
    dq = {}
    for z in zs:
        adv = z["advisory"]
        g30 = gt_severity(actual, z["sensor_indices"], 6)
        g60 = gt_severity(actual, z["sensor_indices"], 12)
        if g30 is None or g60 is None:
            continue
        n += 1
        ok = bool(adv.get("valid")) and not adv.get("suppressed")
        p30 = adv.get("severity_30min") if ok else None
        p60 = adv.get("severity_60min") if ok else None
        s30 += wscore(p30, g30); s60 += wscore(p60, g60)
        e30 += float(p30 == g30); e60 += float(p60 == g60)
        det30 += wscore(z["detector_severity"], g30); det60 += wscore(z["detector_severity"], g60)
        if ok:
            nvalid += 1
            s30v += wscore(p30, g30); s60v += wscore(p60, g60)
            if g30 == "LOW":
                gt_low += 1
                false_alarm += int(p30 in ("HIGH", "MEDIUM"))
            b, t = hallucinated_routes(adv, sorted(set(roads)))
            halluc_bad += b; halluc_tot += t
            ref = [z["forecast_30min"], z["forecast_60min"], z["avg_speed_60min"]]
            text = " ".join(str(v) for v in (adv.get("advisory") or {}).values()) if isinstance(adv.get("advisory"), dict) else ""
            for num in MPH.findall(text):
                mph_total += 1
                faithful += int(min(abs(float(num) - r) for r in ref) <= 5.0)
        attempts.append(adv.get("attempts", 0))
        for gname in adv.get("gates", []) or []:
            gates[gname] = gates.get(gname, 0) + 1
        if z.get("dq"):
            dq[z["dq"]["status"]] = dq.get(z["dq"]["status"], 0) + 1
    if n:
        m.update(wzpa30=s30 / n, wzpa60=s60 / n, zpa30_exact=e30 / n, zpa60_exact=e60 / n,
                 det_wzpa30=det30 / n, det_wzpa60=det60 / n, n_scored_zones=n,
                 valid_rate=nvalid / n)
    if nvalid:
        m.update(wzpa30_valid=s30v / nvalid, wzpa60_valid=s60v / nvalid)
    if gt_low:
        m["false_alarm_rate30"] = false_alarm / gt_low
    if halluc_tot:
        m["hallucinated_route_rate"] = halluc_bad / halluc_tot
    if mph_total:
        m["numeric_faithfulness"] = faithful / mph_total
    m["mean_attempts"] = float(np.mean(attempts)) if attempts else None
    m["gate_triggers"] = gates
    m["dq_status"] = dq

    # CDR (deterministic, identical for variants that share the forecast)
    casc = sorted({s for z in zs for s in z["cascade_risk_raw"]})
    if casc:
        fut = actual[-1, casc]
        valid = fut > C.SENSOR_FAULT_SPEED
        if valid.any():
            m["cdr"] = float(np.mean(fut[valid] < C.SEV_MEDIUM_BELOW))

    # conflicts / CRR (shared detector, task N7)
    ok_z = {z["zone_id"]: z for z in zs if z["advisory"].get("valid") and not z["advisory"].get("suppressed")}
    zone_roads = {z["zone_id"]: z["roads"] for z in zs}
    init_rr = {zid: z.get("initial_reroutes", []) or [] for zid, z in ok_z.items()}
    final_rr = {zid: z["advisory"].get("actionable_reroutes", []) or [] for zid, z in ok_z.items()}
    fp = res.get("footprints")   # present only for segment-level runs (plan §13 F2c)
    fp = {z: {r: frozenset(v) for r, v in f.items()} for z, f in fp.items()} if fp else None
    c_init = detect_conflicts(init_rr, zone_roads, fp)
    c_final = detect_conflicts(final_rr, zone_roads, fp)
    m["n_conflicts_initial"] = len(c_init)
    m["n_conflicts_final"] = len(c_final)
    m["conflict_free"] = float(len(c_final) == 0)
    if c_init:
        resolved = 0
        for c in c_init:
            if c in c_final:
                continue
            dests = {r.get("to_road") for zid in c[2:] for r in final_rr.get(zid, []) if isinstance(r, dict)}
            speeds = [road_speed(actual, roads, d, 6) for d in dests]
            if all(s is not None and s >= C.SEV_MEDIUM_BELOW for s in speeds):
                resolved += 1
        m["crr"] = resolved / len(c_init)
    dests = [r.get("to_road") for rr in final_rr.values() for r in rr if isinstance(r, dict)]
    m["n_reroutes_final"] = len(dests)
    sp = [road_speed(actual, roads, d, 6) for d in dests]
    sp = [s for s in sp if s is not None]
    if sp:
        m["reroute_target_freeflow_rate"] = float(np.mean([s >= C.SEV_MEDIUM_BELOW for s in sp]))
    n_t = n_bad = 0
    for zid, rr in final_rr.items():
        cand = set(ok_z[zid].get("candidates") or [])
        for r in rr:
            if isinstance(r, dict):
                n_t += 1
                n_bad += r.get("to_road") not in cand
    if n_t:
        m["implausible_route_rate"] = n_bad / n_t

    co = res.get("coordination") or {}
    if co:
        m["rounds_used"] = co.get("rounds_used", 0)
        m["messages"] = co.get("messages", 0)
        m["n_neighbor_pairs"] = co.get("n_neighbor_pairs")
        ld = co.get("link_density")
        if ld is not None and ld == ld:
            m["link_density"] = ld
        m["fallback_used"] = float(bool(co.get("fallback_used")))
        m["stop_reason"] = co.get("stop_reason")
        cd = co.get("coordinator") or {}
        if cd.get("kind") == "llm":
            m["coordinator_fallback_zones"] = len(cd.get("fallback_zones", []))
    u = res.get("llm_usage") or {}
    for k in ("calls", "prompt_tokens", "completion_tokens", "latency_s", "parse_errors"):
        if k in u:
            m[f"llm_{k}"] = u[k]
    return m
