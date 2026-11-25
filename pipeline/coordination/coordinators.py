"""Central coordinators (task N5): the centralized cells of the 2 x 2 design and the fallback after negotiation.

rule : greedy by priority (common.priority). Zones are processed from highest to lowest priority; a zone keeps a
       reroute target if no higher-priority zone already uses it, it is not a road of another zone and it is not
       blocked; otherwise it takes its fastest free candidate corridor, or drops the reroute.
llm  : the Global Supervisor (pipeline/generate/supervisor_agent.py) sees every zone's proposal and candidate
       corridors and rewrites reroutes; invalid entries fall back to the zone's own proposal.
Both have a global view (every zone), unlike negotiating agents, which see only their neighbours.
"""
from typing import Dict, List

from pipeline.coordination.agents import ZoneContext
from pipeline.coordination.common import priority
from pipeline.generate.rules import pick_target


def central_rule(proposals: Dict[str, dict], ctxs: Dict[str, ZoneContext], zone_roads: Dict[str, List[str]]) -> dict:
    order = sorted(proposals, key=lambda z: priority(ctxs[z].state), reverse=True)
    used, final, changed = set(), {}, []
    for zid in order:
        p, ctx = proposals[zid], ctxs[zid]
        rr = list(p.get("actionable_reroutes") or []) if p.get("valid") and not p.get("suppressed") else []
        other_roads = {r for z, roads in zone_roads.items() if z != zid for r in roads}
        avoid = used | other_roads | set(ctx.blocked) | set(ctx.state.zone.roads)
        out = []
        for r in rr:
            tgt = r.get("to_road")
            if tgt in avoid:
                tgt = pick_target(ctx.candidates, avoid)
            if tgt:
                used.add(tgt)
                avoid.add(tgt)
                out.append({"from_road": r.get("from_road"), "to_road": tgt,
                            "reason": r.get("reason") if tgt == r.get("to_road") else "reassigned by coordinator"})
        final[zid] = out
        if out != rr:
            changed.append(zid)
    return {"kind": "rule", "final_reroutes": final, "changed_zones": changed}


def central_llm(llm, proposals: Dict[str, dict], ctxs: Dict[str, ZoneContext], zone_roads: Dict[str, List[str]],
                meta: dict) -> dict:
    from pipeline.generate.supervisor_agent import NetworkSupervisorAgent
    dry = central_rule(proposals, ctxs, zone_roads) if getattr(llm, "dry_run", False) else None
    out = NetworkSupervisorAgent(llm).integrate(proposals, ctxs, meta, dry_value=dry)
    out["kind"] = "llm"
    out["changed_zones"] = [z for z in out["final_reroutes"]
                            if out["final_reroutes"][z] != list(proposals[z].get("actionable_reroutes") or [])]
    return out


def coordinate(kind: str, llm, proposals, ctxs, zone_roads, meta) -> dict:
    if kind == "rule":
        return central_rule(proposals, ctxs, zone_roads)
    if kind == "llm":
        return central_llm(llm, proposals, ctxs, zone_roads, meta)
    raise ValueError(f"unknown coordinator {kind}")
