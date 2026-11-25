"""Global Supervisor agent (central LLM coordinator, task N5): sees every zone's proposal, detects routing conflicts
and rewrites reroutes.

The supervisor is NOT given the deterministic conflict list; detection is its job. Its final reroutes are
validated per zone (from_road is a road of the zone, to_road is one of the zone's candidate corridors and not
blocked); invalid entries fall back to the zone's own proposal and the fallback is recorded. Conflicts are scored
independently in evaluation/metrics.py with the shared detector.
"""
import json
from typing import Dict, Optional

from pipeline import config as C
from pipeline.llm_client import LLMClient

SYSTEM = ("You are the Global Supervisor of a network of traffic advisory agents on the PeMS-BAY highway network. "
          "You receive every zone's proposal. Find routing conflicts: two or more zones sending traffic onto the same "
          "road, or a zone sending traffic onto a road covered by another congested zone. Resolve each conflict by "
          "changing the reroutes of the lower-priority zone (priority: HIGH > MEDIUM > LOW, then higher cascade risk), "
          "using only that zone's candidate corridors, or by dropping the reroute. Output ONLY one JSON object.")

OUTPUT_SPEC = {
    "conflicts": "list of {zones: [zone_id, ...], road: road ID, resolution: short text}; empty if none",
    "final_reroutes": "object mapping EVERY zone_id to its final list of {from_road, to_road, reason}",
}


def build_prompt(proposals: Dict[str, dict], ctxs) -> str:
    lines = ["Zones:"]
    blocked = set()
    for zid in sorted(proposals):
        ctx, p = ctxs[zid], proposals[zid]
        blocked |= set(ctx.blocked)
        lines.append(json.dumps({"zone_id": zid, "roads": ctx.state.zone.roads,
                                 "forecast_severity": ctx.state.zone.severity,
                                 "cascade_risk": round(ctx.state.updated_cascade_risk, 3),
                                 "candidate_corridors": ctx.candidates,
                                 "actionable_reroutes": p.get("actionable_reroutes", []) if p.get("valid") else []}))
    if blocked:
        lines += ["", f"Roads of zones forecast HIGH (never reroute onto them): {', '.join(sorted(blocked))}"]
    lines += ["", "Required JSON fields:", json.dumps(OUTPUT_SPEC, indent=1)]
    return "\n".join(lines)


class NetworkSupervisorAgent:
    def __init__(self, llm: LLMClient, model: str = C.SUPERVISOR_MODEL, temperature: float = C.SUPERVISOR_TEMPERATURE):
        self.llm, self.model, self.temperature = llm, model, temperature

    def integrate(self, proposals: Dict[str, dict], ctxs, meta: dict, dry_value: Optional[dict] = None) -> dict:
        dv = None if dry_value is None else {"conflicts": [], "final_reroutes": dry_value["final_reroutes"]}
        out = self.llm.chat_json("supervisor", self.model, SYSTEM, build_prompt(proposals, ctxs),
                                 self.temperature, C.MAX_TOKENS["supervisor"], meta, dry_value=dv)
        final, fallbacks = {}, []
        fr = out.get("final_reroutes") if isinstance(out.get("final_reroutes"), dict) else {}
        for zid, p in proposals.items():
            ctx = ctxs[zid]
            own = list(p.get("actionable_reroutes") or []) if p.get("valid") and not p.get("suppressed") else []
            cand = fr.get(zid)
            ok = isinstance(cand, list) and all(
                isinstance(r, dict) and r.get("from_road") in ctx.state.zone.roads
                and r.get("to_road") in ctx.candidates and r.get("to_road") not in ctx.blocked for r in cand)
            if ok and (p.get("valid") and not p.get("suppressed")):
                final[zid] = cand
            else:
                final[zid] = own
                if p.get("valid") and not p.get("suppressed"):
                    fallbacks.append(zid)
        return {"conflicts_reported": out.get("conflicts", []) if isinstance(out.get("conflicts"), list) else [],
                "final_reroutes": final, "fallback_zones": fallbacks, "parse_error": out.get("_parse_error")}
