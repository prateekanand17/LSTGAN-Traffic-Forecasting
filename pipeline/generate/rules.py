"""Deterministic advisory content (no LLM): used by RuleAgent and by the explicit dry-run stand-in.

Severity: the detector severity at both horizons (generator "rules"), or the config scale applied to the zone's own
+30 / +60 min forecast (generator "rules_h"). Reroute: from the zone's first road to the fastest candidate corridor
(forecast speed at +30 min) that is not blocked, only for HIGH / MEDIUM zones.
"""
from typing import List, Optional, Set

from pipeline import config as C
from pipeline.graph.zone_state import EnrichedZoneState


def pick_target(candidates: List[str], exclude: Set[str]) -> Optional[str]:
    for r in candidates:
        if r not in exclude:
            return r
    return None


def rule_advisory(state: EnrichedZoneState, candidates: List[str], blocked: Set[str],
                  sev30: str = None, sev60: str = None) -> dict:
    z = state.zone
    sev30 = sev30 or z.severity
    sev60 = sev60 or z.severity
    reroutes = []
    if z.severity in ("HIGH", "MEDIUM") and z.roads:
        tgt = pick_target(candidates, set(blocked) | set(z.roads))
        if tgt:
            reroutes.append({"from_road": z.roads[0], "to_road": tgt, "reason": "fastest free candidate corridor"})
    return {
        "zone_id": z.zone_id,
        "severity_30min": sev30,
        "severity_60min": sev60,
        "advisory": {
            "now": f"{z.severity} congestion forecast ({z.avg_speed_60min:.0f} mph). Post warning on message signs.",
            "soon": "Alert downstream operators." if state.updated_cascade_risk > 0.5 else "Monitor zone sensors.",
            "prevent": "Consider upstream ramp metering." if z.trend == "WORSENING" else "Continue monitoring.",
        },
        "actionable_reroutes": reroutes,
        "upstream_alert": bool(state.updated_cascade_risk > C.GATE_CASCADE_BLOCK_RISK),
    }
