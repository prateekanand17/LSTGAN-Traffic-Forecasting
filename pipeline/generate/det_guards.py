"""Deterministic safety gates applied to every zone advisory (stage 5b).

Retry-triggering gates (the violation text is sent back to the worker, up to config.MAX_RETRIES times):
  STRUCTURAL        schema / enum / valid-road violations   (pipeline.generate.schema.structural_errors)
  CONGESTED_TARGET  a reroute sends traffic onto a road covered by a zone forecast HIGH

Override gates (applied after the final attempt; each trigger is recorded in advisory["gates"]):
  SENSOR_FAULT_SKIP  > 50% of the zone's sensors had missing readings in the observed hour -> advisory suppressed
  SEVERE_OVERRIDE    zone mean forecast speed < GATE_SEVERE_OVERRIDE_MPH -> severity_30min and severity_60min = HIGH
  CASCADE_BLOCK      updated cascade risk p' > GATE_CASCADE_BLOCK_RISK -> upstream_alert = true
  RUSH_HOUR_BOOST    weekday rush hour -> rush_hour = true and the PREVENT tier must be non-empty (filled if empty)
"""
from typing import List

from pipeline import config as C
from pipeline.graph.zone_state import EnrichedZoneState
from pipeline.parse import ParsedData


def congested_roads(states: List[EnrichedZoneState]) -> set:
    return {r for s in states if s.zone.severity == "HIGH" for r in s.zone.roads}


def semantic_retry_errors(adv: dict, state: EnrichedZoneState, blocked_roads: set) -> List[str]:
    errs = []
    for i, r in enumerate(adv.get("actionable_reroutes", []) or []):
        if isinstance(r, dict) and r.get("to_road") in blocked_roads and r.get("to_road") not in state.zone.roads:
            errs.append(f"reroute {i}: to_road {r['to_road']} is covered by a zone forecast HIGH; choose another road "
                        f"or no reroute")
    return errs


class DeterministicGuards:
    def apply(self, adv: dict, state: EnrichedZoneState, parsed: ParsedData) -> dict:
        adv = dict(adv)
        gates = list(adv.get("gates", []))
        z = state.zone
        if z.n_faulty > 0.5 * len(z.sensor_indices):
            gates.append("SENSOR_FAULT_SKIP")
            adv["suppressed"] = True
        if z.avg_speed_60min < C.GATE_SEVERE_OVERRIDE_MPH:
            if adv.get("severity_30min") != "HIGH" or adv.get("severity_60min") != "HIGH":
                gates.append("SEVERE_OVERRIDE")
            adv["severity_30min"] = adv["severity_60min"] = "HIGH"
        if state.updated_cascade_risk > C.GATE_CASCADE_BLOCK_RISK and adv.get("upstream_alert") is not True:
            gates.append("CASCADE_BLOCK")
            adv["upstream_alert"] = True
        if parsed.is_rush_hour:
            adv["rush_hour"] = True
            a = adv.get("advisory") if isinstance(adv.get("advisory"), dict) else {}
            if not str(a.get("prevent", "")).strip():
                a = dict(a)
                a["prevent"] = "Rush-hour protocol: keep monitoring and pre-position response for the next 60 min."
                adv["advisory"] = a
                gates.append("RUSH_HOUR_BOOST")
        adv["gates"] = gates
        return adv
