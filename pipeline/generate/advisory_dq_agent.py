"""Advisory DQ agent (evaluation only since v2, plan §9.2): LLM plausibility check of one advisory against the zone state.

Its verdict is recorded, not used to change outputs. An API failure is recorded as status ERROR (the old
version returned PASS on errors). Deterministic faithfulness metrics are computed separately in
evaluation/metrics.py.
"""
import json

from pipeline import config as C
from pipeline.graph.zone_state import EnrichedZoneState
from pipeline.llm_client import LLMClient, LLMError

SYSTEM = "You are a quality-assurance checker for traffic advisories. Output ONLY one JSON object."


class AdvisoryQualityAgent:
    def __init__(self, llm: LLMClient, model: str = C.DQ_MODEL, temperature: float = C.DQ_TEMPERATURE):
        self.llm, self.model, self.temperature = llm, model, temperature

    def check(self, adv: dict, state: EnrichedZoneState, meta: dict) -> dict:
        z = state.zone
        user = (f"Zone state: roads={z.roads}, forecast 60-min mean speed={z.avg_speed_60min:.1f} mph, "
                f"forecast severity={z.severity}, trend={z.trend}.\n"
                f"Advisory: {json.dumps({k: adv.get(k) for k in ('severity_30min', 'severity_60min', 'advisory', 'actionable_reroutes', 'upstream_alert')})}\n"
                "Does the advisory contradict the zone state (wrong speeds, wrong severity wording, impossible "
                "actions)? Return JSON {\"status\": \"PASS\" or \"FAIL\", \"reason\": \"...\"}.")
        try:
            out = self.llm.chat_json("dq", self.model, SYSTEM, user, self.temperature, C.MAX_TOKENS["dq"],
                                     {**meta, "zone_id": z.zone_id},
                                     dry_value={"status": "PASS", "reason": "dry run"} if self.llm.dry_run else None)
        except LLMError as e:
            return {"status": "ERROR", "reason": str(e)}
        status = out.get("status") if out.get("status") in ("PASS", "FAIL") else "INVALID"
        return {"status": status, "reason": out.get("reason", out.get("_parse_error", ""))}
