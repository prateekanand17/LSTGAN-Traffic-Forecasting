"""LLM zone agent (task N3): propose and revise calls with the validation-and-retry loop.

Propose prompt = (1) system context, (2) zone state, (3) neighbour context from GMP (if enabled), (4) the zone's
candidate corridors with forecast speeds and the blocked roads, (5) required fields, (6) two fixed format examples.
Revise prompt = propose prompt + the agent's current proposal, its neighbours' proposals (with priority relative to
its own) and the conflicts visible to it, plus the conflict rule.
Output schema: pipeline.generate.schema.ADVISORY_FIELDS. Retry: structural (to_road must be a candidate corridor)
and CONGESTED_TARGET violations are sent back with the error list, up to config.MAX_RETRIES.
A revision that still fails the gates after all retries is discarded and the previous proposal is kept.
"""
import json
from typing import List, Optional

import numpy as np

from pipeline import config as C
from pipeline.generate.det_guards import semantic_retry_errors
from pipeline.generate.schema import ADVISORY_FIELDS, structural_errors
from pipeline.llm_client import LLMClient

SYSTEM = ("You are a traffic advisory agent for one congestion zone on the PeMS-BAY highway network "
          "(San Francisco Bay Area). You write decision support for Traffic Management Center operators. "
          "Output ONLY one JSON object that matches the required schema. Do not include explanations outside JSON.")

FEW_SHOT = [
    {"zone_id": "zone_example1", "severity_30min": "HIGH", "severity_60min": "HIGH",
     "advisory": {"now": "Severe congestion forecast (22 mph). Post 'slow traffic ahead' on message signs.",
                  "soon": "Activate upstream ramp metering; alert downstream operators.",
                  "prevent": "Recommend alternate corridor for through traffic until speeds recover."},
     "actionable_reroutes": [{"from_road": "101-N", "to_road": "280-N", "reason": "parallel corridor forecast free-flow"}],
     "upstream_alert": True},
    {"zone_id": "zone_example2", "severity_30min": "MEDIUM", "severity_60min": "LOW",
     "advisory": {"now": "Moderate slowdown forecast (44 mph). Inform drivers of delays.",
                  "soon": "Monitor; congestion forecast to ease.",
                  "prevent": "No action needed if recovery continues."},
     "actionable_reroutes": [], "upstream_alert": False},
]

CONFLICT_RULE = ("Conflict rule: the zone with higher priority keeps a contested road (priority = forecast severity, "
                 "then cascade risk); no zone may reroute traffic onto a road covered by a neighbouring zone.")


def zone_forecast_at(inp, idx: List[int], step: int) -> float:
    return float(np.mean(inp.pred_speed[step - 1, idx]))


def build_prompt(ctx) -> str:
    state, inp = ctx.state, ctx.inp
    z = state.zone
    lines = [
        f"Zone ID: {z.zone_id}",
        f"Roads in this zone: {', '.join(z.roads)}",
        f"Sensors: {len(z.sensor_indices)} (with missing readings in the last hour: {z.n_faulty})",
        f"Forecast origin: {getattr(inp, 'timestamp', 'unknown')}" + (" (weekday rush hour)" if ctx.is_rush else ""),
        f"Forecast mean speed: +30 min {zone_forecast_at(inp, z.sensor_indices, 6):.1f} mph, "
        f"+60 min {zone_forecast_at(inp, z.sensor_indices, 12):.1f} mph "
        f"(60-min average {z.avg_speed_60min:.1f} mph, minimum sensor {z.min_speed_60min:.1f} mph)",
        f"Forecast trend: {z.trend}",
        f"Cascade risk (after network messages): {state.updated_cascade_risk:.2f}; network pressure: {state.network_pressure:.3f}",
        f"Severity scale: HIGH < {C.SEV_HIGH_BELOW:.0f} mph, MEDIUM {C.SEV_HIGH_BELOW:.0f}-{C.SEV_MEDIUM_BELOW:.0f} mph, "
        f"LOW >= {C.SEV_MEDIUM_BELOW:.0f} mph (zone mean speed).",
    ]
    if ctx.include_neighbors:
        lines += ["", state.neighbor_context_text]
    if ctx.candidates:
        cand = ", ".join(f"{r} ({ctx.candidate_speeds.get(r, float('nan')):.0f} mph)" for r in ctx.candidates)
        lines += ["", f"Candidate reroute corridors (other freeways within {C.REROUTE_RADIUS_KM:g} km, forecast speed "
                      f"at +30 min, fastest first): {cand}"]
    else:
        lines += ["", "Candidate reroute corridors: none (actionable_reroutes must be empty)."]
    if ctx.blocked:
        lines.append(f"Roads of zones forecast HIGH (never reroute onto them): {', '.join(sorted(ctx.blocked))}")
    lines += ["", "Required JSON fields:", json.dumps(ADVISORY_FIELDS, indent=1),
              "", "Examples of the expected format (roads in the examples are illustrative only):"] + \
             [json.dumps(e) for e in FEW_SHOT]
    return "\n".join(lines)


def describe_conflict(c: tuple, me: str) -> str:
    kind, road, a, b = c
    if kind == "shared":
        other = b if a == me else a
        return f"you and {other} both reroute traffic onto {road}"
    if a == me:
        return f"your reroute sends traffic onto {road}, a road of neighbouring zone {b}"
    return f"{a} reroutes traffic onto {road}, a road of your zone"


def revision_block(ctx, own: dict, inbox, conflicts, rnd: int) -> str:
    from pipeline.coordination.common import priority
    mine = priority(ctx.state)
    keep = ("severity_30min", "severity_60min", "actionable_reroutes", "upstream_alert")
    lines = ["", f"NEGOTIATION ROUND {rnd}.", "Your current proposal: " + json.dumps({k: own.get(k) for k in keep}),
             "Proposals of your neighbouring zones in the communication graph:"]
    if inbox:
        for m in inbox:
            rel = "higher" if m.priority > mine else "lower"
            lines.append(f"- {m.zone_id} (roads {', '.join(m.roads)}; severity {m.severity}; priority {rel} than "
                         f"yours): reroutes {json.dumps(m.reroutes)}")
    else:
        lines.append("- none")
    lines.append("Conflicts you are involved in:")
    lines += [f"- {describe_conflict(c, ctx.zone_id)}" for c in conflicts] or ["- none"]
    lines += [CONFLICT_RULE,
              "Revise your advisory: change or drop the reroutes that break the rule (use only your candidate "
              "corridors). You may also update the severities and the text. Return the full JSON object."]
    return "\n".join(lines)


class ZoneWorkerAgent:
    def __init__(self, llm: LLMClient, model: str = C.WORKER_MODEL, temperature: float = C.WORKER_TEMPERATURE):
        self.llm, self.model, self.temperature = llm, model, temperature

    def _call_with_retry(self, ctx, prompt: str, use_retry: bool, meta: dict, dry_value: Optional[dict]) -> dict:
        attempts = 1 + (C.MAX_RETRIES if use_retry else 0)
        history, adv = [], {}
        zid = ctx.zone_id
        for attempt in range(attempts):
            user = prompt if attempt == 0 else (
                prompt + "\n\nYour previous output was rejected by the validation gate:\n" +
                json.dumps(adv)[:3000] + "\nErrors:\n- " + "\n- ".join(history[-1]) +
                "\nReturn a corrected JSON object.")
            dv = None if dry_value is None else {k: v for k, v in dry_value.items()
                                                  if k not in ("attempts", "valid", "gate_errors")}
            adv = self.llm.chat_json("worker", self.model, SYSTEM, user, self.temperature, C.MAX_TOKENS["worker"],
                                     {**meta, "zone_id": zid, "attempt": attempt}, dry_value=dv)
            errs = structural_errors(adv, zid, ctx.state.zone.roads, ctx.candidates)
            if not errs:
                errs = semantic_retry_errors(adv, ctx.state, ctx.blocked)
            if not errs:
                return {**adv, "attempts": attempt + 1, "valid": True, "gate_errors": history}
            history.append(errs)
        return {**adv, "attempts": attempts, "valid": False, "gate_errors": history}

    def generate(self, ctx, use_retry: bool = True, dry_value: Optional[dict] = None) -> dict:
        return self._call_with_retry(ctx, build_prompt(ctx), use_retry, {**ctx.meta, "round": 0}, dry_value)

    def revise(self, ctx, own: dict, inbox, conflicts, rnd: int, use_retry: bool = True,
               dry_value: Optional[dict] = None) -> dict:
        prompt = build_prompt(ctx) + "\n" + revision_block(ctx, own, inbox, conflicts, rnd)
        out = self._call_with_retry(ctx, prompt, use_retry, {**ctx.meta, "round": rnd}, dry_value)
        if out.get("valid"):
            return {**out, "attempts": own.get("attempts", 0) + out["attempts"]}
        # failed revision: keep the previous (valid) proposal and record the failure
        return {**own, "revision_failures": own.get("revision_failures", 0) + 1,
                "gate_errors": list(own.get("gate_errors", [])) + out.get("gate_errors", [])}
