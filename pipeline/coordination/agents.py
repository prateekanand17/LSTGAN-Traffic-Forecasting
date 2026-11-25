"""Zone agents with one interface, so rule agents and LLM agents run inside the same coordination protocols.

Context passed to an agent (ZoneContext) is local: its own state, its candidate corridors, the globally blocked
roads (safety layer: roads of zones forecast HIGH) and, when revising, ONLY the proposals of its neighbours in the
communication topology plus the conflicts between itself and those neighbours (common.conflicts_visible_to).

Conflict rule shared by both agent types (and by the central rule coordinator):
  - the zone with higher priority (common.priority) keeps a contested target;
  - a zone must not reroute onto a road covered by a neighbouring zone.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from pipeline import config as C
from pipeline.coordination.common import priority, targets_of
from pipeline.generate.rules import pick_target, rule_advisory
from pipeline.graph.zone_state import EnrichedZoneState
from pipeline.inputs import PipelineInput


@dataclass
class ZoneContext:
    state: EnrichedZoneState
    inp: PipelineInput
    is_rush: bool
    candidates: List[str]                     # ranked candidate corridors
    candidate_speeds: Dict[str, float]        # forecast mean speed at +30 min
    blocked: Set[str]                         # roads of zones forecast HIGH (safety layer, visible to every agent)
    include_neighbors: bool = True            # GMP neighbour context in the prompt
    meta: dict = field(default_factory=dict)

    @property
    def zone_id(self) -> str:
        return self.state.zone.zone_id


@dataclass
class NeighborMsg:
    zone_id: str
    roads: List[str]
    severity: str
    cascade_risk: float
    priority: tuple
    reroutes: list


def message_from(ctx: ZoneContext, proposal: dict) -> NeighborMsg:
    s = ctx.state
    return NeighborMsg(zone_id=s.zone.zone_id, roads=list(s.zone.roads), severity=s.zone.severity,
                       cascade_risk=float(s.updated_cascade_risk), priority=priority(s),
                       reroutes=list(proposal.get("actionable_reroutes") or []))


def yield_targets(ctx: ZoneContext, inbox: List[NeighborMsg]) -> Set[str]:
    """Targets this zone must avoid: blocked roads, its own roads, roads of neighbouring zones, and targets held by
    higher-priority neighbours."""
    mine = priority(ctx.state)
    avoid = set(ctx.blocked) | set(ctx.state.zone.roads)
    for m in inbox:
        avoid |= set(m.roads)
        if m.priority > mine:
            avoid |= targets_of(m.reroutes)
    return avoid


def rule_severities(ctx: ZoneContext, per_horizon: bool):
    if not per_horizon:
        return None, None
    idx = ctx.state.zone.sensor_indices
    return (C.speed_to_severity(float(np.mean(ctx.inp.pred_speed[5, idx]))),
            C.speed_to_severity(float(np.mean(ctx.inp.pred_speed[11, idx]))))


class RuleAgent:
    """Deterministic agent. per_horizon=False: detector severity at both horizons ('rules');
    per_horizon=True: config scale on the zone's own +30 / +60 min forecast ('rules_h')."""
    kind = "rule"

    def __init__(self, per_horizon: bool = True):
        self.per_horizon = per_horizon

    def propose(self, ctx: ZoneContext) -> dict:
        s30, s60 = rule_severities(ctx, self.per_horizon)
        adv = rule_advisory(ctx.state, ctx.candidates, ctx.blocked, s30, s60)
        return {**adv, "attempts": 0, "valid": True, "gate_errors": []}

    def revise(self, ctx: ZoneContext, own: dict, inbox: List[NeighborMsg], conflicts: List[tuple], rnd: int) -> dict:
        avoid = yield_targets(ctx, inbox)
        new_rr, used = [], set()
        for r in own.get("actionable_reroutes", []) or []:
            tgt = r.get("to_road")
            if tgt in avoid or tgt in used:
                tgt = pick_target(ctx.candidates, avoid | used)
            if tgt:
                used.add(tgt)
                new_rr.append({"from_road": r.get("from_road"), "to_road": tgt,
                               "reason": r.get("reason") if tgt == r.get("to_road") else "yielded contested corridor"})
        return {**own, "actionable_reroutes": new_rr}


class LLMAgent:
    """Wraps ZoneWorkerAgent (pipeline/generate/zone_worker.py): propose = first advisory, revise = advisory given
    the neighbours' proposals and the visible conflicts. Retry gates run inside every call. In dry-run mode the
    client returns the RuleAgent (per-horizon) output instead of calling a model, so the protocol can be tested."""
    kind = "llm"

    def __init__(self, worker, use_retry: bool = True):
        self.worker, self.use_retry = worker, use_retry
        self._stand_in = RuleAgent(per_horizon=True)

    def _dry(self) -> bool:
        return bool(getattr(self.worker.llm, "dry_run", False))

    def propose(self, ctx: ZoneContext) -> dict:
        dry = self._stand_in.propose(ctx) if self._dry() else None
        return self.worker.generate(ctx, use_retry=self.use_retry, dry_value=dry)

    def revise(self, ctx: ZoneContext, own: dict, inbox: List[NeighborMsg], conflicts: List[tuple], rnd: int) -> dict:
        dry = self._stand_in.revise(ctx, own, inbox, conflicts, rnd) if self._dry() else None
        return self.worker.revise(ctx, own, inbox, conflicts, rnd, use_retry=self.use_retry, dry_value=dry)


def make_agent(kind: str, llm=None, use_retry: bool = True):
    if kind == "rules":
        return RuleAgent(per_horizon=False)
    if kind == "rules_h":
        return RuleAgent(per_horizon=True)
    if kind == "llm":
        from pipeline.generate.zone_worker import ZoneWorkerAgent
        return LLMAgent(ZoneWorkerAgent(llm), use_retry=use_retry)
    raise ValueError(f"unknown agent kind {kind}")
