"""Negotiation loop between zone agents over the communication topology (task N4, plan §9.2).

Round 0: every agent proposes (retry gates inside the call).
Round r = 1..K:
  1. exchange - every agent sends its current proposal to its neighbours N(i) only;
  2. detect   - the global conflict set is computed (common.detect_conflicts); each agent sees only the conflicts
                between itself and a neighbour (common.conflicts_visible_to, finding M32);
  3. revise   - every agent with at least one visible conflict revises, given its neighbours' proposals.
Stops when no agent sees a conflict ('no_visible_conflicts'; conflicts between non-neighbours may remain), when no proposal changed in a round ('stalled'), or after
K rounds ('max_rounds'). Conflicts that no agent can see (between non-neighbours) are left to the coordinator
fallback (if enabled) and to the evaluation.
"""
import json
from typing import Dict, List, Set

from pipeline.coordination.agents import ZoneContext, message_from
from pipeline.coordination.common import conflicts_visible_to, detect_conflicts


def _rr(p: dict) -> list:
    return list(p.get("actionable_reroutes") or []) if p.get("valid") and not p.get("suppressed") else []


def plan_conflicts(proposals: Dict[str, dict], zone_roads: Dict[str, List[str]], footprints=None) -> Set[tuple]:
    return detect_conflicts({zid: _rr(p) for zid, p in proposals.items()}, zone_roads, footprints)


def _snapshot(p: dict) -> str:
    return json.dumps({k: p.get(k) for k in ("severity_30min", "severity_60min", "actionable_reroutes")}, sort_keys=True)


def negotiate(agents: Dict[str, object], ctxs: Dict[str, ZoneContext], proposals: Dict[str, dict],
              nb: Dict[str, Set[str]], zone_roads: Dict[str, List[str]], max_rounds: int, footprints=None) -> dict:
    """Returns {"proposals", "rounds_used", "stop_reason", "transcript", "messages"}. `proposals` is not modified."""
    props = dict(proposals)
    transcript, messages = [], 0
    stop = "max_rounds"
    rounds_used = 0
    for rnd in range(1, max_rounds + 1):
        conflicts = plan_conflicts(props, zone_roads, footprints)
        visible = {zid: conflicts_visible_to(zid, conflicts, nb[zid]) for zid in props}
        if not any(visible.values()):
            stop = "no_visible_conflicts"
            break
        rounds_used = rnd
        inboxes = {zid: [message_from(ctxs[j], props[j]) for j in sorted(nb[zid]) if props[j].get("valid")]
                   for zid in props}
        messages += sum(len(v) for v in inboxes.values())
        changed, revised = [], []
        new_props = dict(props)
        for zid in sorted(props):
            if not visible[zid] or not props[zid].get("valid") or props[zid].get("suppressed"):
                continue
            new = agents[zid].revise(ctxs[zid], props[zid], inboxes[zid], visible[zid], rnd)
            revised.append(zid)
            if _snapshot(new) != _snapshot(props[zid]):
                changed.append(zid)
            new_props[zid] = new
        props = new_props
        transcript.append({
            "round": rnd,
            "conflicts_global": sorted(list(c) for c in conflicts),
            "conflicts_visible": {z: [list(c) for c in v] for z, v in visible.items() if v},
            "messages": sum(len(v) for v in inboxes.values()),
            "revised": revised, "changed": changed,
            "proposals": {z: _rr(p) for z, p in props.items()},
        })
        if not changed:
            stop = "stalled"
            break
    else:
        if not any(conflicts_visible_to(z, plan_conflicts(props, zone_roads, footprints), nb[z]) for z in props):
            stop = "no_visible_conflicts"
    return {"proposals": props, "rounds_used": rounds_used, "stop_reason": stop, "transcript": transcript,
            "messages": messages}
