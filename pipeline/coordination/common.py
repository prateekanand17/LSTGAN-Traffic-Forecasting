"""Shared, deterministic coordination primitives (used by agents, coordinators and evaluation/metrics.py).

- candidate_corridors : plausible reroute targets for a zone (other freeways within REROUTE_RADIUS_KM),
                        ranked by FORECAST speed at +30 min (never ground truth)
- priority            : total order used to settle conflicts (severity, then cascade risk, then zone_id)
- neighbors           : which zones can exchange proposals over the communication topology
- detect_conflicts    : shared-target and into-zone reroute conflicts (global; the single definition used by the
                        negotiation loop, the coordinators and the evaluation metrics)
- conflicts_visible_to: the part of the global conflict set an agent may see: conflicts between itself and a
                        topology neighbour (finding M32: conflicts with non-neighbours are NOT visible)
Robustness options (plan §13 F2; off by default):
- zone_footprints     : per zone and road, the sensors the zone covers (own roads) or would send traffic to
                        (candidate roads: sensors of that road within the reroute radius of the zone centre)
- detect_conflicts(..., footprints=...): segment-level conflicts: a shared-target or into-zone conflict counts only
                        if the two zones' footprints on that road share a sensor (default: whole road-direction)
- corridor_neighbors  : zones are neighbours iff their candidate corridors could produce a conflict
"""
from itertools import combinations
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

from pipeline import config as C
from pipeline.graph.zone_state import EnrichedZoneState
from pipeline.inputs import PipelineInput

SEV_RANK = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}


def _haversine_km(lat1, lng1, lat2, lng2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lng2 - lng1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def candidate_corridors(state: EnrichedZoneState, inp: PipelineInput, radius_km: float = C.REROUTE_RADIUS_KM) -> List[Tuple[str, float]]:
    """[(corridor, forecast mean speed at +30 min over its sensors within the radius)], fastest first.
    Excludes every direction of the zone's own freeways."""
    z = state.zone
    roads = np.asarray(inp.roads)
    own = {r.split("-")[0] for r in z.roads}
    d = _haversine_km(z.center_lat, z.center_lng, inp.lats.astype(float), inp.lngs.astype(float))
    near = np.where(d <= radius_km)[0]
    out = {}
    for i in near:
        r = roads[i]
        if r.split("-")[0] in own:
            continue
        out.setdefault(r, []).append(float(inp.pred_speed[5, i]))
    ranked = sorted(((r, float(np.mean(v))) for r, v in out.items()), key=lambda t: (-t[1], t[0]))
    return ranked


def priority(state: EnrichedZoneState) -> Tuple:
    """Higher tuple = higher priority (wins a conflict)."""
    return (SEV_RANK[state.zone.severity], round(state.updated_cascade_risk, 6), state.zone.zone_id)


def zone_link(A: np.ndarray, a: List[int], b: List[int]) -> float:
    """Undirected link strength between two zones: max of the mean adjacency in either direction."""
    if A is None or not len(a) or not len(b):
        return 0.0
    return float(max(A[np.ix_(a, b)].mean(), A[np.ix_(b, a)].mean()))


def neighbors(states: List[EnrichedZoneState], A: np.ndarray, thr: float = C.NEIGHBOR_THRESHOLD) -> Dict[str, Set[str]]:
    nb = {s.zone.zone_id: set() for s in states}
    for s, t in combinations(states, 2):
        if zone_link(A, s.zone.sensor_indices, t.zone.sensor_indices) > thr:
            nb[s.zone.zone_id].add(t.zone.zone_id)
            nb[t.zone.zone_id].add(s.zone.zone_id)
    return nb


def link_density(nb: Dict[str, Set[str]]) -> float:
    """Share of zone pairs that are neighbours (reported per topology; finding M33)."""
    n = len(nb)
    if n < 2:
        return float("nan")
    return sum(len(v) for v in nb.values()) / (n * (n - 1))


def targets_of(reroutes: Iterable) -> Set[str]:
    return {r.get("to_road") for r in (reroutes or []) if isinstance(r, dict) and r.get("to_road")}


def zone_footprints(states: List[EnrichedZoneState], inp: PipelineInput, candidates: Dict[str, List[str]],
                    radius_km: float = C.REROUTE_RADIUS_KM) -> Dict[str, Dict[str, frozenset]]:
    """{zone: {road: sensor indices}}: own roads -> the zone's sensors on that road; candidate roads -> sensors of
    that road within radius_km of the zone centre (the stretch the zone would send traffic to)."""
    roads = np.asarray(inp.roads)
    out = {}
    for s in states:
        z = s.zone
        d = _haversine_km(z.center_lat, z.center_lng, inp.lats.astype(float), inp.lngs.astype(float))
        fp = {r: frozenset(i for i in z.sensor_indices if roads[i] == r) for r in z.roads}
        for r in candidates.get(z.zone_id, []):
            fp[r] = frozenset(np.where((roads == r) & (d <= radius_km))[0].tolist())
        out[z.zone_id] = fp
    return out


def _overlap(footprints, a: str, b: str, road: str) -> bool:
    if footprints is None:
        return True
    return bool(footprints.get(a, {}).get(road, frozenset()) & footprints.get(b, {}).get(road, frozenset()))


def detect_conflicts(reroutes: Dict[str, list], zone_roads: Dict[str, List[str]], footprints=None) -> Set[tuple]:
    """('shared', road, a, b): zones a and b reroute onto the same road (a < b);
    ('into_zone', road, a, b): zone a reroutes onto a road covered by zone b (any detected zone).
    footprints=None: whole road-direction (default); otherwise only if the footprints on the road overlap."""
    out = set()
    targets = {zid: targets_of(rr) for zid, rr in reroutes.items()}
    for a, b in combinations(sorted(targets), 2):
        for road in targets[a] & targets[b]:
            if _overlap(footprints, a, b, road):
                out.add(("shared", road, a, b))
    for zid, tg in targets.items():
        for other, roads in zone_roads.items():
            if other != zid:
                for road in tg & set(roads):
                    if _overlap(footprints, zid, other, road):
                        out.add(("into_zone", road, zid, other))
    return out


def corridor_neighbors(states: List[EnrichedZoneState], candidates: Dict[str, List[str]],
                       footprints=None) -> Dict[str, Set[str]]:
    """Corridor-overlap topology (F2d): i and j are neighbours iff some choice of candidate targets could make them
    conflict (shared candidate, or a candidate of one is a road of the other), under the same conflict definition."""
    nb = {s.zone.zone_id: set() for s in states}
    for s, t in combinations(states, 2):
        a, b = s.zone.zone_id, t.zone.zone_id
        ca, cb = set(candidates.get(a, [])), set(candidates.get(b, []))
        pot = [(a, b, r) for r in ca & cb] + [(a, b, r) for r in ca & set(t.zone.roads)] + \
              [(b, a, r) for r in cb & set(s.zone.roads)]
        if any(_overlap(footprints, x, y, r) for x, y, r in pot):
            nb[a].add(b)
            nb[b].add(a)
    return nb


def conflicts_visible_to(zid: str, conflicts: Set[tuple], nbrs: Set[str]) -> List[tuple]:
    """Conflicts between `zid` and one of its topology neighbours `nbrs`. Conflicts with zones outside N(i) are
    not visible: under the 'none' topology an agent sees nothing, so the topology ablation isolates communication."""
    out = []
    for c in conflicts:
        a, b = c[2], c[3]
        if (a == zid and b in nbrs) or (b == zid and a in nbrs):
            out.append(c)
    return sorted(out)
