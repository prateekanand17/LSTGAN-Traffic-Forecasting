"""Advisory JSON schema and the structural gate (shared by workers, supervisor and metrics)."""
from typing import List, Tuple

from pipeline import config as C

SEV = list(C.SEVERITY_LEVELS)

ADVISORY_FIELDS = {
    "zone_id": "string, copy exactly from the zone state",
    "severity_30min": "one of HIGH | MEDIUM | LOW: your prediction of the zone's congestion severity in 30 minutes",
    "severity_60min": "one of HIGH | MEDIUM | LOW: your prediction of the zone's congestion severity in 60 minutes",
    "advisory": {"now": "action for the next 15 min", "soon": "action for 15-30 min", "prevent": "action for 30-60 min"},
    "actionable_reroutes": "list of {from_road, to_road, reason}; from_road = a road of this zone, to_road = one of the candidate corridors; may be empty",
    "upstream_alert": "true if upstream operators must be alerted",
}


def structural_errors(adv: dict, zone_id: str, zone_roads: List[str], valid_targets: List[str]) -> List[str]:
    """Return a list of human-readable violations (empty = passes the structural gate).
    valid_targets = the zone's candidate corridors (pipeline.coordination.common.candidate_corridors)."""
    errs = []
    if "_parse_error" in adv:
        return [f"output is not a JSON object ({adv['_parse_error']})"]
    for k in ("zone_id", "severity_30min", "severity_60min", "advisory", "actionable_reroutes", "upstream_alert"):
        if k not in adv:
            errs.append(f"missing field '{k}'")
    if errs:
        return errs
    if adv["zone_id"] != zone_id:
        errs.append(f"zone_id must be '{zone_id}' (got '{adv['zone_id']}')")
    for k in ("severity_30min", "severity_60min"):
        if adv[k] not in SEV:
            errs.append(f"{k} must be one of {SEV} (got {adv[k]!r})")
    a = adv["advisory"]
    if not isinstance(a, dict) or any(not isinstance(a.get(t), str) or not a.get(t).strip() for t in ("now", "soon", "prevent")):
        errs.append("advisory must be an object with non-empty string fields now, soon, prevent")
    if not isinstance(adv["upstream_alert"], bool):
        errs.append("upstream_alert must be true or false")
    rr = adv["actionable_reroutes"]
    if not isinstance(rr, list):
        errs.append("actionable_reroutes must be a list")
    else:
        for i, r in enumerate(rr):
            if not isinstance(r, dict) or "from_road" not in r or "to_road" not in r:
                errs.append(f"reroute {i} must have from_road and to_road")
                continue
            if r["from_road"] not in zone_roads:
                errs.append(f"reroute {i}: from_road {r['from_road']!r} is not a road of this zone {zone_roads}")
            if r["to_road"] not in valid_targets:
                errs.append(f"reroute {i}: to_road {r['to_road']!r} is not one of this zone's candidate corridors "
                            f"{valid_targets}")
            elif r["to_road"] == r.get("from_road"):
                errs.append(f"reroute {i}: to_road equals from_road")
    return errs


def hallucinated_routes(adv: dict, valid_roads: List[str]) -> Tuple[int, int]:
    """(invalid road references, total road references) in actionable_reroutes."""
    bad = tot = 0
    for r in adv.get("actionable_reroutes", []) if isinstance(adv.get("actionable_reroutes"), list) else []:
        if isinstance(r, dict):
            for k in ("from_road", "to_road"):
                if k in r:
                    tot += 1
                    bad += r[k] not in valid_roads
    return bad, tot
