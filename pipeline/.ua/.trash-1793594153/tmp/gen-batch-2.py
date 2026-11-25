import json

ROOT = "D:/Deep Traffic Brain/PART_1/New_ipynb_file/pipeline/.ua/"
IMPORTS = json.load(open(ROOT + "tmp/ua-file-analyzer-input-2.json", encoding="utf-8"))["batchImportData"]

A, CM, CO, N, R, Z = ("coordination/agents.py", "coordination/common.py", "coordination/coordinators.py",
                      "coordination/negotiation.py", "generate/rules.py", "graph/zone_state.py")

FILES = [
    (A, "moderate",
     "Defines the shared zone-agent interface (propose and revise) so deterministic rule agents and LLM-backed agents run inside the same coordination protocols, along with the local context and neighbour messages they exchange.",
     ["agent", "coordination", "factory", "multi-agent", "strategy-pattern"], None),
    (CM, "moderate",
     "Provides the deterministic coordination primitives shared by agents, coordinators and evaluation metrics: candidate corridor ranking, the priority order, communication topologies, and the single definition of reroute conflicts.",
     ["coordination", "utility", "conflict-detection", "graph-topology", "geospatial"], None),
    (CO, "moderate",
     "Implements the centralized coordinators used in the centralized design cells and as the fallback after negotiation: a greedy priority-based rule coordinator and an LLM Global Supervisor, both with a global view of every zone.",
     ["coordination", "centralized", "orchestration", "fallback", "greedy-algorithm"], None),
    (N, "moderate",
     "Runs the round-based negotiation loop in which zone agents exchange proposals only with topology neighbours, see only conflicts with those neighbours, and revise until no conflict is visible, nothing changes, or the round limit is reached.",
     ["coordination", "negotiation", "multi-agent", "protocol", "decentralized"],
     "The negotiation loop uses Python's for-else: the else branch runs only when all rounds complete without a break, which is where the final conflict re-check happens."),
    (R, "simple",
     "Generates deterministic advisory content without an LLM, used by RuleAgent and the dry-run stand-in: severities from the detector or the per-horizon forecast, plus one reroute to the fastest free candidate corridor for HIGH or MEDIUM zones.",
     ["rule-based", "advisory", "deterministic", "utility"], None),
    (Z, "simple",
     "Declares the dataclasses for messages passed between zones and for a zone's state after graph message passing, including its updated cascade risk and neighbour context.",
     ["data-model", "type-definition", "dataclass", "graph-message-passing"], None),
]

# (file, kind, name, start, end, complexity, summary, tags)
SYMBOLS = [
    (A, "class", "ZoneContext", 24, 36, "simple",
     "Holds the local view passed to a zone agent: its enriched state, pipeline input, ranked candidate corridors with forecast speeds, and globally blocked roads.",
     ["data-model", "dataclass", "agent-context"]),
    (A, "class", "NeighborMsg", 40, 46, "simple",
     "Carries the data one zone shares with its neighbours during negotiation: roads, severity, cascade risk, priority and current reroutes.",
     ["data-model", "dataclass", "message"]),
    (A, "function", "message_from", 49, 53, "simple",
     "Builds a NeighborMsg from a zone's context and its current proposal.",
     ["message", "factory", "coordination"]),
    (A, "function", "yield_targets", 56, 65, "simple",
     "Computes the roads a zone must avoid: blocked roads, its own roads, neighbouring zones' roads, and targets held by higher-priority neighbours.",
     ["conflict-resolution", "coordination", "priority"]),
    (A, "function", "rule_severities", 68, 73, "simple",
     "Returns the zone's +30 and +60 minute severities from its mean forecast speed, or None for both when per-horizon mode is off.",
     ["severity", "forecast", "rule-based"]),
    (A, "class", "RuleAgent", 76, 100, "moderate",
     "Deterministic zone agent that proposes a rule-based advisory and, when revising, moves contested reroutes to free candidates or drops them.",
     ["agent", "rule-based", "deterministic"]),
    (A, "class", "LLMAgent", 103, 122, "moderate",
     "Wraps a ZoneWorkerAgent behind the propose and revise interface, supplying RuleAgent output as the dry-run value so the protocol can be tested without a model.",
     ["agent", "llm", "adapter", "dry-run"]),
    (A, "function", "make_agent", 125, 133, "simple",
     "Creates a zone agent for the kind 'rules', 'rules_h' or 'llm', lazily importing the LLM worker and raising on unknown kinds.",
     ["factory", "agent", "lazy-import"]),
    (CM, "function", "candidate_corridors", 36, 51, "moderate",
     "Ranks other freeways within the reroute radius by forecast speed at +30 minutes, fastest first, excluding every direction of the zone's own freeways.",
     ["corridor-ranking", "geospatial", "forecast"]),
    (CM, "function", "priority", 54, 56, "simple",
     "Returns the tuple of severity rank, cascade risk and zone id that gives a total order for settling conflicts.",
     ["priority", "conflict-resolution", "ordering"]),
    (CM, "function", "zone_link", 59, 63, "simple",
     "Computes undirected link strength between two zones as the larger of the mean adjacency in either direction.",
     ["graph-topology", "adjacency", "utility"]),
    (CM, "function", "neighbors", 66, 72, "simple",
     "Builds the communication topology by linking zone pairs whose adjacency-based link strength exceeds a threshold.",
     ["graph-topology", "communication", "adjacency"]),
    (CM, "function", "link_density", 75, 80, "simple",
     "Reports the share of zone pairs that are neighbours in a topology, returning NaN for fewer than two zones.",
     ["metric", "graph-topology", "utility"]),
    (CM, "function", "targets_of", 83, 84, "simple",
     "Extracts the set of target roads from a list of reroutes, ignoring malformed entries.",
     ["utility", "reroute", "parsing"]),
    (CM, "function", "zone_footprints", 87, 100, "moderate",
     "Maps each zone and road to the sensors involved: the zone's own sensors on its roads, and candidate-road sensors within the reroute radius.",
     ["geospatial", "segment-level", "robustness"]),
    (CM, "function", "detect_conflicts", 109, 125, "moderate",
     "Finds shared-target and into-zone reroute conflicts across all zones, optionally counting them only where zone footprints overlap on the road.",
     ["conflict-detection", "coordination", "core-logic"]),
    (CM, "function", "corridor_neighbors", 128, 141, "moderate",
     "Builds an alternative topology in which two zones are neighbours if some choice of candidate targets could make them conflict.",
     ["graph-topology", "conflict-detection", "robustness"]),
    (CM, "function", "conflicts_visible_to", 144, 152, "simple",
     "Filters the global conflict set to those between a zone and its topology neighbours, so conflicts with other zones stay hidden from it.",
     ["conflict-detection", "visibility", "communication"]),
    (CO, "function", "central_rule", 17, 38, "moderate",
     "Greedily resolves reroutes from highest to lowest priority, keeping free targets and otherwise reassigning to the fastest free candidate or dropping the reroute.",
     ["coordinator", "greedy-algorithm", "rule-based"]),
    (CO, "function", "central_llm", 41, 49, "simple",
     "Runs the LLM NetworkSupervisorAgent over all proposals, using central_rule output as the dry-run value, and records which zones changed.",
     ["coordinator", "llm", "supervisor"]),
    (CO, "function", "coordinate", 52, 57, "simple",
     "Dispatches to the rule or LLM coordinator by kind and raises on unknown kinds.",
     ["dispatcher", "coordinator", "entry-point"]),
    (N, "function", "plan_conflicts", 24, 25, "simple",
     "Detects conflicts among the reroutes of proposals that are valid and not suppressed.",
     ["conflict-detection", "coordination", "wrapper"]),
    (N, "function", "negotiate", 32, 75, "complex",
     "Runs up to K negotiation rounds of message exchange, conflict detection and revision, and returns the final proposals, stop reason, message count and a per-round transcript.",
     ["negotiation", "protocol", "core-logic", "multi-agent"]),
    (R, "function", "pick_target", 13, 17, "simple",
     "Returns the first ranked candidate road not in the exclusion set, or None when none is free.",
     ["utility", "reroute", "selection"]),
    (R, "function", "rule_advisory", 20, 41, "moderate",
     "Assembles a complete advisory record from a zone's state, covering severities, templated actions, an optional reroute and the upstream alert flag.",
     ["advisory", "rule-based", "template"]),
    (Z, "class", "ZoneMessage", 7, 13, "simple",
     "Dataclass for the summary a zone sends to neighbours: severity, score, speed forecast, trend and raw cascade probability.",
     ["data-model", "dataclass", "message"]),
    (Z, "class", "EnrichedZoneState", 17, 23, "simple",
     "Dataclass wrapping a CongestionZone with received neighbour messages, updated cascade risk, network pressure, neighbour context text and weights.",
     ["data-model", "dataclass", "zone-state"]),
]

ids = {name: f"{kind}:{path}:{name}" for path, kind, name, *_ in SYMBOLS}
ids.update({
    "speed_to_severity": "function:config.py:speed_to_severity",
    "ZoneWorkerAgent": "class:generate/zone_worker.py:ZoneWorkerAgent",
    "NetworkSupervisorAgent": "class:generate/supervisor_agent.py:NetworkSupervisorAgent",
    "CongestionZone": "class:zone_detection.py:CongestionZone",
})

CALLS = [
    ("message_from", "priority"), ("yield_targets", "priority"), ("yield_targets", "targets_of"),
    ("rule_severities", "speed_to_severity"), ("RuleAgent", "rule_severities"), ("RuleAgent", "rule_advisory"),
    ("RuleAgent", "yield_targets"), ("RuleAgent", "pick_target"), ("LLMAgent", "RuleAgent"),
    ("make_agent", "RuleAgent"), ("make_agent", "LLMAgent"), ("make_agent", "ZoneWorkerAgent"),
    ("neighbors", "zone_link"), ("detect_conflicts", "targets_of"),
    ("central_rule", "priority"), ("central_rule", "pick_target"), ("central_llm", "central_rule"),
    ("central_llm", "NetworkSupervisorAgent"), ("coordinate", "central_rule"), ("coordinate", "central_llm"),
    ("plan_conflicts", "detect_conflicts"), ("negotiate", "plan_conflicts"), ("negotiate", "conflicts_visible_to"),
    ("negotiate", "message_from"), ("rule_advisory", "pick_target"),
]
RELATED = [("LLMAgent", "ZoneWorkerAgent"), ("EnrichedZoneState", "CongestionZone")]


def edge(s, t, ty, w):
    return {"source": s, "target": t, "type": ty, "direction": "forward", "weight": w}


nodes, edges = [], []
for path, cx, summary, tags, notes in FILES:
    n = {"id": f"file:{path}", "type": "file", "name": path.split("/")[-1], "filePath": path,
         "summary": summary, "tags": tags, "complexity": cx}
    if notes:
        n["languageNotes"] = notes
    nodes.append(n)
    edges += [edge(f"file:{path}", f"file:{t}", "imports", 0.7) for t in IMPORTS[path]]
for path, kind, name, start, end, cx, summary, tags in SYMBOLS:
    nid = ids[name]
    nodes.append({"id": nid, "type": kind, "name": name, "filePath": path, "lineRange": [start, end],
                  "summary": summary, "tags": tags, "complexity": cx})
    edges.append(edge(f"file:{path}", nid, "contains", 1.0))
    edges.append(edge(f"file:{path}", nid, "exports", 0.8))
edges += [edge(ids[a], ids[b], "calls", 0.8) for a, b in CALLS]
edges += [edge(ids[a], ids[b], "related", 0.5) for a, b in RELATED]

# Self-checks
node_ids = [n["id"] for n in nodes]
assert len(node_ids) == len(set(node_ids)), "duplicate node ids"
assert sum(e["type"] == "imports" for e in edges) == sum(len(v) for v in IMPORTS.values()), "imports mismatch"
keys = [(e["source"], e["target"], e["type"]) for e in edges]
assert len(keys) == len(set(keys)), "duplicate edges"
external = {ids[k] for k in ("speed_to_severity", "ZoneWorkerAgent", "NetworkSupervisorAgent", "CongestionZone")}
import_targets = {f"file:{t}" for v in IMPORTS.values() for t in v}
for e in edges:
    assert e["source"] != e["target"], e
    assert e["source"] in node_ids, e
    assert e["target"] in node_ids or e["target"] in external or e["target"] in import_targets, e
assert len(nodes) <= 60 and len(edges) <= 120, (len(nodes), len(edges))
for n in nodes:
    assert n["summary"] and 3 <= len(n["tags"]) <= 5, n["id"]

with open(ROOT + "intermediate/batch-2.json", "w", encoding="utf-8") as f:
    json.dump({"nodes": nodes, "edges": edges}, f, indent=2)
json.load(open(ROOT + "intermediate/batch-2.json", encoding="utf-8"))
print("nodes", len(nodes), "edges", len(edges), "imports", sum(e["type"] == "imports" for e in edges))
