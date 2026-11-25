import json

UA = "D:/Deep Traffic Brain/PART_1/New_ipynb_file/pipeline/.ua"
inp = json.load(open(f"{UA}/tmp/ua-file-analyzer-input-1.json", encoding="utf-8"))
imports = inp["batchImportData"]

FILES = {
    "config.py": ("Single source of truth for MASTRA constants: data and forecast file paths, the speed-based severity scale, zone detection and message-passing settings, negotiation limits, safety-gate thresholds, and pinned LLM providers and model IDs.",
                  ["configuration", "constants", "llm-settings", "thresholds"], "moderate",
                  "Model IDs and the provider can be overridden through MASTRA_* environment variables read at import time, so the effective configuration is fixed when the module is first imported."),
    "generate/advisory_dq_agent.py": ("Evaluation-only LLM plausibility check that compares one final advisory with its zone state; the verdict is recorded and never changes the output, and API failures are recorded as ERROR rather than PASS.",
                                      ["service", "llm-agent", "evaluation", "validation"], "simple", None),
    "generate/det_guards.py": ("Deterministic safety gates for zone advisories: finds roads of zones forecast HIGH, produces the CONGESTED_TARGET retry errors, and applies the sensor-fault, severe-speed, cascade-risk and rush-hour overrides.",
                               ["validation", "safety-gates", "deterministic", "rule-based"], "simple", None),
    "generate/schema.py": ("Defines the advisory JSON field specification shown to agents and the structural gate that checks fields, severity enums and reroute road validity; also counts invalid road references for metrics.",
                           ["validation", "schema-definition", "data-model", "llm-output"], "moderate", None),
    "generate/supervisor_agent.py": ("Central LLM coordinator that sees every zone's proposal, detects routing conflicts itself and rewrites reroutes; its output is validated per zone, and invalid entries fall back to the zone's own proposal.",
                                     ["service", "llm-agent", "coordination", "conflict-resolution"], "moderate", None),
    "generate/zone_worker.py": ("LLM zone agent that builds propose and negotiation-revise prompts from zone state and runs the validate-and-retry loop; a revision that still fails the gates is discarded in favour of the previous proposal.",
                                ["service", "llm-agent", "prompt-building", "retry-loop", "negotiation"], "moderate", None),
    "llm_client.py": ("Provider-agnostic client for OpenAI-compatible chat endpoints that logs every call to calls.jsonl, tracks running usage totals, and raises on a missing key or transport failure instead of silently mocking; dry runs must be requested explicitly.",
                      ["service", "llm-client", "logging", "error-handling"], "moderate",
                      "A threading.Lock guards the usage counters and the log file, so one client can be shared safely across threads."),
    "orchestrator.py": ("Main MASTRA pipeline and the single code path used by the evaluation runner and dashboard: parses inputs, detects zones, gathers proposals from zone agents, coordinates them, and applies override gates to produce the final per-zone records.",
                        ["entry-point", "orchestration", "pipeline", "coordination", "experiment-variants"], "complex", None),
}

# (name, start, end, summary, tags, complexity)
FUNCS = {
    "config.py": [("speed_to_severity", 46, 51, "Maps a mean speed in mph to HIGH, MEDIUM or LOW using the shared severity thresholds.", ["utility", "severity-scale", "classification"], "simple")],
    "generate/det_guards.py": [
        ("congested_roads", 20, 21, "Collects the set of roads covered by zones whose detector severity is HIGH.", ["utility", "traffic-zones", "safety-gates"], "simple"),
        ("semantic_retry_errors", 24, 30, "Lists reroutes that send traffic onto a blocked road outside the zone, producing error text that goes back to the agent for a retry.", ["validation", "retry-gate", "reroutes"], "simple"),
    ],
    "generate/schema.py": [
        ("structural_errors", 18, 54, "Returns human-readable violations of the advisory schema, covering parse errors, missing fields, zone ID, severity enums, advisory tiers and reroute road validity; an empty list means the advisory passes.", ["validation", "structural-gate", "llm-output"], "moderate"),
        ("hallucinated_routes", 57, 66, "Counts invalid and total road references in an advisory's reroutes for reporting.", ["metrics", "validation", "reroutes"], "simple"),
    ],
    "generate/supervisor_agent.py": [("build_prompt", 27, 41, "Builds the supervisor prompt from each zone's roads, severity, cascade risk, candidate corridors and reroutes, plus the blocked roads and required output fields.", ["prompt-building", "coordination", "llm-agent"], "simple")],
    "generate/zone_worker.py": [
        ("zone_forecast_at", 43, 44, "Returns the mean forecast speed over a zone's sensors at a given forecast step.", ["utility", "forecast", "traffic-zones"], "simple"),
        ("build_prompt", 47, 76, "Builds the propose prompt from zone state, optional neighbour context, candidate corridors, blocked roads, required fields and format examples.", ["prompt-building", "llm-agent", "traffic-zones"], "moderate"),
        ("describe_conflict", 79, 86, "Turns a conflict tuple into a sentence written from the perspective of the given zone.", ["utility", "negotiation", "prompt-building"], "simple"),
        ("revision_block", 89, 107, "Builds the negotiation section of the revise prompt with the agent's current proposal, neighbour proposals with relative priority, visible conflicts and the conflict rule.", ["prompt-building", "negotiation", "llm-agent"], "moderate"),
    ],
    "llm_client.py": [
        ("_error_body", 126, 130, "Extracts the error dictionary from an API exception's body, or returns an empty dictionary.", ["utility", "error-handling", "llm-client"], "simple"),
        ("is_json_validate_failure", 133, 137, "Detects the provider-side json_validate_failed error so it can be handled as a parse error rather than an abort.", ["error-handling", "llm-client", "validation"], "simple"),
        ("failed_generation", 140, 141, "Returns up to 300 characters of the failed generation text reported by the provider.", ["utility", "error-handling", "llm-client"], "simple"),
    ],
    "orchestrator.py": [
        ("_usage_delta", 78, 82, "Computes the change in LLM usage counters since an earlier snapshot, for per-scenario cost reporting.", ["utility", "llm-usage", "metrics"], "simple"),
        ("_reroutes", 85, 86, "Returns a proposal's reroutes when it is valid and not suppressed, otherwise an empty list.", ["utility", "reroutes", "coordination"], "simple"),
        ("run_scenario", 89, 189, "Runs one scenario end to end under a Variant: parsing, forecast gate, zone detection, message passing, proposals, the chosen coordination mode, override gates and the optional DQ record.", ["entry-point", "orchestration", "pipeline", "coordination"], "complex"),
        ("_single_agent", 192, 228, "Single-LLM baseline in which one call sees every zone and returns all advisories, with retries on missing zones or structural errors.", ["baseline", "llm-agent", "retry-loop"], "moderate"),
    ],
}

CLASSES = {
    "generate/advisory_dq_agent.py": [("AdvisoryQualityAgent", 16, 34, "Asks an LLM whether an advisory contradicts its zone state and returns a PASS, FAIL, INVALID or ERROR status with a reason.", ["llm-agent", "evaluation", "validation"], "simple")],
    "generate/det_guards.py": [("DeterministicGuards", 33, 57, "Applies the override gates to an advisory and records each trigger: suppression on sensor faults, forced HIGH severity, upstream alerts on cascade risk, and a filled PREVENT tier at rush hour.", ["safety-gates", "deterministic", "validation"], "moderate")],
    "generate/supervisor_agent.py": [("NetworkSupervisorAgent", 44, 68, "Calls the supervisor LLM and validates its final reroutes per zone, keeping the zone's own reroutes and recording a fallback when an entry is invalid.", ["llm-agent", "coordination", "conflict-resolution"], "moderate")],
    "generate/zone_worker.py": [("ZoneWorkerAgent", 110, 146, "Wraps LLM propose and revise calls with structural and semantic gate retries, and keeps the previous proposal when a revision fails.", ["llm-agent", "retry-loop", "negotiation"], "moderate")],
    "llm_client.py": [
        ("LLMError", 22, 23, "RuntimeError subclass raised when a key is missing, a call fails, or a dry run has no stand-in output.", ["error-handling", "exception", "llm-client"], "simple"),
        ("LLMClient", 26, 123, "Chat client whose chat_json method returns parsed JSON, reports malformed output as _parse_error for retry gates, logs each call, and keeps usage totals.", ["llm-client", "logging", "service"], "moderate"),
    ],
    "orchestrator.py": [
        ("Variant", 43, 75, "Dataclass of experiment switches for agent type, coordination mode, topology, message passing, gates and robustness settings; validates combinations and reports whether an LLM is needed.", ["data-model", "configuration", "experiment-variants"], "moderate"),
        ("MASTraPipeline", 231, 245, "Compatibility wrapper for the dashboard that fills in missing graph and road inputs and calls run_scenario with live LLM agents and negotiation.", ["compatibility-wrapper", "dashboard", "entry-point"], "simple"),
    ],
}

O, Z = "orchestrator.py", "generate/zone_worker.py"
CALLS = [
    (f"function:{O}:run_scenario", t) for t in [
        "class:parse.py:Parser", "class:validate/forecast_gate.py:ForecastIntegrityGate",
        "class:zone_detection.py:ZoneDetector", "class:graph/message_passing.py:GraphMessagePassingLayer",
        "function:graph/message_passing.py:no_communication", "function:generate/det_guards.py:congested_roads",
        "function:coordination/common.py:candidate_corridors", "function:coordination/common.py:zone_footprints",
        "function:coordination/common.py:corridor_neighbors", "function:coordination/common.py:neighbors",
        "function:coordination/common.py:link_density", "function:coordination/agents.py:make_agent",
        "function:coordination/negotiation.py:negotiate", "function:coordination/negotiation.py:plan_conflicts",
        "function:coordination/coordinators.py:coordinate", "class:generate/det_guards.py:DeterministicGuards",
        "class:generate/advisory_dq_agent.py:AdvisoryQualityAgent", "class:llm_client.py:LLMClient",
        f"function:{O}:_single_agent", f"function:{O}:_reroutes", f"function:{O}:_usage_delta",
    ]
] + [
    (f"function:{O}:_single_agent", f"function:{Z}:build_prompt"),
    (f"function:{O}:_single_agent", "function:generate/schema.py:structural_errors"),
    (f"function:{O}:_single_agent", "class:coordination/agents.py:RuleAgent"),
    (f"class:{O}:MASTraPipeline", f"function:{O}:run_scenario"),
    (f"class:{O}:MASTraPipeline", "function:data_source.py:learned_adj"),
    (f"class:{O}:MASTraPipeline", "function:data_source.py:load_stations"),
    (f"function:{Z}:build_prompt", f"function:{Z}:zone_forecast_at"),
    (f"function:{Z}:revision_block", "function:coordination/common.py:priority"),
    (f"function:{Z}:revision_block", f"function:{Z}:describe_conflict"),
    (f"class:{Z}:ZoneWorkerAgent", f"function:{Z}:build_prompt"),
    (f"class:{Z}:ZoneWorkerAgent", f"function:{Z}:revision_block"),
    (f"class:{Z}:ZoneWorkerAgent", "function:generate/schema.py:structural_errors"),
    (f"class:{Z}:ZoneWorkerAgent", "function:generate/det_guards.py:semantic_retry_errors"),
    (f"class:{Z}:ZoneWorkerAgent", "class:llm_client.py:LLMClient"),
    ("class:generate/advisory_dq_agent.py:AdvisoryQualityAgent", "class:llm_client.py:LLMClient"),
    ("class:generate/supervisor_agent.py:NetworkSupervisorAgent", "function:generate/supervisor_agent.py:build_prompt"),
    ("class:generate/supervisor_agent.py:NetworkSupervisorAgent", "class:llm_client.py:LLMClient"),
    ("class:llm_client.py:LLMClient", "function:llm_client.py:is_json_validate_failure"),
    ("class:llm_client.py:LLMClient", "function:llm_client.py:failed_generation"),
    ("function:llm_client.py:is_json_validate_failure", "function:llm_client.py:_error_body"),
    ("function:llm_client.py:failed_generation", "function:llm_client.py:_error_body"),
]

nodes, edges = [], []


def edge(s, t, typ, w):
    edges.append({"source": s, "target": t, "type": typ, "direction": "forward", "weight": w})


for path, (summary, tags, cx, notes) in FILES.items():
    n = {"id": f"file:{path}", "type": "file", "name": path.split("/")[-1], "filePath": path,
         "summary": summary, "tags": tags, "complexity": cx}
    if notes:
        n["languageNotes"] = notes
    nodes.append(n)
    for kind, table in (("function", FUNCS), ("class", CLASSES)):
        for name, a, b, s, t, c in table.get(path, []):
            nid = f"{kind}:{path}:{name}"
            nodes.append({"id": nid, "type": kind, "name": name, "filePath": path, "lineRange": [a, b],
                          "summary": s, "tags": t, "complexity": c})
            edge(f"file:{path}", nid, "contains", 1.0)
            edge(f"file:{path}", nid, "exports", 0.8)
    for target in imports[path]:
        edge(f"file:{path}", f"file:{target}", "imports", 0.7)
for s, t in CALLS:
    edge(s, t, "calls", 0.8)

# self-checks
ids = [n["id"] for n in nodes]
assert len(ids) == len(set(ids)), "duplicate node ids"
keys = [(e["source"], e["target"], e["type"]) for e in edges]
assert len(keys) == len(set(keys)), "duplicate edges"
assert all(e["source"] != e["target"] for e in edges)
assert sum(e["type"] == "imports" for e in edges) == sum(len(v) for v in imports.values())
assert all(3 <= len(n["tags"]) <= 5 and n["summary"] for n in nodes)
EXTERNAL = {
    "class:parse.py:Parser", "class:validate/forecast_gate.py:ForecastIntegrityGate", "class:zone_detection.py:ZoneDetector",
    "class:graph/message_passing.py:GraphMessagePassingLayer", "function:graph/message_passing.py:no_communication",
    "function:coordination/common.py:candidate_corridors", "function:coordination/common.py:zone_footprints",
    "function:coordination/common.py:corridor_neighbors", "function:coordination/common.py:neighbors",
    "function:coordination/common.py:link_density", "function:coordination/common.py:priority",
    "function:coordination/agents.py:make_agent", "class:coordination/agents.py:RuleAgent",
    "function:coordination/negotiation.py:negotiate", "function:coordination/negotiation.py:plan_conflicts",
    "function:coordination/coordinators.py:coordinate", "function:data_source.py:learned_adj", "function:data_source.py:load_stations",
}
known = set(ids) | EXTERNAL | {f"file:{t}" for v in imports.values() for t in v}
bad = [e for e in edges if e["source"] not in known or e["target"] not in known]
assert not bad, bad

# Step B/C: 123 edges exceeds 120, so split into parts = ceil(max(33/60, 123/120)) = 2
import math
parts = math.ceil(max(len(nodes) / 60, len(edges) / 120))
paths = sorted(FILES)
size = math.ceil(len(paths) / parts)
seen_n, seen_e = 0, 0
for k in range(parts):
    group = set(paths[k * size:(k + 1) * size])
    pn = [n for n in nodes if n["filePath"] in group]
    pids = {n["id"] for n in pn}
    pe = [e for e in edges if e["source"] in pids]
    assert len(pn) <= 60 and len(pe) <= 120, (k, len(pn), len(pe))
    with open(f"{UA}/intermediate/batch-1-part-{k + 1}.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": pn, "edges": pe}, f, indent=1, ensure_ascii=False)
    seen_n += len(pn)
    seen_e += len(pe)
    print(f"part {k + 1}", sorted(group), len(pn), len(pe))
assert seen_n == len(nodes) and seen_e == len(edges)
print(len(nodes), len(edges), sum(e["type"] == "imports" for e in edges))
