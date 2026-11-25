# MASTRA pipeline: execution flow and debugging guide

Study map of `pipeline/` (about 2,000 lines). Read it top to bottom once, then open each file beside its section.
Line numbers are from the code as of Oct 24, 2026.

---

## 1. One-screen map

```
evaluation/run_mastra.py  (or src/app.py via MASTraPipeline)
        │  builds PipelineInput (pipeline/data_source.py) and a Variant
        ▼
pipeline/orchestrator.py :: run_scenario(inp, variant)          ← the ONLY code path
  1  Parser().parse(inp)                      parse.py               per-sensor severity / trend / z-score / faulty
  2  ForecastIntegrityGate().check()          validate/forecast_gate.py   recorded only, never stops the run
  3  ZoneDetector().detect()                  zone_detection.py      DBSCAN on congested sensors → zones (+ BFS cascade)
       └─ no zones → return status NO_CONGESTION (early exit)
  4  GraphMessagePassingLayer().run()         graph/message_passing.py    zones exchange severity → cascade risk p'
       (or no_communication() if variant.gmp is False)
  5  candidate_corridors(), neighbors()       coordination/common.py     reroute targets per zone, who talks to whom
  6  ROUND 0: agent.propose(ctx) per zone     coordination/agents.py     RuleAgent or LLMAgent → zone_worker.py → llm_client.py
  7  COORDINATION                             coordination/negotiation.py | coordinators.py
  8  DeterministicGuards().apply()            generate/det_guards.py     override gates → final records
        ▼
  result dict: status, zones[...], coordination{...}, llm_usage
```

**Mental model in one sentence:** a forecast of speeds becomes congested *zones*; each zone gets an *agent* that writes a JSON *advisory* (including reroutes onto other freeways); agents *negotiate* so two zones don't dump traffic on the same road; *gates* enforce safety rules at the end.

---

## 2. The data that travels

| Object | Defined in | Shape / key fields | Created at stage |
|---|---|---|---|
| `PipelineInput` | `inputs.py` | `pred_speed [12, 325]` (12 steps × 5 min, 325 sensors, mph), `adj_mx [325,325]`, `lats`, `lngs`, `hour`, `day_of_week`; added at runtime: `roads`, `comm_adj`, `observed_speed` | before run |
| `ParsedData` | `parse.py` | per-sensor `speed_levels`, `speed_trends`, `z_scores`, `faulty`; `is_rush_hour` | 1 |
| `CongestionZone` | `zone_detection.py` | `zone_id`, `sensor_indices`, `roads`, `severity`, `trend`, `cascade_risk_raw` (list of sensor ids) | 3 |
| `EnrichedZoneState` | `graph/zone_state.py` | `zone` + `updated_cascade_risk` (p') + `network_pressure` + `neighbor_context_text` | 4 |
| `ZoneContext` | `coordination/agents.py` | `state`, `candidates` (ranked roads), `blocked` (roads of HIGH zones) | 5 |
| advisory `dict` | `generate/schema.py` | `zone_id`, `severity_30min/60min`, `advisory{now,soon,prevent}`, `actionable_reroutes[{from_road,to_road,reason}]`, `upstream_alert`; bookkeeping: `valid`, `suppressed`, `attempts`, `gate_errors`, `gates` | 6–8 |

**Index conventions (the #1 source of off-by-one bugs):**
- `pred_speed[t]` is step `t+1`: index **5 = +30 min**, index **11 = +60 min** (`orchestrator.py:175-176`, `common.py:49`).
- `zone_worker.zone_forecast_at(inp, idx, step)` takes a 1-based *step* and reads `step - 1` (`zone_worker.py:44`). Both conventions coexist: always check which one a function uses.
- Road IDs are strings `"<road_name>-<direction>"`, e.g. `"101-N"`. The freeway is `r.split("-")[0]` (`common.py:41`). Safe only because PeMS-BAY road names are numbers (101, 280, 880, …); a name like `I-880` would break it.

---

## 3. Stage by stage (what, where, what can go wrong)

### Stage 1 — Parse (`parse.py:22`)
- Mean forecast speed per sensor over 12 steps → severity via `config.speed_to_severity`: **HIGH < 35, MEDIUM 35–50, LOW ≥ 50 mph** (`config.py:40-51`).
- Trend = mean of last 3 steps − mean of first 3; below −2 mph → WORSENING, above +2 → IMPROVING.
- `hist_stds == 0` is replaced by `1e-6` to avoid division by zero → z-score can become huge (not a crash, but a giant anomaly score).
- Faulty sensor = any observed reading `<= 0` (PeMS-BAY stores missing as 0).
- Rush hour = Mon–Fri (`day_of_week < 5`) and hour in 7–9 or 16–19 **inclusive**.

### Stage 2 — Forecast gate (`validate/forecast_gate.py:20`)
- Checks NaN/inf, > 50 HIGH sensors, > 40 mph jump between last observation and first forecast.
- **Recorded only** (`orchestrator.py:96`): a failed gate does not stop the run. A classic "why didn't the gate block it?" question.

### Stage 3 — Zone detection (`zone_detection.py:41`)
- Congested sensors = mean forecast < 50 mph (`ZONE_SPEED_THRESHOLD`).
- `DBSCAN(eps=0.015, min_samples=2)` on raw **(lat, lng) in degrees**, Euclidean, not km.
- Label `-1` (noise) is skipped: **an isolated single congested sensor never becomes a zone.**
- `zone_id = "zone_" + sha1(sorted station ids)[:8]`: deterministic, same sensors → same id.
- Zone trend = majority vote with `max(counts, key=counts.get)`; on a tie the first key in dict order wins (WORSENING > STABLE > IMPROVING).
- `_bfs` (`:79`): sensors within 2 hops on the *distance* graph (`adj_mx > 0.3`), outside the zone → `cascade_risk_raw`.

### Stage 4 — Graph message passing (`graph/message_passing.py:33`)
- Each zone broadcasts `(severity score, speed, trend, p)`, with `p = min(1, |cascade sensors| / |zone sensors|)`.
- Link weight from zone j to zone i: `A[rows of z_i, cols of z_j]`, then mean (or max) (`:56`). **Orientation matters**: the learned graph is asymmetric, so swapping the index order silently changes results.
- `normalized` pressure: weighted mean of neighbour severity scores; updated risk `p' = min(1, p + 0.3 · pressure)`.
- Requires `inp.comm_adj`; raises `ValueError` if missing (`:35-36`).

### Stage 5 — Candidates and neighbours (`coordination/common.py`)
- `candidate_corridors` (`:36`): roads of sensors within 5 km (haversine) of the zone centre, **excluding every direction of the zone's own freeways**, ranked by +30-min forecast speed (fastest first; ties by name).
- `neighbors` (`:66`): zones i, j are neighbours if `max(mean A[i,j], mean A[j,i]) > 0`. This decides **who can see whose conflicts**.
- `blocked` = roads of zones forecast HIGH (`det_guards.congested_roads`).

### Stage 6 — Round 0 proposals (`coordination/agents.py`)
- `make_agent` (`:125`): `rules` / `rules_h` → `RuleAgent`; `llm` → `LLMAgent(ZoneWorkerAgent)`.
- `RuleAgent.propose` → `generate/rules.py:rule_advisory`: reroute only for HIGH/MEDIUM zones, from the zone's first road to the fastest candidate not blocked.
- `LLMAgent.propose` → `zone_worker.py:_call_with_retry` (`:114`):
  1. build prompt → `llm.chat_json` → parsed dict, or `{"_parse_error": ...}`
  2. `structural_errors` (schema) → if clean, `semantic_retry_errors` (reroute onto a HIGH zone's road)
  3. errors → append to the prompt and retry, up to `1 + MAX_RETRIES = 3` attempts
  4. still failing → `valid: False`
- **Dry run:** `LLMAgent` passes the RuleAgent output as `dry_value`, so no API call is made. `LLMClient` refuses to run without a key unless `dry_run=True` (`llm_client.py:46`): no silent mock.

### Stage 7 — Coordination
- **Conflict definition** (`common.py:109`, used everywhere):
  - `("shared", road, a, b)`: two zones reroute onto the same road (`a < b` because of `combinations(sorted(...))`);
  - `("into_zone", road, a, b)`: zone a reroutes onto a road that zone b covers.
- **Negotiation** (`negotiation.py:32`), up to K = 3 rounds:
  1. compute global conflicts;
  2. each zone sees only conflicts with its **neighbours** (`conflicts_visible_to`);
  3. zones with a visible conflict call `revise`.
  - Stops on `no_visible_conflicts`, `stalled` (no proposal changed), or `max_rounds`. Note the **`for … else`** at `:71`: the `else` runs only if the loop was *not* broken.
  - Conflicts between non-neighbours stay invisible → left to the fallback coordinator.
- **Priority** (`common.py:54`): tuple `(severity rank, cascade risk, zone_id)`; higher wins; `zone_id` is the deterministic tie-breaker.
- `RuleAgent.revise` (`agents.py:89`): avoid blocked, own roads, all neighbour roads, and targets held by higher-priority neighbours; pick the next free candidate.
- **A failed LLM revision keeps the previous proposal** (`zone_worker.py:144-146`) and increments `revision_failures`.
- **Central coordinator** (`coordinators.py:17`): greedy by priority with a global view; used as a fallback if conflicts remain, or as the main mode (`central_rule` / `central_llm`).

### Stage 8 — Override gates (`generate/det_guards.py:34`)
Applied only if `variant.gates` and the advisory is valid (`orchestrator.py:169`):
- **SENSOR_FAULT_SKIP**: more than 50% of the zone's sensors faulty → `suppressed = True`.
- **SEVERE_OVERRIDE**: zone mean < 15 mph → both severities forced to HIGH.
- **CASCADE_BLOCK**: p' > 0.85 → `upstream_alert = True`.
- **RUSH_HOUR_BOOST**: empty `prevent` text filled during rush hour.
- `adv = dict(adv)` and `a = dict(a)` copy before editing, so the gates never mutate the proposal objects kept elsewhere (a classic aliasing-bug guard).
- `_reroutes(p)` (`orchestrator.py:85`) returns `[]` for invalid **or suppressed** advisories: a suppressed zone's reroutes disappear from conflicts and outputs.

---

## 4. Invariants (if one breaks, you found the bug)

1. Every `to_road` is one of that zone's `candidates`, never its own freeway (`schema.py:49`, `common.py:46`).
2. No reroute targets a road of a HIGH zone (`semantic_retry_errors`, `yield_targets`, `central_rule`).
3. Same input + rule agents ⇒ identical output (sorted iteration, hash ids, tie-break by `zone_id`).
4. `proposals` passed into `negotiate` is not mutated (`props = dict(proposals)`; `initial` is a `deepcopy`).
5. All thresholds come from `config.py`; a hard-coded number in a module is a bug by project rule.
6. An LLM failure never becomes a fake answer: transport errors raise `LLMError`; JSON errors become `_parse_error` → gate → retry → `valid: False`.

---

## 5. Debugging drills (symptom → where to look)

| # | Symptom | Most likely cause | Look at |
|---|---|---|---|
| 1 | Advisory says "+30 min" but numbers match +35 min | index vs step mix-up (`pred_speed[6]` instead of `[5]`) | §2 index conventions |
| 2 | Same conflict counted twice | pair not ordered `a < b` | `common.py:115` `combinations(sorted(...))` |
| 3 | Results change when two zones swap order | missing deterministic tie-break, or unsorted dict iteration | `priority()`, `negotiation.py:51` `sorted(props)` |
| 4 | Zone A "sees" B but B never sees A | asymmetric link used without `max(both directions)` | `common.py:63` |
| 5 | GMP pressure differs between two "identical" runs on the learned graph | rows/cols of `A` swapped | `message_passing.py:56` |
| 6 | A congested sensor shows no zone | DBSCAN noise (`-1`) or `min_samples=2` | `zone_detection.py:48-54` |
| 7 | `ValueError: comm_adj is required` | `PipelineInput.comm_adj` not set by the caller | `message_passing.py:35`, `MASTraPipeline.run` sets it |
| 8 | Initial and final reroutes are always identical | `initial` taken by reference, not `deepcopy`; or coordination = `none` | `orchestrator.py:138` |
| 9 | Gate fires but the run still continues | forecast gate is record-only by design | `orchestrator.py:96` |
| 10 | Negotiation stops after round 1 with conflicts left | `stalled` (nobody changed) or conflicts between non-neighbours | `negotiation.py:68-70`, `conflicts_visible_to` |
| 11 | A reroute vanished from the output | advisory `valid=False` or `suppressed=True` → `_reroutes` returns `[]` | `orchestrator.py:85` |
| 12 | LLM agent output silently equals rule output | dry run is on | `LLMAgent._dry`, `llm_client.py:79` |
| 13 | Huge anomaly score for one sensor | `hist_std = 0` replaced by `1e-6` | `parse.py:24` |
| 14 | Gate edits leak into the round-0 snapshot | dict mutated in place instead of copied | `det_guards.py:35,52` |

---

## 6. How to debug any unfamiliar code (assessment method)

1. **Reproduce first.** Run the failing test or input; read the *whole* traceback bottom-up (last frame = where it broke, earlier frames = how it got there).
2. **State the expected value** before reading code ("this list should have 2 zones").
3. **Follow the data, not the files.** Print the object at each stage boundary (§1) and find the first stage where it is wrong.
4. **Suspect the usual five:** off-by-one/index vs step, wrong comparison (`<` vs `<=`), mutation through a shared reference, wrong default / `None` handling, unordered iteration.
5. **One change, then re-run.** Small diff, explain why it fixes the root cause, not the symptom.
6. Run the tests: `python -m pytest tests/test_coordination.py -q` (19 tests, all pass as of Oct 24).
