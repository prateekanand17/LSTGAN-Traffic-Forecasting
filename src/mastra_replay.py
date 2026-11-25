"""MASTRA replay control room (Streamlit).

Replays the offline study scenario by scenario. For the chosen scenario, forecaster seed and communication topology
the real pipeline (pipeline/orchestrator.run_scenario) is run again with the rule agents and the variant built by
evaluation/run_robustness.variant_for(cell, "base"), i.e. the same code path and settings as the reported runs.
Every replay is checked against the stored record of the same run
(evaluation/results/robustness_f2/records_base_s<seed>.jsonl). No LLM call and no new experiment.

Run from the repository root:  streamlit run src/mastra_replay.py
"""
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pipeline import config as C  # noqa: E402
from pipeline import data_source as D  # noqa: E402
from pipeline.coordination.common import SEV_RANK, _haversine_km, detect_conflicts  # noqa: E402
from pipeline.orchestrator import run_scenario  # noqa: E402
from evaluation.metrics import gt_severity, scenario_metrics  # noqa: E402
from evaluation.run_robustness import KEEP, SCENARIOS, comm_adj, variant_for  # noqa: E402

SEEDS = (42, 123, 456, 789, 1024)        # forecaster seeds with base-setting records
MAIN_SEEDS = (42, 123, 456)              # seeds of the main analysis (report Table 7.2)
RECORDS = os.path.join(C.RESULTS_DIR, "robustness_f2", "records_base_s{}.jsonl")

# label -> cell of evaluation/run_robustness.CELLS
TOPOLOGIES = {
    "Learned graph (forecaster)": "R_neg",
    "Geographic kNN-100 (same density)": "R_neg_knn100",
    "Corridor overlap": "R_neg_corridor",
    "Geographic kNN-10": "R_neg_knn",
    "Distance graph (top-10)": "R_neg_distance",
    "Random rewiring of learned (draw 0)": "R_neg_random0",
    "Fully connected": "R_neg_full",
    "No links": "R_neg_none",
    "Central coordinator (global view)": "R_central",
    "No coordination": "R_none",
}
CELL_LABEL = {v: k for k, v in TOPOLOGIES.items()}
SEV_COLOR = {"HIGH": "#C0392B", "MEDIUM": "#E07A1F", "LOW": "#1B998B"}
OK_COLOR, CONFLICT_COLOR, LINK_COLOR = "#1F3B73", "#D62728", "rgba(90,90,90,0.45)"
MAP_CONFIG = {"scrollZoom": False, "displaylogo": False}   # the wheel scrolls the page, not the map
SPEED_LAYERS = {"Forecast +30 min": ("pred", 6), "Forecast +60 min": ("pred", 12),
                "Observed +30 min": ("actual", 6), "Observed +60 min": ("actual", 12)}


# --- data (cached) ------------------------------------------------------------------------------------------------
@st.cache_data
def load_scenarios():
    return json.load(open(SCENARIOS, encoding="utf-8"))["scenarios"]


@st.cache_data
def load_records(seed: int) -> dict:
    out = {}
    for line in open(RECORDS.format(seed), encoding="utf-8"):
        r = json.loads(line)
        out[(r["cell"], r["sample_idx"])] = r["metrics"]
    return out


@st.cache_resource
def stations():
    return D.load_stations()


@st.cache_resource(max_entries=64)
def base_input(sample: int, seed: int):
    return D.build_input(sample, seed)


@st.cache_data(show_spinner=False, max_entries=512)
def replay(cell: str, sample: int, seed: int):
    """Same steps as evaluation/run_robustness.py (base setting) for one cell and scenario."""
    v = variant_for(cell, "base")
    inp = copy.copy(base_input(sample, seed))
    inp.comm_adj = comm_adj(v.topology, seed)
    inp.gmp_adj = None
    res = run_scenario(inp, v, meta={"run_id": "replay", "sample_idx": sample})
    return res, scenario_metrics(res, inp.actual_speed, inp.roads)


# --- helpers ------------------------------------------------------------------------------------------------------
def zone_labels(res) -> dict:
    """Z1, Z2, ... in priority order (the order that wins a conflict, pipeline.coordination.common.priority)."""
    zs = sorted(res["zones"], key=lambda z: (SEV_RANK[z["detector_severity"]], round(z["updated_cascade_risk"], 6),
                                             z["zone_id"]), reverse=True)
    return {z["zone_id"]: f"Z{i + 1}" for i, z in enumerate(zs)}


def scored_zones(res) -> set:
    return {z["zone_id"] for z in res["zones"] if z["advisory"].get("valid") and not z["advisory"].get("suppressed")}


def frames(res) -> list:
    """[(label, note, reroutes per zone)] from round 0 to the scored final plan."""
    ok = scored_zones(res)
    keep = lambda rr: {z: list(rr.get(z) or []) for z in sorted(ok)}  # noqa: E731
    coord = res["coordination"]
    out = [("Round 0", "Every zone agent proposes on its own: reroute to the fastest free corridor.",
            keep({z["zone_id"]: z["initial_reroutes"] for z in res["zones"]}))]
    for t in coord["transcript"]:
        note = (f"{len(t['revised'])} zones saw a conflict with a neighbour and revised; {len(t['changed'])} changed "
                f"their plan; {t['messages']} messages exchanged.")
        out.append((f"Round {t['round']}", note, keep(t["proposals"])))
    final = keep({z["zone_id"]: z["advisory"].get("actionable_reroutes") for z in res["zones"]})
    if final != out[-1][2]:
        who = "the central coordinator" if coord["coordinator"] else "the override gates"
        out.append(("Final plan", f"Plan after {who}.", final))
    return out


def conflict_parts(conflicts) -> set:
    """(zone, road) pairs that take part in a conflict."""
    parts = set()
    for kind, road, a, b in conflicts:
        parts.add((a, road))
        if kind == "shared":
            parts.add((b, road))
    return parts


def zone_center(z, lats, lngs):
    idx = z["sensor_indices"]
    return float(np.mean(lats[idx])), float(np.mean(lngs[idx]))


def road_point(road, lat0, lng0, inp):
    """Location of a reroute target: mean position of the road's sensors within the reroute radius."""
    roads = np.asarray(inp.roads)
    on = np.where(roads == road)[0]
    if not len(on):
        return None
    d = _haversine_km(lat0, lng0, inp.lats[on].astype(float), inp.lngs[on].astype(float))
    near = on[d <= C.REROUTE_RADIUS_KM]
    pick = near if len(near) else on[[int(np.argmin(d))]]
    return float(inp.lats[pick].mean()), float(inp.lngs[pick].mean())


def fmt_rr(rr) -> str:
    return ", ".join(f"{r['from_road']} → {r['to_road']}" for r in rr) if rr else "—"


# --- map ----------------------------------------------------------------------------------------------------------
def scenario_map(res, inp, rr, conflicts, layer, show_links, height=600, show_scale=True, zoom=10.2, legend=True):
    st_df = stations()
    lats, lngs = inp.lats.astype(float), inp.lngs.astype(float)
    src, step = SPEED_LAYERS[layer]
    speed = (inp.pred_speed if src == "pred" else inp.actual_speed)[step - 1]
    labels = zone_labels(res)
    parts = conflict_parts(conflicts)
    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=lats, lon=lngs, mode="markers", name="Sensors (speed)",
        marker=dict(size=7, color=np.where(speed > 0, speed, np.nan), colorscale="RdYlGn", cmin=10, cmax=70,
                    showscale=show_scale, colorbar=dict(title="mph", thickness=10, len=0.5, x=0.0, xanchor="left")),
        text=[f"{sid} · {r} · {s:.0f} mph" if s > 0 else f"{sid} · {r} · no reading"
              for sid, r, s in zip(st_df["station_id"], inp.roads, speed)],
        hoverinfo="text"))
    centers = {z["zone_id"]: zone_center(z, lats, lngs) for z in res["zones"]}
    if show_links:
        la, lo = [], []
        for z in res["zones"]:
            for n in z["neighbors"]:
                if z["zone_id"] < n:
                    (a1, b1), (a2, b2) = centers[z["zone_id"]], centers[n]
                    la += [a1, a2, None]
                    lo += [b1, b2, None]
        fig.add_trace(go.Scattermap(lat=la, lon=lo, mode="lines", line=dict(width=2, color=LINK_COLOR),
                                    name="Agents that can talk", hoverinfo="skip"))
    for bad in (False, True):
        la, lo, tla, tlo, ttx = [], [], [], [], []
        for zid, rrs in rr.items():
            c = centers[zid]
            for r in rrs:
                if ((zid, r["to_road"]) in parts) != bad:
                    continue
                p = road_point(r["to_road"], c[0], c[1], inp)
                if p is None:
                    continue
                la += [c[0], p[0], None]
                lo += [c[1], p[1], None]
                tla.append(p[0]); tlo.append(p[1]); ttx.append(f"{labels[zid]} → {r['to_road']}")
        col = CONFLICT_COLOR if bad else OK_COLOR
        name = "Reroute in conflict" if bad else "Reroute without conflict"
        fig.add_trace(go.Scattermap(lat=la, lon=lo, mode="lines", line=dict(width=4 if bad else 3, color=col),
                                    name=name, hoverinfo="skip"))
        fig.add_trace(go.Scattermap(lat=tla, lon=tlo, mode="markers+text", marker=dict(size=11, color=col),
                                    text=[t.split("→ ")[1] for t in ttx], textposition="top right",
                                    textfont=dict(size=11, color=col), hovertext=ttx, hoverinfo="text",
                                    showlegend=False))
    for sev in ("HIGH", "MEDIUM", "LOW"):
        zs = [z for z in res["zones"] if z["detector_severity"] == sev]
        if not zs:
            continue
        fig.add_trace(go.Scattermap(
            lat=[centers[z["zone_id"]][0] for z in zs], lon=[centers[z["zone_id"]][1] for z in zs],
            mode="markers+text", name=f"Zone ({sev})",
            marker=dict(size=[16 + 2 * np.sqrt(len(z["sensor_indices"])) for z in zs], color=SEV_COLOR[sev],
                        opacity=0.9),
            text=[labels[z["zone_id"]] for z in zs], textposition="middle center",
            textfont=dict(size=10, color="white"),
            hovertext=[f"{labels[z['zone_id']]} · {', '.join(z['roads'])} · {len(z['sensor_indices'])} sensors · "
                       f"forecast +30 {z['forecast_30min']:.0f} mph" for z in zs], hoverinfo="text"))
    center = dict(lat=float(lats.min() + lats.max()) / 2, lon=float(lngs.min() + lngs.max()) / 2)
    fig.update_layout(map=dict(style="carto-positron", center=center, zoom=zoom),
                      height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=legend,
                      legend=dict(orientation="h", yanchor="top", y=0.99, x=0.01, bgcolor="rgba(255,255,255,0.9)",
                                  font=dict(color="#222", size=11)))
    return fig


def conflict_lines(conflicts, res):
    labels = zone_labels(res)
    nb = {z["zone_id"]: set(z["neighbors"]) for z in res["zones"]}
    rows = []
    for kind, road, a, b in sorted(conflicts):
        visible = b in nb[a]
        what = (f"{labels[a]} and {labels[b]} both send traffic to {road}" if kind == "shared"
                else f"{labels[a]} sends traffic into {labels[b]}'s congested road {road}")
        rows.append((visible, what))
    return rows


def check_against_record(cell, sample, seed, met):
    rec = load_records(seed).get((cell, sample))
    if rec is None:
        return None, []
    diffs = []
    for k in KEEP:
        if k in rec:
            a, b = rec[k], met.get(k)
            same = (abs(a - b) < 1e-9) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a == b
            if not same:
                diffs.append((k, rec[k], met.get(k)))
    return rec, diffs


# --- page ---------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="MASTRA Replay", page_icon="🚦", layout="wide")
st.title("MASTRA replay control room")
st.caption("Offline replay of the PeMS-BAY test set (325 freeway sensors, June 2017). The advisories were never issued "
           "to drivers: each plan is scored against the speeds that were actually recorded. Rule agents only; no "
           "language model is called.")

scen = load_scenarios()
with st.sidebar:
    st.header("Scenario")
    seed = st.selectbox("Forecaster seed (Adaptive-LSTGAN)", SEEDS, index=0,
                        help="Each seed has its own forecasts and its own learned graph.")
    recs = load_records(seed)
    rows = []
    for s in scen:
        g = lambda c: recs.get((c, s["sample_idx"]), {})  # noqa: E731
        rows.append({"sample_idx": s["sample_idx"], "time": s["timestamp"][:16], "regime": s["regime"],
                     "conflicts_r0": g("R_neg").get("n_conflicts_initial", 0),
                     "learned": g("R_neg").get("n_conflicts_final", 0),
                     "knn100": g("R_neg_knn100").get("n_conflicts_final", 0),
                     "corridor": g("R_neg_corridor").get("n_conflicts_final", 0)})
    table = pd.DataFrame(rows)
    order = st.radio("List", ["Largest learned − kNN-100 gap", "Most round-0 conflicts", "Time order"],
                     help="Ordering uses the stored results of this seed.")
    if order == "Largest learned − kNN-100 gap":
        table = table.assign(gap=table.learned - table.knn100).sort_values(["gap", "conflicts_r0"], ascending=False)
    elif order == "Most round-0 conflicts":
        table = table.sort_values("conflicts_r0", ascending=False)
    contested_only = st.checkbox("Only scenarios with round-0 conflicts", value=True)
    if contested_only:
        table = table[table.conflicts_r0 > 0]
    names = {r.sample_idx: f"{r.time} · {r.regime} · learned {r.learned} / kNN-100 {r.knn100} left"
             for r in table.itertuples()}
    pick = st.selectbox(f"Scenario ({len(table)})", list(names), format_func=names.get)
    st.header("Coordination")
    topo_label = st.selectbox("Communication topology", list(TOPOLOGIES), index=0)
    layer = st.selectbox("Sensor colour", list(SPEED_LAYERS), index=0)
    show_links = st.checkbox("Show which agents can talk", value=True)

cell = TOPOLOGIES[topo_label]
sample = int(pick)
s_meta = next(s for s in scen if s["sample_idx"] == sample)
inp = base_input(sample, seed)
with st.spinner("Running the pipeline..."):
    res, met = replay(cell, sample, seed)

tab_replay, tab_compare, tab_study, tab_about = st.tabs(
    ["Replay", "Compare topologies", "Study results", "What this is"])

# --- tab 1: replay ------------------------------------------------------------------------------------------------
with tab_replay:
    if res["status"] != "SUCCESS":
        st.info("No congestion zone in this forecast: nothing to coordinate.")
    else:
        fr = frames(res)
        origin = pd.Timestamp(s_meta["timestamp"])
        st.markdown(f"#### {origin:%A %d %B %Y, %H:%M} · {s_meta['regime'].title()} regime · {res['n_zones']} "
                    f"congestion zones · {'rush hour' if res['is_rush_hour'] else 'outside rush hour'}")
        st.caption(f"{topo_label} · forecaster seed {seed} · scenario {sample} of the frozen test list")

        idx = st.select_slider("Negotiation round", options=list(range(len(fr))), value=0,
                               format_func=lambda i: fr[i][0])
        label, note, rr = fr[idx]
        conflicts = detect_conflicts(rr, {z["zone_id"]: z["roads"] for z in res["zones"]})
        clines = conflict_lines(conflicts, res)
        n0 = len(detect_conflicts(fr[0][2], {z["zone_id"]: z["roads"] for z in res["zones"]}))
        n_vis = sum(v for v, _ in clines)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Conflicts", len(conflicts), delta=len(conflicts) - n0 if idx else None, delta_color="inverse")
        k2.metric("Visible to the agents", n_vis)
        k3.metric("Hidden", len(conflicts) - n_vis, help="Conflicts between zones that are not linked: neither "
                  "agent can see them.")
        k4.metric("Messages sent", res["coordination"]["messages"], help="Proposals exchanged over the whole run.")
        st.markdown(f"**{label}.** {note}")

        left, right = st.columns([2.2, 1])
        with left:
            st.plotly_chart(scenario_map(res, inp, rr, conflicts, layer, show_links), use_container_width=True,
                            config=MAP_CONFIG)
        with right:
            st.subheader("Conflicts")
            box = st.container(height=520)
            if not clines:
                box.success("No conflicting reroutes in this plan.")
            for visible, what in clines:
                box.markdown(f"{'🟠' if visible else '🔴'} {what} · "
                             f"{'*visible: the two agents are linked*' if visible else '**hidden: no link between the two zones**'}")
            stop = res["coordination"]["stop_reason"]
            if stop:
                st.caption(f"Negotiation stopped: {stop.replace('_', ' ')} after {res['coordination']['rounds_used']} "
                           f"round(s) (maximum {C.NEGOTIATION_ROUNDS}).")

        rec, diffs = check_against_record(cell, sample, seed, met)
        if rec is None:
            st.warning("No stored record for this run.")
        elif diffs:
            st.error("Replay differs from the stored record: " +
                     "; ".join(f"{k}: stored {a}, replay {b}" for k, a, b in diffs))
        else:
            st.success(f"Replay reproduces the stored run (`records_base_s{seed}.jsonl`, cell `{cell}`): "
                       f"{met['n_conflicts_initial']} conflicts at round 0, {met['n_conflicts_final']} in the final "
                       f"plan, WZPA@30 {met.get('wzpa30', float('nan')):.3f}, WZPA@60 {met.get('wzpa60', float('nan')):.3f}.")

        st.subheader("Zone agents")
        labels = zone_labels(res)
        final_rr = fr[-1][2]
        zrows = []
        for z in sorted(res["zones"], key=lambda z: int(labels[z["zone_id"]][1:])):
            adv = z["advisory"]
            ok = adv.get("valid") and not adv.get("suppressed")
            zrows.append({
                "Zone": labels[z["zone_id"]], "Roads": ", ".join(z["roads"]), "Sensors": len(z["sensor_indices"]),
                "Forecast +30 (mph)": round(z["forecast_30min"], 1), "Forecast +60 (mph)": round(z["forecast_60min"], 1),
                "Advised +30": adv.get("severity_30min") if ok else "suppressed",
                "Observed +30": gt_severity(inp.actual_speed, z["sensor_indices"], 6),
                "Advised +60": adv.get("severity_60min") if ok else "suppressed",
                "Observed +60": gt_severity(inp.actual_speed, z["sensor_indices"], 12),
                "Round-0 reroute": fmt_rr(z["initial_reroutes"]), "Final reroute": fmt_rr(final_rr.get(z["zone_id"])),
                "Talks to": ", ".join(sorted((labels[n] for n in z["neighbors"]), key=lambda s: int(s[1:]))) or "—",
            })
        st.dataframe(pd.DataFrame(zrows), hide_index=True, use_container_width=True)
        st.caption("Zones are numbered by priority (severity, then cascade risk): a higher-priority zone keeps a "
                   "contested road. 'Observed' is the severity of the recorded speeds, used only for scoring.")
        with st.expander("Advisory text of each zone (final plan)"):
            for z in sorted(res["zones"], key=lambda z: int(labels[z["zone_id"]][1:])):
                adv = z["advisory"]
                text = adv.get("advisory") or {}
                st.markdown(f"**{labels[z['zone_id']]}** ({', '.join(z['roads'])}) · now: {text.get('now', '—')} · "
                            f"soon: {text.get('soon', '—')} · prevent: {text.get('prevent', '—')}"
                            + (f" · gates: {', '.join(adv.get('gates'))}" if adv.get("gates") else "")
                            + (" · upstream alert" if adv.get("upstream_alert") else ""))

# --- tab 2: compare topologies ------------------------------------------------------------------------------------
with tab_compare:
    if res["status"] != "SUCCESS":
        st.info("No congestion zone in this forecast.")
    else:
        st.markdown("Same scenario, same forecast and same round-0 proposals; only the communication graph changes.")
        picks = st.multiselect("Topologies side by side", list(TOPOLOGIES),
                               default=["Learned graph (forecaster)", "Geographic kNN-100 (same density)",
                                        "Corridor overlap"], max_selections=3)
        cols = st.columns(max(1, len(picks)))
        for col, lab in zip(cols, picks):
            r2, m2 = replay(TOPOLOGIES[lab], sample, seed)
            f2 = frames(r2)
            cf = detect_conflicts(f2[-1][2], {z["zone_id"]: z["roads"] for z in r2["zones"]})
            with col:
                st.markdown(f"**{lab}**")
                a, b = st.columns(2)
                a.metric("Final conflicts", m2["n_conflicts_final"],
                         delta=m2["n_conflicts_final"] - m2["n_conflicts_initial"], delta_color="inverse")
                b.metric("Messages", m2.get("messages", 0))
                st.plotly_chart(scenario_map(r2, inp, f2[-1][2], cf, layer, show_links, height=420,
                                             show_scale=False, zoom=9.5, legend=False), use_container_width=True,
                                config=MAP_CONFIG, key=f"cmp_{lab}")
                st.caption(f"Zone-pair link density {m2.get('link_density', 0):.2f} · "
                           f"{m2.get('rounds_used', 0)} round(s) · stop: {m2.get('stop_reason') or '—'}")
        allm = [(lab, replay(c, sample, seed)[1]) for lab, c in TOPOLOGIES.items()]
        bar = go.Figure(go.Bar(x=[m["n_conflicts_final"] for _, m in allm], y=[lab for lab, _ in allm],
                               orientation="h", marker_color=[CONFLICT_COLOR if m["n_conflicts_final"] else OK_COLOR
                                                              for _, m in allm]))
        bar.update_layout(height=360, margin=dict(l=0, r=10, t=30, b=0), title="Final conflicts in this scenario",
                          yaxis=dict(autorange="reversed"), xaxis_title="conflicting reroute pairs left")
        st.plotly_chart(bar, use_container_width=True)

# --- tab 3: study results -----------------------------------------------------------------------------------------
with tab_study:
    st.markdown("Means over all 120 scenarios, recomputed here from the stored records. Unit: a scenario with at "
                "least one round-0 conflict, averaged over the seeds in which it is contested (as in the report).")

    @st.cache_data
    def study_table(seeds):
        per = {}
        for s in seeds:
            for (c, i), m in load_records(s).items():
                if c in CELL_LABEL and m.get("n_conflicts_initial", 0) > 0:
                    per.setdefault(c, {}).setdefault(i, []).append(m["n_conflicts_final"])
        return pd.DataFrame([{"Topology": CELL_LABEL[c], "Scenarios": len(d),
                              "Final conflicts per contested scenario": float(np.mean([np.mean(v) for v in d.values()]))}
                             for c, d in per.items()]).sort_values("Final conflicts per contested scenario")

    which = st.radio("Seeds", ["Main analysis: 42, 123, 456", "All five seeds"], horizontal=True)
    tab = study_table(MAIN_SEEDS if which.startswith("Main") else SEEDS)
    fig = go.Figure(go.Bar(x=tab["Final conflicts per contested scenario"], y=tab["Topology"], orientation="h",
                           marker_color=[CONFLICT_COLOR if "Learned" in t else OK_COLOR for t in tab["Topology"]],
                           text=[f"{v:.2f}" for v in tab["Final conflicts per contested scenario"]],
                           textposition="outside"))
    fig.update_layout(height=420, margin=dict(l=0, r=40, t=10, b=0), yaxis=dict(autorange="reversed"),
                      xaxis_title="final conflicts per contested scenario")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(tab.round(3), hide_index=True, use_container_width=True)
    st.caption("Source: evaluation/results/robustness_f2/records_base_s<seed>.jsonl. For the three main seeds these "
               "means are the values reported in the paper and in the end-semester report (learned 2.10, kNN-100 "
               "1.59, corridor overlap 0.005).")

# --- tab 4: what this is ------------------------------------------------------------------------------------------
with tab_about:
    st.markdown(f"""
**What you see.** For the chosen scenario, the forecast of Adaptive-LSTGAN (seed {seed}) for the next 60 minutes is
turned into congestion zones. Each zone has a rule agent that proposes a reroute onto the fastest nearby corridor
that is not blocked. Agents send their proposals only to the zones they are linked to in the communication graph and
revise when they see a conflict with a linked zone, for up to {C.NEGOTIATION_ROUNDS} rounds. A conflict between two
zones that are not linked stays invisible to both of them.

**How it is computed.** The page calls the project code: `pipeline.orchestrator.run_scenario` with the variant from
`evaluation.run_robustness.variant_for(cell, "base")`, on the frozen scenario list
`evaluation/results/rule_cells_s42/scenarios.json`. The green check under the map compares the replay with the stored
record of the same run.

**What it is not.** It is not a live system and it did not control traffic. The scores compare each plan with the
speeds recorded in June 2017; the effect of the advice on traffic was not measured (that needs a closed-loop
simulator or a field trial). The language-model agents of the proposal are not used here.
""")
