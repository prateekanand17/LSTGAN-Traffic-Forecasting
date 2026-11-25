"""Analysis of the F2 robustness runs (plan §13; review findings M39, M42, M43, M46, M47). No API calls.

Reads evaluation/results/robustness_f2/records_*.jsonl (evaluation/run_robustness.py) and writes summary.json and
summary.md in the same folder. All analyses here are exploratory (added after the first results).

Unit: scenario. Conflict metrics on contested scenario-seeds (>= 1 round-0 conflict in the R_neg cell of the same
setting and seed), averaged over the seeds in which the scenario is contested. Paired differences (cell - R_neg)
with 95% bootstrap CI and Wilcoxon signed-rank p (Holm across all rows of section 1, reported for transparency).

Sections
 1. settings: radius 3/8 km, segment-level conflicts, DBSCAN eps, K, neighbour threshold, fixed priorities
 2. 100 random rewirings per seed: learned-graph percentile (graph-level), learned vs mean of 100 (scenario-level)
 3. five forecaster seeds: per-seed paired differences and a seed-level summary
 4. conflict coverage: paired differences and coverage/density with bootstrap CI
 5. priority order changes caused by the topology (review R-17)
 6. graph statistics: edges, edges per sensor, overlap with the learned graph, edge length; zone-level density of
    the full distance graph; consensus edge counts; rewiring overlap at 10 vs 50 swaps per edge
 7. the alpha = 1 learned graph and the static forecaster as the zone source (status M58)
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from evaluation.summarize_mastra import boot_ci, holm  # noqa: E402

DIR = os.path.join(ROOT, "evaluation", "results", "robustness_f2")
SEEDS3 = (42, 123, 456)
SEEDS5 = (42, 123, 456, 789, 1024)
RAND5 = [f"R_neg_random{i}" for i in range(5)]
RAND100 = [f"R_neg_random{i}" for i in range(100)]
COMPARE = ["R_neg_random_mean", "R_neg_knn100", "R_neg_knn80", "R_neg_corridor", "R_neg_knn", "R_neg_distance",
           "R_neg_none", "R_neg_full", "R_central", "R_none"]
SETTING_ORDER = ["base", "r3", "r8", "seg", "eps010", "eps020", "K5", "K10", "thr1", "thr5", "prio"]


def load():
    data = defaultdict(dict)   # data[(setting, cell)][(seed, sample)] = record
    regime = {}
    for p in sorted(glob.glob(os.path.join(DIR, "records_*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            data[(r["setting"], r["cell"])][(r["seed"], r["sample_idx"])] = r
            regime[r["sample_idx"]] = r["regime"]
    return data, regime


def contested(data, setting):
    return {k for k, r in data.get((setting, "R_neg"), {}).items() if r["metrics"].get("n_conflicts_initial", 0) > 0}


def metric_of(r, m):
    if m == "coverage":
        return r["covered_pairs0"] / r["conflict_pairs0"] if r.get("conflict_pairs0") else None
    v = r["metrics"].get(m)
    return float(v) if isinstance(v, (int, float)) and v == v else None


def values(data, setting, cell, m, seeds, cont):
    """{sample: mean over seeds}; cell may be a mean of random draws ('R_neg_random_mean' / '..._mean100')."""
    if cell == "R_neg_random_mean100":
        cells = RAND100
    elif cell.endswith("_random_mean"):   # mean of the five rewirings, e.g. R_neg_random_mean, R_neg_gwnet_random_mean
        cells = [cell[:-len("_mean")] + str(i) for i in range(5)]
    else:
        cells = [cell]
    acc = defaultdict(list)
    for seed in seeds:
        for (sd, s) in cont:
            if sd != seed:
                continue
            xs = [metric_of(data[(setting, c)][(sd, s)], m) for c in cells if (sd, s) in data.get((setting, c), {})]
            xs = [x for x in xs if x is not None]
            if len(xs) == len(cells):
                acc[s].append(float(np.mean(xs)))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def paired(a, b, rng):
    common = sorted(set(a) & set(b))
    if len(common) < 5:
        return None
    d = np.array([b[s] - a[s] for s in common])
    p = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    return {"n": len(common), "n_nonzero": int((d != 0).sum()), "mean_diff": float(d.mean()), "ci95": boot_ci(d, rng),
            "p": p}


def haversine_km(lat1, lng1, lat2, lng2):
    q = np.pi / 180
    a = np.sin((lat2 - lat1) * q / 2) ** 2 + np.cos(lat1 * q) * np.cos(lat2 * q) * np.sin((lng2 - lng1) * q / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def graph_stats(seeds):
    from pipeline import data_source as D
    st = D.load_stations()
    lat, lng = st["latitude"].values.astype(float), st["longitude"].values.astype(float)
    dist = haversine_km(lat[:, None], lng[:, None], lat[None, :], lng[None, :])
    out = {}
    for seed in seeds:
        L = D.topology("learned", seed=seed) > 0
        np.fill_diagonal(L, False)
        row = {}

        def describe(name, A):
            E = A > 0
            np.fill_diagonal(E, False)
            n = int(E.sum())
            lens = dist[E]
            und_l = L | L.T
            return {"edges": n, "edges_per_sensor": n / len(E), "overlap_with_learned": float((E & L).sum() / n) if n else None,
                    "undirected_overlap_with_learned": float(((E | E.T) & und_l).sum() / max(1, (E | E.T).sum())),
                    "edge_km_median": float(np.median(lens)) if n else None,
                    "edge_km_p90": float(np.percentile(lens, 90)) if n else None,
                    "share_edges_over_10km": float(np.mean(lens > 10)) if n else None}

        for name in ("learned", "knn", "knn:80", "knn:100", "distance", "distance_full", "consensus", "consensus:3",
                     "alpha1"):
            try:
                row[name] = describe(name, D.topology(name, seed=seed))
            except FileNotFoundError:   # gwnet exists for seeds 42/123/456 only
                pass
        r10 = [describe("r", D.topology("random", seed=seed, rand_seed=i)) for i in range(100)]
        row["random_100draws"] = {k: [float(np.min([x[k] for x in r10])), float(np.mean([x[k] for x in r10])),
                                      float(np.max([x[k] for x in r10]))] for k in r10[0]}
        r50 = [describe("r50", D._rewire(D.learned_adj(seed), np.random.default_rng(i), n_swaps_per_edge=50)) for i in range(5)]
        row["random_50swaps_5draws"] = {k: [float(np.min([x[k] for x in r50])), float(np.max([x[k] for x in r50]))]
                                        for k in ("overlap_with_learned", "edge_km_median")}
        out[seed] = row
    return out


def zone_density_distance_full(seeds):
    """Zone-level link density of the full DCRNN distance graph on the zones of the main runs (review R-16)."""
    from pipeline import data_source as D
    from pipeline.coordination.common import zone_link
    A = D.topology("distance_full")
    res = {}
    for seed in seeds:
        p = os.path.join(ROOT, "evaluation", "results", f"rule_cells_s{seed}", "records.jsonl")
        if not os.path.exists(p):
            continue
        dens_all, dens_cont = [], []
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["variant"] != "R_neg" or r["metrics"].get("status") != "SUCCESS":
                continue
            zs = r["result"]["zones"]
            if len(zs) < 2:
                continue
            pairs = [(i, j) for i in range(len(zs)) for j in range(i + 1, len(zs))]
            d = float(np.mean([zone_link(A, zs[i]["sensor_indices"], zs[j]["sensor_indices"]) > 0 for i, j in pairs]))
            dens_all.append(d)
            if r["metrics"].get("n_conflicts_initial", 0) > 0:
                dens_cont.append(d)
        res[seed] = {"all_scenarios": float(np.mean(dens_all)), "n_all": len(dens_all),
                     "contested": float(np.mean(dens_cont)), "n_contested": len(dens_cont)}
    return res


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data, regime = load()
    rng = np.random.default_rng(0)
    out = {"settings": {}, "rewiring": {}, "seeds5": {}, "coverage": {}, "priority": {}, "graphs": {}}
    L = ["# F2 robustness runs (exploratory; rule agents, no API)", "",
         "Source: `evaluation/results/robustness_f2/records_*.jsonl` (`evaluation/run_robustness.py`). Unit: scenario; "
         "conflict metrics on contested scenario-seeds, averaged over seeds; diff = cell − R_neg (learned graph); "
         "95% bootstrap CI; Wilcoxon p, Holm over all rows of section 1.", ""]

    # 1. settings ----------------------------------------------------------------------------------------------
    rows = []
    for setting in SETTING_ORDER:
        cont = contested(data, setting)
        cont3 = {k for k in cont if k[0] in SEEDS3}
        if not cont3:
            continue
        cells = ["R_neg"] + [c for c in COMPARE if c == "R_neg_random_mean" and all((setting, x) in data for x in RAND5)
                             or (setting, c) in data]
        S = {"n_contested_scenario_seeds": len(cont3), "cells": {}}
        ref = values(data, setting, "R_neg", "n_conflicts_final", SEEDS3, cont3)
        init = values(data, setting, "R_neg", "n_conflicts_initial", SEEDS3, cont3)
        S["round0_conflicts"] = float(np.mean(list(init.values())))
        for c in cells:
            v = values(data, setting, c, "n_conflicts_final", SEEDS3, cont3)
            e = {"final_conflicts": float(np.mean(list(v.values()))), "n_scenarios": len(v)}
            for m in ("conflict_free", "messages", "link_density", "coverage"):
                x = values(data, setting, c, m, SEEDS3, cont3)
                if x:
                    e[m] = float(np.mean(list(x.values())))
            if c != "R_neg":
                pr = paired(ref, v, rng)
                if pr:
                    e["vs_learned"] = pr
                    rows.append(pr)
            S["cells"][c] = e
        out["settings"][setting] = S
    for pr, padj in zip(rows, holm([r["p"] for r in rows])):
        pr["p_holm"] = padj
    L += ["## 1. Final conflicts by setting (seeds 42/123/456)", "",
          "| setting | n | round-0 | cell | final conflicts | conflict-free | messages | zone density | coverage | diff vs learned [95% CI] | p | p_Holm |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for setting, S in out["settings"].items():
        for c, e in S["cells"].items():
            vs = e.get("vs_learned")
            f = lambda k: f"{e[k]:.3f}" if k in e else "–"
            L.append(f"| {setting} | {S['n_contested_scenario_seeds']} | {S['round0_conflicts']:.2f} | {c} | {e['final_conflicts']:.3f} | "
                     f"{f('conflict_free')} | {f('messages')} | {f('link_density')} | {f('coverage')} | "
                     + (f"{vs['mean_diff']:+.3f} [{vs['ci95'][0]:+.3f}, {vs['ci95'][1]:+.3f}] | {vs['p']:.2g} | {vs['p_holm']:.2g} |"
                        if vs else "– | – | – |"))

    # 2. 100 rewirings ----------------------------------------------------------------------------------------
    L += ["", "## 2. Learned graph vs 100 degree-preserving rewirings (base setting)", "",
          "| seed | n contested | learned | rewirings: min / 5% / median / 95% / max | draws ≤ learned | graph-level p = (1+#≤)/(1+N) | learned − mean of 100 [95% CI] | Wilcoxon p |",
          "|---|---|---|---|---|---|---|---|"]
    cont = contested(data, "base")
    for seed in SEEDS5:
        cs = {k for k in cont if k[0] == seed}
        have = [c for c in RAND100 if all(k in data.get(("base", c), {}) for k in cs)]
        if not cs or len(have) < 100:
            continue
        lm = float(np.mean([metric_of(data[("base", "R_neg")][k], "n_conflicts_final") for k in cs]))
        dm = np.array([np.mean([metric_of(data[("base", c)][k], "n_conflicts_final") for k in cs]) for c in RAND100])
        le = int((dm <= lm).sum())
        a = values(data, "base", "R_neg_random_mean100", "n_conflicts_final", (seed,), cs)
        b = values(data, "base", "R_neg", "n_conflicts_final", (seed,), cs)
        pr = paired(a, b, rng)
        out["rewiring"][seed] = {"n_contested": len(cs), "learned": lm, "draw_means": dm.tolist(), "draws_le_learned": le,
                                 "graph_p": (1 + le) / (1 + len(dm)), "learned_minus_mean100": pr}
        q = np.percentile(dm, [0, 5, 50, 95, 100])
        L.append(f"| {seed} | {len(cs)} | {lm:.3f} | " + " / ".join(f"{x:.3f}" for x in q) + f" | {le} | {(1 + le) / 101:.3f} | "
                 f"{pr['mean_diff']:+.3f} [{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}] | {pr['p']:.2g} |")
    cs3 = {k for k in cont if k[0] in SEEDS3}
    if all(("base", c) in data for c in RAND100):
        pr = paired(values(data, "base", "R_neg_random_mean100", "n_conflicts_final", SEEDS3, cs3),
                    values(data, "base", "R_neg", "n_conflicts_final", SEEDS3, cs3), rng)
        out["rewiring"]["pooled_3seeds"] = pr
        if pr:
            L.append(f"| 42/123/456 pooled (scenario unit) | {pr['n']} | | | | | {pr['mean_diff']:+.3f} "
                     f"[{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}] | {pr['p']:.2g} |")

    # 3. five seeds ----------------------------------------------------------------------------------------------
    L += ["", "## 3. Per forecaster seed (base setting): cell − learned, final conflicts", "",
          "| seed | n | learned | kNN-100 − learned [CI] | kNN-80 − learned [CI] | random(5) − learned [CI] | corridor − learned [CI] |",
          "|---|---|---|---|---|---|---|"]
    per = {}
    for seed in SEEDS5:
        cs = {k for k in cont if k[0] == seed}
        if not cs:
            continue
        ref = values(data, "base", "R_neg", "n_conflicts_final", (seed,), cs)
        row = {"n": len(ref), "learned": float(np.mean(list(ref.values())))}
        for c in ("R_neg_knn100", "R_neg_knn80", "R_neg_random_mean", "R_neg_corridor"):
            row[c] = paired(ref, values(data, "base", c, "n_conflicts_final", (seed,), cs), rng)
        per[seed] = row
        g = lambda p: f"{p['mean_diff']:+.3f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]" if p else "–"
        L.append(f"| {seed} | {row['n']} | {row['learned']:.3f} | {g(row['R_neg_knn100'])} | {g(row['R_neg_knn80'])} | "
                 f"{g(row['R_neg_random_mean'])} | {g(row['R_neg_corridor'])} |")
    out["seeds5"]["per_seed"] = per
    L += ["", "Seed-level summary (one difference per seed; mean and t-interval over seeds):", ""]
    for c in ("R_neg_knn100", "R_neg_knn80", "R_neg_random_mean", "R_neg_corridor"):
        d = np.array([per[s][c]["mean_diff"] for s in per if per[s].get(c)])
        if len(d) >= 2:
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            out["seeds5"][c] = {"n_seeds": len(d), "mean": float(d.mean()), "t_ci95": [float(d.mean() - h), float(d.mean() + h)],
                                "n_negative": int((d < 0).sum())}
            L.append(f"- {c} − learned: mean over {len(d)} seeds {d.mean():+.3f}, t 95% CI [{d.mean() - h:+.3f}, {d.mean() + h:+.3f}], "
                     f"negative in {(d < 0).sum()}/{len(d)} seeds")
    cs5 = {k for k in cont if k[0] in SEEDS5}
    ref5 = values(data, "base", "R_neg", "n_conflicts_final", SEEDS5, cs5)
    for c in ("R_neg_knn100", "R_neg_random_mean"):
        pr = paired(ref5, values(data, "base", c, "n_conflicts_final", SEEDS5, cs5), rng)
        out["seeds5"][f"{c}_scenario_unit_5seeds"] = pr
        if pr:
            L.append(f"- {c} − learned, scenario unit over 5 seeds: {pr['mean_diff']:+.3f} [{pr['ci95'][0]:+.3f}, "
                     f"{pr['ci95'][1]:+.3f}], Wilcoxon p {pr['p']:.2g}, n = {pr['n']}")

    # 4. coverage ------------------------------------------------------------------------------------------------
    L += ["", "## 4. Conflict coverage (base setting, seeds 42/123/456)", "",
          "coverage = share of round-0 conflicting zone pairs that are neighbours. ratio = mean coverage / mean zone-level density, "
          "95% CI by scenario bootstrap.", "",
          "| cell | coverage | density | ratio [95% CI] | coverage − learned [95% CI] | Wilcoxon p |", "|---|---|---|---|---|---|"]
    cref = values(data, "base", "R_neg", "coverage", SEEDS3, cs3)
    for c in ["R_neg", "R_neg_knn100", "R_neg_knn80", "R_neg_knn", "R_neg_distance", "R_neg_random_mean", "R_neg_corridor",
              "R_neg_consensus"]:
        cv = values(data, "base", c, "coverage", SEEDS3, cs3)
        dn = values(data, "base", c, "link_density", SEEDS3, cs3)
        common = sorted(set(cv) & set(dn))
        if not common:
            continue
        x, y = np.array([cv[s] for s in common]), np.array([dn[s] for s in common])
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(common), len(common))
            if y[i].mean() > 0:
                bs.append(x[i].mean() / y[i].mean())
        ratio = float(x.mean() / y.mean()) if y.mean() > 0 else None
        pr = paired(cref, cv, rng) if c != "R_neg" else None
        out["coverage"][c] = {"coverage": float(x.mean()), "density": float(y.mean()), "ratio": ratio,
                              "ratio_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None,
                              "vs_learned": pr, "n": len(common)}
        e = out["coverage"][c]
        L.append(f"| {c} | {e['coverage']:.3f} | {e['density']:.3f} | "
                 + (f"{ratio:.2f} [{e['ratio_ci95'][0]:.2f}, {e['ratio_ci95'][1]:.2f}]" if ratio else "–") + " | "
                 + (f"{pr['mean_diff']:+.3f} [{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}] | {pr['p']:.2g} |" if pr else "– | – |"))
    # ratio difference kNN-100 vs learned (paired bootstrap)
    a_c, a_d = values(data, "base", "R_neg", "coverage", SEEDS3, cs3), values(data, "base", "R_neg", "link_density", SEEDS3, cs3)
    b_c, b_d = (values(data, "base", "R_neg_knn100", "coverage", SEEDS3, cs3),
                values(data, "base", "R_neg_knn100", "link_density", SEEDS3, cs3))
    common = sorted(set(a_c) & set(a_d) & set(b_c) & set(b_d))
    A1, A2, B1, B2 = (np.array([d[s] for s in common]) for d in (a_c, a_d, b_c, b_d))
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(common), len(common))
        bs.append(B1[i].mean() / B2[i].mean() - A1[i].mean() / A2[i].mean())
    out["coverage"]["ratio_diff_knn100_minus_learned"] = {"diff": float(B1.mean() / B2.mean() - A1.mean() / A2.mean()),
                                                          "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                                          "n": len(common)}
    r = out["coverage"]["ratio_diff_knn100_minus_learned"]
    L.append(f"\nratio(kNN-100) − ratio(learned) = {r['diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] (paired scenario bootstrap, n = {r['n']})")

    # 5. priority order ------------------------------------------------------------------------------------------
    L += ["", "## 5. Priority order differs from the learned-graph cell (contested scenario-seeds, seeds 42/123/456)", "",
          "| setting | cell | differs / n |", "|---|---|---|"]
    for setting in ("base", "prio"):
        cs = {k for k in contested(data, setting) if k[0] in SEEDS3}
        for c in ("R_neg_knn100", "R_neg_random0", "R_neg_none", "R_neg_full", "R_neg_corridor"):
            if (setting, c) not in data:
                continue
            ks = [k for k in cs if k in data[(setting, c)]]
            dif = sum(data[(setting, c)][k].get("priority_order") != data[(setting, "R_neg")][k].get("priority_order") for k in ks)
            out["priority"][f"{setting}/{c}"] = {"differs": dif, "n": len(ks)}
            L.append(f"| {setting} | {c} | {dif} / {len(ks)} |")

    # 7. other learned graphs and another forecaster (status M58, items B1-B3) ------------------------------------
    L += ["", "## 7. Other learned graphs and the static forecaster (status M58)", "",
          "alpha1 = Adaptive-LSTGAN trained with alpha = 1 (Graph WaveNet graph not usable, status M59); "
          "random = mean of five degree-preserving rewirings of that graph. static = zones and severities from the "
          "static LSTGAN forecasts, topology = Adaptive-LSTGAN learned graph of the same seed.", "",
          "| block | seeds | n | cell | final conflicts | messages | zone density | coverage | diff [95% CI] vs | Wilcoxon p |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    out["extra"] = {}
    STATIC_SEEDS = tuple(s for s in SEEDS5 if any(k[0] == s for k in data.get(("static", "R_neg"), {})))
    blocks = [("alpha1", "base", SEEDS5, [("R_neg_alpha1", "R_neg"), ("R_neg_alpha1_random_mean", "R_neg_alpha1"),
                                          ("R_neg_knn100", "R_neg_alpha1")]),
              ("static", "static", STATIC_SEEDS, [("R_neg_random_mean", "R_neg"), ("R_neg_knn100", "R_neg"),
                                                          ("R_neg_knn80", "R_neg"), ("R_neg_corridor", "R_neg"),
                                                          ("R_central", "R_neg"), ("R_none", "R_neg")])]
    for name, setting, seeds, comps in blocks:
        cs = {k for k in contested(data, setting) if k[0] in seeds}
        need = {c for pair in comps for c in pair}
        if not cs or not all((setting, c.replace("_random_mean", "_random0")) in data for c in need):
            L.append(f"| {name} | – | – | (records missing) | | | | | | |")
            continue
        blk = {"seeds": list(seeds), "n_contested_scenario_seeds": len(cs), "cells": {}, "comparisons": []}
        for c in sorted(need):
            v = values(data, setting, c, "n_conflicts_final", seeds, cs)
            e = {"final_conflicts": float(np.mean(list(v.values()))), "n_scenarios": len(v)}
            for m in ("messages", "link_density", "coverage", "wzpa30", "wzpa60"):
                x = values(data, setting, c, m, seeds, cs)
                if x:
                    e[m] = float(np.mean(list(x.values())))
            blk["cells"][c] = e
        for c, ref in comps:
            pr = paired(values(data, setting, ref, "n_conflicts_final", seeds, cs),
                        values(data, setting, c, "n_conflicts_final", seeds, cs), rng)
            blk["comparisons"].append({"cell": c, "ref": ref, **(pr or {})})
            e = blk["cells"][c]
            f = lambda k: f"{e[k]:.3f}" if k in e else "–"
            L.append(f"| {name} | {'/'.join(map(str, seeds))} | {len(cs)} | {c} | {e['final_conflicts']:.3f} | {f('messages')} | "
                     f"{f('link_density')} | {f('coverage')} | "
                     + (f"{pr['mean_diff']:+.3f} [{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}] vs {ref} | {pr['p']:.2g} |" if pr else "– | – |"))
        if name == "static":   # severity accuracy of the static forecaster vs the adaptive one on all scenarios with zones
            for st, lab in (("static", "static"), ("base", "adaptive")):
                allk = {k for k in data.get((st, "R_none"), {}) if k[0] in seeds}
                for m in ("wzpa30", "wzpa60"):
                    x = values(data, st, "R_none", m, seeds, allk)
                    blk[f"{lab}_{m}_all_zones"] = float(np.mean(list(x.values()))) if x else None
            L.append(f"\nSeverity accuracy (R_none, all scenarios with zones, seeds {seeds}): static WZPA@30/60 "
                     f"{blk['static_wzpa30_all_zones']:.3f}/{blk['static_wzpa60_all_zones']:.3f}, adaptive "
                     f"{blk['adaptive_wzpa30_all_zones']:.3f}/{blk['adaptive_wzpa60_all_zones']:.3f}")
        out["extra"][name] = blk

    # 6. graphs ----------------------------------------------------------------------------------------------------
    out["graphs"] = graph_stats(SEEDS3)
    out["distance_full_zone_density"] = zone_density_distance_full(SEEDS3)
    L += ["", "## 6. Graph statistics (sensor level; seeds 42/123/456)", "",
          "| seed | graph | edges | edges/sensor | overlap with learned | median edge km | 90% edge km | share > 10 km |",
          "|---|---|---|---|---|---|---|---|"]
    for seed, row in out["graphs"].items():
        for name, g in row.items():
            if name.startswith("random"):
                continue
            f = lambda k, p=2: "–" if g[k] is None else f"{g[k]:.{p}f}"
            L.append(f"| {seed} | {name} | {g['edges']} | {g['edges_per_sensor']:.1f} | {f('overlap_with_learned', 3)} | "
                     f"{f('edge_km_median')} | {f('edge_km_p90')} | {f('share_edges_over_10km', 3)} |")
        rr = row["random_100draws"]
        L.append(f"| {seed} | random (100 draws, min/mean/max) | {rr['edges'][1]:.0f} | {rr['edges_per_sensor'][1]:.1f} | "
                 f"{rr['overlap_with_learned'][0]:.3f}/{rr['overlap_with_learned'][1]:.3f}/{rr['overlap_with_learned'][2]:.3f} | "
                 f"{rr['edge_km_median'][1]:.2f} | {rr['edge_km_p90'][1]:.2f} | {rr['share_edges_over_10km'][1]:.3f} |")
        r50 = row["random_50swaps_5draws"]
        L.append(f"| {seed} | random, 50 swaps/edge (5 draws, min–max) | | | {r50['overlap_with_learned'][0]:.3f}–"
                 f"{r50['overlap_with_learned'][1]:.3f} | {r50['edge_km_median'][0]:.2f}–{r50['edge_km_median'][1]:.2f} | | |")
    L += ["", "Zone-level link density of the full DCRNN distance graph (all edges), R_neg zones of rule_cells_s*: "
          + "; ".join(f"seed {s}: {v['all_scenarios']:.3f} (n={v['n_all']}), contested {v['contested']:.3f} (n={v['n_contested']})"
                      for s, v in out["distance_full_zone_density"].items())]

    json.dump(out, open(os.path.join(DIR, "summary.json"), "w"), indent=1)
    open(os.path.join(DIR, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
