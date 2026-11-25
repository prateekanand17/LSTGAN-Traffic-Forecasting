"""Real inputs for MASTRA (tasks P1.1/P1.2/M13).

- Forecasts: Adaptive-LSTGAN test-set predictions (not the observed history used by the old runners).
- Timestamps: taken from the test DataFrame index (the old runners assumed the split starts at midnight Monday).
- Topologies for agent communication: learned (per forecaster seed), distance, geographic kNN, fully connected,
  degree-preserving random rewiring of the learned graph, and none - all at matched edge density where possible.
"""
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline import config as C
from pipeline.inputs import PipelineInput


@lru_cache(maxsize=1)
def load_test() -> pd.DataFrame:
    return pd.read_pickle(C.TEST_PKL)


@lru_cache(maxsize=1)
def load_stations() -> pd.DataFrame:
    test = load_test()
    meta = pd.read_csv(C.STATIONS_CSV)
    meta["station_id"] = meta["station_id"].astype(str)
    meta = meta.set_index("station_id").loc[[str(c) for c in test.columns]].reset_index()
    return meta


@lru_cache(maxsize=1)
def load_distance_adj() -> np.ndarray:
    with open(C.DISTANCE_ADJ_PKL, "rb") as f:
        ids, _, adj = pickle.load(f, encoding="latin1")
    test_ids = [str(c) for c in load_test().columns]
    if [str(i) for i in ids] != test_ids:
        raise ValueError("adj_mx_bay.pkl sensor order differs from the test DataFrame columns")
    return adj.astype(np.float32)


@lru_cache(maxsize=8)
def load_forecasts(seed: int, forecaster: str = "adaptive") -> np.ndarray:
    """forecaster: 'adaptive' (Adaptive-LSTGAN, main) or 'static' (static LSTGAN, check only).

    Falls back to assets/forecasts_120scenarios.npz (a compact 120-scenario slice used by Streamlit
    Cloud where the full 375 MB NPZ is not available).  build_input() accounts for the index mapping.
    """
    if forecaster == "adaptive":
        path, key = C.FORECAST_FILES[seed]
        slim = os.path.join(os.path.dirname(path), "..", "..", "assets", "forecasts_120scenarios.npz")
        slim = os.path.normpath(slim)
        if not os.path.exists(path) and os.path.exists(slim):
            arr = np.load(slim)[f"adaptive_s{seed}"]
            return arr.astype(np.float32)
        path_use, key_use = path, key
    else:
        path_use, key_use = C.STATIC_FORECAST_FILES[seed]
    arr = np.load(path_use)[key_use] if key_use else np.load(path_use)
    if arr.shape[1:] != (C.HORIZON, load_test().shape[1]):
        raise ValueError(f"unexpected forecast shape {arr.shape} in {path_use}")
    return arr.astype(np.float32)


def learned_adj(seed: int, form: str = "topk10", name: str = "") -> np.ndarray:
    """name '' = Adaptive-LSTGAN (main); 'alpha1' = extra graph (papers/MASTRA/analysis/extract_extra_graphs.py)."""
    p = os.path.join(C.LEARNED_ADJ_DIR, f"pems_bay_{name + '_' if name else ''}seed{seed}_{form}.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} - run papers/MASTRA/analysis/extract_learned_adj.py first")
    return np.load(p)


# --- topologies ----------------------------------------------------------------------------------------
def _knn_geo(k: int) -> np.ndarray:
    st = load_stations()
    xy = np.c_[st["latitude"].values, st["longitude"].values * np.cos(np.deg2rad(37.5))]
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    A = np.zeros_like(d, dtype=np.float32)
    idx = np.argsort(d, axis=1)[:, :k]
    rows = np.repeat(np.arange(len(d)), k)
    A[rows, idx.ravel()] = 1.0 / k
    return A


def _distance_topk(k: int) -> np.ndarray:
    D = load_distance_adj().copy()
    np.fill_diagonal(D, 0)
    A = np.zeros_like(D)
    for i in range(len(D)):
        nz = np.nonzero(D[i])[0]
        keep = nz[np.argsort(-D[i, nz])[:k]]
        if len(keep):
            A[i, keep] = D[i, keep] / D[i, keep].sum()
    return A


def _rewire(A: np.ndarray, rng: np.random.Generator, n_swaps_per_edge: int = 10) -> np.ndarray:
    """Degree-preserving random rewiring of a directed weighted graph (edge swaps keep in/out degrees)."""
    rows, cols = np.nonzero(A)
    w = A[rows, cols].copy()
    edges = list(zip(rows.tolist(), cols.tolist()))
    present = set(edges)
    m = len(edges)
    for _ in range(n_swaps_per_edge * m):
        a, b = rng.integers(0, m, size=2)
        (u1, v1), (u2, v2) = edges[a], edges[b]
        if u1 == v2 or u2 == v1 or (u1, v2) in present or (u2, v1) in present:
            continue
        present -= {(u1, v1), (u2, v2)}
        present |= {(u1, v2), (u2, v1)}
        edges[a], edges[b] = (u1, v2), (u2, v1)
    R = np.zeros_like(A)
    for (u, v), wt in zip(edges, w):
        R[u, v] = wt
    rs = R.sum(1, keepdims=True)
    return np.divide(R, rs, out=np.zeros_like(R), where=rs > 0)


def consensus_adj(min_seeds: int = C.CONSENSUS_MIN_SEEDS, seeds=tuple(C.FORECAST_FILES)) -> np.ndarray:
    """Seed-consensus graph (task R12): keep a directed edge of the top-k learned graph if it appears in at least
    `min_seeds` of the forecaster seeds; weight = mean weight over the seeds that contain it; rows renormalized.
    Learned graphs differ strongly between seeds (finding M20), so this keeps only edges the learner finds repeatedly."""
    mats = [learned_adj(s, "topk10") for s in seeds]
    present = np.stack([m > 0 for m in mats])
    count = present.sum(0)
    w = np.stack(mats).sum(0) / np.maximum(count, 1)
    A = np.where(count >= min_seeds, w, 0.0).astype(np.float32)
    np.fill_diagonal(A, 0)
    rs = A.sum(1, keepdims=True)
    return np.divide(A, rs, out=np.zeros_like(A), where=rs > 0)


def topology(name: str, seed: int = C.DEFAULT_FORECAST_SEED, rand_seed: int = 0, k: int = 10) -> np.ndarray:
    """Communication graph between sensors used to weight inter-zone messages.

    learned   : Adaptive-LSTGAN learned adjacency (top-k=10, the graph the model propagates over)
    learned_dense : same learner without the top-k mask
    distance  : DCRNN distance-kernel graph, top-k=10 strongest edges per row (matched density)
    distance_full : DCRNN distance-kernel graph as published (all edges)
    knn       : k nearest stations by great-circle-approximated distance (matched density)
    knn:K     : same with K neighbours (K=100 matches the learned graph's zone-level link density, M36)
    full      : fully connected, uniform weights (broadcast-all)
    random    : degree-preserving rewiring of the learned graph (rand_seed selects the draw)
    alpha1    : learned graph of Adaptive-LSTGAN trained with alpha = 1 (learned graph only; M58, B2)
    random_alpha1 : degree-preserving rewiring of that graph
    consensus : edges of the learned top-k graph present in >= CONSENSUS_MIN_SEEDS forecaster seeds (R12)
    random_consensus : degree-preserving rewiring of the consensus graph (density-matched control for it)
    none      : no communication
    """
    n = load_test().shape[1]
    if name == "learned":
        return learned_adj(seed, "topk10")
    if name == "learned_dense":
        return learned_adj(seed, "dense")
    if name == "distance":
        return _distance_topk(k)
    if name == "distance_full":
        D = load_distance_adj().copy()
        np.fill_diagonal(D, 0)
        return D
    if name == "knn":
        return _knn_geo(k)
    if name.startswith("knn:"):
        return _knn_geo(int(name.split(":")[1]))
    if name == "full":
        A = np.full((n, n), 1.0 / (n - 1), dtype=np.float32)
        np.fill_diagonal(A, 0)
        return A
    if name == "alpha1":
        return learned_adj(seed, name=name)
    if name == "random_alpha1":
        return _rewire(learned_adj(seed, name=name.split("_")[1]), np.random.default_rng(rand_seed))
    if name == "random":
        return _rewire(learned_adj(seed, "topk10"), np.random.default_rng(rand_seed))
    if name == "consensus":
        return consensus_adj()
    if name.startswith("consensus:"):
        return consensus_adj(int(name.split(":")[1]))
    if name == "random_consensus":
        return _rewire(consensus_adj(), np.random.default_rng(rand_seed))
    if name == "none":
        return np.zeros((n, n), dtype=np.float32)
    raise ValueError(f"unknown topology {name}")


# --- scenarios -----------------------------------------------------------------------------------------
@dataclass
class Scenario:
    sample_idx: int          # index into the forecast arrays
    t0: int                  # test-set row of the first forecast step
    timestamp: str           # wall-clock time of the forecast origin (last observed step)
    regime: str              # HIGH / MODERATE / CASCADE / NORMAL


def n_samples() -> int:
    return load_test().shape[0] - C.WEEKLY_WINDOW - C.HORIZON + 1


def build_input(sample_idx: int, forecast_seed: int = C.DEFAULT_FORECAST_SEED,
                mode: str = "forecast", comm_adj: Optional[np.ndarray] = None,
                forecaster: str = "adaptive") -> PipelineInput:
    """mode: 'forecast' (Adaptive-LSTGAN), 'reactive' (persist the last observed speed), 'oracle' (ground truth)."""
    test = load_test()
    v = test.values.astype(np.float32)
    t0 = C.WEEKLY_WINDOW + sample_idx
    actual = v[t0:t0 + C.HORIZON].copy()          # copies, so an input does not keep the whole test array alive
    observed = v[t0 - C.HIST_WINDOW:t0].copy()
    if mode == "forecast":
        forecasts = load_forecasts(forecast_seed, forecaster)
        # When using the slim 120-scenario NPZ the array is indexed by position, not sample_idx.
        idx_map_path = os.path.join(os.path.dirname(C.FORECAST_FILES[forecast_seed][0]),
                                    "..", "..", "assets", "forecast_index_map.json")
        idx_map_path = os.path.normpath(idx_map_path)
        if forecasts.shape[0] == 120 and os.path.exists(idx_map_path):
            import json as _json
            idx_map = _json.load(open(idx_map_path))
            pos = idx_map.get(str(sample_idx))
            if pos is None:
                raise KeyError(f"sample_idx {sample_idx} not in slim forecast index map")
            pred = forecasts[int(pos)]
        else:
            pred = forecasts[sample_idx]
    elif mode == "reactive":
        pred = np.tile(observed[-1:], (C.HORIZON, 1))
    elif mode == "oracle":
        pred = actual.copy()
    else:
        raise ValueError(mode)
    hist = v[max(0, t0 - 3 * C.HIST_WINDOW):t0]
    hist_m = np.where(hist > 0, hist, np.nan)
    means = np.nan_to_num(np.nanmean(hist_m, 0), nan=float(np.nanmean(hist_m)))
    stds = np.nan_to_num(np.nanstd(hist_m, 0), nan=1.0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    origin = test.index[t0 - 1]
    st = load_stations()
    inp = PipelineInput(
        pred_speed=pred, actual_speed=actual,
        adj_mx=load_distance_adj(),
        station_ids=[str(c) for c in test.columns],
        lats=st["latitude"].values.astype(np.float32), lngs=st["longitude"].values.astype(np.float32),
        hist_means=means.astype(np.float32), hist_stds=stds.astype(np.float32),
        day_of_week=int(origin.weekday()), hour=int(origin.hour), timestamp_idx=int(t0),
    )
    inp.comm_adj = comm_adj if comm_adj is not None else learned_adj(forecast_seed)
    inp.observed_speed = observed
    inp.timestamp = str(origin)
    inp.roads = (st["road_name"].astype(str) + "-" + st["direction"].astype(str)).tolist()
    return inp


def select_scenarios(n_per_regime: Dict[str, int], stride: int = 6, seed: int = 0) -> List[Scenario]:
    """Stratified, reproducible scenario sample over the usable test window.

    Regimes use OBSERVED speed at the forecast origin (no forecast or future information):
      HIGH     : > 15 sensors below SEV_HIGH_BELOW
      MODERATE : 5-15 sensors below SEV_HIGH_BELOW
      CASCADE  : >= 5 sensors at/above SEV_HIGH_BELOW now that fall below it within 30 min (uses ground truth
                 only to label the regime, never as model input)
      NORMAL   : < 3 sensors below SEV_HIGH_BELOW
    Candidates are taken every `stride` samples; within each regime a random subset is drawn with `seed`.
    """
    v = load_test().values.astype(np.float32)
    test = load_test()
    pools = {"HIGH": [], "MODERATE": [], "CASCADE": [], "NORMAL": []}
    for i in range(0, n_samples(), stride):
        t0 = C.WEEKLY_WINDOW + i
        now = v[t0 - 1]
        valid = now > C.SENSOR_FAULT_SPEED
        low_now = (now < C.SEV_HIGH_BELOW) & valid
        n_low = int(low_now.sum())
        fut = v[min(t0 + 5, len(v) - 1)]
        new_low = int(((now >= C.SEV_HIGH_BELOW) & (fut < C.SEV_HIGH_BELOW) & (fut > 0)).sum())
        if n_low > 15:
            pools["HIGH"].append(i)
        elif 5 <= n_low <= 15:
            pools["MODERATE"].append(i)
        elif new_low >= 5:
            pools["CASCADE"].append(i)
        elif n_low < 3:
            pools["NORMAL"].append(i)
    rng = np.random.default_rng(seed)
    out = []
    for regime, n in n_per_regime.items():
        pool = pools[regime]
        if len(pool) < n:
            raise ValueError(f"regime {regime}: only {len(pool)} candidates for {n} requested")
        for i in sorted(rng.choice(pool, size=n, replace=False).tolist()):
            out.append(Scenario(i, C.WEEKLY_WINDOW + i, str(test.index[C.WEEKLY_WINDOW + i - 1]), regime))
    return out
