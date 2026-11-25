"""Single source of truth for MASTRA constants (task P1.3).

Every threshold, GMP constant, model ID and run setting lives here. Modules must import from this file
instead of hard-coding values. Values marked PENDING need a decision from the authors before the main runs
(tracked in papers/MASTRA/MASTRA_Status.md).
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- Data -------------------------------------------------------------------------------------------
TEST_PKL = os.path.join(ROOT, "assets", "test_5min.pkl")            # PeMS-BAY test split, 10,424 x 325
STATIONS_CSV = os.path.join(ROOT, "assets", "traffic_stations.csv")  # road_name, direction, lat/lng
DISTANCE_ADJ_PKL = os.path.join(ROOT, "assets", "adj_mx_bay.pkl")   # DCRNN distance-kernel graph
LEARNED_ADJ_DIR = os.path.join(ROOT, "assets", "learned_adj")        # written by papers/MASTRA/analysis/extract_learned_adj.py
FORECAST_FILES = {  # Adaptive-LSTGAN test-set forecasts, shape (8397, 12, 325), mph
    42: (os.path.join(ROOT, "papers", "Adaptive_LSTGAN_Paper", "analysis", "paper_inference_data.npz"), "adaptive"),
    123: (os.path.join(ROOT, "papers", "Adaptive_LSTGAN_Paper", "analysis", "global_eval", "pred_pems_adaptive_s123.npy"), None),
    456: (os.path.join(ROOT, "papers", "Adaptive_LSTGAN_Paper", "analysis", "global_eval", "pred_pems_adaptive_s456.npy"), None),
    789: (os.path.join(ROOT, "papers", "Adaptive_LSTGAN_Paper", "analysis", "global_eval", "pred_pems_adaptive_s789.npy"), None),
    1024: (os.path.join(ROOT, "papers", "Adaptive_LSTGAN_Paper", "analysis", "global_eval", "pred_pems_adaptive_s1024.npy"), None),
}
# Static LSTGAN (no learned graph) test-set forecasts from the companion study (status M58, item B3). Seed 123 was
# not saved by the companion study; it is regenerated on Kaggle (Kaggle_MASTRA_F2/check_v3) from the epoch-53
# checkpoint Result_paper_1/static_seed_123/static_seed123.pt (not the epoch-11 file in Results/, status M63).
# Used only to check that the topology results do not depend on which forecaster defines the zones.
STATIC_FORECAST_FILES = {
    42: (os.path.join(ROOT, "papers", "Adaptive_LSTGAN_Paper", "analysis", "paper_inference_data.npz"), "static"),
    123: (os.path.join(ROOT, "papers", "Adaptive_LSTGAN_Paper", "analysis", "global_eval", "pred_pems_static_s123.npy"), None),
    456: (os.path.join(ROOT, "papers", "Adaptive_LSTGAN_Paper", "analysis", "global_eval", "pred_pems_static_s456.npy"), None),
    789: (os.path.join(ROOT, "papers", "Adaptive_LSTGAN_Paper", "analysis", "global_eval", "pred_pems_static_s789.npy"), None),
    1024: (os.path.join(ROOT, "papers", "Adaptive_LSTGAN_Paper", "analysis", "global_eval", "pred_pems_static_s1024.npy"), None),
}
WEEKLY_WINDOW = 2016   # forecast sample i predicts test steps [2016+i, 2016+i+12) (TrafficDataset in the training notebook)
HORIZON = 12           # 12 x 5 min = 60 min
HIST_WINDOW = 12       # observed hour before the forecast origin

# --- Severity scale (one scale everywhere: detector, prompts, gates, metrics) ------------------------
# Q1 decided 2026-10-17 (author accepted default): HIGH < 35, MEDIUM 35-50, LOW >= 50 mph.
SEV_HIGH_BELOW = 35.0
SEV_MEDIUM_BELOW = 50.0
SEVERITY_LEVELS = ("LOW", "MEDIUM", "HIGH")
SEV_SCORES = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.2}


def speed_to_severity(speed_mph: float) -> str:
    if speed_mph < SEV_HIGH_BELOW:
        return "HIGH"
    if speed_mph < SEV_MEDIUM_BELOW:
        return "MEDIUM"
    return "LOW"


# --- Zone detection ---------------------------------------------------------------------------------
ZONE_SPEED_THRESHOLD = SEV_MEDIUM_BELOW   # sensors whose mean forecast speed is below this form zones
DBSCAN_EPS_DEG = 0.015                    # lat/lng Euclidean (as implemented; the paper text must say so)
DBSCAN_MIN_SAMPLES = 2
BFS_HOPS = 2
BFS_ADJ_THRESHOLD = 0.3                   # on the distance graph
TREND_DELTA_MPH = 2.0                     # last-3 minus first-3 forecast steps
SENSOR_FAULT_SPEED = 0.0                  # PeMS-BAY encodes missing readings as 0

# --- Graph message passing --------------------------------------------------------------------------
GMP_EDGE_THRESHOLD = 0.0      # w_ij > threshold to pass a message (0 = any edge of the chosen topology)
GMP_LAMBDA = 0.3              # pressure scale (called alpha in the old paper text; renamed to avoid clash)
GMP_WEIGHT_AGG = "mean"       # "mean" or "max" over sensor pairs (ablation P2.9)
GMP_PRESSURE = "normalized"   # "normalized" or "additive" (ablation P2.9)

# --- Coordination / negotiation ------------------------------------------------------------------------
# Q9 decided 2026-10-17 (default): candidate reroute corridors = other freeways with a sensor within this radius of the zone
# centre (5 km gives >=1 candidate for 95% of zones, median 4, on the 120-scenario set; measured 2026-10-09).
REROUTE_RADIUS_KM = 5.0
NEGOTIATION_ROUNDS = 3          # Q10 decided 2026-10-17 (default): maximum revise rounds after the initial proposal;
COORDINATOR_FALLBACK = True     # Q10 (default): central coordinator runs only if conflicts remain after the rounds
# Q11 default 2026-10-17: 0, to be checked by measuring how many zone pairs it links before N10 (plan §11.5)
NEIGHBOR_THRESHOLD = 0.0        # zones i, j exchange proposals when mean A over their sensor pairs (either direction) > this
# R12: an edge enters the seed-consensus graph if it is in >= this many of the 5 seeds. Measured 2026-10-17 on the
# top-k learned graphs: m=1 14,500 edges, m=2 1,525 (323 sensors with edges), m=3 117 (96), m=4 1, m=5 0.
# m=3 (first proposed) is too sparse to act as a topology; m=2 decided 2026-10-17 (Q12), m=3 kept as an extra cell.
CONSENSUS_MIN_SEEDS = 2

# --- Safety gates -----------------------------------------------------------------------------------
GATE_SEVERE_OVERRIDE_MPH = 15.0
GATE_CASCADE_BLOCK_RISK = 0.85
RUSH_HOURS = ((7, 9), (16, 19))   # inclusive hour ranges, Mon-Fri
MAX_RETRIES = 2

# --- LLM --------------------------------------------------------------------------------------------
# PENDING (Q2): provider and backbones. IDs checked on console.groq.com/docs/models on 2026-10-08.
LLM_PROVIDER = os.environ.get("MASTRA_LLM_PROVIDER", "groq")
LLM_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",       # console.groq.com/docs/openai (2026-10-08)
    "local": os.environ.get("MASTRA_LOCAL_BASE_URL", "http://localhost:8000/v1"),
}
LLM_API_KEY_ENV = {"groq": "GROQ_API_KEY", "local": "MASTRA_LOCAL_API_KEY"}
WORKER_MODEL = os.environ.get("MASTRA_WORKER_MODEL", "openai/gpt-oss-120b")
SUPERVISOR_MODEL = os.environ.get("MASTRA_SUPERVISOR_MODEL", "openai/gpt-oss-120b")
DQ_MODEL = os.environ.get("MASTRA_DQ_MODEL", "openai/gpt-oss-20b")
WORKER_TEMPERATURE = 0.2
SUPERVISOR_TEMPERATURE = 0.2
DQ_TEMPERATURE = 0.0
# gpt-oss models spend completion tokens on reasoning; 400/1200 caused "max completion tokens reached" (M22)
MAX_TOKENS = {"worker": 4000, "supervisor": 8000, "dq": 2000}
REQUEST_TIMEOUT_S = 60
RATE_LIMIT_SLEEP_S = 0.0

# --- Runs -------------------------------------------------------------------------------------------
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
DEFAULT_FORECAST_SEED = 42
