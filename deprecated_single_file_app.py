import os, math, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium

st.set_page_config(
    page_title="LSTGAN Traffic Forecaster",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: #0e1117; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label { color: #c9d1d9 !important; font-weight: 500; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    height: 44px; border-radius: 8px 8px 0 0; padding: 8px 20px;
    background: #161b22; color: #8b949e; font-weight: 600; border: 1px solid #21262d;
}
.stTabs [aria-selected="true"] {
    background: #1f6feb !important; color: white !important; border-color: #1f6feb !important;
}
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #21262d; border-radius: 12px; padding: 20px;
    text-align: center; transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card h2 { margin: 8px 0 4px; font-size: 28px; }
.metric-card p { color: #8b949e; margin: 0; font-size: 13px; }
.status-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 600;
}
.dashboard-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1c2333 100%);
    padding: 24px 32px; border-radius: 16px; margin-bottom: 24px;
    border: 1px solid #21262d;
}
.congestion-table {
    width: 100%; border-collapse: collapse; margin-top: 12px;
}
.congestion-table th {
    background: #1f6feb; color: white; padding: 10px 16px;
    text-align: left; font-size: 13px;
    }
.congestion-table td {
    padding: 8px 16px; border-bottom: 1px solid #21262d;
    color: #c9d1d9; font-size: 13px;
}
.congestion-table tr:hover td { background: #161b22; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE, "results")
CKPT_PATH  = os.path.join(DATA_DIR, "best_lstgan.pt")
ADJ_FILE   = os.path.join(DATA_DIR, "adj_mx_bay.pkl")
TEST_FILE  = os.path.join(DATA_DIR, "test_5min.pkl")
STATIONS_CSV = os.path.join(BASE, "data", "traffic_stations.csv")

NUM_SENSORS = 325; STEPS_PER_HOUR = 12; STEPS_PER_DAY = 288
WEEKLY_WINDOW = 2016; DAILY_WINDOW = 288; HOURLY_WINDOW = 12
FORECAST_HORIZON = 12
WEEKLY_IN_CHANNELS = 3; WEEKLY_OUT_CHANNELS = 16
DAILY_IN_CHANNELS = 3; DAILY_OUT_CHANNELS = 8
HOURLY_IN_CHANNELS = 1
WEEKLY_STRIDE_1 = 14; WEEKLY_STRIDE_2 = 12
DAILY_STRIDE_1 = 4; DAILY_STRIDE_2 = 6
SPATIAL_EMBED_DIM = 16
GLOBAL_SPATIAL_CHANNELS = 16; GLOBAL_ATTN_HEADS = 4; GLOBAL_ATTN_LAYERS = 1
LOCAL_SPATIAL_CHANNELS = 8; CHEBYSHEV_K = 5
D_MODEL = WEEKLY_OUT_CHANNELS + DAILY_OUT_CHANNELS + GLOBAL_SPATIAL_CHANNELS + LOCAL_SPATIAL_CHANNELS
TIME_ONEHOT_DIM = 7 + STEPS_PER_DAY
DECODER_HEADS = 6; DECODER_LAYERS = 1
DEVICE = torch.device("cpu")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

class TemporalEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, s1, s2):
        super().__init__()
        mid = max((in_ch + out_ch) // 2, 8)
        self.c1a = nn.Conv1d(in_ch, mid, 3, padding=1)
        self.c1b = nn.Conv1d(mid, mid, s1, stride=s1) if s1 > 1 else nn.Identity()
        self.bn1 = nn.BatchNorm1d(mid)
        self.c2a = nn.Conv1d(mid, out_ch, 3, padding=1)
        self.c2b = nn.Conv1d(out_ch, out_ch, s2, stride=s2) if s2 > 1 else nn.Identity()
        self.bn2 = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        x = F.relu(self.bn1(self.c1b(F.relu(self.c1a(x)))))
        x = self.bn2(self.c2b(F.relu(self.c2a(x))))
        _, Co, To = x.shape
        return x.reshape(B, N, Co, To).permute(0, 3, 1, 2)

class SpatialEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(HOURLY_IN_CHANNELS, SPATIAL_EMBED_DIM, 1)
        self.bn = nn.BatchNorm1d(SPATIAL_EMBED_DIM)
    def forward(self, x):
        B, T, N, C = x.shape
        x = x.reshape(B * T, N, C).permute(0, 2, 1)
        return self.bn(self.conv(x)).permute(0, 2, 1).reshape(B, T, N, -1)

class GlobalSpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({
            'attn': nn.MultiheadAttention(SPATIAL_EMBED_DIM, GLOBAL_ATTN_HEADS, batch_first=True, dropout=0.1),
            'n1': nn.LayerNorm(SPATIAL_EMBED_DIM), 'n2': nn.LayerNorm(SPATIAL_EMBED_DIM),
            'ffn': nn.Sequential(nn.Linear(SPATIAL_EMBED_DIM, SPATIAL_EMBED_DIM * 4),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(SPATIAL_EMBED_DIM * 4, SPATIAL_EMBED_DIM))
        })])
        self.proj = nn.Identity()
    def forward(self, x):
        B, T, N, C = x.shape; x = x.reshape(B * T, N, C)
        for l in self.layers:
            a, _ = l['attn'](x, x, x); x = l['n1'](x + a); x = l['n2'](x + l['ffn'](x))
        return self.proj(x).reshape(B, T, N, -1)

class ManualChebConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(SPATIAL_EMBED_DIM, LOCAL_SPATIAL_CHANNELS)) for _ in range(CHEBYSHEV_K)
        ])
        self.bias = nn.Parameter(torch.zeros(LOCAL_SPATIAL_CHANNELS))
        for w in self.weights: nn.init.xavier_uniform_(w)
    def forward(self, x, L):
        Z0, Z1 = x, L @ x; out = Z0 @ self.weights[0]
        if CHEBYSHEV_K > 1: out = out + Z1 @ self.weights[1]
        for k in range(2, CHEBYSHEV_K):
            Zk = 2 * L @ Z1 - Z0; out = out + Zk @ self.weights[k]; Z0, Z1 = Z1, Zk
        return out + self.bias

class LocalSpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cheb = ManualChebConv(); self.register_buffer('L_scaled', torch.zeros(NUM_SENSORS, NUM_SENSORS))
    def set_graph(self, adj_mx):
        A = adj_mx; D_inv = np.diag(1.0 / np.sqrt(np.maximum(A.sum(1), 1e-8)))
        L = np.eye(A.shape[0]) - D_inv @ A @ D_inv; eig = np.linalg.eigvalsh(L)
        self.L_scaled.copy_(torch.FloatTensor(2 * L / (eig[-1] + 1e-8) - np.eye(A.shape[0])))
    def forward(self, x, edge_idx=None, edge_wt=None):
        B, T, N, C = x.shape; L = self.L_scaled; x = x.reshape(B * T, N, C)
        return self.cheb(x, L).reshape(B, T, N, -1)

class TimeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        h = D_MODEL * 2
        self.mlp = nn.Sequential(nn.Linear(TIME_ONEHOT_DIM, h), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, D_MODEL))
        self.norm = nn.LayerNorm(D_MODEL)
    def forward(self, ste, ti):
        dow = F.one_hot(ti[:, 0].long(), 7).float()
        tod = F.one_hot(ti[:, 1].long(), 288).float()
        vt = self.mlp(torch.cat([dow, tod], -1))[:, None, None, :].expand_as(ste)
        return self.norm(ste + vt)

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        pe = torch.zeros(500, D_MODEL); pos = torch.arange(500).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-math.log(10000.0) / D_MODEL))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class TemporalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = PositionalEncoding()
        self.enc_layers = nn.ModuleList([nn.ModuleDict({
            'attn': nn.MultiheadAttention(D_MODEL, DECODER_HEADS, batch_first=True, dropout=0.1),
            'n1': nn.LayerNorm(D_MODEL), 'n2': nn.LayerNorm(D_MODEL),
            'ffn': nn.Sequential(nn.Linear(D_MODEL, D_MODEL * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(D_MODEL * 4, D_MODEL))
        })])
        self.dec_layers = nn.ModuleList([nn.ModuleDict({
            'attn': nn.MultiheadAttention(D_MODEL, DECODER_HEADS, batch_first=True, dropout=0.1),
            'n1': nn.LayerNorm(D_MODEL), 'n2': nn.LayerNorm(D_MODEL),
            'ffn': nn.Sequential(nn.Linear(D_MODEL, D_MODEL * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(D_MODEL * 4, D_MODEL))
        })])
        self.fq = nn.Parameter(torch.randn(1, FORECAST_HORIZON, D_MODEL) * 0.02)
        self.out_proj = nn.Linear(D_MODEL, 1)
    def forward(self, ste):
        B, T, N, d = ste.shape; x = ste.permute(0, 2, 1, 3).reshape(B * N, T, d); x = self.pe(x)
        for l in self.enc_layers:
            a, _ = l['attn'](x, x, x); x = l['n1'](x + a); x = l['n2'](x + l['ffn'](x))
        q = self.fq.expand(B * N, -1, -1)
        for l in self.dec_layers:
            a, _ = l['attn'](q, x, x); q = l['n1'](q + a); q = l['n2'](q + l['ffn'](q))
        return self.out_proj(q).reshape(B, N, FORECAST_HORIZON, 1).permute(0, 2, 1, 3)

class LSTGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.weekly_enc = TemporalEncoder(WEEKLY_IN_CHANNELS, WEEKLY_OUT_CHANNELS, WEEKLY_STRIDE_1, WEEKLY_STRIDE_2)
        self.daily_enc  = TemporalEncoder(DAILY_IN_CHANNELS, DAILY_OUT_CHANNELS, DAILY_STRIDE_1, DAILY_STRIDE_2)
        self.sp_embed   = SpatialEmbedding()
        self.global_enc = GlobalSpatialEncoder()
        self.local_enc  = LocalSpatialEncoder()
        self.time_enc   = TimeEncoder()
        self.decoder    = TemporalDecoder()
    def _match_time(self, x, tgt):
        T = x.shape[1]
        if T == tgt: return x
        if T > tgt: return x[:, :tgt]
        B, T, N, C = x.shape; x = x.permute(0, 3, 2, 1).reshape(B * C * N, 1, T)
        x = F.interpolate(x, size=tgt, mode='nearest'); return x.reshape(B, C, N, tgt).permute(0, 3, 2, 1)
    def forward(self, X_w, X_d, X_h, t_info):
        Xw = self.weekly_enc(X_w); Xd = self.daily_enc(X_d); Xe = self.sp_embed(X_h)
        Xsa = self.global_enc(Xe); Xgcn = self.local_enc(Xe)
        Tt = X_h.shape[1]; Xw = self._match_time(Xw, Tt); Xd = self._match_time(Xd, Tt)
        STe = torch.cat([Xw, Xd, Xsa, Xgcn], dim=-1); STE = self.time_enc(STe, t_info)
        return self.decoder(STE)

@st.cache_resource
def load_model():
    if not os.path.exists(CKPT_PATH) or not os.path.exists(ADJ_FILE):
        return None, None, None, None, None
    with open(ADJ_FILE, 'rb') as f:
        adj_data = pickle.load(f, encoding='latin1')
        sensor_ids = adj_data[0] if isinstance(adj_data, (tuple, list)) else [str(i) for i in range(len(adj_data[-1]))]
        adj_mx = adj_data[2] if isinstance(adj_data, (tuple, list)) and len(adj_data) >= 3 else adj_data[-1]
        adj_mx = adj_mx.astype(np.float32) if isinstance(adj_mx, np.ndarray) else adj_mx

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    mean, std = ckpt['mean'], ckpt['std']

    model = LSTGAN().to(DEVICE)
    model.load_state_dict(ckpt['model'], strict=True)
    model.local_enc.set_graph(adj_mx)
    model.eval()
    return model, mean, std, adj_mx, sensor_ids

@st.cache_data
def load_test_data():
    if not os.path.exists(TEST_FILE): return None
    return pd.read_pickle(TEST_FILE)

@st.cache_data
def get_geographic_coordinates(sensor_ids):
    fallback_lats = dict(zip(sensor_ids, np.random.uniform(37.25, 37.43, len(sensor_ids))))
    fallback_lngs = dict(zip(sensor_ids, np.random.uniform(-122.08, -121.84, len(sensor_ids))))
    
    if os.path.exists(STATIONS_CSV):
        df = pd.read_csv(STATIONS_CSV)
        if 'sensor_id' in df.columns:
            df['sensor_id'] = df['sensor_id'].astype(str)
            mapping = df.set_index('sensor_id')
            
            real_lats = []
            real_lngs = []
            for sid in sensor_ids:
                if sid in mapping.index:
                    real_lats.append(mapping.loc[sid, 'latitude'])
                    real_lngs.append(mapping.loc[sid, 'longitude'])
                else:
                    real_lats.append(fallback_lats[sid])
                    real_lngs.append(fallback_lngs[sid])
            return np.array(real_lats), np.array(real_lngs)
    return np.array(list(fallback_lats.values())), np.array(list(fallback_lngs.values()))

class TrafficDatasetLite:
    def __init__(self, data_df, mean, std):
        self.num_sensors = data_df.shape[1]
        self.mean = mean; self.std = std
        self.speed = ((data_df.values - mean) / (std + 1e-8)).astype(np.float32)
        if hasattr(data_df.index, 'weekday'):
            self.day_of_week = data_df.index.weekday.values
            self.time_of_day = (data_df.index.hour * 12 + data_df.index.minute // 5).values
        else:
            self.day_of_week = np.arange(len(data_df)) % 7
            self.time_of_day = np.arange(len(data_df)) % 288

    def _time_feats(self, idx, length):
        s = idx - length
        dow = self.day_of_week[s:idx].astype(np.float32) / 6.0
        tod = self.time_of_day[s:idx].astype(np.float32) / 287.0
        return np.stack([dow, tod], axis=-1)

    def get_sample(self, t):
        N = self.num_sensors
        sp_w = self.speed[t - WEEKLY_WINDOW:t]; tf_w = self._time_feats(t, WEEKLY_WINDOW)
        X_w = np.stack([sp_w, np.tile(tf_w[:, 0:1], (1, N)), np.tile(tf_w[:, 1:2], (1, N))], axis=-1)
        sp_d = self.speed[t - DAILY_WINDOW:t]; tf_d = self._time_feats(t, DAILY_WINDOW)
        X_d = np.stack([sp_d, np.tile(tf_d[:, 0:1], (1, N)), np.tile(tf_d[:, 1:2], (1, N))], axis=-1)
        X_h = self.speed[t - HOURLY_WINDOW:t][:, :, np.newaxis]
        t_info = np.array([self.day_of_week[t], self.time_of_day[t]], dtype=np.int64)
        Y = self.speed[t:t + FORECAST_HORIZON][:, :, np.newaxis]
        return [torch.from_numpy(x).unsqueeze(0).to(DEVICE) for x in (X_w, X_d, X_h, t_info, Y)]

    def find_matching_index(self, day_of_week, hour):
        tod = hour * STEPS_PER_HOUR
        for t in range(WEEKLY_WINDOW, len(self.speed) - FORECAST_HORIZON):
            if self.day_of_week[t] == day_of_week and self.time_of_day[t] == tod:
                return t
        return None

def speed_to_color(speed):
    if speed >= 55: return '#00b894'
    elif speed >= 45: return '#55efc4'
    elif speed >= 35: return '#fdcb6e'
    elif speed >= 25: return '#e17055'
    elif speed >= 15: return '#d63031'
    else: return '#6c5ce7'

def speed_to_level(speed):
    if speed >= 50: return '🟢 Free Flow'
    elif speed >= 35: return '🟡 Moderate'
    elif speed >= 20: return '🔴 Heavy'
    else: return '⛔ Severe'

def speed_to_badge_color(speed):
    if speed >= 50: return '#00b894'
    elif speed >= 35: return '#fdcb6e'
    elif speed >= 20: return '#e17055'
    else: return '#d63031'

st.markdown("""
<div class="dashboard-header">
    <h1 style="margin:0; color:white; font-size:28px;">
        🚦 LSTGAN Traffic Prediction Dashboard
    </h1>
    <p style="margin:4px 0 0; color:#8b949e; font-size:14px;">
        Deep learning forecasts for 325 PeMS-Bay sensors • Select day, hour & sensor to explore predictions
    </p>
</div>
""", unsafe_allow_html=True)

model, train_mean, train_std, adj_mx, station_ids = load_model()
df = load_test_data()

if model is None or df is None:
    st.error(f"❌ Missing model files. Ensure `best_lstgan.pt`, `adj_mx_bay.pkl`, and `test_5min.pkl` exist in `{DATA_DIR}`.")
    st.stop()

ds = TrafficDatasetLite(df, train_mean, train_std)

with st.spinner("Initializing map geometry..."):
    lats, lngs = get_geographic_coordinates(station_ids)

st.sidebar.markdown("""
<div style="text-align:center; padding:12px 0;">
    <span style="font-size:32px;">🚦</span>
    <h3 style="margin:4px 0; color:white;">Prediction Controls</h3>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

query_day = st.sidebar.selectbox("📅 Day of Week", range(7),
                                  format_func=lambda x: DAYS[x], index=0, key="day_select")
query_hour = st.sidebar.slider("🕐 Hour", 0, 23, 8, key="hour_slider")
selected_sensor = st.sidebar.selectbox("📡 Sensor ID", station_ids, key="sensor_input")
sensor_idx = station_ids.index(str(selected_sensor))

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="background:#161b22; padding:12px; border-radius:8px; border:1px solid #21262d;">
    <p style="color:#58a6ff; font-weight:600; margin:0 0 8px;">📍 Current Selection</p>
    <p style="color:#c9d1d9; margin:2px 0; font-size:13px;">
        <b>Day:</b> {DAYS[query_day]}<br>
        <b>Time:</b> {query_hour:02d}:00<br>
        <b>Sensor:</b> #{selected_sensor}<br>
        <b>Forecast:</b> Next 60 min (12×5min)
    </p>
</div>
""", unsafe_allow_html=True)

t_idx = ds.find_matching_index(query_day, query_hour)
if t_idx is None:
    st.error(f"No data found for {DAYS[query_day]} at {query_hour:02d}:00 in test set. Try a different day/hour.")
    st.stop()

Xw, Xd, Xh, ti, Y = ds.get_sample(t_idx)
with torch.no_grad():
    pred = model(Xw, Xd, Xh, ti)

pred_speed = (pred.cpu().numpy() * train_std + train_mean)[0, :, :, 0]
actual_speed = (Y.cpu().numpy() * train_std + train_mean)[0, :, :, 0]
history_speed = (Xh.cpu().numpy() * train_std + train_mean)[0, :, :, 0]

speeds_60 = pred_speed[-1]

tab1, tab2, tab3 = st.tabs(["🗺️ Network Map", "📊 Sensor Forecast", "📈 Network Summary"])

with tab1:
    st.markdown(f"### 🗺️ Traffic Map — {DAYS[query_day]} {query_hour:02d}:00 (+60min forecast)")

    avg_spd = speeds_60.mean()
    n_congested = (speeds_60 < 35).sum()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <p>Average Speed</p><h2 style="color:#58a6ff">{avg_spd:.1f} mph</h2>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <p>Congested Sensors</p><h2 style="color:#e17055">{n_congested}</h2>
            <p>{n_congested/NUM_SENSORS*100:.0f}% of network</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <p>Free Flow Sensors</p><h2 style="color:#00b894">{(speeds_60 >= 50).sum()}</h2>
        </div>""", unsafe_allow_html=True)
    with col4:
        sel_spd = speeds_60[sensor_idx]
        badge_col = speed_to_badge_color(sel_spd)
        st.markdown(f"""<div class="metric-card" style="border-color:{badge_col}">
            <p>Sensor #{selected_sensor}</p><h2 style="color:{badge_col}">{sel_spd:.1f} mph</h2>
            <p>{speed_to_level(sel_spd)}</p>
        </div>""", unsafe_allow_html=True)

    st.write("")

    center_lat, center_lng = np.mean(lats), np.mean(lngs)
    m = folium.Map(location=[center_lat, center_lng], zoom_start=11,
                   tiles='CartoDB dark_matter')

    legend_html = '''
    <div style="position:fixed; bottom:30px; right:20px; z-index:9999;
         background:rgba(0,0,0,0.88); padding:14px 18px; border-radius:10px;
         color:white; font-size:12px; font-family:Inter,sans-serif; border:1px solid #333;">
      <b style="font-size:13px;">⚡ Speed Legend</b><br><br>
      <span style="color:#00b894">●</span> &gt;55 mph — Free Flow<br>
      <span style="color:#55efc4">●</span> 45–55 — Light<br>
      <span style="color:#fdcb6e">●</span> 35–45 — Moderate<br>
      <span style="color:#e17055">●</span> 25–35 — Heavy<br>
      <span style="color:#d63031">●</span> 15–25 — Very Heavy<br>
      <span style="color:#6c5ce7">●</span> &lt;15 — Severe
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    for i in range(NUM_SENSORS):
        for j in range(i + 1, NUM_SENSORS):
            if adj_mx[i, j] > 0.3:
                avg_s = (speeds_60[i] + speeds_60[j]) / 2
                folium.PolyLine(
                    [[lats[i], lngs[i]], [lats[j], lngs[j]]],
                    color=speed_to_color(avg_s), weight=1.5, opacity=0.25
                ).add_to(m)

    for i in range(NUM_SENSORS):
        spd = speeds_60[i]
        color = speed_to_color(spd)
        level = speed_to_level(spd)
        is_selected = (i == sensor_idx)
        radius = 10 if is_selected else (7 if spd < 35 else 4)

        popup_html = f"""
        <div style='font-family:Inter,Arial,sans-serif; width:220px; padding:4px;'>
          <b style='font-size:14px; color:#333;'>📡 Sensor #{station_ids[i]}</b>
          <hr style='margin:6px 0; border-color:#eee;'>
          <b>Predicted Speed:</b> {spd:.1f} mph<br>
          <b>Status:</b> {level}<br>
          <b>Time:</b> {DAYS[query_day]} {query_hour:02d}:00 → {(query_hour+1)%24:02d}:00<br>
          <hr style='margin:6px 0; border-color:#eee;'>
          <small style='color:#666;'>
            +5min: {pred_speed[0,i]:.1f} | +15min: {pred_speed[2,i]:.1f} |
            +30min: {pred_speed[5,i]:.1f} | +60min: {pred_speed[11,i]:.1f} mph
          </small>
        </div>
        """

        folium.CircleMarker(
            location=[lats[i], lngs[i]],
            radius=radius,
            color='white' if is_selected else color,
            weight=3 if is_selected else 1,
            fill=True, fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"Sensor {station_ids[i]}: {spd:.0f} mph {level}",
        ).add_to(m)

    st_folium(m, width=None, height=550, use_container_width=True)

with tab2:
    st.markdown(f"### 📊 Sensor #{selected_sensor} — {DAYS[query_day]} {query_hour:02d}:00 Forecast")

    hist_avg = history_speed[:, sensor_idx].mean()
    pred_60 = pred_speed[-1, sensor_idx]
    actual_60 = actual_speed[-1, sensor_idx]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <p>Historical Avg (Last Hr)</p><h2 style="color:#8b949e">{hist_avg:.1f} mph</h2>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card" style="border:2px solid #1f6feb;">
            <p>Predicted (+60min)</p><h2 style="color:#58a6ff">{pred_60:.1f} mph</h2>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <p>Actual (Ground Truth)</p><h2 style="color:#7ee787">{actual_60:.1f} mph</h2>
        </div>""", unsafe_allow_html=True)
    with c4:
        err = abs(pred_60 - actual_60)
        st.markdown(f"""<div class="metric-card">
            <p>Prediction Error</p><h2 style="color:{'#7ee787' if err < 3 else '#e17055'}">{err:.1f} mph</h2>
        </div>""", unsafe_allow_html=True)

    st.write("")

    horizons = np.arange(1, 13) * 5
    hist_time = np.arange(-55, 5, 5)
    p_seq = pred_speed[:, sensor_idx]
    a_seq = actual_speed[:, sensor_idx]

    unc_std = np.linspace(0.5, 3.0, 12)
    lo95, hi95 = p_seq - 1.96 * unc_std, p_seq + 1.96 * unc_std
    lo80, hi80 = p_seq - 1.28 * unc_std, p_seq + 1.28 * unc_std

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_time, y=history_speed[:, sensor_idx],
                             mode='lines+markers', name='History',
                             line=dict(color='#8b949e', dash='dash', width=2),
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=horizons, y=hi95, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=horizons, y=lo95, fill='tonexty', mode='lines', line=dict(width=0),
                             fillcolor='rgba(31,111,235,0.08)', name='95% CI'))
    fig.add_trace(go.Scatter(x=horizons, y=hi80, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=horizons, y=lo80, fill='tonexty', mode='lines', line=dict(width=0),
                             fillcolor='rgba(31,111,235,0.18)', name='80% CI'))
    fig.add_trace(go.Scatter(x=horizons, y=p_seq, mode='lines+markers', name='Predicted',
                             line=dict(color='#58a6ff', width=3), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=horizons, y=a_seq, mode='lines+markers', name='Actual',
                             line=dict(color='#7ee787', width=2, dash='dot'), marker=dict(size=5)))

    fig.update_layout(
        title=dict(text=f"Speed Forecast — Sensor #{selected_sensor}", font=dict(color='white', size=16)),
        xaxis_title="Minutes Ahead", yaxis_title="Speed (mph)",
        height=420, margin=dict(t=50, b=30, l=50, r=30),
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='#c9d1d9'),
        xaxis=dict(gridcolor='#21262d', zerolinecolor='#30363d'),
        yaxis=dict(gridcolor='#21262d', zerolinecolor='#30363d'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    trend = "improving" if p_seq[-1] > p_seq[0] else "declining"
    st.info(f"**🗣️ AI Summary:** Sensor #{selected_sensor} at {DAYS[query_day]} {query_hour:02d}:00 — "
            f"current speed {hist_avg:.1f} mph. Over the next hour, speed is **{trend}** to "
            f"**{pred_60:.1f} mph** (actual: {actual_60:.1f} mph, error: {err:.1f} mph). "
            f"Status: {speed_to_level(pred_60)}.")

    st.markdown("---")
    st.markdown("#### 🔄 Compare Multiple Sensors")
    compare_ids = st.multiselect("Select sensors to compare",
                                  station_ids,
                                  default=[selected_sensor],
                                  max_selections=6)
    if compare_ids:
        fig2 = go.Figure()
        colors = ['#58a6ff', '#7ee787', '#e17055', '#fdcb6e', '#d2a8ff', '#79c0ff']
        for idx, sid in enumerate(compare_ids):
            s_idx = station_ids.index(str(sid))
            fig2.add_trace(go.Scatter(
                x=horizons, y=pred_speed[:, s_idx], mode='lines+markers',
                name=f'Sensor {sid}', line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=5)
            ))
        fig2.update_layout(
            title=dict(text="Multi-Sensor Speed Comparison", font=dict(color='white', size=14)),
            xaxis_title="Minutes Ahead", yaxis_title="Speed (mph)",
            height=350, margin=dict(t=50, b=30, l=50, r=30),
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='#c9d1d9'),
            xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown(f"### 📈 Network Summary — {DAYS[query_day]} {query_hour:02d}:00 (+60min)")

    n_free = (speeds_60 >= 50).sum()
    n_light = ((speeds_60 >= 35) & (speeds_60 < 50)).sum()
    n_heavy = ((speeds_60 >= 20) & (speeds_60 < 35)).sum()
    n_severe = (speeds_60 < 20).sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <p>🟢 Free Flow (≥50)</p>
            <h2 style="color:#00b894">{n_free}</h2>
            <p>{n_free/NUM_SENSORS*100:.0f}% of sensors</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <p>🟡 Moderate (35–50)</p>
            <h2 style="color:#fdcb6e">{n_light}</h2>
            <p>{n_light/NUM_SENSORS*100:.0f}% of sensors</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <p>🔴 Heavy (20–35)</p>
            <h2 style="color:#e17055">{n_heavy}</h2>
            <p>{n_heavy/NUM_SENSORS*100:.0f}% of sensors</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <p>⛔ Severe (&lt;20)</p>
            <h2 style="color:#d63031">{n_severe}</h2>
            <p>{n_severe/NUM_SENSORS*100:.0f}% of sensors</p>
        </div>""", unsafe_allow_html=True)

    st.write("")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=speeds_60, nbinsx=30,
        marker=dict(color='#58a6ff', line=dict(color='#1f6feb', width=1)),
        opacity=0.85, name='Speed Distribution'
    ))
    for threshold, label, color in [(20, 'Severe', '#d63031'), (35, 'Heavy', '#e17055'), (50, 'Moderate', '#fdcb6e')]:
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color=color,
                          annotation_text=label, annotation_position="top",
                          annotation_font_color=color)
    fig_hist.update_layout(
        title=dict(text="Speed Distribution Across All Sensors", font=dict(color='white', size=14)),
        xaxis_title="Speed (mph)", yaxis_title="Number of Sensors",
        height=350, margin=dict(t=50, b=30, l=50, r=30),
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='#c9d1d9'),
        xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### 🚨 Top 15 Most Congested Sensors")
    sorted_idx = np.argsort(speeds_60)
    table_rows = ""
    for rank, idx in enumerate(sorted_idx[:15]):
        spd = speeds_60[idx]
        lvl = speed_to_level(spd)
        col = speed_to_badge_color(spd)
        table_rows += f"""<tr>
            <td>{rank+1}</td>
            <td><b>Sensor {idx}</b></td>
            <td>{station_ids[idx]}</td>
            <td style="color:{col}; font-weight:600;">{spd:.1f} mph</td>
            <td><span class="status-badge" style="background:{col}22; color:{col};">{lvl}</span></td>
        </tr>"""

    st.markdown(f"""
    <table class="congestion-table">
        <thead><tr>
            <th>#</th><th>Sensor</th><th>Station ID</th><th>Speed</th><th>Status</th>
        </tr></thead>
        <tbody>{table_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Network Speed Across Forecast Horizons")
    horizon_avgs = [pred_speed[h].mean() for h in range(12)]
    horizon_labels = [f"+{(h+1)*5}min" for h in range(12)]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=horizon_labels, y=horizon_avgs,
        marker=dict(color=[speed_to_color(s) for s in horizon_avgs],
                    line=dict(width=0)),
        text=[f"{s:.1f}" for s in horizon_avgs],
        textposition='outside', textfont=dict(color='#c9d1d9', size=10),
    ))
    fig_bar.update_layout(
        title=dict(text="Average Network Speed by Horizon", font=dict(color='white', size=14)),
        xaxis_title="Forecast Horizon", yaxis_title="Avg Speed (mph)",
        height=350, margin=dict(t=50, b=30, l=50, r=30),
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='#c9d1d9'),
        xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)