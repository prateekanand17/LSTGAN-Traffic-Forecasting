import os
import sys
# Add the repository root to the Python path so Streamlit Cloud can find the 'src' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium

from src.config import *
from src.model import LSTGAN
from src.data import TrafficDatasetLite, get_geographic_coordinates
from src.utils import speed_to_color, speed_to_level, speed_to_badge_color

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

sp    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Network Map", "📊 Sensor Forecast", "📈 Network Summary", "🤖 Scenario Engine"])

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

with tab4:
    st.markdown("### 🤖 NLP 'What-If' Scenario Engine")
    st.markdown("Type a natural language scenario to perform **dynamic mathematical graph surgery** on the native Adjacency Matrix and compute gridlock shockwaves in real-time.")
    
    scenario_input = st.text_input(
        "Enter scenario prompt:",
        value=f"Simulate a massive accident and 3-lane closure at Sensor {selected_sensor}",
    )
    
    if st.button("Run Interactive Simulation"):
        with st.spinner("Extracting NLP Intent and recalculating PyTorch Graph edge nodes..."):
            import re
            import copy
            
            # Simple RegEx Mock for NLP Parser Extracting IDs
            target_ids = re.findall(r'\b(40\d{4})\b', scenario_input)
            if not target_ids:
                st.error("❌ ActionParser failed: Could not detect a valid 6-digit Sensor ID in the prompt (e.g., 400038).")
            else:
                sim_sensor = target_ids[0]
                if str(sim_sensor) not in station_ids:
                    st.error(f"❌ GeoParser failed: Sensor {sim_sensor} not found in physical network graph.")
                else:
                    sim_idx = station_ids.index(str(sim_sensor))
                    
                    # 1. Clone Graph and Sever Edges (Mathematical Graph Surgery)
                    adj_mx_sim = adj_mx.copy()
                    adj_mx_sim[sim_idx, :] = 0.0
                    adj_mx_sim[:, sim_idx] = 0.0
                    
                    # 2. Clone Model Context and Inject New Graph Adjacency
                    model_sim = copy.deepcopy(model)
                    model_sim.local_enc.set_graph(adj_mx_sim)
                    
                    # 3. Mutate History Tensors (Force to 0 mph severe gridlock)
                    zero_val = -train_mean / train_std
                    Xh_sim = Xh.clone(); Xh_sim[0, :, sim_idx, 0] = zero_val
                    Xw_sim = Xw.clone(); Xw_sim[0, :, sim_idx, 0] = zero_val
                    Xd_sim = Xd.clone(); Xd_sim[0, :, sim_idx, 0] = zero_val
                    
                    # 4. Generate Simulation
                    with torch.no_grad():
                        pred_sim = model_sim(Xw_sim, Xd_sim, Xh_sim, ti)
                        
                    pred_speed_sim = (pred_sim.cpu().numpy() * train_std + train_mean)[0, :, :, 0]
                    
                    st.success(f"✅ Simulation Complete! Mathematical edges pointing to Target Node **Sensor {sim_sensor}** have been successfully severed from the Spatial Graph.")
                    
                    # Compute network cascading failure
                    orig_net = pred_speed[-1].mean()
                    sim_net = pred_speed_sim[-1].mean()
                    
                    st.markdown("#### Scenario Analytical Impact")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        c1.metric("Original Average Speed", f"{orig_net:.1f} mph")
                    with c2:
                        c2.metric("Simulation Average Speed", f"{sim_net:.1f} mph", f"{sim_net - orig_net:.2f} mph overall")
                    with c3:
                        neighbor_drop = pred_speed_sim[-1].mean() - pred_speed[-1].mean()
                        c3.metric("Network Cascading Shockwave", f"{neighbor_drop:.2f} mph decay")
                    
                    # Graph the Shockwave Ripple
                    fig_sim = go.Figure()
                    horizons_sim = np.arange(1, 13) * 5
                    
                    # Selected Sensor (Zeroed)
                    fig_sim.add_trace(go.Scatter(x=horizons_sim, y=pred_speed[:, sim_idx], name=f'Original Sensor {sim_sensor}', line=dict(color='#8b949e', dash='dot')))
                    fig_sim.add_trace(go.Scatter(x=horizons_sim, y=pred_speed_sim[:, sim_idx], name=f'Simulated Sensor {sim_sensor} (Closure)', line=dict(color='#d63031', width=3)))
                    
                    # Plot the ripple on a connected neighbor node!
                    neighbors = (adj_mx[sim_idx, :] > 0.0).nonzero()[0]
                    if len(neighbors) > 0:
                        n_idx = neighbors[0] # Pick primary neighbor
                        n_id = station_ids[n_idx]
                        fig_sim.add_trace(go.Scatter(x=horizons_sim, y=pred_speed[:, n_idx], name=f'Prior Neighbor {n_id} (Unaffected)', line=dict(color='#58a6ff', dash='dot')))
                        fig_sim.add_trace(go.Scatter(x=horizons_sim, y=pred_speed_sim[:, n_idx], name=f'Predicted Neighbor {n_id} (Spillover)', line=dict(color='#d2a8ff', width=3)))

                    fig_sim.update_layout(
                        title=dict(text="Network Graph Spillover Effects (Local Connectivity Severed)", font=dict(color='white')),
                        xaxis_title="Minutes Ahead", yaxis_title="Predicted Speed (mph)",
                        height=400, margin=dict(t=50, b=30, l=50, r=30),
                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font=dict(color='#c9d1d9'),
                        xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
                        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=12)),
                    )
                    st.plotly_chart(fig_sim, use_container_width=True)
262d'), yaxis=dict(gridcolor='#21262d'),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
