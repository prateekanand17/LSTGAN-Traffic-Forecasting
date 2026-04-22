# Long-Term Spatio-Temporal Graph Attention Network for Urban Traffic Speed Forecasting with Adaptive Graph Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lstgan-traffic-forecasting.streamlit.app/)
**Live Deployment:** [Test the live dashboard here!](https://lstgan-traffic-forecasting.streamlit.app/)

## 1. Overview
This repository contains the complete source code, dataset processing pipeline, and interactive dashboard for my B.Tech CSE Dissertation on macroscopic traffic speed forecasting.

The core of this project is a faithful implementation of the **Long-Term Spatio-Temporal Graph Attention Network (LSTGAN)**, explicitly abstracting the road network as a dynamic Spatial-Temporal Graph $G(V, E, A)$. 

**Primary Novelty:** This project extends the standard architecture with a custom **Adaptive Graph Learning** module. By employing Node Embeddings, Scaled Dot-Product Attention, and Top-K Sparsification, the model dynamically learns hidden behavioral correlations (e.g., parallel highway re-routing or shared rush-hour zones) that are entirely invisible to physical distance-based adjacency matrices.

## 2. Key Features
- **Adaptive Graph Learning:** Dynamically discovers hidden inter-sensor corridors and fuses them with the static physical graph using a differentiable Alpha-blending parameter.
- **Local Spatial Mapping:** Utilizes Chebyshev Polynomial Graph Convolutions (Order K=5) on the normalized Laplacian to mathematically propagate congestion through physically connected intersections.
- **Global Spatial Attention:** Implements a Multi-Head Self-Attention Transformer mechanism to uncover unlinked, city-wide semantic correlations.
- **Multi-Scale Temporal Triad:** Deploys stride-based 1D convolutions across massive Weekly (2016 steps), Daily (288 steps), and Hourly (12 steps) historical sliding windows.
- **Production-Grade UI:** A highly interactive Streamlit interface rendering real-time geospatial Folium maps, automated anomaly detection alerts, 60-minute prediction profiles, and robust confidence intervals.
- **LLM-Powered Scenario Engine:** A fully integrated Llama-3 agent (accessed via Groq API) that allows planners to type natural-language "what-if" queries (e.g., simulating lane closures) for dynamic scenario analysis.

## 3. Tech Stack
- **AI/ML Engine:** PyTorch $\ge$ 2.0 (leveraging Mixed Precision `torch.amp` training to prevent VRAM exhaustion across 7 dense modules)
- **Frontend App:** Streamlit $\ge$ 1.30
- **Visualization:** Folium (Leaflet.js) & Plotly
- **Data Engineering:** Pandas & NumPy
- **Context Environment:** 325 highway loop detectors tracked by the Caltrans PeMS-BAY dataset across 6 months.

## 4. Install and Run Instructions
To run this application locally, you must first have Python 3.9+ installed and clone the repository.

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit Dashboard**
   ```bash
   python -m streamlit run src/app.py
   ```

3. **View the Dashboard**
   - Open your browser and navigate to `http://localhost:8501`

*(Note: We have natively bundled the required pre-trained LSTGAN-Adaptive model and testing dataset inside the repository's `assets/` directory so the dashboard will compile, load, and run inference entirely out-of-the-box).*

## 5. Performance & Evaluation
The Adaptive Graph extension consistently outperforms the static baseline across all horizons on the PeMS-BAY benchmark. 
At the most difficult **60-minute prediction horizon**, the model achieves:
- **MAE:** 2.12 mph *(9.4% error reduction over the static baseline)*
- **RMSE:** 4.75 mph
- **MAPE:** 4.38%

This architecture strictly outperforms established state-of-the-art models including ARIMA, DCRNN, STGCN, and Graph WaveNet.

## 6. Architecture & Developer Notebooks
Our underlying model (239,172 trainable parameters) operates using a highly advanced 7-module composite architecture based on the framework originally proposed by Fang et al (2022).

For judges, examiners, and technical reviewers seeking to inspect the raw foundational model training or the mathematical proofs behind the Adaptive Graph:
- **`Novelty/`**: Contains the isolated PyTorch notebooks for the `AdaptiveGraphLearner`, Alpha Blending, and the Sparse L1 Regularization.
- **`notebooks/LSTGAN_Training.ipynb`**: Original PyTorch model compilation, forward-pass training loops, and validation/testing metrics generation.
- **`notebooks/LSTGAN_Complete_Data_Pipeline.ipynb`**: The massive sliding-window tensor generation algorithm that extracts aligned temporal patches from the raw PeMS binaries.

## 7. Future Directions
- **Exogenous Weather Embeddings:** Extending the Time Encoder to inject multi-channel meteorological variables (rainfall, temperature, visibility) which fundamentally alter speed capacities.
- **Live Streaming Integration:** Replacing static `.pkl` files with Apache Kafka streams connected to live Caltrans web sockets for genuine real-time inference.
- **Edge Deployment:** Optimizing the model via TorchScript and FP8 quantization for deployment directly onto embedded traffic control edge hardware.
