# LSTGAN Traffic Forecaster

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lstgan-traffic-forecasting.streamlit.app/)
**Live Deployment:** [Test the live dashboard here!](https://lstgan-traffic-forecasting.streamlit.app/)
## 1. Overview
The **LSTGAN Traffic Forecaster** is an intelligent, real-time diagnostic dashboard designed to solve one of the most complex challenges in smart city management: predicting macroscopic traffic speeds up to a full hour into the future before congestion even occurs.

**What problem it solves:** Traditional forecasting models analyze traffic strictly as isolated streams of linear data. Real-world traffic, however, is a deeply interconnected web governed by the mathematical geometry of the city (where one crash eventually spills into downstream arteries) and deep historical cycles (where Friday at 5:00 PM looks vastly different from Sunday at 5:00 AM). 

Our solution explicitly abstracts the city as a dynamic **Spatial-Temporal Graph $G(V, E, A)$**. It applies a state-of-the-art **Long-Term Spatio-Temporal Graph Attention Network (LSTGAN)** to seamlessly fuse immediate road network physics (GCNs) with global city-wide semantic correlations (Multi-Head Self Attention).

**Intended users:** City planners, traffic control centers, and logistics companies needing mathematically proven, macro-level insights to proactively reroute fleets, deploy emergency services, and drastically minimize urban congestion pipelines.

## 2. Features
- **Local Spatial Mapping (Graph Convolutions):** Utilizes Chebyshev Polynomial approximations on the road Adjacency Matrix to mathematically "bleed" congestion into physically connected intersections in real-time.
- **Global Spatial Attention:** Implements a Transformer-style Query/Key attention mechanism to uncover invisible, city-wide correlations (e.g., matching a commercial district's behavior with a distant industrial park).
- **Temporal Downsampling:** Deploys 1D Convolutional sliding windows across massive weekly and daily historical subsets to seamlessly filter out minute-by-minute noise and extract underlying macro-rhythms.
- **Dynamic Time Fusions:** Injects exact `Day-of-Week` and `Time-of-Day` timestamps via one-hot encoded matrix addition, explicitly forcing the AI to recognize complex cyclical habits like the morning rush hour.
- **Heavy Horizon Forecasting:** Safely outperforms Baseline models (Graph WaveNet, DCRNN) for massive long-term predictions (15, 30, and 60 minutes out) with minimized Root Mean Square Error (RMSE).
- **Interactive UI Intelligence:** Real-time percentage breakdowns of severe gridlock versus free flow, alongside highly granular 6-sensor multi-comparison capabilities wrapped in a Streamlit interface.

## 3. Tech Stack & Technical Depth
- **AI/ML Engine (PyTorch):** We utilized PyTorch over TensorFlow because its dynamic computational graph natively supports the highly irregular dimensions of Graph Convolutional Networks (GCNs). This solved the problem of tracking back-propagation errors through our 325-node spatial adjacency matrix.
- **Frontend App (Streamlit):** Chosen instead of React/Node.js to tightly couple our Python-based ML layer directly with our View layer. It solved the problem of building massive, stateful geospatial UIs without needing a complex, latency-heavy REST API backend.
- **Visualization (Folium & Plotly):** We integrated Folium (Leaflet.js) to render our graph nodes accurately onto real-world maps. This allowed us to physically verify our graph embeddings rather than relying exclusively on abstract matrix math.
- **Data Engineering (Pandas & NumPy):** Essential for transforming the massive PeMS-Bay datasets into efficient, sliding-window temporal tensors (1D convolutions) before feeding them into the PyTorch engine.

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

*(Note: We have natively bundled the required pre-trained LSTGAN model and testing dataset inside the repository's `assets/` directory so the dashboard will compile and load out-of-the-box).*
## 5. Usage Examples

![Dashboard Preview](Figures/Hackathon_Demo_figure/executive_dashboard.png)
*(Note: Additional dashboard UI layouts and dynamic map rendering examples can be found in our [`Figures/Hackathon_Demo_figure/`](Figures/Hackathon_Demo_figure/) and [`Figures/Map_figure/`](Figures/Map_figure/) directories).*

- **Check Rush Hour**: Use the left sidebar to select `Monday` and `08:00`. Watch the map instantly render the predicted rush hour congestion +60 mins into the future.
- **Deep Dive a Sensor**: Select `Sensor #400001` to view its specific 5-10-15-30-60 minute predictions plotted against true historic states.

## 6. Architecture Notes

![Model Architecture](Figures/Architecture.png)

Our underlying model (LSTGAN) operates using a highly advanced composite architecture:
*(Note: Detailed, per-module schematic diagrams for every component are available in the [`Figures/Architecture_figure/`](Figures/Architecture_figure/) directory).*

- **Spatial Topology:** Employs Chebyshev Graph Convolutional Networks (GCN) coupled with Multi-head Self-Attention to capture immediate upstream/downstream neighbor states *and* distant structural network states simultaneously.
- **Temporal Processing:** Uses dilated 1D-convolutions with distinct paths covering Weekly, Daily, and Hourly historical timeframes.
- **Frontend Segregation:** To maintain code stability, the Streamlit app acts exclusively as the View layer (`src/app.py`), directly sourcing the computational framework from `src/model.py`.
The spatial topology translates the exact Latitude and Longitude (`traffic_stations.csv`) into a 325x325 weight matrix (`adj_mx.pkl`), while the temporal sequences span consecutive 12-horizon (+60 minute) prediction branches.

## 7. Developer & Research Notebooks
For judges and technical reviewers seeking to inspect the raw foundational model training, data preprocessing pipelines, and architectural proofs, we have included our comprehensive research suite.
Please refer to the [`notebooks/`](notebooks/) directory to explore the original developmental workflows:
- **`LSTGAN_Training.ipynb`**: Original PyTorch model compilation, forward-pass training loops, and MAE evaluation generation.
- **`LSTGAN_Architecture_Demo.ipynb`**: Granular layer-by-layer architectural tracing of the Local Spatial Encoders and Global Attention mechanisms.
- **`LSTGAN_Complete_Data_Pipeline.ipynb`**: The massive sliding-window algorithm that generates the historical memory slices from the raw PeMS binaries.

## 8. Limitations
- **Static Dataset Integration:** Live streaming API ingestion (e.g. hooking directly into Caltrans PeMS web sockets) is pending. Predictions run on an offline high-resolution slice of `test_5min.pkl`.
- **CPU Inference Bound:** Streamlit defaults to deploying the model onto the CPU. Larger matrices may experience slow-down without an attached accelerator in production.

## 9. Performance & Evaluation
To validate our system's accuracy, robustness, and baseline comparative strength, we have logged extensive graphical evidence:
- **Baseline Analysis:** See the [`Figures/Complete_analysis_figure/`](Figures/Complete_analysis_figure/) directory for detailed spatial-temporal error distributions, horizon degradation charts, and multi-sensor heatmaps.
- **Prediction Proofs:** See the [`Figures/Visual_figure/`](Figures/Visual_figure/) directory for 24-hour traffic profiles, scatter plots, and direct Ground Truth vs Prediction visualizations.
- **Data Engineering:** See the [`Figures/Data_pipeline_figure/`](Figures/Data_pipeline_figure/) directory for flowchart diagrams illustrating our complex time-series horizon windowing techniques.

## 10. Future Improvements
**LLM-Powered "What-If" Scenario Engine**
If granted more time and resources, we plan to integrate a Natural Language Processing (LLM) interface for city planners. Instead of exclusively viewing static forward-predictions, a planner could type: *"Simulate a 3-lane closure on US-101 North at 8:00 AM"*. The system would dynamically sever the corresponding graph edges within the underlying Adjacency Matrix, forcing the LSTGAN model to re-calculate and visualize the predicted gridlock shockwaves in real-time.

