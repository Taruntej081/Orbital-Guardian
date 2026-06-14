# 🛰️ ORBITAL GUARDIAN
### AI-Powered Space Debris Detection, Tracking & Collision Prevention
**Team LEGION | ASTRAVA Hackathon 2024 | Domain: AI For Space Tech**

---
**APP LINK : https://orbital-guardian-pawd8bq4p9ccc3e5jltypq.streamlit.app/**
## ⚡ QUICK START (Windows)

The HTML FILE - WEBSITE IN BROWSER BETA ( not completely BUild )
Chrome is blocking Celestrack, so coming soon
But you can run WebsiteInBrowserBeta.html it will work ( Not related to Steamlit One )

```powershell
# 1. Open PowerShell in your workspace folder
cd .\Orbital-Guardian-main

# 2. Activate virtual environment
..\ .venv\Scripts\Activate.ps1  # if running from parent folder
# or if already in the project folder: .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## ⚡ QUICK START (if no venv)

```powershell
pip install streamlit pandas numpy scikit-learn plotly requests sgp4
streamlit run app.py
```

---

## 🧠 AI MODELS USED

| Model | Purpose |
|-------|---------|
| **SGP4 Propagator** | Physics-based orbital trajectory prediction (NASA standard) |
| **Isolation Forest** | Anomaly detection — flags satellites with unusual orbital behaviour |
| **Random Forest Classifier** | ML collision probability estimation |
| **3D ECI Distance** | Real-time proximity analysis in Earth-Centered Inertial frame |

## 📡 DATA SOURCES

- **CelesTrak** — Live TLE data (real satellite positions)
- **SGP4 Library** — Same propagator used by NASA/NORAD
- Built-in fallback TLEs for offline demo

## 🔬 FEATURES

1. **Live Globe** — Real-time satellite positions on 3D Earth with risk color coding
2. **Collision Risk Engine** — 3D closest-approach calculation + ML probability
3. **Trajectory Prediction** — SGP4-propagated ground tracks with anomaly overlay
4. **Overpass Finder** — Predict when satellites pass over your location
5. **Fleet Analytics** — Orbit zone distribution, speed histograms, AI anomalies

---

## 🗂️ PROJECT STRUCTURE

```
orbital_guardian/
├── app.py            ← Main Streamlit dashboard (UI)
├── data_fetcher.py   ← Live TLE data + SGP4 position computation
├── predictor.py      ← AI models (Isolation Forest, Random Forest, trajectory)
├── requirements.txt
└── README.md
```

---
*Built for ASTRAVA Hackathon | Team LEGION*