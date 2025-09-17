# ⚡ Smart Energy Forecasting (India) — Stage 1: State-Level Demand

This project is part of a **multi-scale energy forecasting portfolio** (macro → meso → micro).
Stage 1 focuses on **forecasting electricity demand at the state/national level** using weather + demand + generation data.

---

## 🎯 Project Goals

* Predict **daily electricity demand** from weather variables.
* Compare **simple vs. weighted national averages**.
* Build **baseline and advanced forecasting models**.
* Evaluate interpretability (which weather features drive demand).

---

## 📊 Datasets

* **Daily Electricity Demand (2014–2024)** – [Zenodo Dataset](https://zenodo.org/records/14983362)
* **State-wise Power Generation (2013–2023)** – [Kaggle Dataset](https://www.kaggle.com/datasets/krishnadaskv/daily-power-generation-in-india-2013-2023)

**Features included**:

* Weather: temperature (min/mean/max), humidity (dewpoint), wind speed, solar radiation, cloud cover.
* Electricity: maximum demand met, shortages, energy met (MU), state-wise generation.
* Flags: Major holidays, extreme heat events.

📂 `data/sample_data.csv` → contains a **5,000-row sample** for quick experiments.
Full datasets available from the sources above (see `data/README.md`).

---

## 🛠️ Methods

* **EDA**:

  * Correlations between weather and demand.
  * Seasonal trends (summer, monsoon, winter).
  * Demand vs. supply gap analysis.

* **Models**:

  * Baseline: Linear Regression (time-series aware CV).
  * Random Forest + XGBoost: feature importance & hyperparameter tuning.
  * Added engineered features: **Cooling Degree Days (CDD)**, lagged temperatures, rolling averages.

* **Diagnostics**:

  * Residual plots (with holidays & extreme heat highlighted).
  * QQ plot of residuals.
  * Actual vs. predicted scatter & time-series overlays.

---

## 📈 Results

* Best model: **Random Forest (with log-transform)**
* Final metrics (test set):

  * MAE = **2.18**
  * RMSE = **4.58**
  * R² = **0.998**

📊 **Feature Importance** →

1. Temperature (mean, max).
2. Humidity (dewpoint + UTCI).
3. Holiday/Extreme heat flags.
4. Solar radiation, wind (secondary drivers).

**Key Insight**:
Extreme heat events were the **main cause of forecast errors**.
Future improvements should focus on **multi-day heatwave modeling** (lags, CDD variations).

---

## 📂 Repo Structure

```
notebooks/    → Jupyter notebooks (EDA, feature engineering, modeling)
data/         → Sample dataset + source README
results/      → Figures, metrics, residual plots
scripts/      → Reusable Python scripts
models/       → Saved baseline and tuned models (Pickle)
```

---

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/smart-energy-forecasting.git
cd smart-energy-forecasting
pip install -r requirements.txt
jupyter notebook
```

---

## 📌 Next Steps

* Stage 2: **Household-level clustering** + anomaly detection.
* Stage 3: **Appliance-level disaggregation**.
* Deploy interactive **Streamlit dashboard** for demand visualization & forecasting.

---

## 📜 License

MIT License — free to use and modify, attribution required.
