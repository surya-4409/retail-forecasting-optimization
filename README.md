
# 📦 Retail Demand Forecasting & Inventory Optimization Pipeline

**Developer:** Surya (Roll No: 23MH1A4409) | **Role:** Data Scientist  
**🎥 Video Walkthrough:** [INSERT YOUR YOUTUBE/DRIVE LINK HERE]  
**🌐 Live Dashboard:** [INSERT YOUR STREAMLIT LINK HERE]  

---

## 🎯 The Task: Project Overview
Retail margins are constantly eroded by two opposing inventory failures: **stockouts** (which cause immediate revenue loss and customer churn) and **overstocking** (which drives up holding costs and markdowns). 

**The Objective:** Design and deploy an end-to-end, production-grade Machine Learning pipeline that predicts SKU-level demand and mathematically optimizes order quantities to minimize holding costs while strictly constraining stockout risk.

---

## 🧠 Way of Execution: The Methodology
To achieve a senior-level data science solution, this project was executed in five distinct phases, strictly adhering to MLOps and Software Engineering best practices:

### 1. Robust Data Engineering & Validation
* **Configuration Management:** All hyperparameters, feature sets, and pipeline settings are dynamically loaded from a centralized `config.yaml` to decouple logic from code.
* **Defensive Validation:** Automated checks execute prior to processing to halt the pipeline if corrupted data (e.g., negative sales, missing keys) is detected.
* **Signal Preservation:** Historical stockouts were not treated as zero-demand; instead, linear interpolation was used to impute missing sales, preserving the true demand baseline and seasonality.

### 2. Segmented Forecasting Strategy
Demand profiles dictate the modeling approach:
* **Fast-Moving SKUs:** Forecasted using an explicit weighted ensemble combining **XGBoost** (capturing complex non-linear promotional and pricing effects) with **Exponential Smoothing** (extracting robust seasonality).
* **Intermittent SKUs:** Forecasted using the **Syntetos-Boylan Approximation of Croston's Method** to prevent the zero-inflation bias commonly seen in slow-moving inventory.

### 3. Rigorous Evaluation (Walk-Forward CV)
* Standard cross-validation causes data leakage in time-series forecasting. This pipeline utilizes strict **Walk-Forward Cross-Validation** (3 expanding windows) to simulate real-world monthly retraining schedules and objectively measure performance.

### 4. Mathematical Inventory Optimization
* Point forecasts and their residual uncertainties are passed into a **Newsvendor Optimization Model**. 
* The system calculates the Critical Ratio to balance holding costs against stockout penalties, producing dynamically optimized Safety Stock and Reorder Points (ROP) specific to lead times.

### 5. MLOps & Deployment
* **Model Versioning:** The pipeline utilizes Python `logging` and automatically serializes the highest-performing XGBoost artifacts (`.json`) to disk.
* **Interactive UI:** The entire mathematical backend is wrapped in a deployed **Streamlit Dashboard** allowing stakeholders to filter recommendations by store, SKU, and operational constraints.

---

## 🏆 Key Business Results
The system successfully surpassed the strict business performance thresholds:
* **Target:** Mean Absolute Percentage Error (MAPE) < 20% for Fast-Moving Products.
* **Achieved:** **12.1% MAPE** on the target segment.

---

## 📂 Repository Architecture
```text
├── config.yaml          # Centralized configuration (parameters, features, thresholds)
├── data/                # Raw datasets and historical performance exports
├── models/              # Saved MLOps artifacts (xgb_model_v1.json)
├── notebooks/           # Interactive Jupyter notebooks for EDA and Walk-Forward CV
├── reports/             # Technical Executive Summary PDF
├── src/                 # Modularized, production-grade Python package
│   ├── preprocessing.py # Data validation, ETL, and imputation logic
│   ├── features.py      # Dynamic feature engineering (lags, rolling windows)
│   ├── models.py        # Forecasting logic, Croston's, Ensembling, Versioning
│   └── optimization.py  # Newsvendor Critical Ratio and Safety Stock math
├── dashboard/           # Streamlit application
│   └── app.py           # Stakeholder interface
├── requirements.txt     # Python environment dependencies
└── README.md            # Project documentation

```

## ⚙️ How to Reproduce Locally

1. Clone the repository to your local machine.`git clone https://github.com/surya-4409/retail-forecasting-optimization.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. To view the final output, launch the dashboard: `streamlit run dashboard/app.py`

```