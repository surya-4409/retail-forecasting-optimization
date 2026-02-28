
# 📦 Retail Demand Forecasting & Inventory Optimization Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retail-forecasting-optimization-b7et7qbwnypym9etluap82.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/surya-4409/retail-forecasting-optimization.git)

**🎥 Video Walkthrough:** [[WATCH THE VIDEO HERE](https://drive.google.com/file/d/1gO-cSdOXt968wbr44ZVRMwQUzvyb39UA/view?usp=sharing)]  
**🌐 Live Dashboard:** [View Deployed Application Here](https://retail-forecasting-optimization-b7et7qbwnypym9etluap82.streamlit.app/)  

---

## 🎯 Project Overview
Retail margins are constantly eroded by two opposing inventory failures: **stockouts** (which cause immediate revenue loss and customer churn) and **overstocking** (which drives up holding costs and markdowns). 

**The Objective:** Design and deploy an end-to-end, production-grade Machine Learning pipeline that predicts SKU-level demand and mathematically optimizes order quantities to minimize holding costs while strictly constraining stockout risk.

---

## 🧠 Methodology & MLOps Architecture
To achieve a senior-level data science solution, this project was executed with strict adherence to MLOps and Software Engineering best practices:

### 1. Robust Data Engineering & Validation
* **Configuration Management:** All hyperparameters, feature sets, and pipeline settings are dynamically loaded from a centralized `config.yaml` to decouple logic from code.
* **Defensive Validation:** Automated checks execute prior to processing to halt the pipeline if corrupted data (e.g., negative sales, missing keys) is detected.
* **Signal Preservation:** Historical stockouts were not treated as zero-demand. Linear interpolation was used to impute missing sales, preserving true demand baselines and seasonality.

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
* **Containerization:** The entire application is containerized using Docker for flawless reproducibility across diverse computing environments.

---

## 🏆 Key Business Results
The system successfully surpassed the strict business performance thresholds:
* **Target:** Mean Absolute Percentage Error (MAPE) < 20% for Fast-Moving Products.
* **Achieved:** **12.1% MAPE** on the target segment.

---

## 📂 Repository Architecture
```text
├── config.yaml          # Centralized configuration (parameters, features, thresholds)
├── Dockerfile           # Containerization instructions for reproducibility
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

---

## ⚙️ How to Run Locally

You can run this project using either a standard Python environment or Docker.

### Option 1: Standard Python Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/surya-4409/retail-forecasting-optimization.git](https://github.com/surya-4409/retail-forecasting-optimization.git)
cd retail-forecasting-optimization

```


2. **Install dependencies:**
*(It is recommended to use a virtual environment)*
```bash
pip install -r requirements.txt

```


3. **Launch the Dashboard:**
```bash
streamlit run dashboard/app.py

```



### Option 2: Run with Docker (Recommended for Reproducibility)

1. **Clone the repository:**
```bash
git clone [https://github.com/surya-4409/retail-forecasting-optimization.git](https://github.com/surya-4409/retail-forecasting-optimization.git)
cd retail-forecasting-optimization

```


2. **Build the Docker Image:**
```bash
docker build -t retail-forecast .

```


3. **Run the Container:**
```bash
docker run -p 8501:8501 retail-forecast

```


4. **View the App:** Open your web browser and navigate to `http://localhost:8501`.

---

## ✍️ Author

**Surya (Billakurti Venkata Suryanarayana)** * **Roll No:** 23MH1A4409

* **Role:** Data Scientist / Machine Learning Engineer
* **College:** Aditya University

```

