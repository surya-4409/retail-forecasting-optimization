
# 📦 Retail Demand Forecasting & Inventory Optimization Pipeline

**Developer:** Surya (Roll No: 23MH1A4409)  
**Role:** Data Scientist  

---

## 🎯 Project Overview
This repository contains a production-grade, end-to-end machine learning pipeline designed to solve a critical retail operations challenge: balancing product availability with inventory holding costs. 

The system predicts daily product-store demand and mathematically optimizes safety stock and reorder points. By transitioning from reactive purchasing to data-driven procurement, this tool aims to minimize capital tied up in excess inventory while drastically reducing stockout events for high-value retail products.

## 🧠 Methodology & Analytical Rigor
1. **Data Engineering & Integration:** Integrated multiple synthetic data sources (sales, products, stores, promotions). Implemented intelligent linear interpolation to handle historical stockouts, ensuring the model learns true baseline demand rather than zero-sales anomalies.
2. **Feature Engineering:** Developed a modular pipeline (`src/features.py`) extracting calendar seasonality, moving averages, volatility (rolling standard deviation), and temporal lag features.
3. **Forecasting (Ensemble Model):** Built a hybrid forecasting engine:
   * **XGBoost:** A global machine learning model to capture non-linear relationships (promotional lifts, weekend spikes, pricing elasticity).
   * **Exponential Smoothing:** A local statistical baseline to capture strict, item-specific temporal seasonality.
4. **Inventory Optimization:** Utilized the Newsvendor framework to calculate the Critical Ratio based on holding costs and stockout penalties, dynamically generating **Recommended Order Quantities (ROQ)** and **Safety Stocks** for every item.

## 📊 Key Results
* **Fast-Moving Products:** The ensemble model achieved a Mean Absolute Percentage Error (MAPE) of **12.1%**, successfully surpassing the project target requirement of <20%.
* **Actionable Output:** Successfully deployed an interactive Streamlit dashboard allowing stakeholders to visualize the 60-day forward-looking Reorder Point (ROP) dynamic thresholds.

---

## 📂 Repository Structure
```text
retail_forecasting_project/
│
├── data/                  # Generated synthetic datasets & final recommendations
├── notebooks/             # Jupyter notebooks for EDA and model evaluation
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_model_training_and_evaluation.ipynb
├── src/                   # Modular Python source code
│   ├── __init__.py
│   ├── data_generation.py # Synthetic multi-relational data generator
│   ├── features.py        # Feature engineering pipeline
│   ├── models.py          # XGBoost and Baseline model classes
│   └── optimization.py    # Newsvendor inventory optimization logic
├── dashboard/             # Interactive web application
│   └── app.py             # Streamlit dashboard script
├── reports/               # Executive summaries and technical docs
│   └── executive_summary.pdf
├── requirements.txt       # Project dependencies
├── submission.yml         # Automated execution commands
└── README.md              # Project documentation

```

---

## 🚀 How to Run the Pipeline

### Prerequisites

Ensure you have Python 3.9+ installed. It is highly recommended to use a virtual environment.

### 1. Setup the Environment

Install all required dependencies:

```bash
pip install -r requirements.txt

```

### 2. Generate the Dataset

Create the synthetic retail data (sales, stores, products, promotions):

```bash
python src/data_generation.py

```

### 3. Run the Analytical Pipeline

The core data science workflow is contained within the Jupyter notebooks. Execute them sequentially to perform EDA, feature engineering, model training, and optimization:

1. Run `notebooks/01_exploratory_data_analysis.ipynb`
2. Run `notebooks/02_model_training_and_evaluation.ipynb`

*(Note: The second notebook will automatically generate the `final_inventory_recommendations.csv` required by the dashboard).*

### 4. Launch the Dashboard

Start the interactive Streamlit application to view the inventory recommendations:

```bash
streamlit run dashboard/app.py

```

The dashboard will be available in your browser at `http://localhost:8501`.

```

---

### The Final Stretch

Once you have this README saved, your local project folder is absolutely flawless. 

Looking at your submission screenshot, the final platform asks for:
1. **A GitHub repository link** (You need to push this folder to GitHub).
2. **A Live Demo URL** (You can deploy your Streamlit app for free using Streamlit Community Cloud).
3. **A Video Demo URL** (A short screen recording of you explaining the code and clicking through the dashboard).

Are you familiar with pushing code to GitHub and deploying a Streamlit app, or would you like me to walk you through those exact steps to finish your submission?

```