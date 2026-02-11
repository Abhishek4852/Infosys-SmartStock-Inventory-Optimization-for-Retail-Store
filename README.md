# 🛒 Infosys SmartStock Inventory Optimization for Retail Store over Walmart Sales Forecasting dataset

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Ensemble-orange)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

---

## 🎥 Project Demo
 Watch the project walkthrough here: [YouTube Link](https://www.youtube.com/watch?v=dRS5JuUvReA)


---


## 📌 Project Demo Images

### 🖼️ Landing Page
<img width="1470" height="836" alt="landing page" src="https://github.com/user-attachments/assets/1a4c40d5-4fa3-40e4-b4f0-27a2fd6dc834" />



### 📊 Input From user
<img width="1394" height="787" alt="input fields" src="https://github.com/user-attachments/assets/a618c157-f200-4009-99c6-9f964f9e22b9" />



### 📊 Dashboard (Prediction + Charts)
<img width="1470" height="836" alt="Dashboard" src="https://github.com/user-attachments/assets/bc412138-0cbf-4841-b6d1-8149e4b90a20" />



### 🤖 Gemini AI Summary Section
<img width="1470" height="835" alt="Gemini suggestion" src="https://github.com/user-attachments/assets/afff3a51-83b3-4716-a7f6-5ecca034c45e" />

---
## Dataset Used
Dataset: **Walmart Sales Forecasting (Kaggle)**
---
## 📌 Project Overview

This project is an **end-to-end sales forecasting system** built using real Walmart historical data.

It predicts **future weekly sales revenue** and provides:

* 📈 **Sales forecasting**: Predicts next week, next month, and next 3 months.
* 📦 **Inventory optimization**: Calculates safety stock & reorder point (ROP).
* 🤖 **AI-based business suggestions**: Powered by Google Gemini.
* 🌐 **Interactive dashboard**: Clean UI built with FastAPI, Jinja2, and Plotly.
* 🔁 **Automated retraining**: CI/CD pipeline using GitHub Actions for continuous model updates.

The system is designed to be **industry-grade, production-ready, and scalable**.

---

## 🎯 What does this project predict?

✅ **Total Weekly Sales Amount (Revenue)**
❌ Not number of units sold

The model predicts:
`Weekly_Sales → total department revenue ($)`

**Why?**
* The Walmart dataset does **not include unit price or quantity sold**.
* Only aggregated weekly department-level revenue is provided.

---

## 🧠 Models Used

Three models were selected intentionally to cover different learning behaviors:

| Model | Purpose |
| :--- | :--- |
| **LightGBM** | Fast, efficient, high-accuracy tree model for tabular data. |
| **XGBoost** | Strong gradient boosting learner to capture non-linear patterns. |
| **Prophet** | Time-series forecasting specializing in seasonality & trends. |

This combination provides:
* ML-based pattern learning (Store/Dept features).
* Time-series seasonality capture (Date features).
* Stable ensemble prediction through weighting.

---

## 📊 Model Performance

| Model | RMSE | MAE | R² Score |
| :--- | :--- | :--- | :--- |
| **LightGBM** | 2,834.32 | 1,374.27 | 0.9835 |
| **XGBoost** | 2,807.31 | 1,352.54 | 0.9838 |
| **Prophet** | 797.48 | 661.80 | -0.8533 |

> [!NOTE]
> Prophet shows a negative R² because it is trained as a global trend model (Date-only) and evaluates against detailed store/dept variances which it isn't designed to capture alone.

---

## ⚙️ Feature Engineering Strategy

### 🔹 Time Features
`Year`, `Month`, `Week`, `Day`, `Day of week`, `IsWeekend`, etc.

### 🔹 Lag Features
Used to capture short-term sales memory:
- **Lag_1**: Sales of last week
- **Lag_2**: Sales 2 weeks ago
- **Lag_3, Lag_4, Lag_8, Lag_12, Lag_24**

### 🔹 Rolling Window Features
- **RollingMean_4**: Average sales of last 4 weeks.
- **RollingStd_4**: Volatility of last 4 weeks.
- **RollingMean_12 / RollingStd_12**: Captures long-term stability vs fluctuation.

---

## 🔀 Ensemble Learning Strategy

Instead of static weights, the model contribution is calculated dynamically using **Inverse RMSE Weighting**.

### 📐 Formula
`weight_i = (1 / RMSE_i) / Σ(1 / RMSE_all)`

### ✅ Benefits
* **Lower error → higher weight**: Automatically trusts the most accurate model.
* **Auto-Calibration**: No manual weight tuning required after retraining.

---

## 🔄 Automated Retraining Pipeline (MLOps)

The full ML pipeline is automated in `retraining/retrain_pipeline.py`:

1.  **Data Cleaning**: Merges raw data and handles missing values.
2.  **Feature Engineering**: Generates lags, rolling stats, and date features.
3.  **Model Training**: Trains LightGBM, XGBoost, and Prophet.
4.  **Model Evaluation**: Calculates RMSE for each model.
5.  **Auto Ensemble**: Updates weights based on the latest performance.
6.  **Size Check**: Ensures every model file is `< 80MB` for GitHub compatibility.

---

## 🧩 Tech Stack

- **Backend**: FastAPI
- **ML Models**: LightGBM, XGBoost, Prophet
- **Frontend**: HTML5, Vanilla CSS, Jinja2
- **Visualization**: Plotly.js
- **DevOps**: GitHub Actions, Docker
- **Language**: Python 3.10+

---

## 🚀 How to Run Locally

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key

### 2. Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd sales_forcasting

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root:
```env
GEMINI_API_KEY=your_api_key_here
```

### 4. Training & Running
```bash
# Run the retraining pipeline (generates models)
python retraining/retrain_pipeline.py

# Start the FastAPI server
python src/api/main.py
```
Visit `http://localhost:8000` to access the dashboard.

---

## 👨‍💻 Author
**Abhishek Yaduwanshi**
MCA | Machine Learning | Backend Development

⭐ If you found this project helpful, please give it a star!
