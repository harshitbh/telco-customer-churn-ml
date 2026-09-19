# End-to-End Telco Customer Churn Machine Learning Pipeline

A production-ready machine learning project predicting customer churn for telecommunications providers. This repository covers the complete data science lifecycle from exploratory data analysis and advanced feature engineering to comparative model evaluation, model interpretation, and real-time API deployment using FastAPI.

---

## Project Overview
Customer churn is a critical operational challenge in the telecommunications sector. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project builds a robust classification system to identify high-risk customers, allowing retention teams to intervene proactively with targeted incentives.

---

## Key Features & Methodology
1. **Data Cleaning & Leakage Prevention:** Addressed hidden missing values in `TotalCharges`, removed duplicate entries, and strictly eliminated unique identifiers (`customerID`) to prevent data leakage.
2. **Advanced Feature Engineering:** Engineered high-value features inspired by competitive Kaggle solutions:
   - `AvgChargesPerTenure`: Financial burden relative to customer lifespan.
   - `TotalServices`: Aggregates active value-added subscriptions.
   - `ChargesPerService`: Cost-efficiency ratio per active service.
   - `TenureGroup`: Cohort risk bins (0–1 year, 1–2 years, etc.).
   - `Contract_Payment`: High-risk behavioral interaction combining contract type and payment method.
3. **Imbalance-Handling Benchmarking:** Compared two distinct strategies side-by-side:
   - **SMOTE (Synthetic Minority Over-sampling Technique)** inside leakage-free `imblearn` pipelines.
   - **Native Class Weighting (`scale_pos_weight`)** in XGBoost.
4. **Probability Threshold Tuning:** Optimized decision thresholds using Precision-Recall curves to maximize the F1-score for the minority churn class.
5. **Production Deployment:** Serialized optimal model pipelines using `joblib` and served real-time predictions via a **FastAPI** backend with automated Pydantic schema validation.

---

## Tech Stack
* **Language:** Python 3.11
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (v1.4+), XGBoost (v2.0.3), Imbalanced-learn (v0.12.0)
* **Model Serialization:** Joblib
* **API & Backend:** FastAPI, Uvicorn, Pydantic

---

## Repository Structure
```text
customer_churn_project/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
│
├── notebook/
│   └── churn_analysis.ipynb                  # Jupyter notebook (Phases 1-7)
│
├── model/
│   └── churn_model.pkl                       # Serialized production model pipeline
│
├── app.py                                    # FastAPI application script
├── requirements.txt                          # Project dependency versions
└── README.md                                 # Project documentation