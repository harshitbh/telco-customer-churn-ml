# Telco Customer Churn Prediction

This project predicts whether a telecom customer is likely to churn using historical customer behavior, service usage, and billing data.

The workflow includes data cleaning, EDA, feature engineering, class balancing, model comparison, threshold tuning, and deployment via a FastAPI prediction endpoint.

## Project objective
Customer churn is a major cost driver in telecom businesses. This project helps identify high-risk customers so the retention team can act early with offers, upgrades, or support interventions.

## What is included
- Data preprocessing and cleaning
- Duplicate and leakage checks
- Feature engineering for customer risk patterns
- Train-test split with class imbalance handling
- Model benchmarking using Logistic Regression, Decision Tree, Random Forest, and XGBoost
- Recall-focused threshold tuning
- Model interpretation using feature importance
- Saved production pipeline via joblib
- FastAPI app for real-time prediction

## Tech stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- imbalanced-learn
- FastAPI
- Pydantic
- Joblib

## Repository structure
```text
Teleco_Customer_Churn/
├── app.py                     # FastAPI prediction API
├── requirements.txt          # Project dependencies
├── sample_request.json        # Example JSON payload for API testing
├── README.MD                 # Project documentation
├── data/
│   └── TelcoCustomerChurn.csv
├── model/
│   └── churn_model.pkl       # Trained churn model pipeline
├── notebook/
│   └── Customer_Churn_Analysis.ipynb
├── .gitignore
└── venv/                     # local virtual environment (ignored in Git)
```

## How to run the notebook
Open the notebook in VS Code or Jupyter:

```bash
notebook/Customer_Churn_Analysis.ipynb
```

The notebook has path checks so it can run from either the project root or the notebook folder.

## How to run the API
From the project root:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then test it with:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

## Model output
The final model is saved to:

```text
model/churn_model.pkl
```

## Business insight
The project is designed around business risk mitigation, where missing a churner is more expensive than flagging a loyal customer. Because of this, recall is prioritized in the model evaluation and threshold tuning stage.

## Notes
- The assignment PDF is intentionally excluded from Git tracking in [.gitignore](.gitignore).
- The notebook and app have been updated to use stable, project-root-safe file handling for the dataset and model files.