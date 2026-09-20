from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize FastAPI application
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="A production-ready API for predicting customer churn risk.",
    version="1.0"
)

MODEL_PATH = Path(__file__).resolve().parent / "model" / "churn_model.pkl"

# Load the trained model pipeline
model = joblib.load(MODEL_PATH)

# Define input data structure using Pydantic for validation
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    AvgChargesPerTenure: float
    TotalServices: int
    ChargesPerService: float
    TenureGroup: str
    Contract_Payment: str

@app.get("/")
def home():
    return {"message": "Welcome to the Telco Customer Churn Prediction API! Send a POST request to /predict."}

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert incoming JSON request into a Pandas DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Generate prediction and probability
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    churn_status = "Yes" if prediction == 1 else "No"
    risk_level = "High Risk" if probability >= 0.5 else "Low Risk"

    return {
        "churn_prediction": churn_status,
        "churn_probability": round(float(probability), 4),
        "risk_level": risk_level,
        "recommendation": "Retention outreach required" if probability >= 0.5 else "Monitor and continue standard retention"
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
