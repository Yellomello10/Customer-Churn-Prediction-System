import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Initialize FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn using optimized Scikit-learn pipelines.",
    version="1.0.0"
)

# Load the model
MODEL_PATH = "outputs/models/best_model.joblib"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Best model not found at {MODEL_PATH}. Please run the pipeline first.")

model_pipeline = joblib.load(MODEL_PATH)

# Define request schema
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

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }
    }

def create_tenure_groups(df):
    """Recreate the binning logic from feature engineering."""
    df_engineered = df.copy()
    labels = ["0-12 Months", "12-24 Months", "24-48 Months", "48-60 Months", "60+ Months"]
    bins = [-1, 12, 24, 48, 60, 100]
    df_engineered['TenureGroup'] = pd.cut(df_engineered['tenure'], bins=bins, labels=labels)
    return df_engineered

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        # Convert request to dictionary
        data_dict = customer.model_dump()
        
        # Create DataFrame
        df = pd.DataFrame([data_dict])
        
        # Add TenureGroup
        df = create_tenure_groups(df)
        
        # Make predictions
        prob = model_pipeline.predict_proba(df)[0][1]
        prediction = int(model_pipeline.predict(df)[0])
        
        risk_tier = "Low"
        if prob >= 0.70:
            risk_tier = "High"
        elif prob >= 0.40:
            risk_tier = "Medium"
            
        return {
            "churn_prediction": prediction,
            "churn_probability": float(prob),
            "risk_tier": risk_tier,
            "insights": get_customer_insights(customer, prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_customer_insights(customer, prob):
    """Generate dynamic business recommendations based on user input features."""
    recommendations = []
    
    # 1. Contract risk
    if customer.Contract == "Month-to-month":
        recommendations.append(
            "Contract Migration: Customer is on a Month-to-month plan. Offer a $5/month discount or 1 free month to switch to a 1-year contract."
        )
    
    # 2. Payment method friction
    if customer.PaymentMethod == "Electronic check":
        recommendations.append(
            "Autopay Conversion: Electronic check payment shows high churn risk. Offer a $10 bill credit incentive to sign up for Autopay (Credit Card or Bank Transfer)."
        )
        
    # 3. Fiber Optic check
    if customer.InternetService == "Fiber optic":
        recommendations.append(
            "Fiber Quality Check: Fiber optic users experience higher churn. Reach out via customer service to check signal/routing issues and verify price satisfaction."
        )
        
    # 4. Tenure window
    if customer.tenure <= 12:
        recommendations.append(
            "Early Tenure Support: Customer is in their critical first year. Schedule an onboarding check-in call to ensure they are getting full service utility."
        )
        
    # 5. Overcharge check
    if customer.MonthlyCharges > 75.0:
        recommendations.append(
            "High Bill Loyalty Discount: Monthly charges of ${:.2f} are above average. Offer a value-added free feature (e.g., Tech Support or Device Protection) to offset bill shock.".format(customer.MonthlyCharges)
        )
        
    if not recommendations:
        recommendations.append("Healthy Profile: Keep standard engagement; no immediate critical triggers.")
        
    return recommendations

# Serve Static Files (Frontend SPA)
# Make sure the 'static' directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    # Start uvicorn server on port 8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
