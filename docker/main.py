from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Stroke Prediction API",
              description="API for predicting stroke risk based on health metrics",
              version="1.0")

# Load the trained model
model_path = os.getenv("SAVED_MODEL")
try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Define input data model
class PatientData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    smoking_status: str
    
    # Optional fields for feature-engineered categories
    age_group: Optional[str] = None
    bmi_category: Optional[str] = None
    glucose_category: Optional[str] = None
    age_hypertension: Optional[float] = None

@app.get("/")
def read_root():
    return {"message": "Stroke Prediction API"}

@app.post("/predict")
def predict_stroke_risk(patient_data: PatientData):
    """
    Predict stroke risk for a patient
    
    Returns:
        - Probability of stroke (0-1)
        - Risk category (Low/Medium/High)
        - Key contributing factors
    """
    try:
        # Convert input data to DataFrame
        input_data = patient_data.dict()
        df = pd.DataFrame([input_data])
        
        # Calculate feature-engineered fields if not provided
        if 'age_group' not in input_data or input_data['age_group'] is None:
            age_bins = [0, 50, 80, 120]
            age_labels = ['Young adult', 'Middle-aged', 'Very old']
            df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
        
        if 'bmi_category' not in input_data or input_data['bmi_category'] is None:
            bmi_bins = [0, 18.5, 25, 30, 35, 40, 100]
            bmi_labels = ['Underweight', 'Healthy Weight', 'Overweight', 
                         'Class 1 Obesity', 'Class 2 Obesity', 'Class 3 Obesity']
            df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)
        
        if 'glucose_category' not in input_data or input_data['glucose_category'] is None:
            glucose_bins = [0, 70, 85, 100, 110, 126, 140, 300]
            glucose_labels = ['Hypoglycemia', 'Low Normal', 'Normal', 'Elevated', 
                            'Pre-diabetic', 'Borderline Diabetic', 'Diabetic']
            df['glucose_category'] = pd.cut(df['avg_glucose_level'], bins=glucose_bins, 
                                          labels=glucose_labels, right=False)
        
        if 'age_hypertension' not in input_data or input_data['age_hypertension'] is None:
            df['age_hypertension'] = df['age'] * df['hypertension']
        
        # Make prediction
        probability = model.predict_proba(df)[0][1]
        
        # Determine risk category
        if probability < 0.3:
            risk = "Low"
        elif probability < 0.7:
            risk = "Medium"
        else:
            risk = "High"
        
        # Get top contributing factors
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model.best_estimator_.named_steps['classifier'], 'coef_'):
            importances = np.abs(model.best_estimator_.named_steps['classifier'].coef_[0])
        else:
            importances = None
        
        if importances is not None:
            feature_names = model.best_estimator_.named_steps['preprocessing'].get_feature_names_out()
            top_features = sorted(zip(feature_names, importances), 
                                key=lambda x: x[1], reverse=True)[:3]
            contributing_factors = [f[0] for f in top_features]
        else:
            contributing_factors = ["Feature importance not available"]
        
        return {
            "probability": float(probability),
            "risk_category": risk,
            "contributing_factors": contributing_factors
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model_info")
def get_model_info():
    """Return information about the deployed model"""
    try:
        return {
            "model_type": str(type(model.best_estimator_)),
            "model_params": model.best_params_,
            "threshold": 0.3,
            "metrics": {
                "recall": model.best_score_,
                "roc_auc": model.cv_results_['mean_test_roc_auc'][model.best_index_]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
