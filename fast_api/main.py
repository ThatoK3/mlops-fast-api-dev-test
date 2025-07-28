from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import Optional
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import json

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="Stroke Prediction API",
              description="API for predicting stroke risk based on health metrics",
              version="1.0")

# Database connection function
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        return connection
    except Error as e:
        raise RuntimeError(f"Error connecting to MySQL: {e}")

# Create predictions table if not exists
def init_db():
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            gender VARCHAR(10),
            age FLOAT,
            hypertension BOOLEAN,
            heart_disease BOOLEAN,
            avg_glucose_level FLOAT,
            bmi FLOAT,
            smoking_status VARCHAR(20),
            probability FLOAT,
            risk_category VARCHAR(10),
            contributing_factors JSON,
            prediction_data JSON
        )
        """
        cursor.execute(create_table_query)
        connection.commit()
    except Error as e:
        raise RuntimeError(f"Error initializing database: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# Initialize database on startup
init_db()

# Model loading with proper path resolution
model_path = Path(__file__).parent / os.getenv("SAVED_MODEL")
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

def save_prediction_to_db(data: dict, prediction_result: dict):
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        insert_query = """
        INSERT INTO predictions (
            gender, age, hypertension, heart_disease,
            avg_glucose_level, bmi, smoking_status,
            probability, risk_category, contributing_factors,
            prediction_data
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            data['gender'],
            data['age'],
            bool(data['hypertension']),
            bool(data['heart_disease']),
            data['avg_glucose_level'],
            data['bmi'],
            data['smoking_status'],
            prediction_result['probability'],
            prediction_result['risk_category'],
            json.dumps(prediction_result['contributing_factors']),
            json.dumps(data)
        ))
        
        connection.commit()
        return cursor.lastrowid
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

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
        - Prediction ID from database
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
        
        prediction_result = {
            "probability": float(probability),
            "risk_category": risk,
            "contributing_factors": contributing_factors
        }
        
        # Save to database
        prediction_id = save_prediction_to_db(input_data, prediction_result)
        prediction_result['prediction_id'] = prediction_id
        
        return prediction_result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predictions")
def get_predictions(limit: int = 10):
    """
    Retrieve stored predictions
    Parameters:
        - limit: Number of predictions to return (default: 10)
    """
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT %s"
        cursor.execute(query, (limit,))
        results = cursor.fetchall()
        
        # Convert JSON strings back to objects
        for row in results:
            if row['contributing_factors']:
                row['contributing_factors'] = json.loads(row['contributing_factors'])
            if row['prediction_data']:
                row['prediction_data'] = json.loads(row['prediction_data'])
        
        return results
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

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
