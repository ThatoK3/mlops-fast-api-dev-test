import pytest
import requests
import json
import os
from datetime import datetime
import csv

API_URL = "http://localhost:8000/predict"
RESULTS_DIR = "test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_CASES = [
    {
        "name": "high_risk_elderly",
        "args": {
            "gender": "Male",
            "age": 72,
            "hypertension": 1,
            "heart_disease": 1,
            "avg_glucose_level": 210,
            "bmi": 34,
            "smoking_status": "smokes"
        },
        "expected_risk": "High"
    },
    {
        "name": "medium_risk_middle_aged",
        "args": {
            "gender": "Female",
            "age": 55,
            "hypertension": 0,
            "heart_disease": 0,
            "avg_glucose_level": 160,
            "bmi": 28,
            "smoking_status": "formerly smoked"
        },
        "expected_risk": "Medium"
    },
    {
        "name": "low_risk_young",
        "args": {
            "gender": "Female",
            "age": 35,
            "hypertension": 0,
            "heart_disease": 0,
            "avg_glucose_level": 90,
            "bmi": 22,
            "smoking_status": "never smoked"
        },
        "expected_risk": "Low"
    }
]

def save_test_result(test_name, arguments, response, passed):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RESULTS_DIR}/{test_name}_{timestamp}.json"
    
    result = {
        "test_name": test_name,
        "timestamp": timestamp,
        "arguments": arguments,
        "response": response,
        "test_passed": passed
    }
    
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Also save to CSV for aggregate analysis
    csv_file = f"{RESULTS_DIR}/test_results.csv"
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "test_name", "arguments", "response", "passed"])
        writer.writerow([
            timestamp,
            test_name,
            json.dumps(arguments),
            json.dumps(response),
            passed
        ])

@pytest.mark.parametrize("test_case", TEST_CASES)
def test_api_prediction(test_case):
    response = requests.post(API_URL, json=test_case["args"])
    
    assert response.status_code == 200, f"API returned {response.status_code}"
    
    result = response.json()
    
    # Save raw response regardless of test outcome
    save_test_result(
        test_case["name"],
        test_case["args"],
        result,
        passed=result["risk_category"] == test_case["expected_risk"]
    )
    
    assert "probability" in result, "Response missing probability"
    assert "risk_category" in result, "Response missing risk_category"
    assert result["risk_category"] == test_case["expected_risk"], \
        f"Expected {test_case['expected_risk']} risk, got {result['risk_category']}"
