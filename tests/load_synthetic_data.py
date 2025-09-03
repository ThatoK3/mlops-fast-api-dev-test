import pandas as pd
import requests
import json
import time
from tqdm import tqdm

# FastAPI endpoint URL
API_URL = "http://localhost:8000/predict"  # Adjust if your API is running elsewhere

def load_synthetic_data():
    """Load synthetic data from GitHub"""
    url = "https://raw.githubusercontent.com/ThatoK3/mlops-fast-api-dev-test/main/tests/synthetic_health_data.csv"
    print("Loading synthetic data from GitHub...")
    df = pd.read_csv(url)
    print(f"Loaded {len(df)} records")
    return df

def send_prediction_request(row):
    """Send a single prediction request to the FastAPI endpoint"""
    # Prepare the payload according to your PatientData model
    payload = {
        "gender": str(row['gender']),
        "age": float(row['age']),
        "hypertension": int(row['hypertension']),
        "heart_disease": int(row['heart_disease']),
        "avg_glucose_level": float(row['avg_glucose_level']),
        "bmi": float(row['bmi']),
        "smoking_status": str(row['smoking_status']),
        "name": str(row.get('name', 'Synthetic Patient')),
        "country": str(row.get('country', 'Unknown')),
        "province": str(row.get('province', 'Unknown'))
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return None

def main():
    # Load synthetic data
    df = load_synthetic_data()
    
    # Add missing columns if needed
    if 'name' not in df.columns:
        df['name'] = [f"Patient_{i}" for i in range(len(df))]
    if 'country' not in df.columns:
        df['country'] = 'Test Country'
    if 'province' not in df.columns:
        df['province'] = 'Test Province'
    
    print("Sample data:")
    print(df[['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 
              'bmi', 'smoking_status', 'name', 'country', 'province']].head().to_string())
    
    # Test connection first
    print("\nTesting API connection...")
    try:
        test_response = requests.get("http://localhost:8000/", timeout=10)
        if test_response.status_code == 200:
            print("✓ API is reachable")
        else:
            print("✗ API returned non-200 status")
            return
    except requests.exceptions.RequestException:
        print("✗ API is not reachable. Make sure FastAPI is running on localhost:8000")
        return
    
    # Send prediction requests
    successful_requests = 0
    failed_requests = 0
    results = []
    
    print("\nSending prediction requests...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        result = send_prediction_request(row)
        
        if result:
            successful_requests += 1
            results.append({
                'index': index,
                'status': 'success',
                'prediction_id': result.get('prediction_id'),
                'probability': result.get('probability'),
                'risk_category': result.get('risk_category')
            })
        else:
            failed_requests += 1
            results.append({
                'index': index,
                'status': 'failed'
            })
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total requests: {len(df)}")
    print(f"Successful: {successful_requests}")
    print(f"Failed: {failed_requests}")
    print(f"Success rate: {(successful_requests/len(df))*100:.1f}%")
    
    # Save results to file
    with open('prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to prediction_results.json")
    
    # Show some successful predictions
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        print("\nSample successful predictions:")
        for result in successful_results[:5]:
            print(f"ID: {result['prediction_id']}, "
                  f"Probability: {result['probability']:.3f}, "
                  f"Risk: {result['risk_category']}")

if __name__ == "__main__":
    main()
