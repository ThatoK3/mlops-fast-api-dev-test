import random
import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()

# Constants
N = 5000
genders = ["Male", "Female"]
smoking_status = ["never smoked", "smokes", "formerly smoked"]
provinces = [
    "Eastern Cape", "Free State", "Gauteng", "KwaZulu-Natal", 
    "Limpopo", "Mpumalanga", "Northern Cape", "North West", "Western Cape"
]

# Functions for realistic values
def generate_glucose(age, hypertension, heart_disease):
    base = 90 + (0.5 * hypertension + 0.7 * heart_disease) * 30
    noise = np.random.normal(0, 15)
    return max(70, min(300, base + noise + (0.1 * age)))

def generate_bmi(age, smoking):
    base = random.uniform(18, 35)  # normal range
    if smoking == "smokes":
        base -= random.uniform(0, 2)  # smokers sometimes lower BMI
    if age > 50:
        base += random.uniform(0, 3)  # age effect
    return round(base + np.random.normal(0, 2), 1)

# Generate dataset
data = []
for _ in range(N):
    gender = random.choice(genders)
    age = random.randint(18, 85)
    hypertension = random.choice([0, 1])
    heart_disease = random.choice([0, 1])
    smoking = random.choice(smoking_status)
    
    glucose = round(generate_glucose(age, hypertension, heart_disease), 1)
    bmi = generate_bmi(age, smoking)
    
    row = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": smoking,
        "name": fake.name(),
        "country": "South Africa",
        "province": random.choice(provinces)
    }
    data.append(row)

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("synthetic_health_data.csv", index=False)

print("✅ synthetic_health_data.csv generated with", len(df), "rows")
print(df.head())

