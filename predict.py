import pandas as pd
import joblib

# Load Saved Files
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")

print("Model Loaded Successfully!")
print("Total Features:", len(columns))

# Create Sample Customer
sample_data = pd.DataFrame(
    [[0] * len(columns)],
    columns=columns
)

# Scale Data
sample_scaled = scaler.transform(sample_data)

# Predict
prediction = model.predict(sample_scaled)

print("\nPrediction:")
print(prediction)

if prediction[0] == 1:
    print("Customer is likely to Churn")
else:
    print("Customer is likely to Stay")