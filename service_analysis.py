import pandas as pd

df = pd.read_csv("data/churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()

services = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

for service in services:
    print("\n" + "="*50)
    print(service)
    print("="*50)

    result = pd.crosstab(
        df[service],
        df["Churn"],
        normalize="index"
    ) * 100

    print(result)