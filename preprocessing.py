import pandas as pd
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("data/churn.csv")

# Data Cleaning
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()

# Remove customerID
df = df.drop("customerID", axis=1)

# Convert Churn to Numeric
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)

print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)