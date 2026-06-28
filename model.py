import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Dataset
df = pd.read_csv("data/churn.csv")

# Data Cleaning
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
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

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

print("Model Training Completed Successfully!")

# Make Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n" + "="*50)
print("DECISION TREE MODEL")
print("="*50)

# Train Decision Tree
dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

# Predictions
dt_pred = dt_model.predict(X_test)

# Accuracy
dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:")
print(dt_accuracy)

# Confusion Matrix
print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

# Classification Report
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_pred))

print("\n" + "="*50)
print("RANDOM FOREST MODEL")
print("="*50)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)

# Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

# Confusion Matrix
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# Classification Report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Save Logistic Regression Model
joblib.dump(model, "models/churn_model.pkl")

# Save Scaler
joblib.dump(scaler, "models/scaler.pkl")

# Save Column Names
joblib.dump(X.columns.tolist(), "models/columns.pkl")

print("\nModel, Scaler, and Columns Saved Successfully!")