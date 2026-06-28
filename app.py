from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Saved Files
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction_text = ""

    if request.method == "POST":

        # Create empty row
        data = pd.DataFrame(
            [[0] * len(columns)],
            columns=columns
        )

        # Numerical Features
        data["SeniorCitizen"] = int(request.form["SeniorCitizen"])
        data["tenure"] = int(request.form["tenure"])
        data["MonthlyCharges"] = float(request.form["MonthlyCharges"])
        data["TotalCharges"] = float(request.form["TotalCharges"])

        # Gender
        if request.form["gender"] == "Male":
            data["gender_Male"] = 1

        # Partner
        if request.form["Partner"] == "Yes":
            data["Partner_Yes"] = 1

        # Dependents
        if request.form["Dependents"] == "Yes":
            data["Dependents_Yes"] = 1

        # Scale
        scaled_data = scaler.transform(data)

        # Predict
        prediction = model.predict(scaled_data)[0]

        if prediction == 1:
            prediction_text = "Customer is likely to Churn"
        else:
            prediction_text = "Customer is likely to Stay"

    return render_template(
        "index.html",
        prediction_text=prediction_text
    )


if __name__ == "__main__":
    app.run(debug=True)