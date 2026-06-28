# Customer Churn Prediction System

## Project Overview

Customer Churn Prediction System is a Machine Learning project developed to predict whether a telecom customer is likely to leave the company (churn) or continue using the service.

The project uses customer demographic information, service subscriptions, contract details, and billing information to make churn predictions.

## Objectives

* Analyze customer behavior and churn patterns.
* Identify important factors influencing customer churn.
* Build and evaluate machine learning models.
* Deploy the model using Flask for real-time predictions.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Flask
* HTML
* CSS
* Joblib

## Machine Learning Models

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Feature Engineering
6. Model Building
7. Model Evaluation
8. Model Saving
9. Flask Web Application
10. Real-Time Prediction

## Dataset Information

Dataset contains telecom customer information such as:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Internet Service
* Contract Type
* Payment Method
* Monthly Charges
* Total Charges

Target Variable:

* Churn (Yes/No)

## Model Performance

### Logistic Regression

* Accuracy: 78.75%

### Decision Tree

* Accuracy: 72.57%

### Random Forest

* Accuracy: 78.46%

Logistic Regression achieved the best overall performance and was selected for deployment.

## Features

* Real-time churn prediction
* Interactive web interface
* Trained machine learning model
* Data preprocessing pipeline
* Model evaluation and comparison

## Future Improvements

* Add all customer service fields to the prediction form.
* Improve user interface design.
* Deploy application to cloud platforms.
* Add advanced machine learning models.

## Author

Suyash Ketkar
Python Developer & Machine Learning Enthusiast


Customer_Churn_Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── churn.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── src/
│   ├── churn.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── model.py
│   └── predict.py
│
├── templates/
│   └── index.html
│
└── static/
