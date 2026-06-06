# SmartLoan: Loan Approval & Risk Assessment System

An end-to-end Machine Learning project that predicts whether a loan application should be approved based on applicant information and financial history.

The project uses Logistic Regression, Decision Trees, and Random Forest Classifiers to analyze loan applications and assist in risk assessment.

---

##  Problem Statement

Banks receive thousands of loan applications and need to evaluate the risk associated with each applicant.

This project predicts loan approval status using historical applicant data and compares multiple machine learning models to identify the most effective solution.

---

## Features

* Exploratory Data Analysis (EDA)
* Data Cleaning & Preprocessing
* Feature Engineering
* Logistic Regression Model
* Decision Tree Classifier
* Random Forest Classifier
* Model Evaluation & Comparison
* Feature Importance Analysis
* Interactive Streamlit Dashboard
* Loan Approval Prediction Tool

---

##  Dataset

Loan Approval Prediction Dataset

Target Variable:

* Loan_Status

  * Y → Loan Approved
  * N → Loan Rejected

Features include:

* Gender
* Married
* Dependents
* Education
* Self Employed
* Applicant Income
* Coapplicant Income
* Loan Amount
* Credit History
* Property Area

---

##  Models Used

### Logistic Regression

Used as the baseline classification model.

### Decision Tree Classifier

Provides interpretable rule-based predictions and visual decision paths.

### Random Forest Classifier

Combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

##  Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Feature Importance

---

##  Streamlit Dashboard

The dashboard includes:

### Dashboard

* Total Applications
* Approval Rate
* Average Loan Amount
* Interactive Charts

### Loan Predictor

Users can enter applicant information and receive:

* Loan Approval Prediction
* Approval Probability
* Risk Assessment

### Feature Importance

Visualization of the most influential factors affecting loan approval decisions.

---

## Project Structure

SmartLoan/

├── app.py

├── train.csv

├── requirements.txt

├── notebooks/

│ ├── EDA_nb.ipynb

│ ├── preprocessing.ipynb

│ └── model.ipynb

├── artifacts/

│ ├── model.pkl

│ ├── scaler.pkl

│ ├── feature_columns.json

│ └── feature_importance.csv

├── plots/

└── README.md

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Plotly
* Streamlit

---

## 👩‍💻 Author

Ananya Choudhury
