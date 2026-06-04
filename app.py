import streamlit as st
import pandas as pd
import pickle
import json
import plotly.express as px

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="SmartLoan",
    page_icon="💰",
    layout="wide"
)

# =====================================
# LOAD FILES
# =====================================

model = pickle.load(
    open("artifacts/model.pkl", "rb")
)

scaler = pickle.load(
    open("artifacts/scaler.pkl", "rb")
)

feature_importance = pd.read_csv(
    "artifacts/feature_importance.csv"
)

with open(
    "artifacts/feature_columns.json",
    "r"
) as f:
    feature_columns = json.load(f)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("💰 SmartLoan")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Loan Predictor",
        "Feature Importance",
        "About"
    ]
)

# =====================================
# DASHBOARD
# =====================================

if page == "Dashboard":

    st.title(
        "💰 SmartLoan: Loan Approval & Risk Assessment"
    )

    st.markdown("---")

    df = pd.read_csv("train.csv")

    total_apps = len(df)

    approval_rate = (
        (df["Loan_Status"] == "Y")
        .mean()
        * 100
    )

    avg_income = int(
        df["ApplicantIncome"]
        .mean()
    )

    avg_loan = int(
        df["LoanAmount"]
        .mean()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Applications",
        total_apps
    )

    c2.metric(
        "Approval Rate",
        f"{approval_rate:.1f}%"
    )

    c3.metric(
        "Avg Income",
        avg_income
    )

    c4.metric(
        "Avg Loan Amount",
        avg_loan
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            df,
            names="Loan_Status",
            title="Loan Approval Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            df,
            x="Property_Area",
            color="Loan_Status",
            barmode="group",
            title="Property Area vs Approval"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# =====================================
# LOAN PREDICTOR
# =====================================

elif page == "Loan Predictor":

    st.title(
        "Loan Approval Predictor"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            [
                "Graduate",
                "Not Graduate"
            ]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

    with col2:

        income = st.number_input(
            "Applicant Income",
            0,
            100000,
            5000
        )

        co_income = st.number_input(
            "Coapplicant Income",
            0,
            50000,
            0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            0,
            1000,
            120
        )

        credit_history = st.selectbox(
            "Credit History",
            [0, 1]
        )

        property_area = st.selectbox(
            "Property Area",
            [
                "Urban",
                "Semiurban",
                "Rural"
            ]
        )

    if st.button(
        "Predict Loan Status"
    ):

        input_df = pd.DataFrame(
            [[0] * len(feature_columns)],
            columns=feature_columns
        )

        # --------------------------------
        # NUMERIC FEATURES
        # --------------------------------

        if "ApplicantIncome" in input_df.columns:
            input_df[
                "ApplicantIncome"
            ] = income

        if "CoapplicantIncome" in input_df.columns:
            input_df[
                "CoapplicantIncome"
            ] = co_income

        if "LoanAmount" in input_df.columns:
            input_df[
                "LoanAmount"
            ] = loan_amount

        if "Credit_History" in input_df.columns:
            input_df[
                "Credit_History"
            ] = credit_history

        # --------------------------------
        # ONE HOT FEATURES
        # --------------------------------

        cols_to_set = [
            f"Gender_{gender}",
            f"Married_{married}",
            f"Dependents_{dependents}",
            f"Education_{education}",
            f"Self_Employed_{self_employed}",
            f"Property_Area_{property_area}"
        ]

        for col in cols_to_set:

            if col in input_df.columns:

                input_df[col] = 1

        # --------------------------------
        # SCALE
        # --------------------------------

        scaled = scaler.transform(
            input_df
        )

        prediction = model.predict(
            scaled
        )[0]

        probability = (
            model.predict_proba(
                scaled
            )[0][1]
        )

        st.markdown("---")

        st.subheader(
            "Prediction Results"
        )

        st.metric(
            "Approval Probability",
            f"{probability:.2%}"
        )

        st.progress(
            float(probability)
        )

        if prediction == 1:

            st.success(
                "✅ Loan Approved"
            )

        else:

            st.error(
                "❌ Loan Rejected"
            )

# =====================================
# FEATURE IMPORTANCE
# =====================================

elif page == "Feature Importance":

    st.title(
        "Feature Importance"
    )

    st.markdown("---")

    top_features = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features Influencing Loan Approval"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# ABOUT
# =====================================

elif page == "About":

    st.title(
        "About Project"
    )

    st.markdown("---")

    st.markdown("""
### SmartLoan

AI-powered Loan Approval and Risk Assessment System.

### Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### Dataset

Loan Approval Prediction Dataset

### Developed By

Ananya Choudhury
""")