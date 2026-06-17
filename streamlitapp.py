import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Regression
from sklearn.linear_model import LinearRegression

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# XGBoost
from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(page_title="ML Algorithm Comparison App")

st.title("Machine Learning Algorithm Comparison Dashboard")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    algorithm = st.selectbox(
        "Select Algorithm",
        [
            "Linear Regression",
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "SVM",
            "Naive Bayes",
            "XGBoost"
        ]
    )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical columns
    le_dict = {}

    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

    # Encode target if classification
    if y.dtype == "object":
        y_le = LabelEncoder()
        y = y_le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    if st.button("Train Model"):

        if algorithm == "Linear Regression":

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = r2_score(y_test, y_pred)

            st.success(f"R² Score: {score:.4f}")

        elif algorithm == "Logistic Regression":

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = accuracy_score(y_test, y_pred)

            st.success(f"Accuracy: {score:.4f}")

        elif algorithm == "Decision Tree":

            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = accuracy_score(y_test, y_pred)

            st.success(f"Accuracy: {score:.4f}")

        elif algorithm == "Random Forest":

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = accuracy_score(y_test, y_pred)

            st.success(f"Accuracy: {score:.4f}")

        elif algorithm == "SVM":

            model = SVC()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = accuracy_score(y_test, y_pred)

            st.success(f"Accuracy: {score:.4f}")

        elif algorithm == "Naive Bayes":

            model = GaussianNB()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = accuracy_score(y_test, y_pred)

            st.success(f"Accuracy: {score:.4f}")

        elif algorithm == "XGBoost":

            if len(np.unique(y)) > 10:

                model = XGBRegressor()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                score = r2_score(y_test, y_pred)

                st.success(f"R² Score: {score:.4f}")

            else:

                model = XGBClassifier()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                score = accuracy_score(y_test, y_pred)

                st.success(f"Accuracy: {score:.4f}")

        st.session_state["model"] = model
        st.session_state["features"] = X.columns.tolist()

    if "model" in st.session_state:

        st.subheader("Prediction")

        feature_values = []

        for col in st.session_state["features"]:
            val = st.number_input(
                f"{col}",
                value=0.0
            )
            feature_values.append(val)

        if st.button("Predict"):

            prediction = st.session_state["model"].predict(
                [feature_values]
            )

            st.success(
                f"Prediction: {prediction[0]}"
            )
