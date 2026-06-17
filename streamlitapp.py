import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("Machine Learning Algorithm Dashboard")

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

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

    # Fill missing values
    df = df.fillna("Missing")

    # Encode every column
    encoders = {}

    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Force all columns numeric
    df = df.apply(pd.to_numeric)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Classification or Regression
    is_classification = y.nunique() <= 20

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    if st.button("Train Model"):

        try:

            if algorithm == "Linear Regression":

                if is_classification:
                    st.error(
                        "Linear Regression cannot be used for classification datasets."
                    )
                    st.stop()

                model = LinearRegression()

            elif algorithm == "Logistic Regression":

                if not is_classification:
                    st.error(
                        "Logistic Regression requires a classification dataset."
                    )
                    st.stop()

                model = LogisticRegression(max_iter=1000)

            elif algorithm == "Decision Tree":

                if not is_classification:
                    st.error(
                        "Decision Tree is configured for classification."
                    )
                    st.stop()

                model = DecisionTreeClassifier(random_state=42)

            elif algorithm == "Random Forest":

                if not is_classification:
                    st.error(
                        "Random Forest is configured for classification."
                    )
                    st.stop()

                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )

            elif algorithm == "SVM":

                if not is_classification:
                    st.error(
                        "SVM is configured for classification."
                    )
                    st.stop()

                model = SVC()

            elif algorithm == "Naive Bayes":

                if not is_classification:
                    st.error(
                        "Naive Bayes requires a classification dataset."
                    )
                    st.stop()

                model = GaussianNB()

            elif algorithm == "XGBoost":

                if is_classification:

                    model = XGBClassifier(
                        eval_metric="logloss",
                        random_state=42
                    )

                else:

                    model = XGBRegressor(
                        objective="reg:squarederror",
                        random_state=42
                    )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            if is_classification:

                score = accuracy_score(
                    y_test,
                    y_pred
                )

                st.success(
                    f"Accuracy: {score:.4f}"
                )

            else:

                score = r2_score(
                    y_test,
                    y_pred
                )

                st.success(
                    f"R² Score: {score:.4f}"
                )

            st.session_state["model"] = model
            st.session_state["features"] = X.columns.tolist()

        except Exception as e:

            st.error(f"Training Error: {e}")

    if "model" in st.session_state:

        st.subheader("Prediction")

        values = []

        for feature in st.session_state["features"]:

            value = st.number_input(
                feature,
                value=0.0
            )

            values.append(value)

        if st.button("Predict"):

            pred = st.session_state["model"].predict(
                [values]
            )

            st.success(
                f"Prediction: {pred[0]}"
            )
