
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("linear_regression.pkl", "rb"))

st.title("Linear Regression Prediction App")

n_features = model.n_features_in_

inputs = []

for i in range(n_features):
    value = st.text_input(f"Feature {i+1}", "0")
    inputs.append(float(value))

if st.button("Predict"):
    prediction = model.predict([inputs])
    st.success(f"Prediction: {prediction[0]}")
