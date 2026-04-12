import streamlit as st
import pandas as pd

st.title("Heart Disease Prediction AI")
st.sidebar.header("Patient Data")

def user_input_features():
    age = st.sidebar.slider('Age', 10, 100, 30)
    chol = st.sidebar.slider('Cholesterol', 100, 500, 200)
    return pd.DataFrame({'age': [age], 'chol': [chol]})

df = user_input_features()
st.write("### Input Parameters", df)
st.write("---")
st.info("Connect a trained .pkl model to see real-time predictions!")
