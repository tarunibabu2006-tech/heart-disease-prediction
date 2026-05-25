
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
# Make sure 'model.pkl' and 'scaler.pkl' are in the same directory as app.py
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Error: model.pkl or scaler.pkl not found. Please ensure they are in the same directory as app.py")
    st.stop()

# Define the feature names in the correct order as used during training
feature_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Streamlit App Title
st.title('Heart Disease Prediction App')

st.write("Enter the patient's details to predict the likelihood of heart disease.")

# Input fields for patient data
# Use st.columns for better layout

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', options=[('Male', 1), ('Female', 0)], format_func=lambda x: str(x[0]))
    cp = st.selectbox('Chest Pain Type (CP)', options=[(1, 1), (2, 2), (3, 3), (4, 4)], format_func=lambda x: f'Type {x[0]}')
    trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)
    chol = st.number_input('Serum Cholestoral in mg/dl (chol)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[('True', 1), ('False', 0)], format_func=lambda x: str(x[0]))

with col2:
    restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', options=[(0, 0), (1, 1), (2, 2)], format_func=lambda x: f'Result {x[0]}')
    thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina (exang)', options=[('Yes', 1), ('No', 0)], format_func=lambda x: str(x[0]))
    oldpeak = st.number_input('ST depression induced by exercise relative to rest (oldpeak)', min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox('Slope of the peak exercise ST segment (slope)', options=[(1, 1), (2, 2), (3, 3)], format_func=lambda x: f'Slope {x[0]}')
    ca = st.selectbox('Number of major vessels (0-3) colored by flourosopy (ca)', options=[(0, 0), (1, 1), (2, 2), (3, 3)], format_func=lambda x: str(x[0]))
    thal = st.selectbox('Thalassemia (thal)', options=[(3, 3), (6, 6), (7, 7)], format_func=lambda x: f'Type {x[0]}')

# Collect all inputs into a dictionary
input_data = {
    'age': age,
    'sex': sex[1], # Get the numerical value
    'cp': cp[1],
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs[1],
    'restecg': restecg[1],
    'thalach': thalach,
    'exang': exang[1],
    'oldpeak': oldpeak,
    'slope': slope[1],
    'ca': ca[1],
    'thal': thal[1]
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

# Scale the input data
scaled_input = scaler.transform(input_df)

# Make prediction
if st.button('Predict Heart Disease'):
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'The model predicts: **Heart Disease Detected** (Probability: {prediction_proba[0][1]:.2f})')
    else:
        st.success(f'The model predicts: **No Heart Disease** (Probability: {prediction_proba[0][0]:.2f})')

    st.write("--- Feature Values Provided ---")
    st.write(input_df)
