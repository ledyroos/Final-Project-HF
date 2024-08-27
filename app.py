import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p

# Load the model and scaler
model_path = 'model_xgb_reg.pkl'
scaler_path = 'scaler.pkl'  # Assume you've saved the scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Box-Cox lambda value
lam = 0.20

# Function to preprocess the input data
def preprocess_input(data):
    encoders = {
        'cut': LabelEncoder().fit(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']),
        'color': LabelEncoder().fit(['D', 'E', 'F', 'G', 'H', 'I', 'J']),
        'clarity': LabelEncoder().fit(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    }

    data['cut'] = encoders['cut'].transform(data['cut'])
    data['color'] = encoders['color'].transform(data['color'])
    data['clarity'] = encoders['clarity'].transform(data['clarity'])

    # Apply Box-Cox transformation
    data['carat'] = boxcox1p(data['carat'], lam)
    data['table'] = boxcox1p(data['table'], lam)
    data['y'] = boxcox1p(data['y'], lam)
    data['z'] = boxcox1p(data['z'], lam)

    return data

# Dropdown for navigation
page = st.sidebar.selectbox("Navigate", ["Home", "Diamond Price Prediction"])

if page == "Home":
    st.title("Welcome to the Diamond Price Prediction App")
    st.image("diamond.jpg", caption="Shining Bright", use_column_width=True)
    st.write("This app helps you predict the price of diamonds based on their features.")
    st.write("Use the sidebar to navigate to the prediction page.")

elif page == "Diamond Price Prediction":
    st.title("Diamond Price Prediction")

    # User input fields
    carat = st.number_input("Carat", min_value=0.0, step=0.01)
    cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = st.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    depth = st.number_input("Depth", min_value=0.0, step=0.1)
    table = st.number_input("Table", min_value=0.0, step=0.1)
    x = st.number_input("X dimension", min_value=0.0, step=0.01)
    y = st.number_input("Y dimension", min_value=0.0, step=0.01)
    z = st.number_input("Z dimension", min_value=0.0, step=0.01)

    # Create a data dictionary
    data = {
        'carat': [carat],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    }

    # Convert to DataFrame
    data_df = pd.DataFrame(data)

    # Preprocess the data
    preprocessed_data = preprocess_input(data_df)

    # Scaling the data
    preprocessed_data_scaled = scaler.transform(preprocessed_data)

    # Predict using the model
    prediction = model.predict(preprocessed_data_scaled)

    # Apply inverse log transformation to the prediction
    prediction_exp = np.expm1(prediction)

    # Show the prediction
    st.write(f"Predicted Price: ${prediction_exp[0]:,.2f}")
