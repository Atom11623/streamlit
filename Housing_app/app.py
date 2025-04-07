# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="California House Price Predictor", layout="centered")
st.title("🏠 California House Price Predictor")
st.markdown("Enter the housing details below to predict the median house value:")

# 🌊 Ocean proximity input and encoding
ocean_proximity = st.selectbox("Ocean Proximity", [
    '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'
])
ocean_categories = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
ocean_encoded = [1 if ocean_proximity == cat else 0 for cat in ocean_categories]

# 🧮 Numerical feature inputs
longitude = st.number_input("Longitude", -125.0, -113.0, value=-120.0)
latitude = st.number_input("Latitude", 32.0, 43.0, value=36.0)
housing_median_age = st.slider("Housing Median Age", 1, 52, 20)
total_rooms = st.number_input("Total Rooms", 2, 50000, value=2000)
total_bedrooms = st.number_input("Total Bedrooms", 1, 10000, value=400)
population = st.number_input("Population", 1, 50000, value=1000)
households = st.number_input("Households", 1, 10000, value=400)
median_income = st.number_input("Median Income (×$10,000)", 0.0, 20.0, value=3.0)

# 🔢 Combine inputs into array
input_data = np.array([[longitude, latitude, housing_median_age, total_rooms,
                        total_bedrooms, population, households, median_income]])

# Scale the numerical inputs
input_scaled = scaler.transform(input_data)

# Append encoded ocean proximity
final_input = np.concatenate([input_scaled[0], ocean_encoded]).reshape(1, -1)

# 🎯 Predict
if st.button("Predict Price"):
    prediction = model.predict(final_input)[0]
    st.subheader("💵 Predicted Median House Value:")
    st.success(f"${prediction:,.2f}")
