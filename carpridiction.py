import streamlit as st
import pickle as pk
import numpy as np
import pandas as pd
import xgboost as xgb

# Load the trained model from pickle
model = pk.load(open('xgb_saved.pkl','rb'))

# Streamlit app title
st.title("Car Price Prediction")

# User input fields
year = st.number_input("Year of Manufacture", min_value=1900, max_value=2024, value=2015)
name = st.selectbox("Name of the car", ['Ford', 'City', 'Mercides', 'toyata', 'cultus'])
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000)
fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])
owner = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=20.0)
engine = st.number_input("Engine Capacity (CC)", min_value=500, max_value=5000, value=1500)
max_power = st.number_input("Max power  (CC)", min_value=500, max_value=5000, value=1500)


# Create input dataframe
input_data = {
    'year': year,
    'name': name,
    'km_driven': km_driven,
    'fuel': fuel,
    'seller_type': seller_type,
    'transmission': transmission,
    'owner': owner,
    'mileage': mileage,
    'engine': engine,
    'max_power': max_power
}

input_df = pd.DataFrame([input_data])
input_df = pd.get_dummies(input_df, drop_first=True)


# Make a prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    st.write(f"The estimated selling price is: ₹{prediction[0]:,.2f}")
