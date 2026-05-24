import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("sales_model.pkl")

# Page title
st.title("📈 Store Sales Prediction")

st.write("Enter store details below:")

# User inputs
advertising_cost = st.number_input("Advertising Cost", min_value=0.0)

store_size = st.number_input("Store Size", min_value=0.0)

customers = st.number_input("Number of Customers", min_value=0.0)

# Prediction button
if st.button("Predict Sales"):

    # Create dataframe
    data = pd.DataFrame(
        [[advertising_cost, store_size, customers]],
        columns=[
            "Advertising_Cost",
            "Store_Size",
            "Customers"
        ]
    )

    # Prediction
    prediction = model.predict(data)[0]

    # Display result
    st.success(f"Predicted Sales: {round(prediction, 2)}")

