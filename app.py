import streamlit as st
import numpy as np
import joblib
from xgboost import XGBClassifier

# Load your saved model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Online Payment Fraud Detection", layout="centered")

st.title("💳 Online Payment Fraud Detection")
st.write("Enter transaction details below to check if it is **Fraudulent or Genuine**.")

# Example input features (replace with your dataset’s important columns)
col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step (Time step)", min_value=0, value=10)
    amount = st.number_input("Transaction Amount", min_value=0.0, value=500.0)
    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=1000.0)
    newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=500.0)
    oldbalanceDest = st.number_input("Old Balance (Dest) ",min_value=0.0,value=500.0)
    newbalanceDest = st.number_input(" New Balance (Dest)",min_value=0.0,value=5000.0) 

with col2:
    isFlaggedFraud = st.selectbox("Is Flagged Fraud",[0,1])
    CASH_OUT = st.number_input("Cashout",min_value=0.0,value= 0.0,step=1.0)
    DEBIT = st.number_input("Debit",min_value=0.0,value= 0.0,step=1.0)
    PAYMENT = st.number_input("Payment",min_value=0.0,value= 0.0,step=1.0)
    TRANSFER = st.number_input("Transfer",min_value=0.0,value= 0.0,step=1.0)

# Collect inputs into array (order must match training dataset)
input_data = np.array([[step, amount, oldbalanceOrg, newbalanceOrig,oldbalanceDest,newbalanceDest,isFlaggedFraud,CASH_OUT,DEBIT,PAYMENT,TRANSFER]])

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {proba:.2f})")
        
    else:
        st.success(f"✅ Genuine Transaction (Probability: {proba:.2f})")
        
