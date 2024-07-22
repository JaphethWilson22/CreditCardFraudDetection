from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load the model
load_path = '/app/credit_fraud_model1'

# Function to load the model with caching
@st.cache_resource
def load_model(load_path):
    clf_loaded = joblib.load(load_path) 
    return clf_loaded

# Load the model from the file
clf_loaded = load_model(load_path)


# Initialize FastAPI app
app = FastAPI()

# Define request model
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array([[transaction.V1, transaction.V2, transaction.V3, transaction.V4, transaction.V5,
                      transaction.V6, transaction.V7, transaction.V8, transaction.V9, transaction.V10,
                      transaction.V11, transaction.V12, transaction.V13, transaction.V14, transaction.V15,
                      transaction.V16, transaction.V17, transaction.V18, transaction.V19, transaction.V20,
                      transaction.V21, transaction.V22, transaction.V23, transaction.V24, transaction.V25,
                      transaction.V26, transaction.V27, transaction.V28, transaction.Amount]])
    
    prediction = clf_loaded.predict(data)
    confidence = clf_loaded.predict_proba(data)

    result = {
        "prediction": int(prediction[0]),
        "confidence": confidence.tolist()  # Converts numpy array to list for JSON serialization
    }

    return result
