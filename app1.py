import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import os


# Correct path to the model file
load_path = 'credit_fraud_model1'

# Function to load the model with caching
@st.cache_resource
def load_model(load_path):
    clf_loaded = joblib.load(load_path) 
    return clf_loaded
# Load the model from the file
clf_loaded = load_model(load_path)


# Initialize counts
fraudulent_count = 0
legitimate_count = 0



# Function to Predict transactions from CSV file
def predict_transactions(df, clf_loaded):
    # Check for missing values and handle them
    if df.isnull().values.any():
        st.warning("Warning: Missing values detected. Filling missing values with 0.")
        df = df.fillna(0)  # Fill missing values with 0

    # Extract features for prediction
    features = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]

    # Make predictions in batch
    predictions = clf_loaded.predict(features)

    # Count the number of fraudulent and legitimate transactions
    fraudulent_count = (predictions == 1).sum()
    legitimate_count = (predictions == 0).sum()

    # Display the counts
    if legitimate_count == 0:
        st.success(f"Number of legitimate transactions: **{legitimate_count}** 	:sos: ")
    elif legitimate_count < fraudulent_count:
        st.success(f"Number of legitimate transactions: **{legitimate_count}** There's a Problem! :rotating_light: ")    
    else:
        st.success(f"Number of legitimate transactions: **{legitimate_count}** :white_check_mark: ")
        
    if fraudulent_count > 0:
        st.error(f"Number of fraudulent transactions: **{fraudulent_count}** ")
    else:
        st.error(f"Number of fraudulent transactions: **{fraudulent_count}** :raised_hands: ")
    
    #Display the pie chart in Streamlit if Visualize is checked
    if visualize:
        try:
            labels = 'Fraudulent', 'Legitimate'
            sizes = [fraudulent_count, legitimate_count]
            colors = ['red', 'green']
            explode = (0.1, 0)  # explode the 1st slice (Fraudulent)
    
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
            st.pyplot(fig1)
                

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("File has not been read yet!")



#SideBar
st.sidebar.header("Hello, Administrator!")
uploaded_file = st.sidebar.file_uploader("Upload CSV file for Analysis", type=['csv'])


# Main App Code

st.title("Credit Card Fraud Detection")


input_df = st.text_input("*Enter All Required Features' values*")
input_df_splited = input_df.split(',')

# Validate and sanitize user input
if st.button('Predict'):
    if len(input_df_splited) == 29:
        try:
            features = np.asarray(input_df_splited, dtype=np.float64)
            prediction = clf_loaded.predict(features.reshape(1, -1))
        
            if prediction[0] == 1:
                st.error(f"Warning: This transaction is predicted to be **Fraudulent** with a confidence score of **{clf_loaded.predict_proba(features.reshape(1, -1))[0][1]:.2f}**.")
            else:
                st.success(f"This transaction is predicted to be **Legitimate** with a confidence score of **{clf_loaded.predict_proba(features.reshape(1, -1))[0][0]:.2f}**.")
        except ValueError:
            st.error("Error: Invalid input format. Please enter numeric values.")
    else:
        st.error("Error: Please enter all 29 required features.")
        
        
# Add an option to upload a CSV file


show_data = st.checkbox("Show Uploaded Data")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
            
        # Ensure the Amount column exists and is numeric
        if 'Amount' in df.columns and pd.api.types.is_numeric_dtype(df['Amount']):
            # Amount slider filter
            min_amount = df['Amount'].min()
            max_amount = df['Amount'].max()
            if show_data:
                amount_range = st.sidebar.slider("Select Amount Range", float(min_amount), float(max_amount), (float(min_amount), float(max_amount)))

                # Filter by amount range
                df = df[(df['Amount'] >= amount_range[0]) & (df['Amount'] <= amount_range[1])]
                
            
            
            # Convert the filtered DataFrame to CSV
            csv = df.to_csv().encode('utf-8')

            # Create a download button
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='filtered_data.csv',
                mime='text/csv'
            )

        else:
            st.error("The 'Amount' column is missing or not numeric in the uploaded CSV file.")



        
        visualize = st.sidebar.checkbox("Visualize")
        

        
        # Predict and count fraudulent and legitimate transactions
        if st.sidebar.button('Predict file'):
            
            predict_transactions(df, clf_loaded)

    except Exception as e:
        st.error(f"Error reading file: {e}")
        

# Show data if checkbox is checked and file is uploaded
if show_data:
    try:
        st.write(df) # Dsiplays the dataframe
    except Exception as e:
        st.error(f"Error reading file: {e}")
    
if uploaded_file is None and show_data:
    st.error("Please upload your **CSV file!** in the **Sidebar**. The file should include columns like 'V1' to 'V28' and 'Amount'.")
    
    
