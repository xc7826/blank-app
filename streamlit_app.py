import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('cancer_risk_model.pkl')

# Title of the app
st.title('Cancer Risk Classification')

# Input fields for 5 features
st.write('Enter the values for the 5 features:')
feature1 = st.number_input('Feature 1', value=0.0)
feature2 = st.number_input('Feature 2', value=0.0)
feature3 = st.number_input('Feature 3', value=0.0)
feature4 = st.number_input('Feature 4', value=0.0)
feature5 = st.number_input('Feature 5', value=0.0)

# Predict button
if st.button('Predict'):
    # Prepare input data as a numpy array
    input_data = np.array([feature1, feature2, feature3, feature4, feature5]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.write(f'Cancer Risk: {"High Risk" if prediction[0] == 1 else "Low Risk"}')
