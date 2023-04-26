"""
Created on Sun May  8 21:01:15 2022

@authors: Jeanne Mukakalisa and Vanessa Mukamanzi
"""

import numpy as np
import pickle
import streamlit as st

st.title('DIABETES PREDICTION')

loaded_logreModel = pickle.load(open('logreModel.pkl', 'rb'))

def Diagnosis(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_logreModel.predict(input_data_reshaped)

    if (prediction[0]==0):
        return 'You are not diabetic'
    else:
        return 'You are diabetic'

def main():
    st.write('Prediction Model')

    BMI = st.number_input("Enter your BMI")
    Insulin = st.number_input("Enter your insulin level", step=2)
    Glucose = st.number_input("Enter your glucose level", step=2)
    BloodPressure = st.number_input("Enter your blood pressure", step=2)
    Age = st.number_input("Enter your age", step=2)

    disease = ''

    if st.button('Predict'):
        disease = Diagnosis([Glucose, BloodPressure, Insulin, BMI, Age])
        st.write(disease)

if __name__ == '__main__':
     main()
