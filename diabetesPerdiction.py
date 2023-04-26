"""
@authors: Jeanne Mukakalisa and Vanessa Mukamanzi
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
def main():
    # giving a title
    st.title('DIABETES PREDICTION WEB APP')

    # getting the input data from the user
    try:
        Pregnancies = int(st.text_input('Number of Pregnancies'))
        Glucose = int(st.text_input('Glucose Level'))
        BloodPressure = int(st.text_input('Blood Pressure value'))
        SkinThickness = int(st.text_input('Skin Thickness value'))
        Insulin = int(st.text_input('Insulin Level'))
        BMI = float(st.text_input('BMI value'))
        DiabetesPedigreeFunction = float(st.text_input('Diabetes Pedigree Function value'))
        Age = int(st.text_input('Age of the Person'))
    except ValueError:
        st.warning('Please enter valid numerical values for all input fields.')
        return

    # code for Prediction
    diagnosis = ''
    # creating a button for Prediction
    if st.button('Prediction Results'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()
