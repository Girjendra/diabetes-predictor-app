import numpy as np
import pickle
import streamlit as st
import sklearn


# Load the trained SVM model from file
loaded_model = pickle.load(open(r'C:\Users\shiva\Desktop\ML_Youtube\Deploy_models\diabetes_web_app\SVM_trained_model.sav','rb'))

# Function to make prediction
def diabetes_prediction(input_data):

    # Convert input data (strings) to float values
    input_data = [float(x) for x in input_data]

    # Convert list to numpy array
    input_data = np.asarray(input_data)

    # Reshape array for single instance prediction
    input_data = input_data.reshape(1, -1)

    # Make prediction using loaded model
    prediction = loaded_model.predict(input_data)

    # Return result based on prediction
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


# Main function for Streamlit app
def main():

    # App title
    st.title('Test for Diabetes')

    # Input fields for user data
    Pregnancies = st.text_input("No. of Pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood Pressure")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")

    # Variable to store result
    diagnosis = ''

    # Button to trigger prediction
    if st.button('Diabetes test result'):

        # Call prediction function with user input
        diagnosis = diabetes_prediction([
            Pregnancies, Glucose, BloodPressure,
            SkinThickness, Insulin, BMI,
            DiabetesPedigreeFunction, Age
        ])

    # Display result
    st.success(diagnosis)


# Run the app
if __name__ == '__main__':
    main()