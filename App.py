import streamlit as st
import pandas as pd
import joblib

# Load the trained components
scaler = joblib.load("scaler.pkl")
model = joblib.load("logistic_model.pkl")
selected_features = joblib.load("selected_features.pkl")  # Load selected feature names

# Feature descriptions
feature_descriptions = {
    "cp": (
        "Chest Pain Type (cp):"
        " - 0: Typical Angina"
        " - 1: Atypical Angina"
        " - 2: Non-Anginal Pain"
        " - 3: Asymptomatic\n"
        "Acceptable range: 0 to 3."
    ),
    "oldpeak": (
        "ST Depression (oldpeak):\n"
        "The amount of ST depression induced by exercise relative to rest.\n"
        "Indicates abnormal heart activity.\n"
        "Acceptable range: 0.0 to approximately 6.2 (observed max)."
    ),
    "ca": (
        "Number of Major Vessels Colored (ca):\n"
        "Ranges from 0 to 4.\n"
        "Represents the number of major vessels visualized by fluoroscopy.\n"
        "Acceptable range: 0 to 4."
    )
}


# Streamlit app title and description
st.title("Heart Disease Prediction App")
st.write("Input your data to predict the likelihood of heart disease.")

# Input fields with feature descriptions
user_input = []
for feature in selected_features:
    # Display the description of the feature
    st.write(f"**{feature_descriptions[feature]}**")
    
    # Input field for the feature
    value = st.number_input(f"Enter {feature}", value=0.0)
    user_input.append(value)

# Convert user input to DataFrame
input_data = pd.DataFrame([user_input], columns=selected_features)

# Preprocess the input data
try:
    # Transform the input data using the scaler
    scaled_data = scaler.transform(input_data)
except Exception as e:
    st.error(f"Error during preprocessing: {e}")
    st.stop()

# Prediction button
if st.button("Predict"):
    try:
        # Make predictions using the loaded model
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)

        # Display results
        if prediction[0] == 1:
            st.success("The model predicts that you are at risk for heart disease.")
        else:
            st.success("The model predicts that you are not at risk for heart disease.")

        st.write(f"Probability of heart disease: {prediction_proba[0][1]:.2f}")
        st.write(f"Probability of no heart disease: {prediction_proba[0][0]:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")


