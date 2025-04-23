import streamlit as st
import pickle
import pandas as pd

# Load the model and therapy dataset
model = pickle.load(open('svd_model_with_user_features.pkl', 'rb'))

# Load therapy dataset for display purposes
therapies_df = pd.read_csv('therapies.csv')  # Assuming this file exists with therapy details

# Mapping therapy_id to actual therapy names
therapy_names = dict(zip(therapies_df['id'], therapies_df['name']))

st.title("Therapy Recommendation Based on Patient History")

st.subheader("Enter Patient History")
age = st.number_input("Age", min_value=0, max_value=120)
gender = st.selectbox("Gender", ["male", "female", "other"])

# Input for conditions (one-hot encoded as per your model)
conditions = st.multiselect("Known Conditions", ["Condition 1", "Condition 2", "Condition 3"])

# Convert gender to encoded form
gender_encoded = {'male': 0, 'female': 1, 'other': 2}[gender]

# One-hot encode conditions input
conditions_encoded = [1 if condition in conditions else 0 for condition in ["Condition 1", "Condition 2", "Condition 3"]]

# Combine features: age, gender, and conditions
user_input_features = [age, gender_encoded] + conditions_encoded

if st.button("Recommend Therapies"):
    # Generate predictions using the model (assuming it can handle this feature format)
    predictions = model.predict(user_input_features)  # This will depend on your retrained model

    # Display the top recommendations
    st.subheader("Top Therapy Recommendations")
    for therapy_id, score in predictions:
        therapy_name = therapy_names.get(therapy_id, "Unknown Therapy")
        st.write(f"**{therapy_name}** - Predicted Success: {score:.2f}")
