import streamlit as st
import pickle
import random

# Load model and therapies
with open("svd_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("therapies_list.pkl", "rb") as f:
    all_therapies = pickle.load(f)

st.title("Therapy Recommendation System")

# Get user input
user_age = st.number_input("Age", min_value=0, max_value=120, value=30)
user_gender = st.selectbox("Gender", ["male", "female", "other"])
user_blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
user_conditions = st.text_area("Known Conditions (comma separated IDs)", "")
user_tried_therapies = st.text_area("Therapies Already Tried (comma separated IDs)", "")

if st.button("Recommend Therapies"):
    # Generate a pseudo user ID (can be randomized or based on hash)
    pseudo_user_id = str(random.randint(100000, 999999))

    tried_therapies = set(t.strip() for t in user_tried_therapies.split(",") if t.strip())
    recommendations = []

    for therapy in all_therapies:
        if therapy not in tried_therapies:
            prediction = model.predict(pseudo_user_id, therapy)
            recommendations.append((therapy, prediction.est))

    # Sort by predicted score
    recommendations.sort(key=lambda x: x[1], reverse=True)

    st.subheader("Top Recommended Therapies:")
    for therapy, score in recommendations[:5]:
        st.write(f"{therapy}: Predicted Success = {round(score, 2)}")
