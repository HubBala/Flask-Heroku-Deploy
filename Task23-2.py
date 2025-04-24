import json
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle

# Load your dataset
with open("datasetB_sample.json") as f:
    data = json.load(f)

# Prepare interaction data
interaction_data = []

for patient in data["Patients"]:
    patient_id = patient["id"]
    for trial in patient.get("trials", []):
        therapy_id = trial["therapy"]
        success = trial["successful"]
        interaction_data.append([str(patient_id), therapy_id, success])

df = pd.DataFrame(interaction_data, columns=["userID", "itemID", "rating"])

# Train SVD model
reader = Reader(rating_scale=(0, 100))
data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
trainset, _ = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)

# Save model
with open("svd_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save therapies list for use in UI
all_therapies = list(df["itemID"].unique())
with open("therapies_list.pkl", "wb") as f:
    pickle.dump(all_therapies, f)
