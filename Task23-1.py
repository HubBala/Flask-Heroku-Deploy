import json
import pandas as pd
from surprise import Reader, SVD, Dataset
from surprise.model_selection import train_test_split
import pickle

# Load the dataset
with open("datasetB_sample.json", "r") as f:
    data = json.load(f)

# Extract therapy information directly from the JSON file
therapy_names = {therapy['id']: therapy['name'] for therapy in data['Therapies']}

# Extract interaction data for training the model
interaction_data = []

for patient in data['Patients']:
    patient_id = patient['id']
    trials = patient.get('trials', [])

    for trial in trials:
        therapy_id = trial['therapy']
        success = trial['successful']

        if patient_id is not None and therapy_id and success is not None:
            interaction_data.append([patient_id, therapy_id, success])

# Converting those into a pandas DataFrame
df = pd.DataFrame(interaction_data, columns=['patient_id', 'therapy_id', 'success'])

# Prepare data for the Surprise model
reader = Reader(rating_scale=(0, 100))
data = Dataset.load_from_df(df[['patient_id', 'therapy_id', 'success']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Fit the SVD model
model = SVD()
model.fit(trainset)

# Save the trained model
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the DataFrame as CSV for later use in the UI
df.to_csv('patient_therapy_success.csv', index=False)

# Save the therapy names dictionary for use in the UI
with open('therapy_names.pkl', 'wb') as f:
    pickle.dump(therapy_names, f)
