import json 
import pandas as pd 

with open ("datasetB_sample.json", "r") as f:
    data = json.load(f)
'''
print(data.keys())
# View some conditions
print("Conditions Sample:", data['Conditions'][:2])

# View some patients
print("Patients Sample:", data['Patients'][:2])

# View some therapies
print("Therapies Sample:", data['Therapies'][:2])
'''


# extracting the patients trial data 
interaction_data = []

for patient in data['Patients']:
    patient_id = patient['id']
    trials = patient.get('trials', [])

    for trial in trials:
        therapy_id = trial['therapy']
        success = trial['successful']

        if patient_id is not None and therapy_id and success is not None:
            interaction_data.append([patient_id, therapy_id, success])

# converting those into a pandas data frame
df = pd.DataFrame(interaction_data, columns =['patient_id', 'therapy_id', 'success'])

print(df.head())
print(df.info())

# print(f"Total patients: {len(data['Patients'])}")

