import platform
print(platform.architecture())


from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd

# Load the data from your existing DataFrame
# Make sure df is already created as you've done
# df = pd.DataFrame(interaction_data, columns=['patient_id', 'therapy_id', 'success'])

# Define reader for Surprise (rating_scale from 0 to 100)
reader = Reader(rating_scale=(0, 100))
data = Dataset.load_from_df(df[['patient_id', 'therapy_id', 'success']], reader)

# Train/test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Build and train SVD model
model = SVD()
model.fit(trainset)

# Make predictions on testset (optional)
predictions = model.test(testset)

# Predict for a specific patient and therapy
example_patient_id = 0
all_therapies = df['therapy_id'].unique()

# Recommend top-N therapies the patient hasn't tried
def get_recommendations(patient_id, n=5):
    tried_therapies = df[df['patient_id'] == patient_id]['therapy_id'].unique()
    untried = [t for t in all_therapies if t not in tried_therapies]
    
    pred_scores = [(therapy_id, model.predict(patient_id, therapy_id).est) for therapy_id in untried]
    sorted_preds = sorted(pred_scores, key=lambda x: x[1], reverse=True)
    
    return sorted_preds[:n]

# Show top 5 therapy recommendations for patient 0
recommended = get_recommendations(0)
print("Top Therapy Recommendations for Patient 0:")
for therapy_id, score in recommended:
    print(f"Therapy: {therapy_id}, Predicted Success Score: {score:.2f}")
