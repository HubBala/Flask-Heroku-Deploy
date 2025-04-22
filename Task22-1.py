import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle


# Load dataset
df = pd.read_csv("Diabetes dataset.csv")
features = df.drop(columns=["Outcome"]).values
labels = df["Outcome"].values

# Normalize data
scaler = StandardScaler()
features = scaler.fit_transform(features)

# save the scaler 
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Define treatment actions
actions = ["Diet", "Medication", "Exercise"]
n_actions = len(actions)

# Q-table: [states x actions] - simulate with state index
n_states = len(features)
q_table = np.zeros((n_states, n_actions))

# RL hyperparameters
alpha = 0.1        # Learning rate
gamma = 0.9        # Discount factor
epsilon = 1.0      # Exploration
epsilon_decay = 0.995
min_epsilon = 0.1
episodes = 500

# Reward rule
def calculate_reward(label, action):
    success_probs = [0.4, 0.6, 0.5]  # diet, med, ex
    success = np.random.rand() < success_probs[action]
    if success and label == 1:
        return 100
    elif success:
        return 10
    else:
        return -10

# Training loop
for episode in range(episodes):
    total_reward = 0
    for i in range(n_states):
        state = i
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(q_table[state])

        reward = calculate_reward(labels[i], action)
        next_state = state  # since we're using one-step episodes

        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# saving the Q-Reinforcement learning model

with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training complete.")

# Evaluation
print("\nRecommended actions for first 10 patients:")
for i in range(10):
    best_action = np.argmax(q_table[i])
    print(f"Patient {i}: Recommend -> {actions[best_action]}")
