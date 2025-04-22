import pandas as pd
import numpy as np
import random
import gym
from gym import spaces
from sklearn.preprocessing import StandardScaler

# 1. Load and scale the data
df = pd.read_csv("Diabetes dataset.csv")
features = df.drop(columns=['Outcome']).values
labels = df['Outcome'].values 

# standardization 
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Discretize the features
def discretize_state(state, bins):
    """Convert continuous state into a discrete tuple."""
    state_disc = []
    for i in range(len(state)):
        state_disc.append(np.digitize(state[i], bins[i]))
    return tuple(state_disc)

# Create bins for discretization
n_bins = 8  
bins = [np.linspace(-3, 3, n_bins) for _ in range(features.shape[1])]

# 3. Custom Gym Environment
class DiabetesEnv(gym.Env):
    def __init__(self, features, labels):
        super(DiabetesEnv, self).__init__()
        self.features = features
        self.labels = labels
        self.index = 0
        self.state = None
        self.action_space = spaces.Discrete(3)  # Diet, Medication, Exercise
        self.observation_space = spaces.Box(low=-3, high=3, shape=(features.shape[1],), dtype=np.float32)

    def reset(self):
        self.index = random.randint(0, len(self.features) - 1)
        self.state = self.features[self.index]
        return self.state

    def step(self, action):
        original_risk = self.labels[self.index]
        success_probs = [0.4, 0.6, 0.5]
        improved = np.random.rand() < success_probs[action]
        reward = 100 if (improved and original_risk == 1) else (10 if improved else -10)
        done = True
        return self.state, reward, done, {}

# 4. Q-Learning Agent
env = DiabetesEnv(features, labels)
q_table = {}

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.1
episodes = 5000

rewards_history = []

for episode in range(episodes):
    state = env.reset()
    state_disc = discretize_state(state, bins)
    total_reward = 0

    if state_disc not in q_table:
        q_table[state_disc] = np.zeros(env.action_space.n)

    if np.random.rand() < epsilon:
        action = np.random.randint(env.action_space.n)
    else:
        action = np.argmax(q_table[state_disc])

    next_state, reward, done, _ = env.step(action)
    next_state_disc = discretize_state(next_state, bins)

    if next_state_disc not in q_table:
        q_table[next_state_disc] = np.zeros(env.action_space.n)

    best_next_q = np.max(q_table[next_state_disc])
    q_table[state_disc][action] += alpha * (reward + gamma * best_next_q - q_table[state_disc][action])

    rewards_history.append(reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# 5. Example: Print average reward
print("Average Reward over last 100 episodes:", np.mean(rewards_history[-100:]))

import matplotlib.pyplot as plt

plt.plot(rewards_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Over Episodes")
plt.grid(True)
plt.show()
