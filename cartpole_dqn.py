import gymnasium as gym
import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt

# Create the environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]  # 4 features
action_size = env.action_space.n             # 2 actions: left or right

# Build the neural network
def build_model(state_size, action_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, input_shape=(state_size,), activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')  # Q-values for each action
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

# DQN parameters
model = build_model(state_size, action_size)
memory = deque(maxlen=2000)

gamma = 0.95            # discount factor
epsilon = 1.0           # exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 32

episodes = 200
scores = []

# Training loop
for e in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for time_t in range(500):
        # Choose action (epsilon-greedy)
        if np.random.rand() <= epsilon:
            action = random.randrange(action_size)
        else:
            q_values = model.predict(state[np.newaxis, :], verbose=0)
            action = np.argmax(q_values[0])

        # Take action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {e+1}/{episodes} - Score: {total_reward}, Epsilon: {epsilon:.2f}")
            break

    # Replay from memory to train
    if len(memory) >= batch_size:
        minibatch = random.sample(memory, batch_size)
        for s, a, r, s_next, done in minibatch:
            target = r
            if not done:
                target += gamma * np.amax(model.predict(s_next[np.newaxis, :], verbose=0)[0])
            target_f = model.predict(s[np.newaxis, :], verbose=0)
            target_f[0][a] = target
            model.fit(s[np.newaxis, :], target_f, epochs=1, verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    scores.append(total_reward)

# Save trained model
model.save("cartpole_dqn_model.keras")
print("✅ Model saved as cartpole_dqn_model.keras")

# Plot training scores
plt.plot(scores)
plt.title("CartPole DQN Training Scores")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()
