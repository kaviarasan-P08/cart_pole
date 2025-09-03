import gymnasium as gym
import numpy as np
import tensorflow as tf
import time

# Load trained model
print("🔍 Loading model...")
model = tf.keras.models.load_model("cartpole_dqn_model.keras")
print("✅ Model loaded successfully.")

# Create environment with real-time GUI rendering
print("🎮 Creating environment in human render mode...")
env = gym.make("CartPole-v1", render_mode="human")
print("✅ Environment ready. Watch the agent play!\n")

# Initialize environment
state, _ = env.reset()
done = False
total_reward = 0

# Run the agent through one episode
while not done:
    # Predict action using the model
    q_values = model.predict(state[np.newaxis, :], verbose=0)
    action = np.argmax(q_values[0])

    # Take action in the environment
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state
    total_reward += reward

    # Render the environment (popup window)
    env.render()

    # Slight delay to slow down the playback
    time.sleep(0.02)

# Close environment and print result
env.close()
print(f"🏁 Episode finished. Final Score: {total_reward}")
