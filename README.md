CartPole DQN – Deep Reinforcement Learning Agent

CartPole DQN is a Deep Reinforcement Learning project where a neural network (Deep Q-Network) is trained to balance a pole on a cart in the CartPole-v1 environment from Gymnasium. The agent learns to select optimal actions using Q-learning with a neural network, experience replay, and an ε-greedy policy.

Features

Implements Deep Q-Network (DQN) for reinforcement learning

Uses experience replay to stabilize training

ε-greedy strategy for exploration vs. exploitation

Real-time environment rendering to visualize agent performance

Fully trainable using TensorFlow

Technologies

Python 3.9+

TensorFlow 2.x

Gymnasium (Classic Control: CartPole-v1)

NumPy

Matplotlib

Installation

Clone the repository:

git clone <your-repo-url>
cd cartpole-dqn


Create a virtual environment (optional but recommended):

python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows


Install dependencies:

pip install tensorflow numpy matplotlib gymnasium[box2d] pygame

Usage
Training the Agent

Run the training script:

python train_agent.py


Trains a DQN agent for 200 episodes (configurable).

Saves the trained model as cartpole_dqn_model.keras.

Plots training scores over episodes.

Testing / Watching the Agent

Run the test script to watch the trained agent play:

python test_agent.py


Uses the saved model to select optimal actions.

Renders the environment in real-time (GUI popup).

Prints the final score of the episode.

How It Works

Q-Network: Neural network approximates Q-values for each state-action pair.

ε-Greedy Policy: Chooses random actions with probability ε to explore; otherwise, exploits learned policy.

Experience Replay: Stores transitions (state, action, reward, next_state, done) to train in batches for stable learning.

Bellman Equation Update: Updates Q-values using:

Q(s,a)←r+γa′max​Q(s′,a′)

Training Loop: Iterates over episodes, updating network weights and decaying ε.

Results

Trained agent achieves high scores (~500 steps per episode).

Model generalizes well to unseen episodes.

Real-time visualization allows observation of balancing behavior.
