import gymnasium as gym
import minatar # Ensures MinAtari environments are registered
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# ==========================================
# 1. Hyperparameters & Setup
# ==========================================
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995       # Multiplier for epsilon after each episode
TARGET_UPDATE = 1000    # Steps between target network syncs
MEMORY_SIZE = 100000
LR = 1e-4
NUM_EPISODES = 2000
PRE_FILL_STEPS = 5000   # Random steps to fill buffer before training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# ==========================================
# 2. Neural Network Architecture (CNN)
# ==========================================
class DQN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 10 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) # Flatten for the linear layers
        return self.fc(x)

# ==========================================
# 3. Replay Buffer
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 4. Helper Functions
# ==========================================
def preprocess_state(state):
    """Converts MinAtari (10,10,4) boolean array to PyTorch (4,10,10) float array."""
    # Convert to float and swap axes to CHW
    return np.transpose(state.astype(np.float32), (2, 0, 1))

def select_action(state, epsilon, policy_net, n_actions):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.randrange(n_actions)
    else:
        with torch.no_grad():
            # Add batch dimension: (4,10,10) -> (1,4,10,10)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    # Q(s, a) from policy net
    current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))

    # max Q(s', a') from target net
    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(1)[0]
        expected_q_values = rewards + (GAMMA * max_next_q_values * (1 - dones))

    # Huber Loss
    loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0) # Prevent exploding gradients
    optimizer.step()

# ==========================================
# 5. Main Training Loop
# ==========================================
env = gym.make('MinAtari/Breakout-v1')
n_actions = env.action_space.n

policy_net = DQN(in_channels=4, n_actions=n_actions).to(device)
target_net = DQN(in_channels=4, n_actions=n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict()) # Sync initially

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

# --- Pre-fill Buffer Phase ---
print("Pre-filling Replay Buffer...")
state, _ = env.reset()
state = preprocess_state(state)

for _ in range(PRE_FILL_STEPS):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    next_state = preprocess_state(next_state)
    
    memory.push(state, action, reward, next_state, done)
    
    if done:
        state, _ = env.reset()
        state = preprocess_state(state)
    else:
        state = next_state

print("Starting Training...")
# --- Training Phase ---
epsilon = EPS_START
total_steps = 0
best_mean_reward = 0
recent_rewards = deque(maxlen=100)

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = preprocess_state(state)
    episode_reward = 0
    done = False
    
    while not done:
        action = select_action(state, epsilon, policy_net, n_actions)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = preprocess_state(next_state)
        
        memory.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        total_steps += 1
        
        # Optimize at every step
        optimize_model(policy_net, target_net, memory, optimizer)
        
        # Sync Target Network
        if total_steps % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    recent_rewards.append(episode_reward)
    epsilon = max(EPS_END, epsilon * EPS_DECAY) # Decay epsilon
    
    # Display progress
    if episode % 10 == 0:
        mean_reward = np.mean(recent_rewards)
        print(f"Episode {episode} | Epsilon {epsilon:.3f} | Mean Reward (100 eps): {mean_reward:.2f}")
        
        # Save Best Model
        if mean_reward > best_mean_reward and episode > 100:
            best_mean_reward = mean_reward
            torch.save(policy_net.state_dict(), 'best_dqn_breakout.pth')
            print(f"--> Saved new best model! Score: {best_mean_reward:.2f}")

print("Training Finished!")
env.close()