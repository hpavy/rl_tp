import numpy as np
import torch
import gymnasium as gym
from minatar import Environment
from minatar.gym import register_env
import torch.nn as nn
import torch.functional as F
import random

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

if "MinAtar/Breakout-v1" not in gym.envs.registry:
    register_envs()


def random_policy(observation, num_actions):
    return np.random.randint(num_actions)

def breakout_heuristic_policy(observation, num_actions):
    paddle_channel = observation[:, :, 0]
    ball_channel = observation[:, :, 1]

    paddle_pos = np.where(paddle_channel == 1)
    ball_pos = np.where(ball_channel == 1)

    if len(paddle_pos[1]) == 0 or len(ball_pos[1]) == 0:
        return 0

    paddle_x = paddle_pos[1].mean()
    ball_x = ball_pos[1].mean()

    if ball_x < paddle_x:
        return 1
    elif ball_x > paddle_x:
        return 2
    else:
        return 0

def run_episode(env, policy_fn):
    observation, info = env.reset()
    num_actions = env.action_space.n

    total_reward = 0
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        action = policy_fn(observation, num_actions)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    return total_reward, steps


def evaluate_policy(env, policy_fn, num_episodes=100):
    rewards = []
    steps_list = []

    for _ in range(num_episodes):
        total_reward, steps = run_episode(env, policy_fn)
        rewards.append(total_reward)
        steps_list.append(steps)

    return np.array(rewards), np.array(steps_list)

env = gym.make("MinAtar/Breakout-v1")
env.reset()

print(f"Observation shape: {env.observation_space.shape}")
print(f"Number of actions: {env.action_space.n}")


import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML


def render_frame(env):
    frame = env.render()
    return (frame * 255).astype(np.uint8)


def run_episode_with_frames(env, policy_fn, max_steps=500):
    observation, info = env.reset()
    num_actions = env.action_space.n
    frames = [render_frame(env)]

    total_reward = 0
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        action = policy_fn(observation, num_actions)
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(render_frame(env))
        total_reward += reward
        steps += 1

    return frames, total_reward, steps


def animate_episode(frames, interval=50):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    img = ax.imshow(frames[0], interpolation="nearest")

    def update(frame):
        img.set_array(frame)
        return [img]

    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=interval, blit=True
    )
    plt.close(fig)
    return anim


##########################################

class Q_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*10*10, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # x: batchx10x10x4
        x = F.relu(self.conv1(x))  # batchx10x10x16 
        x = F.relu(self.conv2(x))  # batchx10x10x32
        x = F.relu(self.fc1(x.view(x.shape[0], 10*10*32)))  # batchx128
        x = F.relu(self.fc2(x)) # batchx4
        return x


class ReplayBuffer():
    def __init__(self, max_episodes):
        self.buffer = []
        self.max_episodes = max_episodes
    
    def add(self, old_observation, new_observation):
        if len(self.buffer) > self.max_episodes:
            self.buffer.pop(0)
        self.buffer.append((old_observation, new_observation))

    def get_batch(self, len_batch):
        """give a batch from the replay buffer"""
        return random.sample(self.buffer, len_batch)


class HyperParamDeep:
  def __init__(self, alpha, gamma, epsilon_init, d, nb_epochs, display_every, punition_terminated, reward_truncated):
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon_init = epsilon_init
    self.d = d
    self.nb_epochs = nb_epochs
    self.display_every = display_every
    self.punition_terminated
    self.reward_truncated


def run_episode_Q_learning_training(env, Q, epoch, hyper_param, find_Q_policy):
    """Run one episode of Q learning"""
    observation, info = env.reset()
    num_actions = env.action_space.n

    total_reward = 0
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        epsilon = hyper_param.epsilon_init * (hyper_param.d**epoch)
        policy = find_Q_policy(Q, epsilon, with_exploration=True)
        action = policy(observation, num_actions)
        state = find_state_Q(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        new_state = find_state_Q(observation)
        Q = update_Q(Q, state, action, new_state, reward, hyper_param, terminated)
        total_reward += reward
        steps += 1

    return total_reward, steps, Q




