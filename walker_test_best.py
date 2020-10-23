#test program to run the walker from a pre trained network

import math
import random
import gym
import pybulletgym 
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt

from multiprocessing_env import SubprocVecEnv

device   = torch.device("cpu")

num_envs = 4

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

enviorment_name = "Walker2DPyBulletEnv-v0"

env = gym.make(enviorment_name)
env.render(mode='human', close=False)  # ?? 

def test_model(number_of_runs):
    rewards = []
    distance = []
    for i in range(number_of_runs):
        done = False
        state = env.reset()
        total_reward = 0
        while not done:
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            normal_dist, critic_value = model.forward(state)
            action = normal_dist.sample()
            action_ = action.squeeze().detach().numpy()
            next_state, reward, done, _ = env.step(action_)
            env.camera_adjust()
            state = next_state
            total_reward += reward
            time.sleep(1/60)
        rewards.append(total_reward)
        distance.append(env.return_distance())
    return np.mean(rewards), np.max(distance)

# Parameters
num_inputs  = env.observation_space.shape[0] # 22
num_outputs = env.action_space.shape[0] # 6
hidden_size = 64

model = ActorCritic(num_inputs, num_outputs, hidden_size)
model.load_state_dict(torch.load("best_model_V2_continue"))

rewards, max_distance = test_model(5)
print('The mean reward is ', rewards, ' and the max distance is ', max_distance)