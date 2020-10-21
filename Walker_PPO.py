#inspierd by https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb and openai baselines

import math
import random
import gym
import pybulletgym  # register PyBullet enviroments with open ai gym
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

use_cuda = torch.cuda.is_available()
device   = torch.device("cpu")

num_envs = 4

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        '''
        self.combined_parameters = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
        )
        '''
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
        #x_combined = self.combined_parameters(x)
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

enviorment_name = "Walker2DPyBulletEnv-v0"

env = gym.make(enviorment_name)
env.render(mode='human', close=False)  # ?? 

def make_env_list():
    def env_multiprocessing():
        env = gym.make(enviorment_name)
        return env

    return env_multiprocessing

envs = [make_env_list() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

def GAE(next_critic_value, rewards, masks, values, gamma, lambda_):
    gae = 0
    values_ = values+ [next_critic_value]
    returns = []
    for k in reversed(range(len(rewards))):
        re = torch.transpose(rewards[k].unsqueeze(1),0,1)
        gv = gamma * torch.transpose(values_[k+1],0,1) * masks[k]
        vv =  torch.transpose(values_[k],0,1)
        delta = re +gv - vv
        gae = delta + gamma * lambda_ * masks[k] * gae
        returns.append(gae + vv)
    return list(reversed(returns))
  
def model_update(number_of_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, clip_param=0.2):
    for i in range(number_of_epochs):
        for j in range(states.size(0)//mini_batch_size):
            states_batch, actions_batch, old_log_probs_batch, returns_batch, advantage_batch = get_batch(mini_batch_size, states, actions, log_probs, returns, advantage)
            
            normal_dist, critic_value = model.forward(states_batch)
            log_probs_batch = normal_dist.log_prob(actions_batch)
            policy_ratio = (log_probs_batch - old_log_probs_batch).exp()

            surrogate_1 = policy_ratio * advantage_batch
            surrogate_2 = torch.clamp(policy_ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage_batch
            actor_objective = torch.min(surrogate_1, surrogate_2).mean()

            critic_objectiv = (returns_batch - critic_value).pow(2).mean()

            entropy = normal_dist.entropy().mean()

            loss = - actor_objective + 0.5*critic_objectiv - 0.01*entropy

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()




def get_batch(mini_batch_size, states, actions, log_probs, returns, advantage):
    max_index = states.size(0)
    indeces = np.random.randint(0, max_index, mini_batch_size)
    return states[indeces,:], actions[indeces,:], log_probs[indeces,:], returns[indeces], advantage[indeces]
    
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
lr = 5e-5
mini_batch_size = 64 
num_steps_for_batch = 2048
num_inputs  = envs.observation_space.shape[0] # 22
num_outputs = envs.action_space.shape[0] # 6
hidden_size = 64


model = ActorCritic(num_inputs, num_outputs, hidden_size)
#model.load_state_dict(torch.load("best_model_V2_continue"))
optimizer = optim.Adam(model.parameters(), lr=lr)

state = envs.reset()
number_of_test_runs = 10
total_steps = 0 
test_rewards = [] #
last_test_reward = 0 #
early_stop = False
gamma = 0.99
lambda_ = 0.95
number_of_epochs = 10
completed = False
distance_walked = []
fintess_save= []
target_distance = 100

#test_reward, max_distance = test_model(number_of_test_runs) 

while not completed:

    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []

    for i in range(num_steps_for_batch):
        state = torch.tensor(state, dtype=torch.float32, device=device)

        normal_dist, critic_value = model.forward(state)
        action = normal_dist.sample()

        next_state, reward, done, _ = envs.step(action.detach().numpy()) # detatch() ??? fuck shit up??

        states.append(state)

        actions.append(action)
        rewards.append(reward) # torch.FloatTensor(reward).unsqueeze(1).to(device)
        masks.append((1-done)) #
        log_probs.append(normal_dist.log_prob(action))
        values.append(critic_value)

        state = next_state
        total_steps += 1
        # test, validation   
        
        if total_steps % 4000 == 0:
            test_reward, max_distance = test_model(number_of_test_runs) 
            test_rewards.append(test_reward)
            fintess_save.append(test_reward)
            print('Test Reward:', test_reward, 'max dist', max_distance)
            distance_walked.append(max_distance)
            if test_reward > last_test_reward:
               #torch.save(model.state_dict(),"best_model_V2_continue")
               last_test_reward = test_reward
            if max_distance > target_distance:
                completed = True
                break

    
    _, next_critic_value = model.forward(torch.tensor(state, dtype=torch.float32, device=device))
    rewards = torch.tensor(rewards, dtype=torch.float32)
    masks = torch.tensor(masks, dtype=torch.float32)

    returns = GAE(next_critic_value, rewards, masks, values, gamma, lambda_)


    rewards = torch.flatten(rewards).unsqueeze(1)
    masks = torch.flatten(masks).unsqueeze(1)
    values = torch.cat(values).detach()
    log_probs = torch.cat(log_probs).detach()

    returns = torch.cat(returns).detach()
  
    returns = torch.flatten(returns).unsqueeze(1)
    advantage = returns - values
    #states = torch.tensor(states, dtype=torch.float32)
    #states = torch.flatten(states, 0,1)
    states = torch.cat(states)

    actions = torch.cat(actions)
    model_update(number_of_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)






plt.figure()
plt.subplot()
plt.title('Training progress distance')
plt.plot(distance_walked)
plt.xlabel('steps x 4000')
plt.ylabel('distance walked')
plt.show()



plt.figure()
plt.subplot()
plt.title('Training progress fitness')
plt.plot(fintess_save)
plt.xlabel('steps x 4000')
plt.ylabel('fitness')
plt.show()



