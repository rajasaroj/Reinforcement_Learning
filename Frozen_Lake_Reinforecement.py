# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 08:34:50 2020


envirnment space
SFFF
FHFH
FFFH
HFFG

State	Description	Reward
S	Agent’s starting point - safe	0
F	Frozen surface - safe	0
H	Hole - game over	0
G	Goal - game over	1

@author: Raja
"""

import numpy as np
import gym
import random
import time
from IPython.display import clear_output


#Create enviornment
env = gym.make("FrozenLake-v0")

#get the action and state space
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size,action_space_size))

#Global parameters initialized
discount_rate = 0.99
learning_rate = 0.1
exploration_rate = 1
exploration_decay_rate = 0.01
max_exploration_decay_rate = 1
min_exploration_decay_rate = 0.01
number_of_episode = 20000
time_steps = 100
reward_all_episode=[]

#iterate across episode
for current_episode in range(number_of_episode):
    
    #reset the agent postion in env
    state =  env.reset()
    done = False
    expected_reward = 0
    
    # iterate over the timestep
    for current_step in range(time_steps):
        
        #Exploration and Expliotation trade off
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
        
        #take the action
        new_state, reward, done, info = env.step(action)
        
        #update the q_value
        q_table[state,action] = q_table[state,action]*(1-learning_rate) + learning_rate * (reward + discount_rate* np.max(q_table[new_state,:]))
        
        #add the reward & transition to new_state
        expected_reward += reward
        state = new_state
        
        if done == True:
            break
    
    #Exploration rate decay
    exploration_rate = min_exploration_decay_rate + (max_exploration_decay_rate - min_exploration_decay_rate) * np.exp(-exploration_decay_rate*current_episode)
    
    #add expected reward to the list
    reward_all_episode.append(expected_reward)


#Calculate avg reward per thousand episode
reward_per_thousand_episode = np.split(np.array(reward_all_episode), number_of_episode/1000)
count = 1000
print("********Average reward per thousand episodes********\n")
for rewards in reward_per_thousand_episode:
    print( str(count), " : ", str(sum(rewards/1000)))
    count += 1000