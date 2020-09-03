# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:51:37 2020

@author: Raja
"""
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from Agent import Agent
from EpsilonGreedyStrategy import EpsilonGreedyStrategy
from ExperienceReplay import ExperienceReplay
from DQN import DQN
from CartPoleEnvManager import CartPoleEnvManager
from Qvalues import Qvalues


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

#initialize hyperparameters parameters
batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
lr = 0.001
num_episode = 9000
target_update = 10
memory_size = 100000

#Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#intialize all objects
em = CartPoleEnvManager(device)
policy_net = DQN(em.get_height(), em.get_width()).to(device)
target_net = DQN(em.get_height(), em.get_width()).to(device)


Experience = namedtuple("Experience", ('state','action','new_state','reward'))



def extract_tensor(batch_of_experiences):
    unwrapped_batch_of_experiences = Experience(*zip(*batch_of_experiences))
    t1 = torch.cat(unwrapped_batch_of_experiences.state)
    t2 = torch.cat(unwrapped_batch_of_experiences.action)
    t3 = torch.cat(unwrapped_batch_of_experiences.new_state)
    t4 = torch.cat(unwrapped_batch_of_experiences.reward)
    
    return (t1, t2, t3, t4)


def plot(values, moving_avg_period):    
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(values)
    #plt.plot(get_moving_average(moving_avg_period,values))
    #plt.pause(0.001)
    moving_avg = get_moving_average(moving_avg_period,values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    
    print("Episode", len(values),"\n", moving_avg_period, "episode moving avg:", moving_avg[-1] )
    if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:   
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:   
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()



#copy weight from poicy to target net
target_net.eval()
target_net.load_state_dict(policy_net.state_dict())

memory = ExperienceReplay(memory_size)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.number_of_action_available(), device)
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)


#intailize replay memory capacity
episode_duration=[]

#Start the training
for episode in range(num_episode):
    em.reset()
    #print("Episode: " + str(episode))
    state = em.get_state()
    #print(" state before memory push: "+ str(state.shape))
    
    # iterate over each time step
    for time_step in count():
        
        #print(" state size while doing memory push 1: "+ str(state.shape))
        #print(time_step)
        #select an action
        #Via exploration or exploitation
        action = agent.select_action(state, policy_net)
        #print(" state size while doing memory push: 2"+ str(state.shape))
        #take action
        #Execute selected action in an emulator.
        #Observe reward and next state.
        reward = em.take_action(action)
        next_state = em.get_state()
        
        #print(" state size while doing memory push 3: "+ str(state.shape))
        #Store experience in replay memory.
        memory.push(Experience(state,action,next_state,reward) )
        
        #update current state to next state
        state = next_state
        #print(" state size while doing memory push 4:  "+ str(state.shape))
        
        
        #Sample random batch from replay memory.
        if memory.can_provide_sample(batch_size):
            batch_of_experiences = memory.sample(batch_size)
            
            
            #Preprocess states from batch.
            states, actions, next_states, rewards = extract_tensor(batch_of_experiences)
            
            #print("extract_tensor state: "+ str(states.shape))
            #print("extract_tensor action: "+ str(actions.shape))
            #Pass batch of preprocessed states to policy network.
            current_qvalue = Qvalues.get_current(policy_net,states,actions)
            next_qvalues = Qvalues.get_next(target_net, next_states)
            
            #calculate target_qvalue from formula E[R(t+1) + Gamma X max qvalues(s`,a`)]
            target_qvalue = reward + (gamma*next_qvalues)
            
            #Calculate loss between output Q-values and target Q-values.
            loss = F.mse_loss(current_qvalue, target_qvalue.unsqueeze(1))
            
            #set the gradients for eg (d(fw)/dw, d(fy)/dy etc) to zero so that they doesn't get accumulated 
            #(above comment continued) in each backprop
            optimizer.zero_grad()
            
            #run the backprop
            loss.backward()
            
            #update weights
            optimizer.step()
            
            #print(" state size while doing memory push 5: "+ str(state.shape))
            
        if em.done:
            episode_duration.append(time_step)
            plot(episode_duration, 100)
            break
        
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
            
        
em.close()
            
            
            
            
            
            
            
            
            



