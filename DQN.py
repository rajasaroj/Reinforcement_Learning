# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 21:20:57 2020

@author: Raja
"""

#import gym
#import math
#import random
#import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
#from itertools import count
#from PIL import Image
import torch
import torch.nn as nn
#import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T


class DQN(nn.Module):
    
    def __init__(self,height, breath):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=breath*height*3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)
        
    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t
    


    def plot(values, moving_avg_period):
        plt.figure(2)
        plt.clf()
        plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(values)
        plt.plot(get_moving_average(moving_avg_period,values))
        plt.pause(0.001)
        if is_ipython: display.clear_output(wait=True)

    def get_moving_average(period, values):
        values = torch.tensor(values, dtype=torch.FloatType)
        
        if(len(values) >= period):
            moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
            moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
            return moving_avg.numpy() 
        else:
            moving_avg = torch.zeros(len(values))
            return moving_avg.numpy()

    Experience = namedtuple("Experience", ('state','action','new_state','reward'))    