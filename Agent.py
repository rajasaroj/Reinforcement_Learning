# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 10:30:24 2020

@author: Raja
"""
import random
import torch

class Agent():
    def __init__(self, strategy, num_actions,device):
        self.strategy = strategy
        self.num_actions = num_actions
        self.current_step = 0 
        self.device = device
        
    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_strategy(self.current_step)
        self.current_step += 1
        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device) # explore
        else:
            with torch.no_grad():
                #print("in arg")
                return policy_net(state).argmax(dim=1).to(self.device) # exploit
                
