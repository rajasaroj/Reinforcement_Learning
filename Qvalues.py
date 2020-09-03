# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 12:05:29 2020

@author: Raja
"""
import random
import torch


class Qvalues:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_current(policy_net, states, action):
        #print("pt")
        return policy_net(states).gather(dim=1, index=action.unsqueeze(-1))
    
    @staticmethod
    def get_next(target_net, next_state):
        final_state_location = next_state.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_location = (final_state_location == False)
        non_final_state = next_state[non_final_state_location]
        batch_size = next_state.shape[0]
        values = torch.zeros(batch_size).to(Qvalues.device)
        values[non_final_state_location] = target_net(non_final_state).max(dim=1)[0].detach()
        return values