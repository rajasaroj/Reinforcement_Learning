# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:31:22 2020

@author: Raja
"""
import math

class EpsilonGreedyStrategy():
    
    def __init__(self,start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_strategy(self, current_step):
        return self.end + (self.start - self.end)*math.exp(-1.*current_step*self.decay)
        