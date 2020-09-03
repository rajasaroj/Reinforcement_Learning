# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:07:29 2020

@author: Raja
"""

import random

class ExperienceReplay():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count  = 0
        
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count +=1
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
    
    
    
        
        