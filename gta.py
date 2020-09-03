# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:14:04 2020

@author: Raja
"""
import numpy as np


distance = np.arange(0,1.2, 0.2)
velocity = np.arange(0,1.2, 0.2)

def moonLander(distance, velocity):
    dist_reward = 1 - distance ** 0.4
    vel_discount = ( 1 - np.maximum(velocity,0.1))
    vel_distance = 1/np.maximum(distance, 0.1)
    vel_combi = vel_discount ** vel_distance
    reward = vel_combi * dist_reward
    return reward


# [0.0 0.2 0.4 0.6 0.8 1.0 ]
#Case 1 velocity less and distance rolling
    

for x in distance:
    print("distance :"+ str(x)+ " reward: " + str(moonLander(x, 0.1)))

print(moonLander(1.0, 0.1))


#print(distance)