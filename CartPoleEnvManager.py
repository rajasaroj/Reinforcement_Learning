# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 07:55:04 2020

@author: Raja
"""

import gym
#import math
#import random
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
#from collections import namedtuple
#from itertools import count
#from PIL import Image
import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
import torchvision.transforms as T


class CartPoleEnvManager:
    
    def __init__(self, device):
        self.env = gym.make('CartPole-v0').unwrapped
        self.done = False
        self.current_screen = None
        self.device = device
        self.env.reset()
    
    def reset(self):
        self.env.reset()
        self.current_screen = None
    
    def close(self):
        self.env.close()
        
    def render(self, mode='human'):
        return self.env.render(mode)
        
    def number_of_action_available(self):
        return self.env.action_space.n
    
    def take_action(self, action):
       # print("take_action")
        #print(action.shape)
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)
    
    def just_starting(self):
        return self.current_screen is None
    
    def get_state(self):
        #print("before if")
        if self.just_starting() or self.done:
            #print("first in")
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            #print("in act")
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            s3 = s2-s1
            #print("in act  diff s1, s2 = s3: "+ str(s3.shape))
            return s3
        
    def get_height(self):
        #screen = get processed screen
        screen = self.get_processed_screen()
        return screen.shape[2]
    
    def get_width(self):
        #screen = get processed screen
        screen = self.get_processed_screen()
        return screen.shape[3]
    
    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2,0,1))
        screen = self.crop_screen(screen)
        return self.transform_screen(screen)
        
    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        top = int(screen_height*0.4)
        bottom = int(screen_height*0.8)
        screen = screen[:,top:bottom,:]
        return screen
        
    def transform_screen(self, screen):
        screen = np.ascontiguousarray(screen, dtype=np.float32)/255
        screen = torch.from_numpy(screen)
        resize = T.Compose([T.ToPILImage(),  T.Resize((40,90)), T.ToTensor()])
        #print("done")
        return resize(screen).unsqueeze(0).to(self.device)