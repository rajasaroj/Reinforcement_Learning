# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 08:42:51 2020

@author: Raja
"""

import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import torch
#import time
from CartPoleEnvManager import CartPoleEnvManager

#cartpoler = CartPoleEnvManager('cpu')

def non_proc():
    cartpoler.env.reset()
    screen = cartpoler.render("rgb_array")
    plt.figure()
    plt.imshow(screen)
    plt.show()
#non_proc()
#print(cartpoler.env.render().shape)



def process_screen():
    screen = cartpoler.get_processed_screen()
    plt.figure()
    plt.imshow(screen.squeeze(0).permute(1,2,0).cpu(), interpolation='none')
    plt.show()

#process_screen()

def get_state_1():
    screen = cartpoler.get_state()
    plt.figure()
    plt.imshow(screen.squeeze(0).permute(1,2,0).cpu(), interpolation='none')
    plt.show()

#get_state_1()
    
def take():
    
    for i in range(5):
        print(i)
        cartpoler.take_action(torch.tensor([1]) )
        print("out")
    screen = cartpoler.get_state()
    #plt.figure()
    plt.imshow(screen.squeeze(0).permute(1,2,0).cpu(), interpolation='none')
    #plt.show()
    
#take()
    

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = CartPoleEnvManager(device)
    em.env.reset()  
    for i in range(500):
    
        if(i%10 == 0):
            em.env.reset()    
            #em.env.step(1)
            #em.env.render()
            em.take_action(torch.tensor([1]))
            screen = em.get_state()
            #time.sleep(0.01)
            em.close()
            #plt.figure()
            #plt.imshow(screen.squeeze(0).permute(1, 2, 0).cpu(), interpolation='none')
            #plt.title('Non starting state example')
            #plt.show()

def dolna():
    t = torch.tensor([
                        [        [1, 2, 0],
                                 [3, 4, 0],
                                 [1, 2, 0]],
                        
                        [        [1, 1, 1],
                                 [3, 1, 0],
                                 [1, 2, 1]],
                                 
                        [        [0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0]],
                        
                        [        [1, 1, 1],
                                 [3, 1, 0],
                                 [1, 2, 1]],
                         
                        [        [1, 2, 0],
                                 [3, 4, 0],
                                 [1, 2, 0]],
                        
                        [        [0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0]],
                                 
                        [        [1, 2, 0],
                                 [3, 4, 0],
                                 [1, 2, 0]],
                        
                        [        [1, 1, 1],
                                 [3, 5, 0],
                                 [1, 2, 1]],

                        [        [1, 2, 0],
                                 [3, 4, 0],
                                 [1, 2, 0]],
                        
                        [        [0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0]]                         
                         
                        
                    ])
    #print(t.unfold(dimension=0, size=2 ,step=1))
    #print(t.unfold(dimension=0, size=2 ,step=1).shape)
    
    print(t.flatten(start_dim=1))
    print("\n")
    print(t.flatten(start_dim=1).max(dim=1)[0])
    
    a = t.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
    
    print(t.shape)
    print(a)
    print("index in t")
    non_final = (a==False)
    print(a==False)
    
    print("batch out") 
    batch_size = t.shape
    vale = torch.zeros(batch_size)
    
    print("non final")
    print(non_final)
    
    print("vale zie after indexfinal")
    print(vale[non_final].shape)
    b = vale[non_final].unsqueeze()
    print(b.shape)
    #print(vale[non_final].shape) 
    
    #print(t[non_final])
    
    #print(t[non_final].max(dim=1)[0])
    #vale[non_final] = t[non_final].max(dim=1)[0]
    #vale[non_final] = t[a==False].max(dim=1)[0].detach()
    #print(vale)
    
    
#dolna()





def fast():
    t = torch.tensor(np.random.rand(2,100,100))
    print(t.shape)
    print(t.unfold(dimension=0, size=10 ,step=1))
#fast()   
# Utitlity function


import matplotlib
import matplotlib.pyplot as plt
    
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

    
    
    
def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(values)
    moving_avg = get_moving_average(moving_avg_period,values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    
    print("Episode", len(values),"\n", moving_avg_period, "episode moving avg:", moving_avg[-1] )
    
    if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    
    if(len(values) >= period):
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy() 
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()
    
    
plot(np.random.rand(300),100)