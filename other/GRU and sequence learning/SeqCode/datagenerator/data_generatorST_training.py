#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:09:46 2019

@author: Zhewei Zhang
"""
import os 
import copy
import datetime
import numpy as np
import scipy.io as sio

NumTrials = int(7.5e5+1)
trans_prob = 0.8 # from A1-B1, from A2-B2
reward_prob = 0.8
block_size = 50
Double = True

info = {'NumTrials':NumTrials,'reward_prob':reward_prob,'block_size':block_size,'trans_prob':trans_prob}

temp = np.hstack((np.ones((block_size,)), np.zeros((block_size,)) ))
blocks = np.tile(temp,int(NumTrials/(block_size*2))) # 0: B1 with large reward prob, 1:b2 with large reward prob
# 
lost_trialsNum = NumTrials - blocks.size
if lost_trialsNum <= block_size:
    temp = np.ones((lost_trialsNum,))
else:
    temp = np.hstack((np.ones((block_size,)), np.zeros((lost_trialsNum - block_size,))))
blocks = np.hstack((blocks, temp))

choices  = np.random.choice([0,1], NumTrials) # 0:A1; 1:A2
trans_probs = trans_prob*choices + (1-trans_prob)*(1-choices) # probability of transition to the B2 in stage 2

temp = np.random.rand(NumTrials,)
state2 = trans_probs>temp # 0: B1; 1: B2

reward_prob_B1 = (1-reward_prob)*state2*(1-blocks) + reward_prob*(1-state2)*(1-blocks)
reward_prob_B2 = reward_prob*state2*blocks + (1-reward_prob)*(1-state2)*blocks 

reward_prob_state2 = reward_prob_B1 + reward_prob_B2 # reward probability of the observation in stage 2 

temp1 = np.random.rand(NumTrials,)
reward_all = reward_prob_state2>temp1
state_all = copy.deepcopy(state2)+1 # 1: B1; 2: B2



# In[]:    
"""
first seven inputs represent visual stimulus 

1st~2nd inputs representing the options

3rd~4th inputs representing the intermeidate outcome

6th~8th inputs representing the movement/choice

9th~10th inpust denotes the reward states
"""

data_ST = []
n_input = 10;
trial_length = 13
shape_Dur = 3; # period for shape presentation
choice_Dur = 2; #period for shape interval
for nTrial in range(NumTrials):
    inputs = np.zeros((n_input,trial_length))
    inputs[0:2,2:5] = 1 # the three-five time points representing the first epoch
    
    if choices[nTrial]==0:
        inputs[6,5:7] = 1
    elif choices[nTrial]==1:
        inputs[7,5:7] = 1
    
    if state_all[nTrial]==1:
        inputs[2,7:10] = 1
    elif state_all[nTrial]==2:
        inputs[3,7:10] = 1
    
    if reward_all[nTrial]==1:
        inputs[8,10:12] = 1
    
    inputs[4,:] = inputs[0:4,:].sum(axis = 0)
    inputs[4,np.where(inputs[4,:]!=0)] = 1
    inputs[4,:] = 1 - inputs[4,:]
    inputs[5,:] = 1 - inputs[6:8,:].sum(axis = 0)
    inputs[9,:] = 1 - inputs[8,:]
    if nTrial!=0:
        data_ST.append([np.hstack((inputs_prev,inputs)).T])
    inputs_prev = copy.deepcopy(inputs)
        
if Double:
    training_guide = np.array([reward_all[1:], 2*trial_length+np.zeros((len(reward_all)-1,))]).squeeze().astype(np.int).T.tolist()
else:
    training_guide = np.array([reward_all[1:], trial_length+np.zeros((len(reward_all)-1,))]).squeeze().astype(np.int).T.tolist()
    
data_ST_Brief = {'choices':choices,'state_all':state_all,
                'reward':reward_all, 'trans_prob':trans_prob,
                'shape_Dur':shape_Dur,'choice_Dur':choice_Dur,'Double':Double,
                'training_guide':training_guide}

# data  saving
pathname = "../data/"
file_name = datetime.datetime.now().strftime("%Y_%m_%d")
data_name = 'SimpTwo_TrainingSet-' + file_name

n = 0
while 1: # save the model
    n += 1
    if not os.path.isfile(pathname+data_name+'-'+str(n)+'.mat'):
        sio.savemat(pathname+data_name+'-'+str(n)+'.mat',
                {'data_ST':data_ST,
                 'data_ST_Brief':data_ST_Brief,
                 'info':info})
        print("_" * 36)
        print("training file for simplified two step task is saved")
        print("file name:" + pathname+data_name+'-'+str(n)+'.mat')
        break
filename = pathname + data_name + '-' + str(n) + '.mat'
#data = sio.loadmat(filename)
#brief = data['data_TS_Brief'][0,0]
#brief['reward']
