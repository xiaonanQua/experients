#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20180809

@author: Zhewei Zhang


Reproduce fake data in the reaction time version of shape task
The input channels are the same as shape tasks
target configuration/present shape/choice is random 
Only the correct trials are included
shape number for each trial is controlled by the logLR and Boundary
saccade is allowed after min_shapenumth shapes appear
If last shape appear and eye movement are at the same time, I believe
the last shape is not used for decision
no break. only error happens on wrong choices
Input channel: Visual Input 1:fixation point;
2/3-4/5: left-right target is red/green, 0 no target, 1 red/green
6-15:shapes; 16:no visual input; Eye Single/Motor information 17: fixate on FP
18-19: fixate on left/right target; 20: fixation on other position
Reward Signal: 21: reward appear; 22:no reward
training data generator

"""

import os
import time
import datetime
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


np.random.seed(int(time.time()))

numTrials = int(7.5e5)

B=1.5 
exp_decay = 0
lin_decay = 0.1
shape_Dur = 3 # period for shape presentation
shape_inter = 2 # period for shape interval
min_shapenum = 1 # minimun number of shapes in each trial
max_shapenum = 15 # maximun number of shapes in each trial
n_timepoint = 77+6 # should be >= 8 + max_shapenum*(shape_inter+shape_Dur)+3;

weight = np.linspace(-0.9, 0.9, 10) # positive for red shape
prob_red = np.array([0.0056, 0.0166, 0.0480, 0.0835, 0.1771, 0.2229, 0.1665, 0.1520, 0.0834, 0.0444]) # block1: red is rewarded
prob_green = np.flip(prob_red) # block2: green is rewarded
n_input = 20

# prepare the trial condition, choice and reward

trialCondi = []
total_wegihts = np.zeros((numTrials,1))
shape_num = np.zeros((numTrials,1)) # shape_num for each trial is random, from min_shapenum to max_shapenum
choose_left = np.zeros((numTrials,1)) # choice; 1/choose red;0/choose green
trialtype = np.around(np.random.rand(numTrials,1)) # 0: draw from green distribution,1: from red
for nTrial in range(numTrials):
    if trialtype[nTrial]:
        cdf_curr = np.cumsum(prob_red)
    else:
        cdf_curr = np.cumsum(prob_green)

    shapes = []# last shape is not used for decision
    temp_sumweight = 0
    n_shape = 0
    while 1:
        temp_B = (B*np.exp(-exp_decay*n_shape) - lin_decay*n_shape).round(4)
        n_shape = n_shape+1;
        tepmp_shape = np.where(np.random.rand()<cdf_curr)[0][0]
        shapes.append(tepmp_shape)
        temp_sumweight += weight[tepmp_shape].round(4)
        if np.abs(temp_sumweight) >= temp_B or n_shape >= max_shapenum:
            if temp_sumweight>0:
                 choose_left[nTrial] = 1
            elif temp_sumweight<0:
                 choose_left[nTrial] = 0
            break
        
    shape_num[nTrial] = int(n_shape)
    trialCondi.append(shapes)

np.mean(choose_left)
plt.figure()
plt.hist(shape_num, np.linspace(1,25,25), density = True);

choosen_target = (2-choose_left).astype(np.int)# 1: choose left; 2:choose right
reward =  choose_left==trialtype

saccadetime = 6 
required_Dur = (saccadetime+(shape_num-1)*(shape_inter+shape_Dur)+1).astype(np.int)
## save the brief information
training_guide = np.array([reward, required_Dur+saccadetime]).squeeze().T.tolist()

tarColLeft = np.ones((numTrials,1))
data_RT_Brief = {'trialtype':trialtype,'shape_num':shape_num, 'tarColLeft':tarColLeft,
                 'trialCondi':trialCondi,'choosen_target':choosen_target,
                 'saccadetime':saccadetime,'reward':reward, 'training_guide':training_guide}
info = {'boundary':B, 'exp_decay':exp_decay, 'min_rt':min_shapenum,
        'max_rt':max_shapenum, 'lin_decay':lin_decay}
              
'''
 arrange the data into time sequence
fixation is one time step later than fixation point appear
fixation stop is one time step later than fixation point disappear
Time point: 1: fixation point appears; 2: acquire fixation 3 target on;
    4:5:5*shape_num-1:shape on; 7:5:5*shape_num+2 shape off; 

'''
data_RT = []
for ntrial in range(numTrials):
    required_Dur_curr = int(required_Dur[ntrial])
    inputs = np.zeros((n_input,n_timepoint)) # inputs*time points
    
    inputs[0,:required_Dur_curr+1] = 1 # fixation point appear
    inputs[1,2:required_Dur_curr+1+1] = 1 # left target
    inputs[2,2:required_Dur_curr+1+1] = 1 # right target

   # shapes appear 
    for nshape in range(int(shape_num[ntrial])):
        inputs[3+trialCondi[ntrial][nshape], (nshape+1)*(shape_inter+shape_Dur)-2:(nshape+1)*(shape_inter+shape_Dur)+shape_Dur-2] = 1

    inputs[14,1:required_Dur_curr] = 1 # fixate on FP
    inputs[14+choosen_target[ntrial],required_Dur_curr:required_Dur_curr+3] = 1 # fixate on target
    inputs[18,2+required_Dur_curr:2+required_Dur_curr+2] = reward[ntrial] # reward is deliverd one time step later than fixation point disappear
    inputs[3-choosen_target[ntrial],1+required_Dur_curr] = 0 # unchosen target disappear
    
    inputs[13,:] = 1-np.any(inputs[0:13,:]!=0, axis =0) # channel: no visual Input
    inputs[17,:] = 1-np.any(inputs[14:17,:]!=0, axis=0) # channel: fixate somewhere else
    inputs[19,:] = 1-inputs[18,:]!=0 # channel: no reward

    data_RT.append([inputs.T])
    

pathname = "../data/"
file_name = datetime.datetime.now().strftime("%Y_%m_%d")
data_name = 'SeqLearning_TrainingSet-' + file_name

n = 0
while 1: # save the model
    n += 1
    if not os.path.isfile(pathname+data_name+'-'+str(n)+'.mat'):
        sio.savemat(pathname+data_name+'-'+str(n)+'.mat',
                    {'data_RT':data_RT,
                     'data_RT_Brief':data_RT_Brief,
                     'info':info})
        break
print("_" * 36)
print("training file for shape task saved")
filename = pathname + data_name + '-' + str(n) + '.mat'


