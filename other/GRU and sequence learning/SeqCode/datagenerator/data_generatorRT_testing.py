#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20180328

@author: Zhewei Zhang

Reproduce fake data in the reaction time version of shape task for validation

The input channels are the same as shape tasks
target configuration/present shape/choice is random
shape number for each trial is random, from min_shapenum to max_shapenum
saccade is allowed after min_shapenumth shapes appear

If last shape appear and eye movement are at the same time, I believe the last shape is not used for decision

no break. only error happens on wrong choices

Input channel: Visual Input 1:fixation point;
     2/3-4/5: left-right target is red/green, 0 no target, 1 red/green
     6-15:shapes; 16:no visual input; Eye Single/Motor information 17: fixate on FP
     18-19: fixate on left/right target; 20: fixation on other position
     Reward Signal: 21: reward appear; 22:no reward
     validate data

30 shapes, no choice no reward
"""

import os
import datetime
import numpy as np
import scipy.io as sio

numTrials = 5000

B=1.5 
exp_decay = 0
lin_decay = 0.1
shape_Dur = 3 # period for shape presentation
shape_inter = 2 # period for shape interval
shape_num = 30
n_input = 20
n_timepoint = 170 # should be >= 8 + max_shapenum*(shape_inter+shape_Dur)+3;


weight = np.linspace(-0.9, 0.9, 10) # positive for red shape
prob_red = np.array([0.0056, 0.0166, 0.0480, 0.0835, 0.1771, 0.2229, 0.1665, 0.1520, 0.0834, 0.0444]) # block1: red is rewarded
prob_green = np.flip(prob_red) # block2: green is rewarded

# prepare the trial condition, choice and reward

trialCondi, temp_wegihts = [], []
total_wegihts = np.zeros((numTrials,1))
trialtype = np.around(np.random.rand(numTrials,1)) # 0: draw from green distribution,1: from red
for nTrial in range(numTrials):
    if trialtype[nTrial]:
        cdf_curr = np.cumsum(prob_red)
    else:
        cdf_curr = np.cumsum(prob_green)
    
    condicurr = np.zeros((shape_num,1))
    for i in range(shape_num):
        new_shape = np.where(np.random.rand()<cdf_curr)[0][0]
        condicurr[i] = new_shape
        
    temp_wegihts.append(weight[condicurr.astype(np.int)])
    trialCondi.append(condicurr.astype(np.int))



saccadetime = 6 
required_Dur = saccadetime+(shape_num-1)*(shape_inter+shape_Dur)+1
## save the brief information
tarColLeft = np.ones((numTrials,1))
data_RT_Brief = {'trialtype':trialtype,'shape_num':shape_num, 'trialCondi':trialCondi,
                 'temp_wegihts':temp_wegihts,'tarColLeft':tarColLeft}
info = {'1':[]}
              
'''
 arrange the data into time sequence
fixation is one time step later than fixation point appear
fixation stop is one time step later than fixation point disappear
Time point: 1: fixation point appears; 2: acquire fixation 3 target on;
    4:5:5*shape_num-1:shape on; 7:5:5*shape_num+2 shape off; 

'''

data_RT = []
required_Dur = int((shape_num-1)*(shape_inter+shape_Dur)+7)
for ntrial in range(numTrials):

    #    inputs = np.zeros((n_input,n_timepoint)) # inputs*time points
    inputs = np.zeros((n_input,3+required_Dur)) # inputs*time points
    
    inputs[0,:required_Dur+1] = 1 # fixation point appear
    inputs[1,2:required_Dur+1+1] = 1 # left target
    inputs[2,2:required_Dur+1+1] = 1 # right target

    # shapes appear 
    for nshape in range(shape_num):
        inputs[3+trialCondi[ntrial][nshape], (nshape+1)*(shape_inter+shape_Dur)-2:(nshape+1)*(shape_inter+shape_Dur)+shape_Dur-2] = 1

    inputs[14,1:required_Dur] = 1 # fixate on FP
    
    
    inputs[13,:] = 1-np.any(inputs[0:13,:]!=0, axis =0) # channel: no visual Input
    inputs[17,:] = 1-np.any(inputs[14:17,:]!=0, axis=0) # channel: fixate somewhere else
    inputs[19,:] = 1-inputs[18,:]!=0 # channel: no reward

    data_RT.append([inputs.T])
    

pathname = "../data/"
file_name = datetime.datetime.now().strftime("%Y_%m_%d")
data_name = 'SeqLearning_TestingSet-' + file_name
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
print("testing file for shape task saved")
filename = pathname + data_name + '-' + str(n) + '.mat'




