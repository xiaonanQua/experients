#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 21:13:54 2019


@author: Zhewei Zhang

Reproduce fake data of the sure target task(Kiani, 2009))

in half of trials, it is the typical random dots task
in the other half, a sure target appear, the sure target lead to a small reward

V2: different duration of the random dots stimuli

1st-14th: about the visual stimili
    first input represents the fixation point
    second/third inputs denote the two choice targets
    the fourth one is the sure target
    5th-9th: are the neurons prefer the left direction
    10th-14th: are the neurons prefer the right direction
    15th: no visual stimuli

16th-20th: about the movement; fixation/left tar/right tar/sure target/break

21th-22th: reward/no reward

"""
import os
import time
import datetime
import numpy as np
import scipy.io as sio

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#std_con = 0.05
#threshold = 0.05*25

np.random.seed(int(time.time()))

NumTrials = int(7.5e5)
sure_prop = 0.5
std_con = 2.5
randots_dur_list = [1,2,3,4,6,8]
num_neurons_dots = 10
threshold = 7.5
linear_inc = 2.5
sure_reward = 0.5
coherence_list = [-51.2, -25.6, -12.8, -6.4, -3.2, 0, 3.2, 6.4, 12.8, 25.6, 51.2]
randots_dur = np.random.choice(randots_dur_list,NumTrials)


coherences = np.random.choice(coherence_list,NumTrials)
sure_trials = np.random.rand(NumTrials,) < sure_prop
# number of inputs about random dots-by-the number of time points-by-the number of trials

neuron_resp = []
acc_evi = []
for nTrial in range(NumTrials):
    resp = std_con*np.random.randn(num_neurons_dots, randots_dur[nTrial])
    resp[:int(num_neurons_dots/2),:] += coherences[nTrial]/10
    resp[int(num_neurons_dots/2):,:] += -coherences[nTrial]/10
    
    evi = np.sum(resp[:int(num_neurons_dots/2),:] - resp[int(num_neurons_dots/2):,:])
        
    acc_evi.append(evi)
    neuron_resp.append(resp)


acc_evi = np.array(acc_evi)

choices = []
rewards = []
training_guide = []

for nTrial in range(NumTrials):
    choice = None
    if sure_trials[nTrial]==1:
        if acc_evi[nTrial] > threshold + linear_inc*(randots_dur[nTrial]-1):
            choice = 1
        elif acc_evi[nTrial] < -(threshold + (linear_inc*(randots_dur[nTrial]-1))):
            choice = 2
        else:
            choice = 3
    else :
        choice = 1 if acc_evi[nTrial] > 0 else 2
        
    choices.append(choice)

    if choice==3:
        reward = sure_reward
    elif (coherences[nTrial]>0) == (choice == 1):
        reward = 1
    elif (coherences[nTrial]<0) == (choice == 1):
        reward = 0
    elif coherences[nTrial]==0 and choice!=3:
        reward = np.random.rand()>0.5
        
    rewards.append(reward)
    if reward != 0:
        training_guide.append(1)
    else:
        training_guide.append(0)

# In[]:
n_input = 22
trial_length = 17 + randots_dur
n_timepoint = 17 + np.max(randots_dur)
data_Sure = []
for nTrial in range(NumTrials):
    inputs = np.zeros((n_input,n_timepoint)) # inputs*time points

    inputs[0,1:9+randots_dur[nTrial]] = 1 # fixation period
    inputs[1:3,3:11+randots_dur[nTrial]] = 1 # two choice targets
    inputs[4:14,5:5+randots_dur[nTrial]] = sigmoid(neuron_resp[nTrial])
    if sure_trials[nTrial]==1:
        inputs[3,5+randots_dur[nTrial]+2:5+randots_dur[nTrial]+6] = 1
        
    inputs[15,2:10+randots_dur[nTrial]] = 1
    choice_prd = [5+randots_dur[nTrial]+5, 5+randots_dur[nTrial]+8]
    if choices[nTrial]==1:
        inputs[16, choice_prd[0] : choice_prd[1]] = 1
        inputs[1, 5+randots_dur[nTrial]+6] = 1
    elif choices[nTrial]==2:
        inputs[17, choice_prd[0] : choice_prd[1]] = 1
        inputs[2, 5+randots_dur[nTrial]+6] = 1
    elif choices[nTrial]==3:
        inputs[18, choice_prd[0] : choice_prd[1]] = 1
        inputs[3, 5+randots_dur[nTrial]+6] = 1
        
    if rewards[nTrial]==1:
        inputs[20,5+randots_dur[nTrial]+8:5+randots_dur[nTrial]+10] = 1
    if rewards[nTrial]==sure_reward:
        inputs[20,5+randots_dur[nTrial]+8:5+randots_dur[nTrial]+10] = sure_reward
    
    inputs[14,:] = 1-inputs[0:4,:].sum(axis=0)
    inputs[14,inputs[14,:]!=1] = 0
    inputs[19,:] = 1-inputs[15:19,:].sum(axis = 0)
    inputs[21,:] = 1
    inputs[21,inputs[20,:]!=0] = 0
    
    data_Sure.append([inputs.T])


training_guide = np.array([training_guide, trial_length]).squeeze().T.tolist()

data_Sure_Brief = {'coherences':coherences, 'choices':choices,'reward': rewards, 'training_guide':training_guide,
                   'acc_evi':acc_evi,'sure_trials':sure_trials,'randots_dur':randots_dur}


info = {'NumTrials':NumTrials,'sure_propt':sure_prop,'std_cont':std_con,
        'randots_durt':randots_dur,'num_neurons_dotst':num_neurons_dots,
        'threshold':threshold,'sure_reward':sure_reward,
        'coherence_list':coherence_list}

pathname = "../data/"
file_name = datetime.datetime.now().strftime("%Y_%m_%d")

data_name = 'SureTask_TrainingSet-' + file_name

n = 0
while 1: # save the model
    n += 1
    filename = pathname + data_name + '-' + str(n) + '.mat'
    if not os.path.isfile(filename):
        sio.savemat(filename,{'data_Sure':data_Sure,'data_Sure_Brief':data_Sure_Brief,'info':info})
        break
print("_" * 36)
print("training file {:6s} for sure task is saved".format(filename))

#data = sio.loadmat(filename)
#brief = data['data_TS_Brief'][0,0]
#brief['reward']

choices = np.array(choices)

for i in coherence_list:
    print('coherence:{:6f} | left:{:6d} | right:{:6d} | sure:{:6d}'.format(
            i,
            np.where(choices[coherences==i]== 1)[0].shape[0],
            np.where(choices[coherences==i]== 2)[0].shape[0],
            np.where(choices[coherences==i]== 3)[0].shape[0]
            ))

print((choices==3).mean())











