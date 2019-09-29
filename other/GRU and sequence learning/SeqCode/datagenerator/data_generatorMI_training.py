#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:41:47 2019

@author: Zhewei Zhang


Reproduce fake data of the multisensory integration task

1st-12th: about the visual stimili
    first input represents the fixation point
    second/third inputs denote the two choice targets
    4th-11th: are the neurons with different perfer orientation for modality 1
    12th-19th: are the neurons with different perfer orientation for modality 1
    20th: no visual stimuli

21th-24th: about the movement; fixation/left tar/right tar/break

25th-26th: reward/no reward


The difference from version 1 is the way of simulating the neural response; AND 
    add the training guide for variable trial length


"""

import os
import math
import time
import datetime
import numpy as np
import scipy.io as sio
import scipy.stats as sts
import matplotlib.pyplot as plt
#%matplotlib inline

np.random.seed(int(time.time()))


# In[0]: hyperparameters

s_vi=45
s_ve=45
g_vi=300
g_ve=300
NumNeurons = 8
tuning_curve_disc=np.linspace(0, 180, NumNeurons, endpoint = True)


NumTrials = int(7.5e5)
NumTimes = 5 # the maximum value of NumTimes is 5
per_ori_set = tuning_curve_disc

NumStimuli = 180
Stimuli_range = [0, 180]

# In[0]: toolbox

def sigmoid(x):
  return 1 / (1 + np.exp(1-x))

#def sigmoid(x):
#  return 1 / (1 + np.exp(-x))

def norm_pdf(x, mean, std):
    return np.exp(-(x-mean)**2/(2*std**2))/(std*np.sqrt(2*np.pi))

def rates_2(s):
    vi_rates=g_vi*norm_pdf(s,tuning_curve_disc,s_vi)
    ve_rates=g_ve*norm_pdf(s,tuning_curve_disc,s_ve)
    return vi_rates, ve_rates


#def rates(s):
#    vi_rates=g_vi*sts.norm.pdf(s,tuning_curve_disc,s_vi)
#    ve_rates=g_ve*sts.norm.pdf(s,tuning_curve_disc,s_ve)
#    return vi_rates, ve_rates

def counts(s):
#    vi_rates,ve_rates=rates(s)  
    vi_rates,ve_rates=rates_2(s)  # faster
    vi_sp=np.random.poisson(vi_rates)
    ve_sp=np.random.poisson(ve_rates)
    return vi_sp,ve_sp


# generate the rates_list for shorter running time
rates_list = []
post_len=180
ss=np.linspace(0,180,post_len)
for sc in range(post_len):
    s=ss[sc]
    rates_list.append(rates_2(s))

def get_posterior(r_counts, post_len=180):
        
#    ss=np.linspace(0,180,post_len) 
    log_posterior_com=np.zeros(post_len)
    log_posterior_ve=np.zeros(post_len)
    log_posterior_vi=np.zeros(post_len)

    for sc in range(post_len):
        rates_s = rates_list[sc]
        log_posterior_vi[sc]=np.sum(-rates_s[0]+r_counts[0]*np.log(rates_s[0]))
        log_posterior_ve[sc]=np.sum(-rates_s[1]+r_counts[1]*np.log(rates_s[1]))
    log_posterior_com = log_posterior_vi+log_posterior_ve

    a = [log_posterior_vi,log_posterior_ve, log_posterior_com]
    for i, k in enumerate(a):
        k=np.exp(k)/sum(np.exp(k))
        a[i]=k
        
    estimates=[]
    for k in enumerate(a):
#        estimate = np.dot(ss,k[1])
#        estimate2 = ss[np.argmax(k[1])]
        estimate3 = k[1][:int(len(k[1])/2)].sum() > k[1][int(len(k[1])/2):].sum()
        estimates.append(estimate3)
    return a, estimates


def package(resp, signal = True):
    baseline = 0.5
    if signal:
        return sigmoid(resp)
    else:
        return baseline + np.zeros(resp.shape)

# In[0]: parameters
stimulus_set = np.linspace(Stimuli_range[0], Stimuli_range[1], NumStimuli, endpoint = True)#[:-1]
directions_idx = np.random.choice(range(len(stimulus_set)), NumTrials)
directions = stimulus_set[directions_idx]

directions = np.random.choice(stimulus_set, NumTrials)
modality = np.random.choice([0,1], NumTrials)
resp = np.zeros((NumTrials, NumNeurons*2, NumTimes))
estimates = np.zeros((NumTrials, 3))
#posteriors = []
start = time.time()
for nTrial in range(NumTrials):
    r_counts_vi, r_counts_ve = [], []
    if (nTrial + 1) % 25000 == 0 :
        print("Trial: {:6d} || time consumed: {:6f} ".format(nTrial + 1, time.time() - start))
        start = time.time()
    for nTime in range(NumTimes):
        r_count_vi, r_count_ve = counts(directions[nTrial])
        r_counts_vi.append(r_count_vi)
        r_counts_ve.append(r_count_ve)
        if modality[nTrial]==0:
            resp[nTrial,:NumNeurons,nTime] = package(r_count_vi)
            resp[nTrial,NumNeurons:,nTime] = package(r_count_ve, signal = False)
        elif modality[nTrial]==1:
            resp[nTrial,:NumNeurons,nTime] = package(r_count_vi, signal = False)
            resp[nTrial,NumNeurons:,nTime] = package(r_count_ve)
    r_counts = [r_counts_vi, r_counts_ve]
    _, estimate = get_posterior(r_counts)
    estimates[nTrial,:] = estimate
    
# In[]:

n_input = 26
trial_length = 17
rewards = []
choices = []
data_MI = []
for nTrial in range(NumTrials):
    inputs = np.zeros((n_input,trial_length))
    inputs[0,1:10] = 1 # fixation period
    inputs[1:3,3:12] = 1 # two choice targetsdata_MI_Brief = {'directions':directions, 'modality':modality,
    inputs[3:19,5:5+NumTimes] = resp[nTrial,:,:]
    
    inputs[20,1:11] = 1
    choices.append(estimates[nTrial,modality[nTrial]])
    
    if directions[nTrial] < 90:
        temp = 1 if choices[-1]==1 else 0
    elif directions[nTrial] > 90:
        temp = 1 if choices[-1]==0 else 0
    else:
        temp = 1 if np.random.rand()>0.5 else 0
    rewards.append(temp)
        
    if choices[-1]==0:
        inputs[21,11:14] = 1
        inputs[1,12] = 1
    elif choices[-1]==1:
        inputs[22,11:14] = 1
        inputs[2,12] = 1
        
    if rewards[-1]==1:
        inputs[24,14:16] = 1
    if rewards[-1]==0:
        inputs[25,14:16] = 1
    
    inputs[19,:] = 1-inputs[0:3,:].sum(axis=0)
    inputs[19,inputs[19,:]!=1] = 0
    inputs[23,:] = 1-inputs[20:23,:].sum(axis = 0)
    inputs[25,:] = 1-inputs[24,:]
    data_MI.append([inputs.T])


training_guide = np.array([rewards, trial_length+np.zeros((len(rewards),))]).squeeze().astype(np.int).T.tolist()


data_MI_Brief = {'directions':directions, 'modality':modality,
                 'choices':choices,'reward':rewards,'estimates':estimates, 'training_guide':training_guide}
info = {'NumTrials':NumTrials,'NumStimuli':NumStimuli,'NumNeurons':NumNeurons,
        'Stimuli_range':Stimuli_range,'stimulus_set':stimulus_set,
        'per_ori_set':per_ori_set,'s_vi':s_vi,'s_ve':s_ve,'g_vi':g_vi,'g_ve':g_ve}

pathname = "../data/"
file_name = datetime.datetime.now().strftime("%Y_%m_%d")
data_name = 'MultInt_TrainingSet-' + file_name
n = 0
while 1: # save the model
    n += 1
    filename = pathname + data_name + '-' + str(n) + '.mat'
    if not os.path.isfile(filename):
        sio.savemat(filename, 
                    {'data_MI':data_MI, 'data_MI_Brief':data_MI_Brief, 'info':info})
        break
print("_" * 36)
print("training file is saved:  {:6s}".format(filename))

