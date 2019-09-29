
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:24:16 2018

@author: YangLab_ZZW

only used for target dimension reducetion, segamented the firing rate in epoches,
and do linear regression on chocie 
"""
import numpy as np
import copy
from lrtools import linearReg
import matplotlib.pyplot as plt

def plglm_constructer(trials, choices, shapes, rewards):

    choice_label = []
    possible_choice_loc = []
    total_length = 0
    for i in range(shapes.count().ontime):
        if not choices.status.iloc[i]:
            continue
        # sum weight
        shape_ontime = shapes.ontime.iloc[i]
        possible_choice_loc.append(shape_ontime+3+total_length) # time point: before choice, the prediction of choice
        if choices.left.iloc[i]==1:
#            choice_label.append([0]*(shapes.rt.iloc[i]-1)+[1])
            choice_label.append([1]*shapes.rt.iloc[i])
        else:
#            choice_label.append([0]*(shapes.rt.iloc[i]-1)+[-1])
            choice_label.append([-1]*shapes.rt.iloc[i])
        
        total_length += trials.length.iloc[i]

    choice_label = np.concatenate(np.array(choice_label))# time point: before choice, the prediction of choice
    possible_choice_loc = np.concatenate(np.array(possible_choice_loc))

    return choice_label,possible_choice_loc    


def lm_fit(choice_label,possible_choice_loc,resp):
    params = []
    p_values = []
    for i in range(5):
        # choice; first--fifth time point in each epoch
        x = choice_label.reshape(-1,1)
        result = linearReg(x,resp,possible_choice_loc-i)
        params.append(result['params'].tolist())
        p_values.append(result['p_values'].tolist())

    return params,p_values


def regress_tdr(all_resp_cell, trial, choice, shapes, reward):
    
    all_resp_array = np.concatenate(all_resp_cell, axis=0)
    assert all_resp_array.shape[0] == trial.length.sum()
    
    num_Neuron = all_resp_array.shape[1]
    result_all = {'params':[],'p_values':[]}
    choice_label, possible_choice_loc = plglm_constructer(trial, choice, shapes, reward)

    for i in range(num_Neuron):
        resp = copy.deepcopy(all_resp_array[:,i])
        params,p_values = lm_fit(choice_label, possible_choice_loc, resp)
        result_all['params'].append(params)
        result_all['p_values'].append(p_values)
        if (i+1)%(2**4) == 0:
            print('regression:{}th neuron'.format(i+1))
    return result_all

