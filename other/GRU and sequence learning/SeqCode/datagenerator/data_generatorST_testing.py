#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:50:37 2019

@author: Zhewei Zhang

5 inputs represent visual stimulus 

6th~9th inputs representing the movement/choice

9th-10th inpust denotes the reward states
"""

import os
import datetime
import numpy as np
import scipy.io as sio

data_ST = []
n_input = 10
trial_length = 6

NumTrials = 5000
trans_prob = 0.8
reward_prob = 0.8
block_size = 70

info = {'NumTrials':NumTrials,'reward_prob':reward_prob,'block_size':block_size,'trans_prob':trans_prob}

temp = np.hstack((np.ones((block_size,)), np.zeros((block_size,)) ))
blocks = np.tile(temp,int(NumTrials/(block_size*2)))
# 
lost_trialsNum = NumTrials - blocks.size
if lost_trialsNum <= block_size:
    temp = np.ones((lost_trialsNum,))
else:
    temp = np.hstack((np.ones((block_size,)), np.zeros((lost_trialsNum - block_size,))))
blocks = np.hstack((blocks, temp))

trans_probs = trans_prob*np.ones(NumTrials,)
reward_probs = reward_prob*blocks + (1-reward_prob)*(1-blocks)


inputs = [
        [0., 0., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0.]
        ]


data_ST = [[np.array(inputs).T]]*NumTrials

data_ST_Brief = {'reward_prob_1':reward_probs, 'trans_probs':trans_probs, 'block_size':block_size, 'block':blocks}


pathname = "../data/"
file_name = datetime.datetime.now().strftime("%Y_%m_%d")
data_name = 'SimpTwo_TestingSet-' + file_name
n = 0
while 1: # save the model
    n += 1
    if not os.path.isfile(pathname+data_name+'-'+str(n)+'.mat'):
        sio.savemat(pathname+data_name+'-'+str(n)+'.mat',
                {'data_ST':data_ST,
                 'data_ST_Brief':data_ST_Brief,
                 })
        print("_" * 36)
        print("testing file for simplified two step task is saved")
        print("file name:" + pathname+data_name+'-'+str(n)+'.mat')
        break
filename = pathname + data_name + '-' + str(n) + '.mat'







