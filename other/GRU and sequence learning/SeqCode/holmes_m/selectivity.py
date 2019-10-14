# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:24:16 2018

@author: YangLab_ZZW
"""
import copy
import tkinter as tk
import numpy as np
import pandas as pd
from scipy import linalg, stats
from tkinter import filedialog
from lrtools import linearReg

"""
demonstrate the neuronal selectivity of the logLR, abs(logLR), urgency and choice

"""
from toolkits_2 import get_hidden_resp_all, get_bhvinfo, load

def plglm_constructer(trials, choices, shapes):

    sum_weight, sum_weight_loc, urgence, choice, choice_loc = [], [], [], [], []
    total_length = 0
    for i in range(shapes.count().ontime):
        if not choices.status.iloc[i]:
            continue
        # sum weight
        shape_ontime = shapes.ontime.iloc[i]
        shape_num = shapes.on.iloc[i]
        tempweight = shapes.tempweight.iloc[i][shape_num-1]
        sumweight = np.cumsum(tempweight)
        sum_weight.append(sumweight)
        sum_weight_loc.append(shape_ontime+3+total_length) # time point: before choice, the prediction of choice
        # urgency
        urgence.append(shape_num)
        # choice
        choice_opportunity = np.zeros(shape_num.shape).tolist()
        if choices.left.iloc[i]==1:
            choice_opportunity[-1] = 1
            choice.extend(choice_opportunity)
        else:
            choice_opportunity[-1] = -1
            choice.extend(choice_opportunity)
        choice_loc.append(choices.time.iloc[i]-1+total_length)# time point: before choice, the prediction of choice
        total_length += trials.length.iloc[i]
        
    sum_weight = np.concatenate(np.array(sum_weight))
    sum_weight_loc = np.concatenate(np.array(sum_weight_loc))
    urgence = np.concatenate(np.array(urgence))
    urgence_loc = sum_weight_loc

    return sum_weight, sum_weight_loc, urgence, urgence_loc, choice, choice_loc    


def lm_fit(sum_weight, sum_weight_loc, urgence, urgence_loc, choice, choice_loc, resp):
    params = []
    p_values = []
    # summed weight/abs(summed weight)/urgency/choice, all in one
    x = np.hstack([sum_weight.reshape(-1,1),
                   np.abs(sum_weight.reshape(-1,1)), 
                   np.array(urgence).reshape(-1,1),
                   np.array(choice).reshape(-1,1)])
    for i in range(5):
        result = linearReg(x,resp,sum_weight_loc+i)
        params.append(result['params'].tolist())
        p_values.append(result['p_values'].tolist())
        
    return params,p_values

def regress_all(all_resp_cell, trial, choice, shapes):
    sum_weight,sum_weight_loc, urgence, urgence_loc, choice, choice_loc = plglm_constructer(trial, choice, shapes)
    all_resp_array = np.concatenate(all_resp_cell, axis=0)
    num_Neuron = all_resp_array.shape[1]

    result_all = {'params':[],'p_values':[]}

    for i in range(num_Neuron):
        resp = copy.deepcopy(all_resp_array[:,i])
        params,p_values = lm_fit(sum_weight,sum_weight_loc,
                               urgence, urgence_loc,
                               choice, choice_loc,
                               resp)
        result_all['params'].append(params)
        result_all['p_values'].append(p_values)
        
        if (i+1)%(2**4) == 0:
            print('regression:{}th neuron'.format(i+1))
    return result_all

def selectivity_test(file_paths):
    df_selectivity = pd.DataFrame([], columns = {'label','evid','conf','urgy','choi'})
    df_numSign = pd.DataFrame([], columns = {'label','evid','conf','urgy','choi'})
    df_orderSign = pd.DataFrame([], columns = {'label','evid','conf','urgy','choi'})
    p_threshold = 0.05/(128*4)
    for i, file in enumerate(file_paths):
        paod, trial_briefs = load(file)
        trial, choice, shapes, _, _ = get_bhvinfo(paod,trial_briefs)
        all_resp_cell = get_hidden_resp_all(paod,trial_briefs) # response of neurons in hidden layer
        result_all = regress_all(all_resp_cell, trial, choice, shapes)
        
        p_value = np.ones([5,128,5])# 5 parameters (4+1 bais term)by 128 neurons by 5 time point
        params = np.ones([5,128,5])
        for ii in range(128): 
            p_value[:,ii,:] = np.array(result_all['p_values'][ii]).T
            params[:,ii,:] = np.array(result_all['params'][ii]).T
                
        df_selectivity.loc[i] = {'label': file,
                          'evid':[p_value[1,:,:], params[1,:,:]],
                          'conf':[p_value[2,:,:], params[2,:,:]],
                          'urgy':[p_value[3,:,:], params[3,:,:]],
                          'choi':[p_value[4,:,:], params[4,:,:]]
                           }
        df_numSign.loc[i] = {'label': file,
                          'evid': np.sum(np.all(p_value[1,:,:]<p_threshold, axis =1)),
                          'conf': np.sum(np.all(p_value[2,:,:]<p_threshold, axis =1)),
                          'urgy': np.sum(np.all(p_value[3,:,:]<p_threshold, axis =1)),
                          'choi': np.sum(np.all(p_value[4,:,:]<p_threshold, axis =1)),
                           }
        df_orderSign.loc[i] = {'label': file,
                          'evid': np.where(np.all(p_value[1,:,:]<p_threshold, axis =1)),
                          'conf': np.where(np.all(p_value[2,:,:]<p_threshold, axis =1)),
                          'urgy': np.where(np.all(p_value[3,:,:]<p_threshold, axis =1)),
                          'choi': np.where(np.all(p_value[4,:,:]<p_threshold, axis =1)),
                           }
    return df_selectivity, df_numSign, df_orderSign

def main():
    print("start")
    print("select the files")
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(parent = root,
                                            title = 'Choose a file',
                                            filetypes = [("HDF5 files", "*.hdf5")]
                                            )
    ##
    df_selectivity, df_numSign, df_orderSign = selectivity_test(file_paths)
    print("evidence encoding: {:6f}  + {:6f} neuons".format(df_numSign.evid.mean(),stats.sem(df_numSign.evid)))
    print("confidence encoding: {:6f}  + {:6f} neuons".format(df_numSign.conf.mean(),stats.sem(df_numSign.conf)))
    print("urgency encoding: {:6f}  + {:6f} neuons".format(df_numSign.urgy.mean(),stats.sem(df_numSign.urgy)))
    print("choice encoding: {:6f}  + {:6f} neuons".format(df_numSign.choi.mean(),stats.sem(df_numSign.choi)))

if __name__ == '__main__':
    main()


