#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:36:57 2019

@author: Zhewei Zhang

test the activities variability in the pca space

"""

import copy
import numpy as np
import pandas as pd
import tkinter as tk
from scipy import stats
from tkinter import filedialog
import matplotlib.pyplot as plt
from toolkits_2 import load, vectorVar, pca, get_hidden_resp_all, get_bhvinfo
from selectivity import selectivity_test, regress_all
from lrtools import sim_linreg
from scipy.stats import spearmanr

def pac_space(all_resp_cell):
    ## pca
    pca_data = np.concatenate(all_resp_cell, axis=0).T
    eigvects, eigvals, mean_value = pca(pca_data)
    
    return eigvects, eigvals


def variance(resp):
    resp = np.dstack(resp)
    nTime, nNeuron, nTrial = resp.shape

    mean_var, ste_var = np.zeros((nNeuron,nTime)), np.zeros((nNeuron,nTime))
    for i in range(nNeuron):
        resp_curr = resp[:,i,:]
#        mean_var[i] = np.nanmean(np.abs(resp_curr - np.tile(resp_curr.mean(axis=1),[nTrial,1]).T), axis = 1)
        mean_var[i] = np.nanstd(resp_curr,axis=1)
        ste_var[i] = stats.sem(resp_curr,axis=1,nan_policy = 'omit')
    return mean_var, ste_var

def resp_grouping(all_resp_cell, choice, rt,eigvects):
    resp_group = {"choice_left":[],"choice_right":[],"choice_all":[],
                  "start_left":[], "start_right":[] ,"start_all":[]}
    distance = copy.deepcopy(resp_group)
    ste_distance = copy.deepcopy(resp_group)
    
    for curr_trial in range(len(all_resp_cell)):
        resp_curr_trial = all_resp_cell[curr_trial]
#        resp = np.matmul(resp_curr_trial, eigvects)
        resp = resp_curr_trial
        if rt[curr_trial] < 6:
            continue
        if choice[curr_trial] == 1.0:
            resp_group["start_left"].append(resp[:28,:])
            resp_group["choice_left"].append(resp[-30:,:])
        elif choice[curr_trial] == -1.0:
            resp_group["start_right"].append(resp[:28,:])
            resp_group["choice_right"].append(resp[-30:,:])
        else:
            raise Exception ('unknown choice')
        resp_group["start_all"].append(resp[:28,:])
        resp_group["choice_all"].append(resp[-30:,:])
        
    for i, value in enumerate(resp_group.keys()):
        distance[value], ste_distance[value]  = variance(resp_group[value])
#        distance[value], ste_distance[value]  = vectorVar(resp_group[value])

    return resp_group, distance, ste_distance


def plot_variance(df_distance, df_signNeuron):
    
    variance = {'evid_pos':pd.DataFrame(),'evid_neg':pd.DataFrame()}
    for neuoron_label, sigNeuron in df_signNeuron.items():
        if neuoron_label == 'label':
            continue
        for sigNeuron_curr, (index, distance) in zip(sigNeuron, df_distance.iterrows()):
            for time_label, distance_per in distance.items():
                if time_label == 'label':
                    continue
                if time_label not in variance[neuoron_label].columns:
                    variance[neuoron_label][time_label] = [[distance_per[sigNeuron_curr,:]]]
                else:
                    variance[neuoron_label][time_label][0].append(distance_per[sigNeuron_curr,:])
        
    colorset = {'start_all':'b', 'choice_pref':'r', 'choice_nonpref':'g'}
    groups_value = {'start_all':[[],[],[],[],[]], 'choice_left':[[],[],[],[],[]], 'choice_right':[[],[],[],[],[]]}
    inver = {'choice_left':'choice_right','choice_right':'choice_left'}
    for neuoron_label in variance:
        for time_label in groups_value:
            if time_label == 'label':
                continue
            distance_value = np.vstack(variance[neuoron_label][time_label][0])
            if 'start' in time_label:
                for ii in [0,1,2,3,4]:
                    groups_value[time_label][ii].extend(distance_value[:,3+5*ii:3+5+5*ii])
            elif 'choice' in time_label:
                if 'neg' in neuoron_label:
                    time_label_  = inver[time_label]
                else:
                    time_label_  = time_label
                for ii in [0,1,2,3,4]:
                    groups_value[time_label_][ii].extend(distance_value[:,5*ii:5+5*ii])
   
    groups_value = {'start_all':groups_value['start_all'],
                    'choice_pref':groups_value['choice_left'],
                    'choice_nonpref':groups_value['choice_right']}
    fig = plt.figure()
    for i, (label, values) in enumerate(groups_value.items()):
        print(label)
        values =  np.array(groups_value[label]).squeeze()
        n_runs = values.shape[1]
        n_epoch = values.shape[0]
        n_time = values.shape[2]
        if 'start' in label:
            x = np.repeat(np.arange(n_epoch),n_runs).reshape(-1,1)
            y = values[:,:,-2:].mean(axis=2).reshape(-1,1)
            coef, p = spearmanr(x, y)
            print(coef, p)
            
            x = np.arange(5)
            y = values[:,:,-2:].reshape(values.shape[0],-1).mean(axis=1)
            y_err = stats.sem(values[:,:,-2:].reshape(values.shape[0],-1), axis=1)
        if 'choice' in label:
            x = np.repeat(np.arange(n_epoch-1),n_runs).reshape(-1,1)
            y = values[:n_epoch-1,:,-2:].mean(axis=2).reshape(-1,1)
            result = sim_linreg(x,y)
            coef, p = spearmanr(x, y)
#            print(result)
            print(coef, p)
            
            x = np.arange(8,13,1)
            y = values[:,:,-2:].reshape(values.shape[0],-1).mean(axis=1)
            y_err = stats.sem(values[:,:,-2:].reshape(values.shape[0],-1), axis=1)
        plt.errorbar(x,y,yerr=y_err,label=label)
    plt.legend()
    fig.savefig('../figs/std_pref_non.eps', format='eps', dpi=1000)

    
    
    

def var_ce(file_paths, threshold = 0.8):
    
    df_distance = pd.DataFrame([], columns = {'label',
                               'choice_left','choice_right','choice_all',
                               'start_left','start_right','start_all'})
    df_signNeuron = pd.DataFrame([], columns = {'label','evid_pos','evid_neg'})

    for i, file in enumerate(file_paths):
        ## load files
        paod, trial_briefs = load(file)
        trial, choice, shape, _, _= get_bhvinfo(paod,trial_briefs)

        all_resp_cell = get_hidden_resp_all(paod, trial_briefs)
        eigvects, eigvals = pac_space(all_resp_cell)
        explained_var = np.cumsum(eigvals/np.sum(eigvals))
        n_dimension = np.where(explained_var>threshold)[0][0]
    
        ## group the response
        _, distance, _ = resp_grouping(all_resp_cell, choice.left, shape.rt, eigvects[:,:n_dimension])
        df_distance.loc[i] = {'label': file,
                       'choice_left':distance['choice_left'],
                       'choice_right':distance['choice_right'],
                       'choice_all':distance['choice_all'],
                       'start_left':distance['start_left'],
                       'start_right':distance['start_right'],
                       'start_all':distance['start_all']
                       }
        
        result_all = regress_all(all_resp_cell, trial, choice, shape)
        
        p_value = np.ones([5,128,5])
        params = np.ones([5,128,5])# 5 parameters (4+1 bais term)by 128 neurons by 5 time point
        for ii in range(128): 
            p_value[:,ii,:] = np.array(result_all['p_values'][ii]).T
            params[:,ii,:] = np.array(result_all['params'][ii]).T
        p_threshold = 0.05/(128*5)
        signNeuron = np.where(np.all(p_value[1,:,:]<p_threshold, axis =1))
        df_signNeuron.loc[i] = {'label': file, 
                                'evid_pos': np.intersect1d(signNeuron, np.where(np.all(params[1,:,:]>-1e-10, axis =1))),
                                'evid_neg': np.intersect1d(signNeuron, np.where(np.all(params[1,:,:]< 1e-10, axis =1)))
                                }
    return df_distance, df_signNeuron


def main():
    print("start")
#    %matplotlib auto
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
            parent=root,title='Choose a file',
            filetypes=[("HDF5 files", "*.hdf5")]
            )
    print("select the files")
    
    df_distance, df_signNeuron = var_ce(file_paths)
    
    plot_variance(df_distance, df_signNeuron)

    

if __name__ == '__main__':
    main()

