# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:37:22 2019

@author: YangLab_ZZW


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



def pac_space(all_resp_cell):
    ## pca
    pca_data = np.concatenate(all_resp_cell, axis=0).T
    eigvects, eigvals, mean_value = pca(pca_data)
    
    return eigvects, eigvals

def resp_grouping(all_resp_cell, choice, rt,eigvects):
    resp_group = {"choice_left":[],"choice_right":[],"choice_all":[],
                  "start_left":[], "start_right":[] ,"start_all":[]}
    distance = copy.deepcopy(resp_group)
    ste_distance = copy.deepcopy(resp_group)
    
    for curr_trial in range(len(all_resp_cell)):
        resp_curr_trial = all_resp_cell[curr_trial]
        resp = np.matmul(resp_curr_trial, eigvects)
#        resp = resp_curr_trial
        if rt[curr_trial] < 6:
            continue
        if choice[curr_trial] == 1.0:
            resp_group["start_left"].append(resp[:18,:])
            resp_group["choice_left"].append(resp[-20:,:])
        elif choice[curr_trial] == -1.0:
            resp_group["start_right"].append(resp[:18,:])
            resp_group["choice_right"].append(resp[-20:,:])
        else:
            raise Exception ('unknown choice')
        resp_group["start_all"].append(resp[:18,:])
        resp_group["choice_all"].append(resp[-20:,:])
        
    for i, value in enumerate(resp_group.keys()):
        distance[value], ste_distance[value]  = vectorVar(resp_group[value])

    return resp_group, distance, ste_distance

def plot_varce(distance):
    fig = plt.figure()
    for i, value in enumerate(distance.columns.values):
        distance_value = np.vstack(distance[value].values)
        if 'start' in value:
            x = range(distance_value.shape[1])
        elif 'choice' in value:
            x = np.linspace(30, 30+distance_value.shape[1]-1, distance_value.shape[1])
        else:
            continue                
        plt.errorbar(x,distance_value.mean(axis=0),yerr = stats.sem(distance_value, axis=0),label = value)
    plt.legend()
#    fig.savefig('../figs/varce.eps', format='eps', dpi=1000)
    

def var_ce(path_files, threshold = 0.8):
    
    df_distance = pd.DataFrame([], columns = {'label',
                               'choice_left','choice_right','choice_all',
                               'start_left','start_right','start_all'})

    for i, file in enumerate(path_files):
        ## load files
        paod, trial_briefs = load(file)
        _, choice, shape, _, _= get_bhvinfo(paod,trial_briefs)

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

    return df_distance


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
    
    df_distance = var_ce(file_paths)
    
    plot_varce(df_distance)

    

if __name__ == '__main__':
    main()

