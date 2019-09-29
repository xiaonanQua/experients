# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:24:16 2018

@author: YangLab_ZZW
"""
import os
import tkinter as tk
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from tkinter import filedialog
from toolkits_2 import get_bhvinfo, load
from toolkits import base642obj3
from task_info import rtshapebrief_config

"""

for lesion study, show how many trials in which the network does not a decision, 
and its error type

"""

input_setting = rtshapebrief_config()
choice_pos = input_setting['choice']
def groups_files(file_paths):
    
    # group the files based on the lesion type
    group = []
    for nfile in file_paths:
        path, file = os.path.split(nfile)
        group.append(file.split('-')[0])
        
    files_pd = pd.DataFrame([list(file_paths),group],['name','lesion'])
    files_pd = files_pd.T
    files_groups = files_pd.name.groupby([files_pd.lesion])
    ncondition = files_pd.lesion.nunique()
    condition = files_pd.lesion.unique()

    return files_groups, condition, ncondition

def trial_type(file_paths, num_Trials=5000):
    df_trials = pd.DataFrame([], columns = {'label','er_resp', 'no_resp', 'abt_fix', 'abt_cho', 'wrg_tim','finished','trial_type'})
    largest_possile_triallength = 145
    for ith, file in enumerate(file_paths):
        paod, trial_briefs = load(file)
        trials, choices, shapes, _, _ = get_bhvinfo(paod,trial_briefs)
        trial_type = [] # left or right
        er_resp = [] # make a choice before the first shape onset
        no_resp = [] # no eye movement within 25 epochs
        abt_fix = [] # abort during fixation on FP
        abt_cho = [] # abort during fixation on choice
        wrg_tim = [] # make a choice at a wrong time
#        cg_mind = [] # change mind after the decision has been made
        for i in range(num_Trials):
            fd, gd, rd, nd = paod.get_neuron_behavior_pair(i)
            brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
            trial_type.append(brief_info['trialtype'])
            choice_vector = gd[:,choice_pos[0]:choice_pos[-1]+1]# fixtion/left target/right target/
            first_left  = np.where(choice_vector[:, 1]==1)[0][0] if np.where(choice_vector[:, 1]==1)[0].size!=0 else 1e5
            first_right = np.where(choice_vector[:, 2]==1)[0][0] if np.where(choice_vector[:, 2]==1)[0].size!=0 else 1e5
            first_nofix = np.where(choice_vector[:, 3]==1)[0][0] if np.where(choice_vector[:, 3]==1)[0].size!=0 else 1e5
            
            if first_left>1e3 and first_right>1e3 and first_nofix>1e3: 
                #keep fixation
                no_resp.append(i)
            elif first_nofix < min(first_left, first_right, largest_possile_triallength): 
                # no fixation before choices, abort fixtion from FP
                abt_fix.append(i)
            elif (first_left < 6 or first_right<6) and first_nofix>min(first_left, first_right):
                # make a choice before first shape onset, and no abort before choice
                er_resp.append(i)
            elif min(first_left, first_right) < largest_possile_triallength and (min(first_left, first_right)-1)%5!=0 :
                # make a choice in a wrong time step
                wrg_tim.append(i)
            elif first_left < first_right:
                # choose left and not keep fixation on the left
                if choice_vector[first_left+1,1]!=1 or choice_vector[first_left+2,1]!=1:
                    abt_cho.append(i) 
            elif first_right <first_left:
                # choose right and not keep fixation on the right
                if choice_vector[first_right+1,2]!=1 or choice_vector[first_right+2,2]!=1:
                    abt_cho.append(i)

        all_defined_trial = np.sort(np.hstack((trials.num.values, er_resp, no_resp, abt_fix, wrg_tim, abt_cho)))
        if np.unique(all_defined_trial).size != num_Trials or all_defined_trial.size != num_Trials:
            raise Exception('some not defined or doubly defined elements are found')
        df_trials.loc[ith] = {'label'  : file,
                            'er_resp': er_resp, 
                            'no_resp': no_resp,
                            'abt_fix': abt_fix,
                            'abt_cho': abt_cho,
                            'wrg_tim': wrg_tim,
                            'finished': trials.num.values,
                            'trial_type':trial_type,
                            }
        
    return df_trials

def plotting(df_trials, ncondition):
    fig_w, fig_h = (15, 10)
    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
    labels = ['er_resp','no_resp','abt_fix','abt_cho','wrg_tim','finished']
    # dirty codes........
    for i, label in enumerate(labels):
        if i == 0:
            left_value      = np.vstack(df_trials[label].values)[:,0]
            left_value_sem  = np.vstack(df_trials[label+'_sem'].values)[:,0]
            right_value     = np.vstack(df_trials[label].values)[:,1]
            right_value_sem = np.vstack(df_trials[label+'_sem'].values)[:,1]
        else:
            left_value      = np.vstack((left_value, np.vstack(df_trials[label].values)[:,0]))
            left_value_sem  = np.vstack((left_value_sem, np.vstack(df_trials[label+'_sem'].values)[:,0]))
            right_value     = np.vstack((right_value, np.vstack(df_trials[label].values)[:,1]))
            right_value_sem = np.vstack((right_value_sem, np.vstack(df_trials[label+'_sem'].values)[:,1]))

    fig = plt.figure()
    for i, label in enumerate(labels):
        plt.subplot(211)
        plt.bar(1.5*np.arange(ncondition)+0.15*i-0.4, left_value[i,:], yerr = left_value_sem[i,:], width=0.12,label = label)
        plt.subplot(212)
        plt.bar(1.5*np.arange(ncondition)+0.15*i-0.4, right_value[i,:], yerr = right_value_sem[i,:], width=0.12,label = label)
    plt.legend()
    plt.title('right target is correct')
    plt.xticks(1.5*np.arange(ncondition),df_trials.label.values)
    plt.subplot(211)
    plt.title('left target is correct')
    plt.legend()
    plt.xticks(1.5*np.arange(ncondition),df_trials.label.values)
    fig.savefig('../figs/trial type_2.eps', format='eps', dpi=1000)

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
    files_groups, condition, ncondition = groups_files(file_paths)
    
    df = pd.DataFrame([], columns = {'label','er_resp', 'no_resp', 'abt_fix', 'abt_cho', 'wrg_tim','finished',
                      'er_resp_sem', 'no_resp_sem', 'abt_fix_sem', 'abt_cho_sem', 'wrg_tim_sem','finished_sem'})
    for ith, files_group in enumerate(files_groups):
        df_trials = trial_type(files_group[1])

        left = {'er_resp': [], 'no_resp': [], 'abt_fix': [], 'abt_cho': [], 'wrg_tim': [], 'finished':[]}
        right = {'er_resp': [], 'no_resp': [], 'abt_fix': [], 'abt_cho': [], 'wrg_tim': [], 'finished':[]}
        labels = []
        for i in range(df_trials.label.count()):
            num_left = np.sum(np.array(df_trials.iloc[i].trial_type)==1)
            num_right = np.sum(np.array(df_trials.iloc[i].trial_type)==0)
            for ii, key in enumerate(df_trials.columns):
                if 'label' in key or 'trial_type' in key:
                    continue
                labels.append(key)
                left[key].append(len(np.intersect1d(np.where(np.array(df_trials.iloc[i].trial_type)==1)[0], df_trials.iloc[i][key]))/num_left)
                right[key].append(len(np.intersect1d(np.where(np.array(df_trials.iloc[i].trial_type)==0)[0], df_trials.iloc[i][key]))/num_right)
        
        df.loc[ith] = {'label':                files_group[0],
                    'er_resp':               [np.mean(left['er_resp']),   np.mean(right['er_resp'])],
                    'er_resp_sem':           [stats.sem(left['er_resp']), stats.sem(right['er_resp'])],
                    'no_resp':               [np.mean(left['no_resp']),   np.mean(right['no_resp'])],
                    'no_resp_sem':           [stats.sem(left['no_resp']), stats.sem(right['no_resp'])],
                    'abt_fix':               [np.mean(left['abt_fix']),   np.mean(right['abt_fix'])],
                    'abt_fix_sem':           [stats.sem(left['abt_fix']), stats.sem(right['abt_fix'])],
                    'abt_cho':               [np.mean(left['abt_cho']),   np.mean(right['abt_cho'])],
                    'abt_cho_sem':           [stats.sem(left['abt_cho']), stats.sem(right['abt_cho'])],
                    'wrg_tim':               [np.mean(left['wrg_tim']),   np.mean(right['wrg_tim'])],
                    'wrg_tim_sem':           [stats.sem(left['wrg_tim']), stats.sem(right['wrg_tim'])],                 
                    'finished':        [np.mean(left['finished']),   np.mean(right['finished'])],
                    'finished_sem':    [stats.sem(left['finished']), stats.sem(right['finished'])]              
                    }
    
        
    plotting(df, ncondition)

if __name__ == '__main__':
    main()


