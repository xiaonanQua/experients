#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:36:20 2019

@author: Zhewei Zhang
"""

import os
import numpy as np
import pandas as pd
import tkinter as tk
from scipy import stats
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tkinter import filedialog
from toolkits_2 import get_multinfo, load, get_hidden_resp_sure
from scipy.signal import savgol_filter
"""

representing the relation between the logRT and rt; 

fit the psychmetric curve

fig1 a/c
"""

def cum_gaussian(x, x0, k):
    return stats.norm.cdf((x-x0)/k)

def psych_curve(directions, choices, modality):
    """
    fitting with a cumulative Gaussian function
    return: psychmetric curve
    """
    fit_params = []
    for i in np.unique(modality):
        tartrials = np.where(modality == i)[0]
        prpt, pcov = curve_fit(cum_gaussian, directions[tartrials], 2-choices[tartrials])
        fit_params.append(prpt)
    return fit_params


def data_extract(file_paths):
    """
    
    """
    df_detail = []
    df_summary = pd.DataFrame([], columns = {'cho_prob','prpt','prpt_theo'})
    for nth, file in enumerate(file_paths):

        paod, trial_briefs = load(file)
        trials = get_multinfo(paod,trial_briefs)
        files_pd = pd.DataFrame([trials["choice"],trials["reward"],trials["chosen"],
                                 trials["modality"],trials["direction"],trials["estimates"]],
                                ['choice','reward','chosen','modality','direction','estimates'])
        files_pd = files_pd.T
        df_detail.append(files_pd)
        
        cho_prob = [[],[],[]]
        for i in np.unique(trials["direction"]):
            tar_trials = np.where(trials["direction"]==i)[0]
            for ii in range(3):
                temp = np.intersect1d(tar_trials, np.where(trials["modality"]==ii))
                cho_prob[ii].append(np.mean(trials["choice"][temp]==1))
        
        modality_choice = trials["modality"]
        theo_choice = np.diag(np.vstack(trials["estimates"].values)[:,modality_choice])+1
        
        prpt = psych_curve(trials["direction"], trials["choice"], trials["modality"])
        prpt_theo = psych_curve(trials["direction"], theo_choice, trials["modality"])
        df_summary.loc[nth] = {'cho_prob':cho_prob,'prpt':prpt,'prpt_theo':prpt_theo}
    return df_detail, df_summary
    

def bhv_plot(df_summary):
    fig_w, fig_h = (10, 7)
    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
    
    numfiles = df_summary.cho_prob.values.shape[0]
    cho_prob = np.array([df_summary.cho_prob.values[i] for i in range(numfiles)])

    fig1 = plt.figure('choices')
    x = np.arange(1,181,1)
    labels = ['modality 1','modality 2','combined']
    color = ['r','g','k']
    
    x_ = np.tile(x, len(df_summary))   
    for i in range(3):
        y_  = np.hstack(cho_prob[:,i,:].reshape(-1,))
        y_[np.where(np.isnan(y_))]=1
        prpt, pcov = curve_fit(cum_gaussian, x_, y_)
        plt.plot(np.mean(cho_prob[:,i,:], axis=0), '.', label = labels[i], color = color[i])
        plt.plot(x, cum_gaussian(x, prpt[0], prpt[1]), color = color[i])
    fig1.legend()
    plt.title('psychmetric curve')
    plt.xlabel('motion direction')        
    plt.ylabel('probability of choosing left')        
    fig1.savefig('../figs/MI-psych_curve', format='eps', dpi=1000)
    plt.show()
    
    
    prpt = np.vstack([(df_summary.prpt.values[i]) for i in range(numfiles)])
    prpt_theo = np.vstack([(df_summary.prpt_theo.values[i]) for i in range(numfiles)])
    fig2 = plt.figure('threshold')
    plt.bar([0,2,4],
            [np.mean(prpt[::3,1]),np.mean(prpt[1::3,1]),np.mean(prpt[2::3,1])],
            yerr = [stats.sem(prpt[::3,1]), stats.sem(prpt[1::3,1]), stats.sem(prpt[2::3,1])],
            label = 'model')
    plt.bar([1,3,5],
            [np.mean(prpt_theo[::3,1]),np.mean(prpt_theo[1::3,1]),np.mean(prpt_theo[2::3,1])],
            yerr = [stats.sem(prpt_theo[::3,1]), stats.sem(prpt_theo[1::3,1]), stats.sem(prpt_theo[2::3,1])],
            label = 'bayesian')
    plt.legend()
    plt.ylabel('threshold (degree)')
    plt.xticks([0.5,2.5,4.5], {'visual','vestibular','combined'})
    fig2.savefig('../figs/MI-threshold', format='eps', dpi=1000)
    plt.show()
    
def main(file_paths=None):
    print("start")
    print("select the files")
    root = tk.Tk()
    root.withdraw()
    if file_paths == None:
        file_paths = filedialog.askopenfilenames(parent = root,
                                                title = 'Choose a file',
                                                filetypes = [("HDF5 files", "*.hdf5")]
                                                )
    ##
    df_detail, df_summary = data_extract(file_paths)
    bhv_plot(df_summary)
    return df_summary

if __name__ == '__main__':
    savepath = '../figs/'
    sel_psth = False
    main()
