#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:11:24 2019

@author: YangLab_ZZW
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk
from toolkits_2 import load, get_bhvinfo, get_hidden_resp_all, array2list
from scipy import stats
from selectivity import regress_all
"""
for plotting psth and sort trials by the finnal choice and logLR

Used for one signle file

fig3 a/b
"""

def psth_plot(psth_resp_all, savepath = './'):
    
    colorset = ['r','m','c','g']
    plt.figure()

    urgency_pos, urgency_neg = [], []
    for decomp in psth_resp_all:
        urgency_pos.append(decomp[0])
        urgency_neg.append(decomp[1])
    
    urgency_pos = np.concatenate(urgency_pos)
    urgency_neg = np.concatenate(urgency_neg)
    nshape = range(urgency_pos.shape[1])

    plt.title('urgency signal')
    plt.errorbar(nshape, np.nanmean(urgency_pos[:,:,0], axis=0), yerr = stats.sem(urgency_pos[:,:,0], axis=0, nan_policy = 'omit'), color = 'r',label = 'choosing left')
    plt.errorbar(nshape, np.nanmean(urgency_pos[:,:,1], axis=0), yerr = stats.sem(urgency_pos[:,:,1], axis=0, nan_policy = 'omit'), color = 'g',label = 'choosing right')
    plt.xticks([0,4,9,14,19],('1','5','10','15','20'))
    plt.legend()
    plt.savefig(savepath + '-pos.eps', format='eps', dpi=1000)
        
    plt.show()
    plt.clf()
    plt.title('urgency signal')
    plt.errorbar(nshape, np.nanmean(urgency_neg[:,:,0], axis=0), yerr = stats.sem(urgency_neg[:,:,0], axis=0, nan_policy = 'omit'), color = 'r',label = 'choosing left')
    plt.errorbar(nshape, np.nanmean(urgency_neg[:,:,1], axis=0), yerr = stats.sem(urgency_neg[:,:,1], axis=0, nan_policy = 'omit'), color = 'g',label = 'choosing right')
    plt.xticks([0,4,9,14,19],('1','5','10','15','20'))
    plt.legend()
    plt.savefig(savepath + '-neg.eps', format='eps', dpi=1000)
    plt.show()
    
    
    evidence_first, evidence_last= [], []
    for decomp in psth_resp_all:
        evidence_first.append(decomp[2])
        evidence_last.append(decomp[3])
    
    evidence_first = np.concatenate(evidence_first)
    evidence_last = np.concatenate(evidence_last)
    
    plt.clf()
    start_time_first, end_time_first = [0, 8, 13], [8, 13, 18] 
    start_time_last, end_time_last = [35, 30, 25], [44, 35, 30]
    length_first, length_last = [8,5,5],[9,5,5]
    for nepoch in range(3):
        for ngroup in range(4):
            start_time, end_time = start_time_first[nepoch], end_time_first[nepoch]
            plt.errorbar(np.arange(start_time,end_time,1), evidence_first[:,nepoch, ngroup,:length_first[nepoch]].mean(axis=0),
                         yerr = stats.sem(evidence_first[:,nepoch, ngroup,:length_first[nepoch]], axis=0), color = colorset[ngroup])

            start_time, end_time = start_time_last[nepoch], end_time_last[nepoch]
            plt.errorbar(np.arange(start_time,end_time,1), evidence_last[:,nepoch, ngroup,:length_last[nepoch]].mean(axis=0),
                         yerr = stats.sem(evidence_last[:,nepoch, ngroup,:length_last[nepoch]], axis=0), color = colorset[ngroup])

    plt.title('PSTH - log LR')
    plt.axvline(3,linestyle='-.',color='k',label = 'shape on')
    plt.axvline(39,linestyle='-.',color='k',label = 'choice')
    bottom, top = plt.ylim()
    plt.ylim([bottom, 1])
    plt.legend()
    plt.savefig(savepath + 'psth-choice.eps', format='eps', dpi=1000)
    plt.show()

    pass


def sort_trials(label,resp):
    num_group_logLR = 4
    min_, max_ = np.min(label), np.max(label)

    resp_all = []
    for ngroup in range(num_group_logLR):
        lower_bound = min_ + (max_ - min_)/num_group_logLR*ngroup
        higher_bound = max_ - (max_ - min_)/num_group_logLR*(num_group_logLR-ngroup-1)
        lowerGroup = np.where(label>=lower_bound) 
        higherGroup = np.where(label<=higher_bound)
        Group = np.intersect1d(lowerGroup,higherGroup)
        if not Group.size:
            continue
        resp_all.append(resp[Group,:,:].mean(axis=0))
    return np.array(resp_all).transpose([2,0,1])
        
def psth_align(all_resp_cell, trial, choice, shape, urgency_neuron, evidence_neuron):
    '''
    prepare the response for plotting
    '''
    
    num_trials = shape.rt.shape[0]
    num_neurons = all_resp_cell[0].shape[1]

    num_epoch_logLR = 3
    num_group_logLR = 4
    
    ## urgency signal
    nshape_max = 18#shape.rt.max()
#    urgency_neuron_num = urgency_neuron['pos'].shape[0] +  urgency_neuron['neg'].shape[0]
    resp_left, resp_right = np.zeros((num_neurons, nshape_max)), np.zeros((num_neurons, nshape_max))
    
    for nshape in range(nshape_max):#
        start_time = 3+5*nshape
        end_time = 3+5*(1+nshape)
        
        temp_resp_left, temp_resp_right = [], []
        for i in range(num_trials):
            if shape.rt[i]>=nshape:
                if choice.left[i] == 1.0:
                    temp_resp_left.append(np.nanmean(all_resp_cell[i][start_time:end_time,:], axis=0))
                elif choice.left[i] == -1.0:
                    temp_resp_right.append(np.nanmean(all_resp_cell[i][start_time:end_time,:], axis=0))
                else:
                    raise Exception('unknown choices')

        resp_left[:,nshape] = np.nanmean(np.array(temp_resp_left), axis=0)
        resp_right[:,nshape] = np.nanmean(np.array(temp_resp_right), axis=0)
    
    neuron_count = -1
#    resp_pos_urg, resp_neg_urg = np.zeros((urgency_neuron_num, nshape_max,2)),  np.zeros((urgency_neuron_num, nshape_max,2))
    resp_pos_urg, resp_neg_urg = [[],[]], [[],[]]
    for label, Neurons in urgency_neuron.items():
        if label == 'pos':
            for neuron  in Neurons:
                neuron_count += 1
#                resp_pos_urg[neuron_count,:,0] = resp_left[neuron,:]
#                resp_pos_urg[neuron_count,:,1] = resp_right[neuron,:]
                resp_pos_urg[0].append(resp_left[neuron,:])
                resp_pos_urg[1].append(resp_right[neuron,:])
        elif label == 'neg':
            for neuron  in Neurons:
                neuron_count += 1
#                resp_neg_urg[neuron_count,:,0] = resp_left[neuron,:]
#                resp_neg_urg[neuron_count,:,1] = resp_right[neuron,:]
                resp_neg_urg[0].append(resp_left[neuron,:])
                resp_neg_urg[1].append(resp_right[neuron,:])
    resp_pos_urg = np.array(resp_pos_urg).transpose(1,2,0)
    resp_neg_urg = np.array(resp_neg_urg).transpose(1,2,0)
# =============================================================================
#      logLR diff
    min_shape = 6
    epoch_len = 5
    resp_first = np.zeros((num_neurons, num_epoch_logLR, num_group_logLR, epoch_len*2))
    resp_last = np.zeros((num_neurons, num_epoch_logLR, num_group_logLR, epoch_len*2))

    for nepoch in range(num_epoch_logLR):
        start_time_first, end_time_first = 3+epoch_len*nepoch, 3+epoch_len*(1+nepoch)
        start_time_last, end_time_last = -6-epoch_len*(nepoch+1)+1, -6-epoch_len*nepoch+1
        if nepoch == 0:
            start_time_first, end_time_last = 0, -1

        # get the reponses and labels before plot; in first three/last three epochs
        temp_resp_first, temp_resp_last = [], []
        label_first, label_last = [], []
        for ntrial in range(num_trials):
            nshape = shape.rt.iloc[ntrial]
#            if nshape < nepoch+1:
            if nshape < min_shape:
                continue
            temp_resp_first.append(all_resp_cell[ntrial][start_time_first:end_time_first,:])
            label_first.append(np.sum(shape.tempweight.iloc[ntrial][:nepoch+1]))
            
            temp_resp_last.append(all_resp_cell[ntrial][start_time_last:end_time_last,:])
            label_last.append(np.sum(shape.tempweight.iloc[ntrial][:nshape - nepoch]))
        
        temp_resp_first, label_first, temp_resp_last, label_last = array2list(temp_resp_first, label_first, temp_resp_last, label_last)
        
        # sort trials
        temp = sort_trials(label_first, temp_resp_first)
        resp_first[:,nepoch,:,:temp.shape[-1]] = temp
#        print(temp.shape)
        temp = sort_trials(label_last, temp_resp_last)
        resp_last[:,nepoch,:,:temp.shape[-1]] = temp
#        print(temp.shape)


    neuron_count = -1
    evidence_neuron_num = evidence_neuron['pos'].shape[0] + evidence_neuron['neg'].shape[0]
    resp_first_evi = np.zeros((evidence_neuron_num, num_epoch_logLR, num_group_logLR, epoch_len*2))
    resp_last_evi = np.zeros((evidence_neuron_num, num_epoch_logLR, num_group_logLR, epoch_len*2))

    for label, Neurons in evidence_neuron.items():
        if label == 'pos':
            for neuron  in Neurons:
                neuron_count += 1
                resp_first_evi[neuron_count,:,:,:] = resp_first[neuron,:,:,:]
                resp_last_evi[neuron_count,:,:,:] = resp_last[neuron,:,:,:]
        elif label == 'neg':
            for neuron  in Neurons:
                neuron_count += 1
                temp = resp_first[neuron]
                resp_first_evi[neuron_count,:,:,:] = temp[:,[3,2,1,0],:]#reserve the group order
                temp = resp_last[neuron]
                resp_last_evi[neuron_count,:,:,:] = temp[:,[3,2,1,0],:]
        
    return resp_neg_urg, resp_pos_urg, resp_first_evi, resp_last_evi

def extract(path_file):


    p_threshold = 0.05/(128*4*len(path_file))
    psth_resp_all = []
    urgency_neuron = {'pos':[],'neg':[]}
    evidence_neuron = {'pos':[],'neg':[]}
    urgency_pos_num = []
    urgency_neg_num = []
    evidence_num = []
    for i, file in enumerate(path_file):
        paod, trial_briefs = load(file)
        trial, choice, shapes, _, _ = get_bhvinfo(paod,trial_briefs)
        all_resp_cell = get_hidden_resp_all(paod,trial_briefs)
        
        result_all = regress_all(all_resp_cell, trial, choice, shapes)
        
        p_value = np.ones([5,128,5])# 5 parameters (4+1 bias term)by 128 neurons by 5 time point
        params = np.ones([5,128,5])
        for ii in range(128): 
            p_value[:,ii,:] = np.array(result_all['p_values'][ii]).T
            params[:,ii,:] = np.array(result_all['params'][ii]).T

        urgency_neuron['pos'] = np.intersect1d(np.where(np.all(p_value[3,:,:]<p_threshold, axis =1)), 
                                        np.where(np.all(params[3,:,:]>-1e-10, axis =1)))
        urgency_neuron['neg'] = np.intersect1d(np.where(np.all(p_value[3,:,:]<p_threshold, axis =1)), 
                                        np.where(np.all(params[3,:,:]<1e-10, axis =1)))
        
        evidence_neuron['pos'] = np.intersect1d(np.where(np.all(p_value[4,:,:]<p_threshold, axis =1)), 
                                        np.where(np.all(params[4,:,:]>-1e-10, axis =1)))
        evidence_neuron['neg'] = np.intersect1d(np.where(np.all(p_value[4,:,:]<p_threshold, axis =1)), 
                                        np.where(np.all(params[4,:,:]<1e-10, axis =1)))
        
        urgency_pos_num = []
        urgency_neg_num = []
        evidence_num
        
        resp_pref_urg, resp_nonpref_urg, resp_first_evi, resp_last_evi = psth_align(all_resp_cell, trial, choice, shapes, urgency_neuron, evidence_neuron)
        psth_resp_all.append([resp_pref_urg, resp_nonpref_urg, resp_first_evi, resp_last_evi])
    return psth_resp_all


def main():
    print("start")
#    %matplotlib auto
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilenames(
            parent=root,title='Choose a file',
            filetypes=[("HDF5 files", "*.hdf5")]
            )
    print("select the files")
    #
    psth_resp_all = extract(file_path)
    psth_plot(psth_resp_all, savepath = '../figs/')
    
if __name__ == '__main__':
    fig_w, fig_h = (10, 7)
    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
    main()


















