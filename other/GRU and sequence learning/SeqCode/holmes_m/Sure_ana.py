#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:49:27 2019

@author: Zhewei Zhang
"""

import os
import numpy as np
import pandas as pd
import tkinter as tk
from scipy import stats
import matplotlib.pyplot as plt
from tkinter import filedialog
from toolkits_2 import get_sureinfo, load, get_hidden_resp_sure
import pdb
import pandas as pd
import matplotlib.pyplot as plt

"""
representing the relation between the logRT and rt; 

fit the psychmetric curve

fig1 a/c
"""

coherence_list = [-51.2, -12.8, -6.4, -3.2, 0, 3.2, 6.4, 12.8, 51.2]
    

def selectivity_test(all_resp, trials):
    """
    whether neurons encode the motion direction during the motion viewing period
        or predict the choices in the no sure target trials
    """
    numTrials = all_resp.shape[0]
    numNeurons = all_resp[0].shape[1]
    
    dots_dur = trials.randots_dur.values
#    moive_on, moive_off, cho_on = 5, 5+dots_dur, 5+dots_dur+5
    moive_on, moive_off, cho_on = 4, 4+dots_dur, 4+dots_dur+5
    # actually, it is one step eariler than the the event happen, but it is about prediction, anyway, be careful about this time step
    
    nosure_left = np.intersect1d(np.where(trials.choice==1), np.where(trials.sure_trial==0)) 
    nosure_right = np.intersect1d(np.where(trials.choice==2), np.where(trials.sure_trial==0))  

    sign_neuron_pos, sign_neuron_neg = {'motion':[],'choice':[],'both':[]}, {'motion':[],'choice':[],'both':[]}
    
    for nNeuron in range(numNeurons):
        rvs_m1, rvs_m2 = [], []
        rvs_c1, rvs_c2 = [], []
        
        for nTrial in range(numTrials):
            if nTrial in nosure_left:
                rvs_m1.extend(all_resp[nTrial][moive_on:moive_off[nTrial], nNeuron])
                rvs_c1.extend(all_resp[nTrial][cho_on[nTrial]:cho_on[nTrial]+1, nNeuron])
            elif nTrial in nosure_right:
                rvs_m2.extend(all_resp[nTrial][moive_on:moive_off[nTrial], nNeuron])
                rvs_c2.extend(all_resp[nTrial][cho_on[nTrial]:cho_on[nTrial]+1, nNeuron])
        
        
        for rvs, label in zip([[rvs_m1, rvs_m2], [rvs_c1, rvs_c2]],['motion','choice']):
            rvs1, rvs2 = np.array(rvs[0]), np.array(rvs[1])        
            t, p = stats.ttest_ind(rvs1,rvs2)
#            t, p = stats.ttest_rel(rvs1,rvs2)
            if p < 0.05/(2*128):
                if rvs1.mean() > rvs2.mean():
                    sign_neuron_pos[label].append(nNeuron)
                else:
                    sign_neuron_neg[label].append(nNeuron)
    sign_neuron_pos["both"] = np.intersect1d(sign_neuron_pos["motion"],  sign_neuron_pos["choice"])
    sign_neuron_neg["both"] = np.intersect1d(sign_neuron_neg["motion"],  sign_neuron_neg["choice"])

    return sign_neuron_pos, sign_neuron_neg
        

def psth(all_resp_all, df_detail, file_paths, Neurons = None, mean = False, label = None):

    baseline_motion = 4
    
    numFile = len(all_resp_all)
    numNeurons = all_resp_all[0][0].shape[1]
    
    if np.all(Neurons == None):
        Neurons = range(numNeurons)
    
    
    colorset = ['r','g','c','y']
    fig = plt.figure()
    
    
    if mean:
        if numFile ==1:
            path, file = os.path.split(file_paths[0])
            task_time = file.split('-')[0]
        else:
            task_time = 'combined'

        resp_plot = {'sure':    {'left': {'mov':[],'cho':[]}, 
                                 'right':{'mov':[],'cho':[]}, 
                                 'sure_left': {'mov':[],'cho':[]},
                                 'sure_right':{'mov':[],'cho':[]}},
                     'no_sure': {'left':{'mov':[],'cho':[]}, 
                                 'right':{'mov':[],'cho':[]}}
                     }
        for all_resp, trials, neurons_curr in zip(all_resp_all, df_detail, Neurons):
            i = 0
            numTrials = all_resp.shape[0]
            dots_dur = trials.randots_dur.values.astype(np.int)
            num_tartrials = np.sum(dots_dur>=baseline_motion)
            
            resp_mov = np.zeros((neurons_curr.size, np.sum(num_tartrials), 6))
            resp_cho = np.zeros((neurons_curr.size, np.sum(num_tartrials), 7))
            moive_on, moive_off, cho_on = 5, 5+dots_dur, 5+dots_dur+5


            sure_trial= trials["sure_trial"][dots_dur>=baseline_motion].values
            choice    = trials["choice"][    dots_dur>=baseline_motion].values
            coherence = trials["coherence"][ dots_dur>=baseline_motion].values
            
            no_sure = {'left': np.intersect1d(np.where(sure_trial==0), np.where(choice==1)),
                       'right': np.intersect1d(np.where(sure_trial==0), np.where(choice==2))}
            sure = {'left': np.intersect1d(np.where(sure_trial==1), np.where(choice==1)),
                    'right': np.intersect1d(np.where(sure_trial==1), np.where(choice==2)),
                    'sure_left':np.intersect1d(np.where(coherence>0), np.where(choice==3)),
                    'sure_right':np.intersect1d(np.where(coherence<0), np.where(choice==3))}

            for nTrial, dur in zip(range(numTrials), dots_dur):
            
                if dur < baseline_motion:
                        continue
                for nth, nNeuron in enumerate(neurons_curr):
                    resp_mov[nth, i, :] = all_resp[nTrial][moive_on-2:moive_on+4 ,nNeuron]
                    resp_cho[nth, i, :] = all_resp[nTrial][cho_on[nTrial]-5:cho_on[nTrial]+2 ,nNeuron]
                    # normalization
#                    resp_mov[nth, i, :] = all_resp[nTrial][moive_on-2:moive_on+4 ,nNeuron]-(all_resp[nTrial][moive_on-2:moive_on ,nNeuron].mean())
#                    resp_cho[nth, i, :] = all_resp[nTrial][cho_on[nTrial]-5:cho_on[nTrial]+2 ,nNeuron]-(all_resp[nTrial][moive_on-2:moive_on ,nNeuron].mean())
                i += 1
            
            for labels, title in zip([sure, no_sure],['sure', 'no_sure']):
                for key, value in labels.items():
                    resp_mov_curr = resp_mov[:,value,:].reshape(-1, 6)
                    resp_cho_curr = resp_cho[:,value,:].reshape(-1, 7)
                    resp_plot[title][key]['mov'].append(resp_mov_curr)
                    resp_plot[title][key]['cho'].append(resp_cho_curr)

        for labels, value__s in resp_plot.items():
            title_label = task_time + '-averaged-' + labels + '-' + label
            plt.clf()
            plt.title(title_label)
            i=0
            for key, value in value__s.items():
                resp_mov = np.concatenate(value['mov'])
                resp_cho = np.concatenate(value['cho'])
                plot_errorbar(range(6), resp_mov, color = colorset[i], label = key)
                plot_errorbar(np.linspace(7,13,7), resp_cho, color = colorset[i])
                i+=1
        
            plt.axvline(2,linestyle='-.',color='k',label = 'moive onset')
            plt.axvline(12,linestyle='-.',color='k',label = 'choice')
            plt.legend()
            fig.savefig(savepath + task_time + '-signNeuron-'+ labels + label + '.eps', format='eps', dpi=1000)
#            plt.savefig(savepath + task_time + '-signNeuron-'+ title + label +'.png')
            plt.show()
        return resp_plot
    
    for all_resp, trials, Neurons_curr in zip(all_resp_all, df_detail, Neurons):
        # the dataset for all neurons maybe very large, so iterate Neuron first
        if numFile ==1:
            path, file = os.path.split(file_paths[0])
            task_time = file.split('-')[0]
        else:
            task_time = 'combined'

        i = 0
        numTrials = all_resp.shape[0]
        dots_dur = trials.randots_dur.values.astype(np.int)
        num_tartrials = np.sum(dots_dur>=baseline_motion)
        
        moive_on, moive_off, cho_on = 5, 5+dots_dur, 5+dots_dur+5


        sure_trial= trials["sure_trial"][dots_dur>=baseline_motion].values
        choice    = trials["choice"][    dots_dur>=baseline_motion].values
        coherence = trials["coherence"][ dots_dur>=baseline_motion].values
        
        no_sure = {'left': np.intersect1d(np.where(sure_trial==0), np.where(choice==1)),
                   'right': np.intersect1d(np.where(sure_trial==0), np.where(choice==2))}
        sure = {'left': np.intersect1d(np.where(sure_trial==1), np.where(choice==1)),
                'right': np.intersect1d(np.where(sure_trial==1), np.where(choice==2)),
                'sure_left':np.intersect1d(np.where(coherence>0), np.where(choice==3)),
                'sure_right':np.intersect1d(np.where(coherence<0), np.where(choice==3))}
        for Neuron_curr in Neurons_curr:
            resp_mov, resp_cho = [], []
            for nTrial, dur in zip(range(numTrials), dots_dur):
                if dur < baseline_motion:
                    continue
                resp_mov.append(all_resp[nTrial][moive_on-2:moive_on+4 ,Neuron_curr])
                resp_cho.append(all_resp[nTrial][cho_on[nTrial]-5:cho_on[nTrial]+2 ,Neuron_curr])
            
            
            resp_mov = np.array(resp_mov)
            resp_cho = np.array(resp_cho)
            
            for labels, title in zip([sure, no_sure],['sure', 'no_sure']):
                title_label = task_time + '-neuron:' + str(Neuron_curr) + '-' + title
                plt.clf()
                plt.title(title_label)
                i = 0
                for key, value in labels.items():
                    plot_errorbar(range(6), resp_mov[value,:], color = colorset[i], label = key)
                    plot_errorbar(np.linspace(7,13,7), resp_cho[value,:], color = colorset[i])
                    i += 1
                
                plt.axvline(2,linestyle='-.',color='k',label = 'moive onset')
        #        plt.axvline(10,linestyle='-.',color='k',label = 'moive offset')
                plt.axvline(12,linestyle='-.',color='k',label = 'choice')
                plt.legend()
    #            fig.savefig(savepath+'/prob_sure.eps', format='eps', dpi=1000)
#                plt.savefig(savepath + task_time + '-' + label + '-' + str(Neuron_curr) + '-'+ title +'.png')
                plt.show()

def plot_errorbar(x, response,  color = 'b', label=None):
    plt.errorbar(x, np.nanmean(response,axis=0), 
                 np.nanstd(response, axis=0)/np.sqrt(response.shape[0]),
                 color = color,
                 label = label)
    

def data_extract(file_paths):
    """
    
    """
    df_detail = []
#    df_detail = pd.DataFrame([], columns = {'choice', 'chosen', 'reward', 'sure_trial', 'coherence', 'acc_evi','randots_dur'})

    df_summary = pd.DataFrame([], columns = {'choice'})
    all_resp_all = []
    sign_neuron_all = {'pos':[], 'neg':[],'pos_motion':[], 'neg_motion':[],'pos_choice':[], 'neg_choice':[]}
    for i, file in enumerate(file_paths):

        paod, trial_briefs = load(file)
        trials = get_sureinfo(paod,trial_briefs)

        files_pd = pd.DataFrame([trials["choice"],trials["reward"],trials["randots_dur"],
                                 trials["sure_trial"],trials["coherence"]],
                                ['choice','reward','randots_dur','sure_trial','coherence'])
        files_pd = files_pd.T
        df_detail.append(files_pd)
        
        choice =[]
        for ii in coherence_list:
            choice.append([
                    ii,
                    np.where(trials["choice"][trials["coherence"]==ii]== 1)[0].shape[0],
                    np.where(trials["choice"][trials["coherence"]==ii]== 2)[0].shape[0],
                    np.where(trials["choice"][trials["coherence"]==ii]== 3)[0].shape[0]
                    ])
        choice = np.array(choice)
        df_summary.loc[i] = {'choice':choice}        

#        # plot the psth
        all_resp = get_hidden_resp_sure(paod, trial_briefs) # response of neurons in hidden layer
        all_resp_all.append(all_resp)
        
        sign_neuron_pos, sign_neuron_neg = selectivity_test(all_resp, trials)
        sign_neuron_all['pos'].append(sign_neuron_pos['both'])
        sign_neuron_all['neg'].append(sign_neuron_neg['both'])
        sign_neuron_all['pos_motion'].append(np.array(sign_neuron_pos['motion']))
        sign_neuron_all['neg_motion'].append(np.array(sign_neuron_neg['motion']))
        sign_neuron_all['pos_choice'].append(np.array(sign_neuron_pos['choice']))
        sign_neuron_all['neg_choice'].append(np.array(sign_neuron_neg['choice']))

#    print('positive neuron')
#    psth(all_resp_all, df_detail, file_paths, Neurons = sign_neuron_all['pos'], mean = True, label = 'pos')
#    print('negative neuron')
#    psth(all_resp_all, df_detail, file_paths, Neurons = sign_neuron_all['neg'], mean = True, label = 'neg')
#
#    print('positive motion neuron')
#    psth(all_resp_all, df_detail, file_paths, Neurons = sign_neuron_all['pos_motion'], mean = True, label = 'pos_motion')
#    print('negative motion neuron')
#    psth(all_resp_all, df_detail, file_paths, Neurons = sign_neuron_all['neg_motion'], mean = True, label = 'neg_motion')
        
    print('positive choice neuron')
    resp_plot_cho_pos = psth(all_resp_all, df_detail, file_paths, Neurons = sign_neuron_all['pos_choice'], mean = True, label = 'pos_choice')
    print([i.size for i in sign_neuron_all['pos_choice']])

    print('negative choice neuron')
    resp_plot_cho_neg = psth(all_resp_all, df_detail, file_paths, Neurons = sign_neuron_all['neg_choice'], mean = True, label = 'neg_choice')
    print([i.size for i in sign_neuron_all['neg_choice']])
    return df_detail, df_summary

#    colorset = ['r','g','c','y']
#    keys = ['left','right','sure_left','sure_right']
#    key_inv = ['right','left','sure_right','sure_left']
#    fig = plt.figure()
#    for i, key in enumerate(keys):
#       resp_mov = copy.deepcopy(resp_plot_cho_pos['sure'][key]['mov'])
#       resp_mov.extend(resp_plot_cho_neg['sure'][key_inv[i]]['mov'])
#       resp_mov = np.vstack(resp_mov)
#       
#       resp_cho = copy.deepcopy(resp_plot_cho_pos['sure'][key]['cho'])
#       resp_cho.extend(resp_plot_cho_neg['sure'][key_inv[i]]['cho'])
#       resp_cho = np.vstack(resp_cho)
#       
#       plot_errorbar(range(6), resp_mov, color = colorset[i], label = key)
#       plot_errorbar(np.linspace(7,13,7), resp_cho, color = colorset[i])
#        
#       plt.axvline(2,linestyle='-.',color='k',label = 'moive onset')
#       plt.axvline(12,linestyle='-.',color='k',label = 'choice')
#    plt.legend()
#    fig.savefig(savepath+'/prob_sure_pref.eps', format='eps', dpi=1000)
#    #                plt.savefig(savepath + task_time + '-' + label + '-' + str(Neuron_curr) + '-'+ title +'.png')
#    plt.show()

def figure_plot(df_detail, savepath = './'):
#    fig_w, fig_h = (10, 7)
#    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
    cho_prop_sure_all = []
    cr_sure_all = []
    cr_nosure_all = []
    for files_pd in df_detail:
                
        cho_by_coh_dur = files_pd.choice.groupby([files_pd.coherence, files_pd.randots_dur])
        cho_by_coh_dur_sure = files_pd.choice.groupby([files_pd.coherence, files_pd.randots_dur, files_pd.sure_trial])
        coherence_list = np.unique(files_pd["coherence"]).tolist()
        randots_dur_list = np.unique(files_pd["randots_dur"]).tolist()
        
        cho_prop_sure = np.zeros((len(coherence_list), len(randots_dur_list)))
        cr_sure = np.zeros((len(coherence_list), len(randots_dur_list)))
        cr_nosure = np.zeros((len(coherence_list), len(randots_dur_list)))
        
#        plt.hist(trials["acc_evi"],50,density =True)
        
        for i, coh_dur in enumerate(cho_by_coh_dur):
            coh = coh_dur[0][0]
            dur = coh_dur[0][1]
            x = np.where(np.array(coherence_list) == coh)[0][0]
            y = np.where(np.array(randots_dur_list) == dur)[0][0]
            
            cho_prop_sure[x,y] = np.mean(coh_dur[1]==3)
        
            
        for i, coh_dur_sure in enumerate(cho_by_coh_dur_sure):
            coh = coh_dur_sure[0][0]
            dur = coh_dur_sure[0][1]
            sure = coh_dur_sure[0][2]
            x = np.where(np.array(coherence_list) == coh)[0][0]
            y = np.where(np.array(randots_dur_list) == dur)[0][0]
#            print(x,y,sure,coh_dur_sure[1].values.shape[0])
            numTrials = np.sum(coh_dur_sure[1]==1) + np.sum(coh_dur_sure[1]==2)
            if sure:
                cr_sure[x,y] = np.sum(coh_dur_sure[1]==1)/numTrials if coh>0 else np.sum(coh_dur_sure[1]==2)/numTrials
            else:
                cr_nosure[x,y] = np.sum(coh_dur_sure[1]==1)/numTrials if coh>0 else np.sum(coh_dur_sure[1]==2)/numTrials

        cho_prop_sure_all.append(cho_prop_sure)
        cr_sure_all.append(cr_sure)
        cr_nosure_all.append(cr_nosure)

    cr_sure_all = np.array(cr_sure_all)
    cr_nosure_all = np.array(cr_nosure_all)
    cho_prop_sure_all = np.array(cho_prop_sure_all)

    color = ['g','r','y','c','b','m','k']
        
    fig = plt.figure('probability sure target')
    for i, coh in enumerate(coherence_list):
        if i <5:
            cho_prop_sure_all[:,10-i,:] = (cho_prop_sure_all[:,i,:]+cho_prop_sure_all[:,10-i,:])/2
        else:
#            plt.plot([0,1,2,3,5,7], np.mean(cho_prop_sure_all[:,i,:], axis=0), '-', color = color[10-i], label = np.abs(coh))
            plt.errorbar([0,1,2,3,5,7], np.mean(cho_prop_sure_all[:,i,:], axis=0), 
                         yerr = stats.sem(cho_prop_sure_all[:,i,:], axis=0),
                         fmt= '-', color = color[10-i], label = np.abs(coh))
    plt.xticks([0,1,2,3,5,7], {100,200,300,400,600,800})
    plt.ylabel('probability sure target')
    plt.legend()
    fig.savefig(savepath+'/prob_sure.eps', format='eps', dpi=1000)
#    fig.savefig(savepath + 'prob_sure.png')
    
    fig2 = plt.figure('probability correct')
    for i, coh in enumerate(coherence_list):
        if i<5:
            cr_sure_all[:,10-i,:] = (cr_sure_all[:,i,:]+cr_sure_all[:,10-i,:])/2
            cr_nosure_all[:,10-i,:] = (cr_nosure_all[:,i,:]+cr_nosure_all[:,10-i,:])/2
        else:
#            plt.plot([0,1,2,3,5,7], np.mean(cr_sure_all[:,i,:], axis=0), fmt='-', color = color[10-i], label = np.abs(coh))
#            plt.plot([0,1,2,3,5,7], np.mean(cr_nosure_all[:,i,:], axis=0), fmt='-.', color = color[10-i])
            plt.errorbar([0,1,2,3,5,7], np.mean(cr_sure_all[:,i,:], axis=0), 
                         yerr = stats.sem(cr_sure_all[:,i,:], axis=0),
                         fmt='-', color = color[10-i], label = np.abs(coh))
            plt.errorbar([0,1,2,3,5,7], np.mean(cr_nosure_all[:,i,:], axis=0),
                         yerr = stats.sem(cr_nosure_all[:,i,:], axis=0),
                         fmt='-.', color = color[10-i])
    plt.legend()
    plt.ylabel('probability correct')
    plt.xticks([0,1,2,3,5,7], {100,200,300,400,600,800})
    fig2.savefig(savepath +'/prob_cr.eps', format='eps', dpi=1000)
#    fig2.savefig(savepath + 'prob_cr.png')
    
    plt.show()
#    choice = np.array([df_summary.choice.values[i] for i in range(df_summary.choice.values.shape[0])])
#    fig2 = plt.figure('choices')
#    nums = range(choice.shape[1])
#    plt.errorbar(nums, np.mean(choice[:,:,1], axis=0), np.std(choice[:,:,1], axis=0), label = 'left target')
#    plt.errorbar(nums, np.mean(choice[:,:,2], axis=0), np.std(choice[:,:,2], axis=0), label = 'right target')
#    plt.errorbar(nums, np.mean(choice[:,:,3], axis=0), np.std(choice[:,:,3], axis=0), label = 'sure target')
#    plt.xticks(nums, coherence_list)        
#    fig2.legend()
#    fig2.savefig('../figs/hist effect', format='eps', dpi=1000)
#    plt.show()



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
    figure_plot(df_detail, savepath = savepath)

    return df_summary

if __name__ == '__main__':
    savepath = '../figs/'
    main()
