#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 09:36:29 2019

@author: Zhewei Zhang
"""

import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from scipy import stats
from scipy.special import erf
from scipy.signal import savgol_filter
from scipy.optimize import minimize, curve_fit
        
from toolkits_2 import get_stinfo, load, get_resp_ts, stayprob

"""
plot the stay probability in the two step task

"""

def softmax_prob(state, q, beta, rep = None):
    """
    Softmax policy: selects action probabilistically depending on the value.
    Args:
        state: an integer corresponding to the current state.
        q: a matrix indexed by state and action.
        params: a dictionary containing the default parameters.
    Returns:
        an array corresponding to the probability of choosing each options.
    """
    value = q[state,:]
    prob = np.exp(value * beta)# beta is the inverse temperature parameter
    if rep != None:
        prob[rep[0]] = np.exp(value[rep[0]] * beta + rep[1]) # tendency of repeating the previous choice
        
    prob = prob / np.sum(prob)  # normalize            
    return prob

def cr_block(trials):
    """
    input: common: the status of transition from the first to the second stage
           reward: reward feedback
           choice: the choice in the first stage
           prob  : the reward probability of four targets in the second stage
    test how the choices is affected by the correct/stay/outcome/
        transition:common or rate/outcome*transition
    """
    numTrials = trials["choice"].size
    trial_sep = np.where(np.diff(trials["block"])!=0)[0]
    numBlocks = np.sum(np.diff(trials["block"])!=0)+1
    correct_rate = []
    cho_prop = [[],[]]
    for nblock in range(numBlocks):
        if nblock == 0:
            continue
        trial_start = trial_sep[nblock-1]+1 if nblock > 0 else 0
        trial_end = trial_sep[nblock]+1 if nblock+1 < numBlocks else numTrials

        curr_block = trials["block"][trial_start]
#        cr = trials["choice"][trial_start:trial_start+30].values
##        cr = trials["choice"][trial_start:trial_end].values
#        cr = cr-1 if curr_block == 0 else 2-cr

        trials_range = np.arange(trial_start-10,trial_start+30)
        cr = [trials["choice"][trial]-1 if trials["block"][trial] == 0 else 2-trials["choice"][trial] for trial in trials_range]



        cho_prop[int(curr_block)].append(trials["choice"][trial_start:trial_end].mean()-1)
        correct_rate.append(cr)
     
    correct_rate = np.vstack(correct_rate)
    
    return np.nanmean(correct_rate, axis = 0)


def hist_effect(common, reward, choice, hist_len = 10):
    """
    input: common: the status of transition from the first to the second stage
           reward: reward feedback
           choice: the choice in the first stage
    test how the choices is affected by the interaction between the outcome and
           the transition status(common or rare) in the last 10 trials 
    """
    numTrials = common.size
    regressors = np.zeros((hist_len*4, numTrials))

    stay = np.diff(choice)
    stay[stay!=0] = 1
    stay = 1-stay
    
    states = reward*2 + 1-common
    for i in range(4):
        target_trials_base = np.where(states==i)[0]
        for ii in range(hist_len):
            temp = np.zeros((1, numTrials))
            target_trials = target_trials_base + ii
            temp[0,target_trials[target_trials<numTrials-1]] = 1
            regressors[i*hist_len+ii,:] = temp
    
#    lm = LogisticRegression(fit_intercept=False, penalty='l1', solver='liblinear')
    lm = LogisticRegression(fit_intercept=True)
    lm.fit(regressors[:,:-1].T, stay)
    return lm.coef_

def factors_eff(common, reward, choice, block):
    """
    input: common: the status of transition from the first to the second stage
           reward: reward feedback
           choice: the choice in the first stage
           prob  : the reward probability of four targets in the second stage
    test how the choices is affected by the correct/stay/outcome/
        transition:common or rate/outcome*transition
    """
    correct = block[:-1] == choice[:-1]
    repeat = np.ones(choice[:-1].size)
    outcome = reward[:-1]*1/2
    outcome[outcome==0] = -1/2
    trans = common[:-1]-0.5
    trans_out = trans*outcome*2
    
    stay = np.diff(choice)
    stay[stay!=0] = 1
    stay = 1-stay
    
    regressors = np.vstack((correct, repeat, outcome, trans, trans_out))
    # l1 is used owing to the correlation between the outcome and the trans_outcome
#    lm = LogisticRegressionCV(cv=10, fit_intercept=False, penalty='l1', solver='liblinear')
    lm = LogisticRegression(fit_intercept=True)
    lm.fit(regressors.T, stay)
    return lm.coef_

def cost_mfmb(params, choices, blocks, states, rewards, ReturnQ = False):
    """
    return the negative log likelihood given the choices and reward feedback
    
    """
    nll = 0
    gama = 1
    trans_prob = 0.8
    n_states, n_arms = 2, 2

    w1, lr_MB, lr_MF, lr_lambda, beta1, rep = params[0], params[1], params[2], params[3], params[4]*10, params[5]*10
    w2 = 1 - w1

    NumTrials = choices.size
    Q_value_based = np.zeros((n_states,n_arms))
    Q_value_free = np.zeros((n_states,n_arms))

    Q_value_all = []
    action_prob_all = []
#    max_nll = 0
#    nll_all = np.zeros(NumTrials,)
    for ntrial in range(1,NumTrials):
        ######## True action and states
        state = states[ntrial]
        action = choices[ntrial]
        reward = rewards[ntrial]
        ######## choice probability based on MB/MF-RL
        Q_value = w1*Q_value_based + w2*Q_value_free
        action_prob = softmax_prob(0, Q_value, beta1, [choices[ntrial-1], rep])
        ######## calculating the log likelihood 
        nll += - ((1-action)  * np.log(action_prob[0]  + 1e-10) + action  * np.log(action_prob[1]  + 1e-10))
#        nll_all[ntrial]  = - ((1-action)  * np.log(action_prob[0]  + 1e-10) + action  * np.log(action_prob[1]  + 1e-10))
#        max_nll = np.max([nll_all[ntrial], max_nll])
#        
#        if nll_all[ntrial].round(3) == 0.693:
#            a=1

        Q_value_all.append(Q_value)
        action_prob_all.append(action_prob)
        
        ####### model-based update
        Q_value_based[1, state] += lr_MB*(reward - Q_value_based[1, state])
                
        Q_value_based[0, 0] = trans_prob * Q_value_based[1, 0] + (1-trans_prob) * Q_value_based[1, 1]
        Q_value_based[0, 1] = trans_prob * Q_value_based[1, 1] + (1-trans_prob) * Q_value_based[1, 0]
        ####### model free update
        second_q_error = reward - Q_value_free[1, state]
        Q_value_free[1, state] += lr_MF*lr_lambda*(second_q_error)
       
        first_q_error = Q_value_free[1,state] - Q_value_free[0, action]
        Q_value_free[0, action] += gama*lr_MF*first_q_error

    if ReturnQ:
        return nll, np.array(Q_value_all), action_prob_all
    else:
        return nll



    
    
def data_extract(file_paths):
    """
    
    """
    df_details = pd.DataFrame([], columns = {'state', 'choice', 'reward', 'block','resp_output'})
    df_summary = pd.DataFrame([], columns = {'stay_num', 'type_num', 'coef_hist','coef_factors','fit_params','cr'})
    
    for i, file in enumerate(file_paths):
        paod, trial_briefs = load(file)
        
        trials = get_stinfo(paod, trial_briefs, completedOnly = False)
        cr = cr_block(trials)
        
        trials = get_stinfo(paod, trial_briefs)
        stay_num, type_num, common_all = stayprob(trials["choice"].values-1, trials["state"].values, trials["reward"])
        
        print('complete rate:{:6d}'.format(trials["reward"].size))
        print('bias rate:{:6f}'.format(trials["choice"].mean()-1))
                
#        reward = trials["reward"]
#        choice = trials["choice"].values-1
#        state = trials["state"].values
#        type_3_num = np.zeros((2,2,2)) # considering choice/reward/state
#        for i in [0,1]:
#            for ii in [0,1]:
#                for iii in [0,1]:
#                    labels = np.intersect1d(np.intersect1d(np.where(choice==i), np.where(reward==ii)), np.where((state-1)==iii))
#                    type_3_num[i,ii,iii] = labels.size
#        
#        choice_diff = np.diff(choice)
#        choice_diff[choice_diff.round(1)!=0]=1
#        choice_diff = np.hstack(([0], choice_diff))
#        stay = 1-choice_diff # wich trial is stay
#        
#        labels = np.intersect1d(np.where(common_all==0), np.where(reward==1))
#        labels = labels[labels<(common_all.size-1)]
#        labels = labels[labels>0]
#        asd = stay[labels+1] # + 2*reward[labels+1]
#        
#        print([np.unique(asd), [np.sum(asd==i) for i in np.unique(asd)]])

#        coef_hist,coef_factors=[],[]
        coef_hist = hist_effect(common_all, trials["reward"], trials["choice"], hist_len = 5)
        coef_factors = factors_eff(common_all, trials["reward"], trials["choice"], trials["block"])
#        
        ## fit the reinforcement learning model
        bounds=[[0,1],[0,1],[0,1],[0,1],[0,3],[0,3]]
        cons = []#construct the bounds in the form of constraints
        for factor in range(len(bounds)):
            l = {'type': 'ineq','fun': lambda x: x[factor]-bounds[factor][0]}
            u = {'type': 'ineq','fun': lambda x: bounds[factor][1]-x[factor]}
            cons.append(l)
            cons.append(u)
        
        nll_wrapper = lambda parameters: cost_mfmb(parameters,
                                                   trials["choice"]-1,
                                                   trials["block"], 
                                                   trials["state"]-1,
                                                   trials["reward"]
                                                   )
        params = np.array([0.7, 0.5 , 0.8, 0.5, 0.1 , 0.1])+np.random.randn(6,)*0.1
#        cho_prob = 0
#        asd = trials["choice"].values-1
#        for i, value in enumerate(first_action_prob_all2):
#            cho_prob += value[asd[i]]
        
#        res = minimize(nll_wrapper, x0=params, method='COBYLA', constraints=cons)
        res2 = minimize(nll_wrapper, x0=params, method='SLSQP' , bounds=bounds, constraints=cons)
        print(res2)
#        res2, Q, first_action_prob_all2 = cost_mfmb(params,
#                                                   trials["choice"]-1,
#                                                   trials["block"], 
#                                                   trials["state"]-1,
#                                                   trials["reward"],
#                                                   ReturnQ = True
#                                                   )
#        first_action_prob_all2 = np.array(first_action_prob_all2)
#        prob = first_action_prob_all2[:,0] -first_action_prob_all2[:,1]
#        plt.plot(prob,trials["choice_stage1"][1:],'.')
        
        # keep summary data of each file for plotting
        resp_hidden, resp_output = get_resp_ts(paod, trial_briefs) 

        df_summary.loc[i] = {
                'stay_num':    stay_num, 
                'type_num':    type_num,
                'coef_hist':   coef_hist,
                'coef_factors':coef_factors,
                'fit_params':  params,#res2.x,
                'cr':       cr
                }
        
        df_details.loc[i] = {
                'block':  trials["block"].values, 
                'state':  trials["state"].values,
                'choice': trials["choice"].values,
                'reward': trials["reward"].values,
                'resp_output':   resp_output[:,:,6:8]
                }

    return df_details, df_summary


def plot_value_coding(df_details, df_summary):

    """
    test the linear relation between the Q value and activities of the units 
    in the hidden layer and output layer

    """
#    from lrtools import sim_linreg
    Q_value_group_all, resp_group_all = [],[]
    time_fir_cho = 4
    n_group = 10
    for i in range(df_summary.shape[0]):
        resp_output = df_details.resp_output[i]
        nll, Q_value, _ = cost_mfmb(df_summary.fit_params[i],
                                    df_details["choice"][i]-1,
                                    df_details["block"][i], 
                                    df_details["state"][i]-1,
                                    df_details["reward"][i],
                                    ReturnQ = True
                                    )
        
        Q_value_diff = Q_value[:,0,0]-Q_value[:,0,1]
        resp_diff = resp_output[:,time_fir_cho,0]-resp_output[:,time_fir_cho,1]
        argsort = np.argsort(Q_value_diff)
        groups = np.linspace(0,df_details["block"][i].size,n_group+1).astype(np.int)
        
        Q_value_group = []
        resp_group = []
        for start, end in zip(groups[:-1], groups[1:]):
            order = argsort[start:end]
            Q_value_group.append(Q_value_diff[order].mean())
            resp_group.append(resp_diff[order+1,].mean())
            
        Q_value_group_all.append(Q_value_group)
        resp_group_all.append(resp_group)

    
    resp_group_all = np.array(resp_group_all)
    Q_value_group_all = np.array(Q_value_group_all)
    
    fig = plt.figure('q value coding')
    plt.errorbar(Q_value_group_all.mean(axis=0),resp_group_all.mean(axis=0),
                 fmt='o', xerr =stats.sem(Q_value_group_all, axis=0, nan_policy = 'omit'),
                 yerr = stats.sem(resp_group_all,axis=0, nan_policy = 'omit'))

#        plt.plot(range(q.shape[1]), q[0])
#        plt.plot(range(q.shape[1]), savgol_filter(resp[0],51,3))
    plt.xlabel('Q value diff')
    plt.ylabel('response diff')
    fig.savefig('../figs/supp2_ TS-q value coding.eps', format='eps', dpi=1000)
    plt.show()
    plt.clf#
    
    
    fit_params = np.vstack(df_summary.fit_params)
    fig2 = plt.figure('parameters')
    plt.boxplot(fit_params)
    plt.xticks(np.arange(1,7), ('w1', 'lr_MB', 'lr_MF', 'lambda', 'beta/10', 'p/10'))
    fig2.savefig('../figs/supp2_ TS-model fitting.eps', format='eps', dpi=1000)
    plt.show()
    print(fit_params.mean(axis=0))
    print(stats.sem(fit_params,axis=0))
    

        

    
    return 
def plot_bhv(df_summary):
    '''
    plot the stay probability ; historical effect;  and factors affect the choices
    '''
#    fig_w, fig_h = (10, 7)
#    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
    stay_prob = np.vstack(df_summary.stay_num.values)/np.vstack(df_summary.type_num.values)

    fig = plt.figure('stay probability')
    plt.bar([1,4], np.mean(stay_prob[:,[0,2]], axis=0), yerr = stats.sem(stay_prob[:,[0,2]], axis=0), label = 'Common')
    plt.bar([2,5], np.mean(stay_prob[:,[1,3]], axis=0), yerr = stats.sem(stay_prob[:,[1,3]], axis=0), label = 'Rare')
    plt.legend()
    plt.xticks([1.5,4.5], ('Rewarded', 'Unrewarded'))
    plt.ylabel('stay probability')
#    plt.ylim(0.3,0.7)
    plt.yticks(np.linspace(0,1,num=6,endpoint = True))
    fig.savefig('../figs/supp2_ ST-stay probability.eps', format='eps', dpi=1000)
    plt.show()
    plt.clf
    
    coef = np.vstack(df_summary.coef_hist.values)
    labels = ['common unrewarded','rare unrewarded','common rewarded','rare rewarded']
    fig2 = plt.figure('hist effect')
    hist_len = int(coef.shape[1]/4) # four groups
    asd = []
    for i in range(20):
        asd.append(stats.ttest_1samp(coef[:,i],0))
    print(asd)
    for i in range(4):# four groups
        plt.errorbar(range(hist_len),
                     np.mean(coef[:,hist_len*i:hist_len*(i+1)], axis=0), 
                     yerr = stats.sem(coef[:,hist_len*i:hist_len*(i+1)], axis=0),
                     marker = 'o',
                     markersize = 20,
                     markerfacecolor='none',
                     label = labels[i])
    plt.xticks(range(5), ('-1', '-2', '-3', '-4', '-5'))
    plt.xlabel('lag (trials)')
    plt.ylabel('log odds')
    plt.yticks(np.linspace(-2,2,num=9,endpoint = True))
    fig2.legend()
    fig2.savefig('../figs/supp2_ ST-hist effect.eps', format='eps', dpi=1000)
    plt.show()
    
    coef = np.vstack(df_summary.coef_factors.values)
    asd = []
    fig3 = plt.figure('coef_factors')
    for i in range(5):
        asd.append(stats.ttest_1samp(coef[:,i],0))
        plt.errorbar(i, np.mean(coef[:,i], axis=0), yerr = stats.sem(coef[:,i], axis=0))
        plt.plot(i, np.mean(coef[:,i], axis=0), 'o')
    print(asd)
    plt.xticks(range(5), ['correct', 'repeat', 'outcome', 'trans', 'trans_out'])
    plt.ylabel('log odds')
    plt.yticks(np.linspace(0,3.5,num=8,endpoint = True))

    fig3.savefig('../figs/supp2_ ST-factors effect.eps', format='eps', dpi=1000)
    plt.show()
    
    
    cr = np.vstack(df_summary.cr.values)
    fig4 = plt.figure('cr')
    plt.errorbar(range(cr.shape[1]), np.mean(cr, axis=0), yerr = stats.sem(cr, axis=0))
    plt.ylabel('correct rate')
    plt.xlabel('lag (trials)')
    plt.yticks(np.linspace(0,1,num=6,endpoint = True))
    fig4.savefig('../figs/supp2_ ST-correct rate.eps', format='eps', dpi=1000)
    plt.show()
    
def main(file_paths = None):
    print("start")
    print("select the files")
    if file_paths == None:
        root = tk.Tk()
        root.withdraw()
        file_paths = filedialog.askopenfilenames(parent = root,
                                                title = 'Choose a file',
                                                filetypes = [("HDF5 files", "*.hdf5")]
                                                )
    ##
    df_details, df_summary = data_extract(file_paths)
    plot_bhv(df_summary)
    plot_value_coding(df_details, df_summary)

#    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#        print(df_trials)

if __name__ == '__main__':
    main()

