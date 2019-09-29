# -*- coding: utf-8 -*-
# @Author: Zhewei Zhang
# @Date: 27/07/2018, 19:11

import os
import math
import numpy as np
import pandas as pd
from toolkits import PaoDing, base642obj3
from task_info import rtshapebrief_config
################################
"""
general tools
"""
def pca(X):
    """
    :param X: input: array, dimension * time point
    :return: new space, mean_value of each dimension
    """
    # X = X - dot(np.ones((int(X.shape[0]), 1)), X.mean(axis=0))
    mean_value = X.mean(axis=0)
    X = X - mean_value
    covariancematrix = np.matmul(X, X.transpose())/X.shape[1]
    eigvals, eigvects = np.linalg.eig(covariancematrix)
    # Sort the eigenvalues and recompute matrices
    order = np.argsort(eigvals.real)[::-1]
    eigvects = eigvects[:, order].real
    eigvals = eigvals[order].real
    mean_value = mean_value[order]
    return eigvects, eigvals, mean_value

def array2list(*kwargs):
    re = []
    for var in kwargs:
        re.append(np.array(var))
    return re

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def load(path_file):
    ## load files
    path, file = os.path.split(path_file)
    
    task_time = ''
    for i in range(len(file.split('-'))-1):
        if i == len(file.split('-'))-2:
            task_time = task_time + file.split('-')[i]
        else:
            task_time = task_time + file.split('-')[i] + '-'
    task_name = file.split('-')[-1][:-5]
    print("current file: {:6s}".format(path_file))
    paod = PaoDing(task_name)
    paod.load_material(path+ '/',task_time)
    record_name = path + '/'+ task_time + "-"+ task_name +".validation_brief.csv"
    trial_briefs = pd.read_csv(record_name, index_col=0)

    return paod, trial_briefs

#################################
"""
tools for the shape task
"""
def get_bhvinfo(paod,trial_briefs,choices_trials = True):
    """
    :param paod: from class paoding
    :return: choice: choose left or right; finish choice or not; when do it make a choice
    shapes: how many shapes are used for choice; temporal weight ; summed weight; shape on time; shape on
    trials: whole trial length
    """
    ###
    num_Trials = len(paod.index_records)
    choice = {'status': [], 'time': [], 'left': [],'correct':[],'correct_logLR':[]}
    shapes = {'rt': [], 'tempweight': [], 'sumweight': [], 'ontime': [], 'order':[], 'on': []} # 
    trial = {'length': [],'num':[],'tarOn_time': []} #
    reward = {'pred': [], 'time': []}
    shape_weight_pair = {-0.9:1,-.7:2,-.5:3,-.3:4,-.1:5,.0:0,.1:6,.3:7,.5:8,.7:9,.9:10}
    shapeRT_max = 30
    time_foreachshape = 5
    a = 0
    for i in range(num_Trials):
        brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
        # only keep the trials in which the decision is made
        if choices_trials:
            if not brief_info['completed']:
                a += 1
                continue
        else:
            if brief_info['completed']:
                a += 1
                continue
        
        n_shape_curr = brief_info['rt']
        tmpweight_tmp = np.zeros(shapeRT_max,)
        tmpweight_tmp[:n_shape_curr] = brief_info['tmp_weight_sequence']
        tmpweight_tmp = tmpweight_tmp.round(2)

        shapes['rt'].append(n_shape_curr)
        shapes['tempweight'].append(tmpweight_tmp)
        shapes['sumweight'].append(np.sum(tmpweight_tmp))
        shapes['ontime'].append(np.arange(3, time_foreachshape * n_shape_curr, time_foreachshape, dtype=int))
        shapeorder_tmp = []
        for ii in range(shapeRT_max):
            shapeorder_tmp.append(shape_weight_pair[tmpweight_tmp[ii]])
        shapes['order'].append(np.array(shapeorder_tmp))
        shapes['on'].append(np.arange(n_shape_curr) + 1)
        ##
        choice['status'].append(brief_info['chosen'])
        choice['time'].append(brief_info['choice_time'][0])
        choice['left'].append(3-2*brief_info['choice']) # 1 if left else -1
        choice['correct_logLR'].append(True if (np.sum(brief_info['tmp_weight_sequence'])>=0)==(brief_info['choice']==1) else False)
        choice['correct'].append(True if brief_info['trialtype']==(2-brief_info['choice']) else False)
        ##
        trial['length'].append(brief_info['choice_time'][0]+6)
        trial['num'].append(i)
        trial['tarOn_time'].append(2)
        ##
        reward['pred'].append(brief_info['reward'])
        reward['time'].append(brief_info['choice_time'][-1])
        
    trial = pd.DataFrame(trial) 
    choice = pd.DataFrame(choice)
    shapes = pd.DataFrame(shapes)
    reward = pd.DataFrame(reward)
    finish_rate = 0 if num_Trials==0 else 1-a/num_Trials
    return trial, choice, shapes, reward, finish_rate

def get_bhvinfo2(paod,trial_briefs,choices_trials = True):
    """
    
    older one
    
    :param paod: from class paoding
    :return: choice: choose left or right; finish choice or not; when do it make a choice
    shapes: how many shapes are used for choice; temporal weight ; summed weight; shape on time; shape on
    trials: whole trial length
    """
    input_setting = rtshapebrief_config()
    choice_position = input_setting['choice']
    reward_position = input_setting['reward']
    ###
    num_Trials = len(paod.index_records)
    choice = {'status': [], 'time': [], 'left': [],'correct':[],'correct_logLR':[]}
    shapes = {'rt': [], 'tempweight': [], 'sumweight': [], 'ontime': [], 'order':[], 'on': []} # 
    trial = {'length': [],'num':[],'tarOn_time': []} #
    reward = {'pred': [], 'time': []}
    shape_weight_pair = {-0.9:1,-.7:2,-.5:3,-.3:4,-.1:5,.0:0,.1:6,.3:7,.5:8,.7:9,.9:10}
    shapeRT_max = 30
    time_foreachshape = 5
    a = 0
    for i in range(num_Trials):
        brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
        # only keep the trials in which the decision is made
        if choices_trials:
            if not brief_info['made_choice']:
                a += 1
                continue
        else:
            if brief_info['made_choice']:
                a += 1
                continue

        fd, gd, rd, _ = paod.get_neuron_behavior_pair(i)
        if (gd.shape[0]-3)%5 !=0:
            continue
        
        n_shape_curr = len(brief_info['tmp_weight_sequence'])
        ##
        tmpweight_tmp = np.zeros(shapeRT_max,)
        tmpweight_tmp[:n_shape_curr] = brief_info['tmp_weight_sequence']
        tmpweight_tmp = np.round(tmpweight_tmp,2)
        shapes['rt'].append(n_shape_curr)
        shapes['tempweight'].append(tmpweight_tmp)
        shapes['sumweight'].append(np.sum(tmpweight_tmp))
        shapes['ontime'].append(np.arange(3, time_foreachshape * n_shape_curr, time_foreachshape, dtype=int))
        shapeorder_tmp = []
        for ii in range(shapeRT_max):
            shapeorder_tmp.append(shape_weight_pair[tmpweight_tmp[ii]])
        shapes['order'].append(np.array(shapeorder_tmp))
        shapes['on'].append(np.arange(n_shape_curr) + 1)
        ##
        choice['status'].append(brief_info['made_choice'])
        choice['time'].append(time_foreachshape * n_shape_curr + 2)
        choice['left'].append(2*gd[choice['time'][-1], choice_position[1]]-1) # 1 if left else 0
        choice['correct_logLR'].append(True if (np.sum(brief_info['tmp_weight_sequence'])>=0)==brief_info['chosen_red'] else False)
        choice['correct'].append(True if brief_info['trialtype']==brief_info['chosen_red'] else False)
        ##
        trial['length'].append(gd.shape[0])
        trial['num'].append(i)
        trial['tarOn_time'].append(2)
        ##
        if fd.shape[0] > choice['time'][-1] + 2:
            reward['pred'].append(gd[choice['time'][-1] + 2, reward_position[0]])
            reward['time'].append(choice['time'][-1] + 2)
        else:
            raise Exception('fd.shape is not correct')
    trial = pd.DataFrame(trial) 
    choice = pd.DataFrame(choice)
    shapes = pd.DataFrame(shapes)
    reward = pd.DataFrame(reward)
    finish_rate = 0 if num_Trials==0 else 1-a/num_Trials
    return trial, choice, shapes, reward, finish_rate


def vectorVar(x, *args):
    """
    :param x: number of point by number of dimension
    :param args: optional, represent the central point
    :return: mean/ste distance batween the dots and central point
    """
    if isinstance(x, list):
        x = np.asarray(x)
#    if not args:
#        central_point = np.mean(x,axis=0)
#    else:
#        central_point = args
    nTimepoints = x.shape[1]
    nTrials = x.shape[0]
    distance = {}
    for i in range(nTimepoints):
        distance[i] = []
        central_point = np.mean(x[:,i,:],axis=0)
        for ii in range(nTrials):
            distance[i].append(np.linalg.norm(x[ii,i,:]-central_point))
            
    mean_dis = [np.mean(distance[i]) for i in range(nTimepoints)]
    ste_dis = [np.std(distance[i])/np.sqrt(nTrials) for i in range(nTimepoints)]

    return mean_dis, ste_dis


#######################
def get_hidden_resp_all(paod,trial_briefs,gates = False):
    """
    :param resp_hidden_group: from class paoding
    :return: response of neurons in the hidden layer
    """
    all_resp = []
    num_Trials = len(paod.index_records)
    if gates:
        ig_all, rg_all, ng_all = [],[],[]
        for i in range(num_Trials):
            brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
            # only keep the trials in which the decision is made
            if not brief_info['completed']:
                continue
            _, _, _, nd, rg, ig, ng = paod.get_neuron_behavior_pair(i,gates = True)
            all_resp.append(nd)
            rg_all.append(rg)
            ig_all.append(ig)
            ng_all.append(ng)
        all_resp = np.array(all_resp)
        rg_all = np.array(rg_all)
        ig_all = np.array(ig_all)
        ng_all = np.array(ng_all)
        return all_resp, rg_all,ig_all, ng_all
    else:
        for i in range(num_Trials):
            brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
            fd, gd, rd, nd = paod.get_neuron_behavior_pair(i)
            if not brief_info['completed']:
                continue
            if (gd.shape[0]-3)%5 !=0:
                b=1

            nd = nd.squeeze()
            all_resp.append(nd)
        all_resp = np.array(all_resp)
        return all_resp
    
def get_hidden_resp_all2(paod,trial_briefs,gates = False):
    """
    :param resp_hidden_group: from class paoding
    :return: response of neurons in the hidden layer
    """
    all_resp = []
    num_Trials = len(paod.index_records)
    if gates:
        ig_all, rg_all, ng_all = [],[],[]
        for i in range(num_Trials):
            brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
            # only keep the trials in which the decision is made
            if not brief_info['made_choice']:
                continue
            _, _, _, nd, rg, ig, ng = paod.get_neuron_behavior_pair(i,gates = True)
            if (nd.shape[0]-3)%5 !=0:
                continue
            all_resp.append(nd)
            rg_all.append(rg)
            ig_all.append(ig)
            ng_all.append(ng)
        all_resp = np.array(all_resp)
        rg_all = np.array(rg_all)
        ig_all = np.array(ig_all)
        ng_all = np.array(ng_all)
        return all_resp, rg_all,ig_all, ng_all
    else:
        for i in range(num_Trials):
            brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
            fd, gd, rd, nd = paod.get_neuron_behavior_pair(i)
            if not brief_info['made_choice']:
                continue
            if (nd.shape[0]-3)%5 !=0:
                continue
            nd = nd.squeeze()
            all_resp.append(nd)
        all_resp = np.array(all_resp)
        return all_resp


def get_hidden_resp_group(paod):
    """
    :param paod: from class paoding
    :return: the response of neurons in the hidden layer, and grouped by the number of shapes
    """
    resp_hidden_group = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[]
        , 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: [], 27: [],28: [], 29: [], 30: []}
    num_Trials = len(paod.index_records)
    for i in range(num_Trials):
        _, _, _, nd = paod.get_neuron_behavior_pair(i)
        brief_info = paod.analyst.decode_trial_brief(i, paod.trial_briefs)
        nd = nd.squeeze()
        n_shape_curr = len(brief_info['tmp_weight_sequence'])
        resp_hidden_group[n_shape_curr].append(nd)
    return resp_hidden_group

def get_hidden_resp_group2(paod):
    """
    :param paod: from class paoding
    :return: the response of neurons in the hidden layer, and grouped by the number of shapes
    """
    resp_hidden_group = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[]
        , 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: [], 27: [],28: [], 29: [], 30: []}
    num_Trials = len(paod.index_records)
    for i in range(num_Trials):
        _, _, _, nd = paod.get_neuron_behavior_pair(i)
        brief_info = paod.analyst.decode_trial_brief(i, paod.trial_briefs)
        nd = nd.squeeze()
        n_shape_curr = len(brief_info['tmp_weight_sequence'])
        resp_hidden_group[n_shape_curr].append(nd)
    return resp_hidden_group


#######################
"""
tools for the two step task
"""

def get_resp_ts(paod,trial_briefs,gates = False):
    """
    :param resp_hidden_group: from class paoding
    :return: response of neurons in the hidden layer
    """
    resp_hidden = []
    resp_output = []
    num_Trials = len(paod.index_records)
    if gates:
        pass
    else:
        for i in range(num_Trials):
            brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
            if brief_info['completed'] is not True:
                continue
            fd, gd, rd, nd = paod.get_neuron_behavior_pair(i)
            resp_hidden.append(nd.squeeze())
            resp_output.append(rd.squeeze())
#        print([resp_output[i].shape[0] for i in range(len(resp_output))])
        return np.array(resp_hidden), np.array(resp_output)

def get_tsinfo(paod,trial_briefs):
    """
    :param paod: from class paoding
    :return: first/second stage choices, stage2 state, reward probability, reward state
    """
#    input_setting = tsDawbrief_config()
#    choice_position = input_setting['choice']
#    reward_position = input_setting['reward']
    ###
    num_Trials = len(paod.index_records)
    trials = {'stage2_state':[], 'choice_stage1':[], 'choice_stage2':[], 'reward':[], 'prob':[]}
    for i in range(num_Trials):
        brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
        if brief_info['completed'] != True:
            continue
        # only keep the trials in which the decision is made
        trials['prob'].append(brief_info['prob'])
        trials['reward'].append(brief_info['trial_reward'])
        trials['stage2_state'].append(brief_info['stage2_state'])
        trials['choice_stage1'].append(brief_info['choice_stage1'])
        trials['choice_stage2'].append(brief_info['choice_stage2'])
    trials = pd.DataFrame(trials)
    return trials


def stayprob(action_all,state_all,reward_all):
    NumTrials = len(reward_all)
    type_num = np.array([0,0,0,0])
    stay_num = np.array([0,0,0,0])
#    common_all = []
    
    common_all = (action_all == (state_all-1))
#    for ntrial in range(NumTrials):
#        if action_all[ntrial] == 0:
#            if state_all[ntrial] == 1:
#                common_all.append(1) # common
#            elif state_all[ntrial] == 2:
#                common_all.append(0) # rare
#            else:
#                raise('unknown state')
#        elif action_all[ntrial] == 1:
#            if state_all[ntrial] == 1:
#                common_all.append(0) # rare
#            elif state_all[ntrial] == 2:
#                common_all.append(1) # common
#            else:
#                raise('unknown state')
#        else:
#            raise('unknown action')
            
    for ntrial in range(NumTrials-1):
        if common_all[ntrial]==1:
            if reward_all[ntrial]==1:
                type_num[0] += 1
                if action_all[ntrial+1] == action_all[ntrial]:
                    stay_num[0] +=  1
            elif reward_all[ntrial]==0:
                type_num[2] += 1
                if action_all[ntrial+1] == action_all[ntrial]:
                    stay_num[2] +=  1
        else:
            if reward_all[ntrial]==1:
                type_num[1] += 1
                if action_all[ntrial+1] == action_all[ntrial]:
                    stay_num[1] +=  1
            elif reward_all[ntrial]==0:
                type_num[3] += 1
                if action_all[ntrial+1] == action_all[ntrial]:
                    stay_num[3] +=  1

    return stay_num, type_num, common_all

#######################

"""
tools for the sure target task
"""
def get_sureinfo(paod,trial_briefs):
    """
    :param paod: from class paoding
    :return: coherence, sure trials, choice, reward, evidence difference for each trial 
    """
#    input_setting = tsDawbrief_config()
#    choice_position = input_setting['choice']
#    reward_position = input_setting['reward']
    ###
    num_Trials = len(paod.index_records)
    trials = {'choice':[], 'chosen':[], 'reward':[], 'sure_trial':[], 'coherence':[],'acc_evi':[],'randots_dur':[]}
    for i in range(num_Trials):
        brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
        if brief_info['completed'] != True:
            continue
        # only keep the trials in which the decision is made
        trials['choice'].append(brief_info['choice'])
        trials['chosen'].append(brief_info['chosen'])
        trials['reward'].append(brief_info['reward'])
        trials['coherence'].append(brief_info['coherences'])
        trials['sure_trial'].append(brief_info['sure_trial'])
        trials['randots_dur'].append(brief_info['randots_dur'])        
#        trials['completed'].append(brief_info['completed'])
        trials['acc_evi'].append(brief_info['acc_evi'])
    trials = pd.DataFrame(trials)
    return trials


def get_hidden_resp_sure(paod,trial_briefs,gates = False):
    """
    :param resp_hidden_group: from class paoding
    :return: response of neurons in the hidden layer
    """
    all_resp = []
    num_Trials = len(paod.index_records)
    for i in range(num_Trials):
        brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
        fd, gd, rd, nd = paod.get_neuron_behavior_pair(i)
        if brief_info['completed'] != True:
            continue
        nd = nd.squeeze()
        all_resp.append(nd)
    all_resp = np.array(all_resp)
    return all_resp


#######################

"""
tools for the multisensory integration task
"""
def get_multinfo(paod,trial_briefs):
    """
    :param paod: from class paoding
    :return: coherence, sure trials, choice, reward, evidence difference for each trial 
    """
#    input_setting = tsDawbrief_config()
#    choice_position = input_setting['choice']
#    reward_position = input_setting['reward']
    ###
    num_Trials = len(paod.index_records)
    trials = {'choice':[], 'chosen':[], 'reward':[], 'modality':[], 'direction':[],'estimates':[]}
    for i in range(num_Trials):
        brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
        if brief_info['completed'] != True:
            continue
        # only keep the trials in which the decision is made
        trials['choice'].append(brief_info['choice'])
        trials['chosen'].append(brief_info['chosen'])
        trials['reward'].append(brief_info['reward'])
        trials['modality'].append(brief_info['modality'])
        trials['direction'].append(brief_info['directions'])
#        trials['completed'].append(brief_info['completed'])
        trials['estimates'].append(brief_info['estimates'])
    trials = pd.DataFrame(trials)
    return trials


def get_hidden_resp_mult(paod,trial_briefs,gates = False):
    """
    :param resp_hidden_group: from class paoding
    :return: response of neurons in the hidden layer
    """
    all_resp = []
    num_Trials = len(paod.index_records)
    for i in range(num_Trials):
        brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
        fd, gd, rd, nd = paod.get_neuron_behavior_pair(i)
        if brief_info['completed'] != True:
            continue
        nd = nd.squeeze()
        all_resp.append(nd)
    all_resp = np.array(all_resp)
    return all_resp


#######################

"""
tools for the reversal learning task
"""
def get_rvlrinfo(paod,trial_briefs):
    """
    :param paod: from class paoding
    :return: coherence, sure trials, choice, reward, evidence difference for each trial 
    """
#    input_setting = tsDawbrief_config()
#    choice_position = input_setting['choice']
#    reward_position = input_setting['reward']
    ###
    num_Trials = len(paod.index_records)
#    num_Trials = 4900
    trials = {'choice':[], 'chosen':[], 'reward':[], 'block':[], 'completed':[]}
    for i in range(num_Trials):
        brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
        if brief_info['completed']:
            trials['choice'].append(brief_info['choice'])
            trials['chosen'].append(brief_info['chosen'])
            trials['reward'].append(brief_info['reward'])
        if not brief_info['completed']:
            trials['choice'].append(np.nan)
            trials['chosen'].append(np.nan)
            trials['reward'].append(np.nan)
        trials['completed'].append(brief_info['completed'])
        trials['block'].append(brief_info['block'])
    trials = pd.DataFrame(trials)
    return trials

#######################

def get_stinfo(paod, trial_briefs, completedOnly = True):
    """
    :param paod: from class paoding
    :return: first/second stage choices, stage2 state, reward probability, reward state
    """

    num_Trials = len(paod.index_records)
    trials = {'state':[], 'choice':[], 'reward':[], 'block':[], 'common':[]}
    if completedOnly:
        for i in range(num_Trials):
            brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
            if brief_info['completed'] != True:
                continue
            # only keep the trials in which the decision is made
            trials['block'].append(brief_info['block'])
            trials['state'].append(brief_info['state'])
            trials['reward'].append(brief_info['reward'])
            trials['choice'].append(brief_info['choice'])
            trials['common'].append(brief_info['common'])
        trials = pd.DataFrame(trials)
        return trials
    else:
        for i in range(num_Trials):
            brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
            if brief_info['completed'] != True:
                trials['block'].append(brief_info['block'])
                trials['state'].append(np.nan)
                trials['reward'].append(np.nan)
                trials['choice'].append(np.nan)
                trials['common'].append(np.nan)

            else:
                trials['block'].append(brief_info['block'])
                trials['state'].append(brief_info['state'])
                trials['reward'].append(brief_info['reward'])
                trials['choice'].append(brief_info['choice'])
                trials['common'].append(brief_info['common'])
        trials = pd.DataFrame(trials)
        return trials


#######################

def get_frinfo(paod, trial_briefs, completedOnly = True):
    """
    :param paod: from class paoding
    :return: first/second stage choices, stage2 state, reward probability, reward state
    """

    num_Trials = len(paod.index_records)
    trials = {'chosen':[], 'reward':[], 'trial_length':[], 'choice_num':[],'choices':[]}
    for i in range(num_Trials):
        brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
        if brief_info['completed'] != True:
            continue
        # only keep the trials in which the decision is made
        trials['chosen'].append(brief_info['chosen'])
        trials['reward'].append(brief_info['reward'])
        trials['choices'].append(brief_info['choices'])
        trials['choice_num'].append(brief_info['choice_num'])
        trials['trial_length'].append(brief_info['trial_length'])
    trials = pd.DataFrame(trials)
    return trials

def get_output_resp_fixratio(paod,trial_briefs,gates = False):
    """
    :param resp_hidden_group: from class paoding
    :return: response of neurons in the hidden layer
    """
    all_resp = []
    num_Trials = len(paod.index_records)
    for i in range(num_Trials):
        brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
        fd, gd, rd, nd = paod.get_neuron_behavior_pair(i)
        if brief_info['completed'] != True:
            continue
        gd = rd.squeeze()
        all_resp.append(gd)
    all_resp = np.array(all_resp)
    return all_resp
#######################

def get_pdinfo(paod, trial_briefs, completedOnly = True):
    """
    :param paod: from class paoding
    :return: all actions
    """
    num_Trials = len(paod.index_records)
    trials = {'choices':[]}
    for i in range(num_Trials):
        brief_info = base642obj3(trial_briefs['trial_brief_base64'][i])
        trials['choices'].append(brief_info['choices'])
    trials = pd.DataFrame(trials)
    return trials

def get_resp_pendan(paod,trial_briefs, gates = False):
    """
    :param resp_hidden_group: from class paoding
    :return: response of neurons in the hidden layer
    """
    num_Trials = len(paod.index_records)
    if gates:
        all_rg = []
        all_ig = []
        all_ng = []
        for i in range(num_Trials):
            fd, gd, rd, nd, rg, ig, ng = paod.get_neuron_behavior_pair(i, gates = True)
            rg = rg.squeeze()
            ig = ig.squeeze()
            ng = ng.squeeze()
            all_rg.append(rg)
            all_ig.append(ig)
            all_ng.append(ng)
        all_rg, all_ig, all_ng = np.array(all_rg), np.array(all_ig), np.array(all_ng)
        return all_rg, all_ig, all_ng
    else:
        all_resp = []
        for i in range(num_Trials):
            fd, gd, rd, nd = paod.get_neuron_behavior_pair(i)
            nd = nd.squeeze()
            all_resp.append(nd)
        all_resp = np.array(all_resp)
        return all_resp
