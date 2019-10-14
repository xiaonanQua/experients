# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 20:16:38 2018

@author: Zhewei Zhang


find the when and which neurons in the hidden layer of the network
"""

## load model
import os
import torch
import numpy as np
from tkinter import filedialog
import tkinter as tk
import yaml
from scipy import stats
from task_info import rtshapebrief_config
from netools import neurons_inNetwork
os.chdir('/home/tyang/Documents/SeqLearning_code/seqrnn_multitask')
from model import GRUNetwork, RNNNetwork, LSTMNetwork
os.chdir('../holmes_m')


def model_selecting():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilenames(
                parent=root,title='Choose the model',
                filetypes=[("model files", "*_250000-*-2.pt")]
                )
    model_paths, models = [], []
    for i, model_path in enumerate(file_path):
        model_path, model = os.path.split(file_path[i])
        model_paths.append(model_path)
        models.append(model)
    print('*'*49)
    print('model path:', model_path)
    print('model name:', model)
    print('*'*49)
    return model_paths, models

def model_loading(model_path, model):
    model_parameters = {
            "nhid": 128,
            "nlayers": 1,
            "input_size": input_size,
            "batch_size": 1,
            "clip": 0.25,
            "lr": 0.6
        }
    if model_mode == 'GRU':
        rnn_model = GRUNetwork(model_parameters["input_size"],
                               model_parameters["nhid"],
                               model_parameters["batch_size"],
                               model_parameters["nlayers"],
                               model_parameters["lr"],
                               )
        rnn_model.load_state_dict(torch.load(model_path + '/' + model))
        ho_weight = rnn_model.decoder.weight.detach().numpy()
        w_hr, w_hi, w_hn = rnn_model.rnn.weight_hh_l0.chunk(3, 0)
        return ho_weight, w_hr, w_hi, w_hn
    
    
def when_which_constructer(ho_weight): # ho_weight, choice_pos,model_mode,*net_weight
    """
    when neuron: when should I make a choice
    which neuron: which target should I choose

    """
    when_value = ho_weight[choice_pos[1]:choice_pos[-1],:].mean(axis = 0) - ho_weight[choice_pos[0],:]    
#    when_value = ho_weight[choice_pos[1]:choice_pos[-1],:].mean(axis = 0) 
    which_value = ho_weight[choice_pos[1],:] - ho_weight[choice_pos[2],:]

    when = neurons_inNetwork(when_value, neuron_proportion)
    which = neurons_inNetwork(which_value, neuron_proportion)

    posneg_all = [np.sum(when_value>0), np.sum(when_value<0), np.sum(which_value>0), np.sum(which_value<0)]
    return when, which, posneg_all

def datasaving(model_path, model, when, which, rnn_weight, ho_weight):
    saving_name = model_path + '/saving_'+ model[:-3] + '-old_threshold'+ str(neuron_proportion) +'.yaml'
    data = {'model':model,'when':when,'which':which,'threshold':neuron_proportion,'rnn_weight':rnn_weight,'output_weight':ho_weight}
        
    with open(saving_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    print('lesion file saved')
    print('file path, file name:',saving_name)
    

def main(model_paths, models):
    posneg_total = []
    for model_path, model in zip(model_paths, models):

        if model_mode == 'GRU':
            ho_weight, w_hr, w_hi, w_hn = model_loading(model_path, model)
            when, which, posneg_all = when_which_constructer(ho_weight)
            datasaving(model_path, model, when, which, w_hr, ho_weight) 
        posneg_all.extend([(when.pos.shape[0]+
                           when.neg.shape[0]+ 
                           which.pos.shape[0]+
                           which.neg.shape[0])/128,
                           when.pos.shape[0],
                           when.neg.shape[0], 
                           which.pos.shape[0], 
                           which.neg.shape[0]])
        posneg_total.append(posneg_all)
    posneg_total = np.array(posneg_total)
    print('mean:', posneg_total.mean(axis=0))
    print('sem:' , stats.sem(posneg_total))
    
    
if __name__ == '__main__':
    
    input_setting = rtshapebrief_config()
    choice_pos = input_setting['choice']
    input_size = input_setting['input_size']
    
    model_mode = 'GRU'
    neuron_proportion = 0.5
    model_paths, models = model_selecting()
    main(model_paths, models)
    