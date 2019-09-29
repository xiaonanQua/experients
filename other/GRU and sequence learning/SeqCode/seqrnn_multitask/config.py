#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017

@author: Huzi Cheng, Zhewei Zhang

configuration file for reaction time version of the shape task

"""
import copy
import os

rt_shape = {
        "name": "reaction_shape_lr",
        "model_mode":'GRU',
        "optimizer":'Adam',        
        "data_file": "../data/",
        "validation_data_file": "../data/",
        "data_attr": "data_RT",
        "data_brief": "data_RT_Brief",
        "log_path": "../log_m/RT/",
        "model_path": "../save_m/RT/model-reaction_shapelr_",
        "load_model": "",
        "train_num": 2.5e5,
        "input_size": 20,
        "batch_size": 128,
        "reset_hidden":True,
        "lr": 1e-3,
        "record": 1,
        "need_train": 3,
        "need_validate": 1,
        "lesion": "",
        "anlys_file":""
    }

task_configurations = []
for i in range(20):
    task_configuration = copy.deepcopy(rt_shape)
    ## for each run
    task_configuration["data_file"] = '../data/'
    task_configuration["validation_data_file"] = '../data/'
    task_configurations.append(task_configuration)

# =============================================================================
# for lesion simulation

## a list of models has been trained
model = ['/home/tyang/Documents/SeqLearning_code/save_m/RT/model-reaction_shapelr.pt']

rt_shape_lesion = {
        "name": "reaction_shape_lr",
        "model_mode":'GRU',
        "optimizer":'Adam',        
        "data_file": "../data/",
        "validation_data_file": "../data/",
        "data_attr": "data_RT",
        "data_brief": "data_RT_Brief",
        "log_path": "../log_m/RT/back_lesion/",
        "model_path": "../save_m/RT/back_lesion/model-reaction_shapelr_",
        "load_model": "",
        "train_num": 0,
        "input_size": 20,
        "batch_size": 128,
        "reset_hidden":True,
        "lr": 1e-3,
        "record": 1,
        "need_train":0,
        "need_validate": 1,
        "lesion": ['output', 'when_pos', 'when_neg', 'which_pos', 'which_neg'],
#        "lesion": ['output','when_pos', 'when_neg'],
        "anlys_file":""
    }



task_configurations_lesion = []
#for lesion_prop in ['0.4','0.3','0.2','0.1']:
for lesion_prop in ['0.5']:
    for i, model_file in enumerate(model):
        task_configuration = copy.deepcopy(rt_shape_lesion)
        ## self-define
        task_configuration["load_model"] = model_file
        task_configuration["data_file"] = '../data/'
        task_configuration["validation_data_file"] = '../data/'
        
        path, file = os.path.split(model_file)
        task_configuration["anlys_file"] = path+"/saving_model.yaml"
        
        task_configurations_lesion.append(task_configuration)


