#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:05:37 2019

@author: Zhewei Zhang 

configuration file for multisensory integration task
    
"""
import copy
import os

MultInt_TestingSet = {
        "name": "mult_int",
        "model_mode":'GRU',
        "optimizer":'Adam',        
        "data_file": "../data/",
        "validation_data_file": "../data/",
        "data_attr": "data_MI",
        "data_brief": "data_MI_Brief",
        "log_path": "../log_m/MI/",
        "model_path": "../save_m/MI/model-mult_int_",
        "load_model": "",
        "train_num":7.5e5,
        "input_size": 26,
        "batch_size": 128,
        "reset_hidden":True,
        "lr": 1e-3,
        "record": 1,
        "need_train":1,
        "need_validate": 1,
        "lesion": "",
        "anlys_file":""
    }

task_configurations = []
for i in range(20):
    task_configuration = copy.deepcopy(MultInt_TestingSet)
    # for each run
    task_configuration["data_file"] = "../data/"
    task_configuration["validation_data_file"] = "../data/"
    task_configurations.append(task_configuration)
    