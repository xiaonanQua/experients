#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:07:45 2019

@author: Zhewei Zhang

configuration file for simplfied two step task

"""
import copy
import os

rv_lr = {
        "name": "smp_ts",
        "model_mode":'GRU',
        "optimizer":'Adam',        
        "data_file": "../data/",
        "validation_data_file": "../data/",
        "data_attr": "data_ST",
        "data_brief": "data_ST_Brief",
        "log_path": "../log_m/ST/",
        "model_path": "../save_m/ST/model-sp_ts_",
        "load_model": "",
        "train_num":7.5e5,
        "input_size": 10,
        "batch_size": 1,
        "reset_hidden":True,
        "lr": 1e-4,
        "record": 1,
        "need_train":1,
        "need_validate": 1,
        "lesion": "",
        "anlys_file":""
    }

task_configurations = []
for i in range(20):
    task_configuration = copy.deepcopy(rv_lr)
    # for each run
    task_configuration["data_file"] = "../data/"
    task_configuration["validation_data_file"] =  "../data/"
    task_configurations.append(task_configuration)


