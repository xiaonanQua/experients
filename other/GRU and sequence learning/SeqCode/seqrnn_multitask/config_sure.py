#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:12:59 2019

@author: Zhewei Zhang

cofiguration file for the sure task adapted from Kiani 2009

"""
import copy
import os

sure = {
        "name": "sure_task",
        "model_mode":'GRU',
        "optimizer":'Adam',
        "data_file": "../data/",
        "validation_data_file": "../data/",
        "data_attr": "data_Sure",
        "data_brief": "data_Sure_Brief",
        "log_path": "../log_m/Sure/",
        "model_path": "../save_m/Sure/model-sure_",
        "load_model": "",
        "train_num": 75e4,
        "input_size": 22,
        "batch_size": 128,
        "reset_hidden":False,
        "lr": 1e-3,
        "record": 1,
        "need_train":1,
        "need_validate": 1,
        "lesion": "",
        "anlys_file":""
    }

task_configurations = []
for i in range(20):
    task_configuration = copy.deepcopy(sure)
    # for each run
    task_configuration["data_file"] = "../data/"
    task_configuration["validation_data_file"] = "../data/"
    task_configurations.append(task_configuration)