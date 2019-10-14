#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:26:41 2019

@author: Zhewei Zhang

"""

from __future__ import division

import sys
import os
import time
import copy
import argparse
import importlib
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import data_util
from evaluator import SeqAgent, Calculator, interactive_sample, match_rate
from model import GRUNetwork, RNNNetwork, LSTMNetwork
from lesion import lesion_rnn

"""
Main script of training and validating RNN with LSTM/GRU.
"""
training_interval = 12800
t_interval = 1000
save_interval = 12800
clip_setting = 0.25

# Training
def train(model, calculator, trial_data, hidden, tmp_reward, clip=clip_setting, bsz = None):
    """
    trial_data -> state -> encode -> rnn-> output-> decode -> state
    trial_data.shape: (?,16)
    """
    model.zero_grad()  # reset the grad
    hidden_sequence, raw_prediction, predicted_trial, total_loss, hidden = calculator.get_output(model, trial_data, hidden, tmp_reward, bsz = bsz)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # gradient clip

    # apply gradient descent
    model.optimizer.step()
        
    correct_rate = match_rate(trial_data[1:,:,:], predicted_trial)
    return total_loss.item(), correct_rate, hidden

def model_save(model, save_path, model_name):
    if not os.path.isfile(save_path + model_name):
        torch.save(model.state_dict(), save_path + model_name)
    else:
        n = 0
        while 1: # save the model
            n += 1
            if not os.path.isfile(save_path + model_name.split('.')[0] + '-' + str(n) + '.pt'):
                torch.save(model.state_dict(), save_path + model_name.split('.')[0] + '-' + str(n) + '.pt')
                break
    print("_" * 36)
    print("model saved")

def repackage_hidden(h):
    """Wrap hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def validate_stage(trial, trial_settings, task_env, sqa):
    
    
    task_env.configure(trial, trial_settings)
    
    raw_rec, tmp_loss, correct_rate = interactive_sample(sqa, task_env)  #[3,4,5] [7,8,9] [13,14,15] [17,18,19]
    trial_abstract = task_env.extract_trial_abstractb64()

    wining = task_env.is_winning()
    completed = task_env.is_completed()
    task_env.reset_configuration()

    return wining, completed, tmp_loss, correct_rate, trial_abstract, raw_rec

def validate(dp, model, sqa, validation_data, task, record, truncate=1e800, to_console=True,
             cuda_enabled=False, suffix = ''):
    
    wining_counts, completed_counts, trial_counts = 0, 0, 0
    validation_helper = task.ValidationHelper(validation_data["validation_set"], validation_data["validation_conditions"])
    validation_set = validation_helper.validation_set
    validation_conditions = validation_helper.validation_conditions

    global_low = 0
    record_saved = 0
    average_loss_v, average_correct_rate_v = 0, 0
    try:
        neuron_shape = list(map(lambda x: int(x), list(model.init_hidden(bsz = 1).data.shape))) # bsz is equal to 1 for validation, for GRU
#        neuron_shape = list(map(lambda x: int(x), list(model.init_hidden(bsz = 1)[0].data.shape))) # for LSTM
        behavior_shape = list(validation_set[0].shape)
        behavior_shape.pop(0)
        if record and (not record_saved):
            print("creating raw records...")
            dp.create_hdf5raw(behavior_shape, neuron_shape, suffix = suffix)
        
        task_env = task.Task()
        for step, trial in enumerate(validation_set):
            if step >= truncate:
                break
            if reset_hidden:
                sqa.init_hidden()

            trial_settings = validation_conditions[step]
            wining, completed, tmp_loss, correct_rate, trial_abstract, raw_rec = validate_stage(trial, trial_settings, task_env, sqa)
            
            trial_counts = trial_counts + 1
            wining_counts = wining_counts + wining
            completed_counts = completed_counts + completed

            task_env.reset_configuration()

            average_loss_v = average_loss_v + tmp_loss
            average_correct_rate_v = average_correct_rate_v + correct_rate

            if record:
                # Notice: all these arrays is np.ndarray
                behavior_data = raw_rec["sensory_sequence"]
                prediction_data = raw_rec["predicted_trial"]
                raw_output = raw_rec["raw_records"]
                neuron_data = raw_rec["hidden_records"]
                tmp_high = global_low + behavior_data.shape[0]

                index_data = np.array([global_low, tmp_high]).reshape(1, 1, 2)
                dp.append_hdf5raw(behavior_data, prediction_data, raw_output,
                                  neuron_data, index_data)

                # save experiment result
                global_low = tmp_high
                dp.write_log(step, correct_rate, tmp_loss, training=0)
                dp.write_validation_brief(step, trial_abstract)
            
            if (step + 1) % t_interval == 0 and step > 0 and to_console:
                print("(Validate)STEP: {:6d} | AVERAGE CORRECT RATE: {:6f} | AVERAGE LOSS(without loss): {:.6f} ".format(
                    step + 1,
                    average_correct_rate_v / t_interval,
                    average_loss_v / t_interval
                ))
                average_loss_v = 0
                average_correct_rate_v = 0
    except KeyboardInterrupt:
        print("KeyboardInterrupt Detected!\n")
        if record and (not record_saved):
            print("Saving validating set...")
            record_saved = dp.close_hdf5raw()

    if record and (not record_saved):
        record_saved = dp.close_hdf5raw()

    wining_rate = wining_counts / trial_counts
    completed_rate = completed_counts / trial_counts
    return wining_rate, completed_rate


def suffle_repeat(training_set,training_conditions,train_truncate):
    '''
    repeat trials if training_set is smaller than what config asked
    suffle the trials 
    
    Specific for the shape task, 1000 trials version
    '''
#    repeats = np.random.choice(len(training_set),int(train_truncate)).astype(np.int)
#    training_set_copy = []
#    for trial in repeats:
#        training_set_copy.append(training_set[trial])
#    training_set = copy.deepcopy(training_set_copy)
#    for i in range(8):
#        if i ==3:
#            training_conditions[i] = training_conditions[i][:,repeats]
#        elif i ==5:
#            training_conditions[i] = training_conditions[i]
#        else:
#            training_conditions[i] = training_conditions[i][repeats,:]
#            
    return training_set,training_conditions

def train_stage(dp, model, sqa, save_path, training_data, validation_data, log_path, task,
                record=1, train_truncate=1e800, batch_size=1, clip=clip_setting, cuda_enabled=False):
    try:
        calculator = Calculator(batch_size = batch_size, cuda_enabled=cuda_enabled)
        training_set, training_conditions = suffle_repeat(training_data["training_set"], 
                                                          training_data["training_conditions"],
                                                          train_truncate)
        
        training_helper = task.TrainingHelper(training_set, training_conditions)
        train_set = training_helper.training_set
        training_guide = training_helper.training_guide
        average_loss, average_correct_rate = 0, 0

        trials, tmp_reward = [], []
        tmp_loss = 0
        try:
            hidden = model.init_hidden(batch_size)
            for step, trial in enumerate(train_set):
                if step >= train_truncate-1:
                    model_name = str(step+1) + '-' + dp.get_logname() + ".pt"
                    model_save(model, save_path, model_name)
                    break
                # prepare the data for each batch
                trials.append(trial)
                tmp_reward.append(training_guide[step])
                
                if (step+1) % batch_size == 0:
                    if reset_hidden:
                        hidden = model.init_hidden(batch_size)
                    trials = np.array(trials).transpose([1,0,2]) # trials: time step by batch size by input number
                    tmp_reward = np.array(tmp_reward)
                    tmp_loss, correct_rate, hidden = train(model, calculator, trials, hidden, tmp_reward, clip=clip)
                    trials, tmp_reward = [], []
                else:
                    continue

                hidden = repackage_hidden(hidden) # necessary for the task keep the hidden unreset
                average_loss = average_loss + tmp_loss
                average_correct_rate = average_correct_rate + correct_rate

                if record:
                    dp.write_log(step, correct_rate, tmp_loss, training=1)
                if (step + 1) % training_interval == 0 and step > 0:
                    print("STEP: {:6d} | AVERAGE CORRECT RATE: {:6f} | AVERAGE LOSS: {:.6f} ".format(
                            step + 1,
                            average_correct_rate / training_interval * batch_size,
                            average_loss / training_interval
                            ))

                    average_loss, average_correct_rate = 0, 0
                    
                    set_size = 100
                    wining_rate, completed_rate  = validate(dp, model, sqa, validation_data, task, record=0, truncate=set_size, to_console=False)
                    print("(Test)WINING RATE: {:6f} | COMPLETED RATE: {:6f} ON {} SAMPLES".format(wining_rate, completed_rate, set_size))

                if (step+1) % save_interval == 0 and step > 0 and record:
                    model_name = str(step+1) + '-' + dp.get_logname() + ".pt"
                    model_save(model, save_path, model_name)

        except KeyboardInterrupt:
            if record:
                model_save(model, save_path, model_name)
            print("stop training, jump to validating...")
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt Detected!\n Now exit the script...")
        sys.exit(0)

def main(task_configurations):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", help="if cuda acceleration is enabled",
                        default=1,
                        type=bool)
    args = parser.parse_args()

    cuda_enabled = args.cuda

    model_parameters = {
        "nhid": 128,
        "nlayers": 1,
        "input_size": task_configurations[0]["input_size"],
        "batch_size": task_configurations[0]["batch_size"],
        "clip": clip_setting,
        "lr": task_configurations[0]["lr"]
    }
    print("Model Parameters: {}".format(model_parameters))
    
    for nth, task_setting in enumerate(task_configurations):
        model_parameters["lr"] = task_setting["lr"]
        if task_setting['model_mode'] == 'GRU':
            rnn_model = GRUNetwork(model_parameters["input_size"],
                               model_parameters["nhid"],
                               model_parameters["batch_size"],
                               model_parameters["nlayers"],
                               model_parameters["lr"],
                               cuda_enabled=cuda_enabled,
                               )
        elif task_setting['model_mode'] == 'LSTM':
            print('LSTM')
            rnn_model = LSTMNetwork(model_parameters["input_size"],
                               model_parameters["nhid"],
                               model_parameters["batch_size"],
                               model_parameters["nlayers"],
                               model_parameters["lr"],
                               cuda_enabled=cuda_enabled,
                               )
        else:
            raise Exception('unknown model mode')
        
        if len(task_setting["load_model"]) > 0:
            rnn_model.load_state_dict(torch.load(task_setting["load_model"]))

        print("{}th repeat:".format(nth+1))
        print(rnn_model)
        
        rnn_model_lesion = lesion_rnn(rnn_model,task_setting['lesion'],task_setting['anlys_file'])
        
        for label, model in rnn_model_lesion.items():
            # global settings
            log_path = task_setting["log_path"] # "../log_m/"
            
            data_name = time.strftime("%Y%m%d_%H%M", time.localtime())
            if label: 
                data_name = label + '-'+ data_name
            print(data_name)
            dp = data_util.DataProcessor(log_name = data_name)
            training_data_path = task_setting["data_file"]
            validation_data_path = task_setting["validation_data_file"]
            
            data_attr = task_setting["data_attr"]
            data_brief_attr = task_setting["data_brief"]
            
            training_set, training_conditions = dp.load_data_v2(
                training_data_path, data_attr, data_brief_attr)
                
            validation_set, validation_conditions = dp.load_data_v2(
                validation_data_path, data_attr, data_brief_attr)
                
            training_data = {
                "training_set": training_set,
                "training_conditions": training_conditions
            }
            
            validation_data = {
                "validation_set": validation_set,
                "validation_conditions": validation_conditions
            }

            if task_setting["record"]:
                dp.create_log(log_path, task_setting["name"])  # create log first
                dp.save_condition(model_parameters, task_setting, training_data_path, validation_data_path)
            
            task = importlib.import_module('tasks.' + task_setting["name"])
            sqa = SeqAgent(model, batch_size = 1, cuda=cuda_enabled) # test the trial with the batch_size being 1
            
            if task_setting["need_train"] > 0:
                print('-' * 89)
                print("START TRAINING: {}".format(task_setting["name"]))
                print('-' * 89)
                for i in range(task_setting["need_train"]):
                    train_stage(
                                dp, model, sqa, task_setting["model_path"], training_data, validation_data, log_path, 
                                task, record=task_setting["record"], train_truncate=task_setting["train_num"], 
                                batch_size=model_parameters["batch_size"], clip=model_parameters["clip"], cuda_enabled=cuda_enabled
                                )
                    
            if task_setting["need_validate"] > 0:
                print('-' * 89)
                print("START VALIDATION: {}".format(task_setting["name"]))
                print('-' * 89)
                wining_rate, completed_rate = validate(dp, model, sqa, validation_data, task, task_setting["record"], 
                                                       truncate=5e3)
                print("wining_rate :{}|completed_rate :{}".format(wining_rate, completed_rate))
                
if __name__ == '__main__':
    os.chdir('/home/tyang/Desktop/submit/seqrnn_multitask')
    config = importlib.import_module('config')
    
#    config = importlib.import_module('config_sure')
#    config = importlib.import_module('config_mi')
#    config = importlib.import_module('config_st')
#
    task_configurations = config.task_configurations
#    task_configurations = config.task_configurations_lesion
    
    reset_hidden = task_configurations[0]['reset_hidden']
    main(task_configurations)
