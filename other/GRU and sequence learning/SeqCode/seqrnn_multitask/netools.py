# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 20:35:50 2018

@author: YangLab_ZZW
"""

import numpy as np
import yaml

def wh_loading(saving_name):
    f = open(saving_name)
#    yaml_load = yaml.load
    yaml_load = lambda x: yaml.load(x, Loader=yaml.Loader)
    y = yaml_load(f)
    f.close()
    return y

def cal_dPLI_PLI(theta):
    """
    #  code for calculating dPLI and PLI
    # theta must has dimension TxN, where T is the length of time points and N is the number of nodes
    # outputs PLI matrix containing PLIs between all pairs of channels, and dPLI matrix containg dPLIs between all pairs of channels
    """
    N = theta.shape[1]
    T = theta.shape[0]
    dPLI = np.zeros((N,N))
    #     _,PDiff_mat = phase_diff_mat(theta)
    for idx1 in range(N):
        for idx2 in range(idx1+1,N):
            PDiff = theta[:,idx1]-theta[:,idx2]
            PDiff_exp_angle = np.angle(np.exp(1j*PDiff))
            dPLI[idx1,idx2] = np.sum(np.sign(PDiff_exp_angle))/T
            dPLI[idx2,idx1] = -dPLI[idx1,idx2]
    PLI = np.abs(dPLI)
    return PLI,dPLI

def cal_mat_thresholded(data_mat,threshold):
    prob = 1 - threshold
    data_mat_nneg = np.absolute(data_mat)
    data_value_sorted = np.sort(data_mat_nneg,axis=None)
    data_value_threshold = data_value_sorted[int(np.round(len(data_value_sorted)*prob))]
    data_mat_thresholded = data_mat*(np.abs(data_mat)>=data_value_threshold)
    data_mat_binary = np.sign(data_mat_thresholded)
    data_mat_binary = np.abs(np.sign(data_mat_thresholded))
    return data_mat_binary



class neurons_inNetwork:
    def __init__(self, value, threshold, *con_mat):
        self.threshold = threshold
        self.value = value
        self.log = ''
        self.pos = []
        self.neg = []
        self.neuron_constructer()
        self.num_mat = int(len(con_mat)/2)
        self.mat = []
        self.mat_name = []
        for n_mat in range(self.num_mat):
            self.mat_name.append(con_mat[int(n_mat*2)])
            self.mat.append(con_mat[int(n_mat*2+1)])
        self.neuron_constructer()
            
    def __getitem__(self,key):
        return self.dict[key]
    
    def __setitem__(self,key,value):
        self.dict[key] = value
        
    def neuron_constructer(self):
        index = np.argsort(self.value)[::-1]
            
        pos_value = self.value[np.where(self.value > 0)]
        pos_value_cumsum = np.sort(pos_value)[::-1].cumsum()
        necePosProportion = self.threshold*pos_value.sum()
        numPosSelected = np.where(pos_value_cumsum>=necePosProportion)[0][0]+1
        self.pos = index[:numPosSelected]
            
        neg_value_abs = np.abs(self.value[np.where(self.value < 0)])
        neg_value_cumsum = np.sort(neg_value_abs)[::-1].cumsum()
        neceNegProportion = self.threshold*neg_value_abs.sum()
        numNegSelected = np.where(neg_value_cumsum>=neceNegProportion)[0][0]+1
        self.neg = index[-numNegSelected:]
            

    
