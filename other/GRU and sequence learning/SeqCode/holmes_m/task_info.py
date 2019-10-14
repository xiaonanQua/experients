# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 09:09:33 2018

@author: YangLab_ZZW
"""
"""
basic info of the input
"""
    
def rtshapebrief_config():
    fp = 0
    input_size = 20
    target = [1,2]
    shapes = [3,4,5,6,7,8,9,10,11,12]
    novisual = 13
    choice = [14,15,16,17]
    reward = [18,19]
    input_setting = {'fp':fp,
                     'target':target,
                     'shapes':shapes, 
                     'novisual':novisual,
                     'choice':choice,
                     'reward':reward,
                     'input_size':input_size
                     }
    return input_setting
