#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:40:06 2019

@author: tyang
"""



import os
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from bhv_check import bhv_extract


def groups_files(file_paths):
    
    # group the files based on the lesion type
    group = []
    for nfile in file_paths:
        path, file = os.path.split(nfile)
        group.append(file.split('-')[0])
        
    files_pd = pd.DataFrame([list(file_paths),group],['name','lesion'])
    files_pd = files_pd.T
    files_groups = files_pd.name.groupby([files_pd.lesion])
    ncondition = files_pd.lesion.nunique()
    condition = files_pd.lesion.unique()

    return files_groups, condition, ncondition

def main(file_paths = None):
    print("start")
    print("select the files")
    root = tk.Tk()
    root.withdraw()
    if file_paths == None:
        file_paths = filedialog.askopenfilenames(parent = root,
                                                title = 'Choose a file',
                                                filetypes = [("HDF5 files", "*-0.5-*201908*.hdf5")]
                                                )
    files_groups, condition, ncondition = groups_files(file_paths)
    
    df = pd.DataFrame([], columns = {'label','rt','rt_sem','cr','cr_sem','fr','fr_sem','cr_log','cr_log_sem','choice_prop','choice_prop_sem'})
    
    for i, files_group in enumerate(files_groups):
        df_basic, _, _ = bhv_extract(files_group[1])
#        print('#'*48)
#        print(files_group[0])
        print('fr:{:6f} || cr:{:6f} || cr-log:{:6f} || choice_prop:{:6f} || rt:{:6f} '.format(df_basic.fr.mean(),
                                                                                       df_basic.cr.mean(), 
                                                                                       df_basic.cr_log.mean(), 
                                                                                       df_basic.choice_prop.mean(),
                                                                                       df_basic.rt_mean.mean()))
        print('fr         :{:6s} || cr         :{:6s} || cr-log     :{:6s} || choice_prop:{:6s} || rt         :{:6s} '.format(str(df_basic.fr.values),
                                                                                       str(df_basic.cr.values), 
                                                                                       str(df_basic.cr_log.values), 
                                                                                       str(df_basic.choice_prop.values),
                                                                                       str(df_basic.rt_mean.values)))

        df.loc[i] = {'label':           files_group[0],
                    'rt':               df_basic.rt_mean,
                    'rt_sem':           np.round(df_basic.rt_mean.sem(),3),
                    'cr':               np.round(df_basic.cr.mean(),3),
                    'cr_sem':           np.round(df_basic.cr.sem(),3),
                    'fr':               np.round(df_basic.fr.mean(),3),
                    'fr_sem':           np.round(df_basic.fr.sem(),3),
                    'cr_log':           np.round(df_basic.cr_log.mean(),3),
                    'cr_log_sem':       np.round(df_basic.cr_log.sem(),3),
                    'choice_prop':      np.round(df_basic.choice_prop.mean(),3),
                    'choice_prop_sem':  np.round(df_basic.choice_prop.sem(),3)                   
                    }
        
#    plt.errorbar()
    fig = plt.figure()
#    plt.errorbar(range(ncondition), df.rt.values, yerr = df.rt_sem.values,fmt='o')
    plt.boxplot(np.vstack(df.rt.values).T)

    plt.xticks(np.arange(ncondition)+1,df.label.values)
    plt.ylabel('reaction time')
    fig.savefig('../figs/lesion_effect_rt2.eps', format='eps', dpi=1000)
    
    fig2 = plt.figure()
    plt.bar(np.arange(ncondition)-0.2, df.cr.values, yerr = df.cr_sem.values, width=0.15,label = 'cr')
    plt.bar(np.arange(ncondition), df.cr_log.values, yerr = df.cr_log_sem.values, width=0.15,label = 'cr_log')
    plt.bar(np.arange(ncondition)+0.2, df.choice_prop.values, yerr = df.choice_prop_sem.values, width=0.15,label = 'choice_prop')
#    plt.plot(np.arange(5)+0.15, df.fr,'o',label = 'fr')
#    plt.xticks(range(ncondition),('control','when neg','when pos','which neg','which pos'))
    plt.ylabel('proportion(%)')
    plt.legend()
    fig2.savefig('../figs/lesion_effect_cr.eps', format='eps', dpi=1000)
    plt.show()
    return df
    
if __name__ == '__main__':
    main()










