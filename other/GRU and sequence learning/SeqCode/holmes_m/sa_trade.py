#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:10:19 2019

@author: Zhzhewei Zhang
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
os.chdir('/home/tyang/Documents/SeqLearning_code/holmes_m')
from bhv_lesion import main
from bhv_check import main as main2 



cr = np.zeros((11,2))
cr_sem = np.zeros((11,2))
rt = np.zeros((11,2))
rt_sem = np.zeros((11,2))
cr_log = np.zeros((11,2))
cr_log_sem = np.zeros((11,2))


for step, i in enumerate(['0.5','0.4','0.3','0.2','0.1']):
    print('label:  '+i)
    files = glob.glob('/home/tyang/Documents/SeqLearning_code/log_m/RT/lesion/when*'+i+'*.hdf5')# files in which when units are inactivated
    files.extend(glob.glob('/home/tyang/Documents/SeqLearning_code/log_m/RT/lesion/output*'+i+'*.hdf5'))# control groups
    
    df = main(file_paths = files)
    cr[step, 0] = df.loc[df.label == 'output_les_con'].cr.values[0]
    cr_sem[step, 0] = df.loc[df.label == 'output_les_con'].cr_sem.values[0]
    cr_log[step, 0] = df.loc[df.label == 'output_les_con'].cr_log.values[0]
    cr_sem[step, 0] = df.loc[df.label == 'output_les_con'].cr_log_sem.values[0]
    rt[step, 0] = df.loc[df.label == 'output_les_con'].rt[0].mean() 
    rt_sem[step, 0] = df.loc[df.label == 'output_les_con'].rt[0].std()
    
    cr[10-step, 0] = df.loc[df.label == 'output_les_con'].cr.values[0]
    cr_sem[10-step, 0] = df.loc[df.label == 'output_les_con'].cr_sem.values[0]
    cr_log[10-step, 0] = df.loc[df.label == 'output_les_con'].cr_log.values[0]
    cr_log_sem[10-step, 0] = df.loc[df.label == 'output_les_con'].cr_log_sem.values[0]
    rt[10-step, 0] = df.loc[df.label == 'output_les_con'].rt[0].mean() 
    rt_sem[10-step, 0] = df.loc[df.label == 'output_les_con'].rt[0].std()
    
#    for i, label in enumerate(['when_neg_output_les', 'when_pos_output_les'])
    cr[step, 1] = df.loc[df.label == 'when_neg_output_les'].cr.values[0]
    cr_sem[step, 1] = df.loc[df.label == 'when_neg_output_les'].cr_sem.values[0]
    cr_log[step, 1] = df.loc[df.label == 'when_neg_output_les'].cr_log.values[0]
    cr_log_sem[step, 1] = df.loc[df.label == 'when_neg_output_les'].cr_log_sem.values[0]
    rt[step, 1] = df.loc[df.label == 'when_neg_output_les'].rt[1].mean() 
    rt_sem[step, 1] = df.loc[df.label == 'when_neg_output_les'].rt[1].std()
    
    cr[10-step, 1] = df.loc[df.label == 'when_pos_output_les'].cr.values[0]
    cr_sem[10-step, 1] = df.loc[df.label == 'when_pos_output_les'].cr_sem.values[0]
    cr_log[10-step, 1] = df.loc[df.label == 'when_pos_output_les'].cr_log.values[0]
    cr_log_sem[10-step, 1] = df.loc[df.label == 'when_pos_output_les'].cr_log_sem.values[0]
    rt[10-step, 1] = df.loc[df.label == 'when_pos_output_les'].rt[2].mean() 
    rt_sem[10-step, 1] = df.loc[df.label == 'when_pos_output_les'].rt[2].std()


files = glob.glob('/home/tyang/Documents/SeqLearning_code/log_m/RT/raw/*.hdf5') # without any lesion
df,_,_ = main2(file_paths = files)
cr[5, :] = df.cr.mean()
cr_sem[5, :] = df.cr.sem()
cr_log[5, :] = df.cr_log.mean()
cr_log_sem[5, :] = df.cr_log.sem()
rt[5, :] = df.rt_mean.mean() 
rt_sem[5, :] = df.rt_mean.sem()

fig0, ax1 = plt.subplots()
ax2 = ax1.twinx()
plt.ylim([0.7,1.25])
ax1.errorbar(range(11), cr[:,1], cr_sem[:,1], label = 'cr')
ax1.errorbar(range(11), cr_log[:,1], cr_log_sem[:,1], label = 'cr_log')
ax1.plot(3, 1.02)
ax1.legend()

ax2.errorbar(range(11), rt[:,1], rt_sem[:,1],color = 'k', label = 'rt')
plt.ylim([2,10.5])
ax2.legend()

fig0.savefig('../figs/speed_accuracy_all.eps', format='eps', dpi=1000)



fig = plt.figure()
plt.errorbar(range(11), cr[:,0], cr_sem[:,0], label = 'control')
plt.errorbar(range(11), cr[:,1], cr_sem[:,1], label = 'lesion')
plt.legend()
plt.ylabel('correct rate')
plt.xticks(range(11),(np.linspace(-0.5,0.5,11).round(2)))
fig.savefig('../figs/speed_accuracy_cr.eps', format='eps', dpi=1000)

plt.show()

    
fig2 = plt.figure()
plt.errorbar(range(11), rt[:,0], rt_sem[:,0], label = 'control')
plt.errorbar(range(11), rt[:,1], rt_sem[:,1], label = 'lesion')
plt.legend()
plt.ylabel('reaction time')
plt.xticks(range(11),(np.linspace(-0.5,0.5,11).round(2)))
fig2.savefig('../figs/speed_accuracy_rt.eps', format='eps', dpi=1000)
plt.show()

