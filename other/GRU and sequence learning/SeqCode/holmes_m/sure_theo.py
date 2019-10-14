#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:33:11 2019

@author: Zhewei Zhang
"""
import pandas as pd
import scipy.io as sio
import tkinter as tk

from tkinter import filedialog
from Sure_ana import figure_plot

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilenames(
            parent=root,title='Choose the sure task training file',
            filetypes=[("sure task training file", "SureTask_Training*.mat")]
            )

df_detail = []
for file in file_path:
    data = sio.loadmat(file)
    files_pd = pd.DataFrame([data['data_Sure_Brief']["choices"][0,0][0].tolist(),
                            data['data_Sure_Brief']["reward"][0,0][0].tolist(),
                            data['data_Sure_Brief']["randots_dur"][0,0][0].tolist(),
                            data['data_Sure_Brief']["sure_trials"][0,0][0].tolist(),
                            data['data_Sure_Brief']["coherences"][0,0][0].tolist()],
                            ['choice','reward','randots_dur','sure_trial','coherence']
                            )
    files_pd = files_pd.T
    df_detail.append(files_pd)

figure_plot(df_detail, savepath = '/home/tyang/Documents/SeqLearning_code/figs/sure')
