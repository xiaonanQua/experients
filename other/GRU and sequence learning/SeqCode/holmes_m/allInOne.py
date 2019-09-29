#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:46:05 2019

@author: tyang

all analysis in one except lesion studies 
"""

import tkinter as tk
from tkinter import filedialog

from bhv_check import bhv_extract, plot_bhvBasic
from pred_shape import shape_pred, plot_shapepred
from targetDimReducation import TDR, plot_TDR
from subweight import shape_extract, plot_subweight
from varce import var_ce, plot_varce


print("start")
print("select the files")
root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(parent = root,
                                        title = 'Choose a file',
                                        filetypes = [("HDF5 files", "*.hdf5")]
                                        )

# basic behavioral analysis
df_basic, df_logRT, df_psycv = bhv_extract(file_paths)
plot_bhvBasic(df_logRT, df_psycv)

# subjective value
df_epoch, df_subweight = shape_extract(file_paths)
plot_subweight(df_epoch, df_subweight)


# shape prediction
df_pred = shape_pred(file_paths)
plot_shapepred(df_pred)

# target dimension reduction
df_tdr = TDR(file_paths)
plot_TDR(df_tdr)

# var ce
df_distance = var_ce(file_paths)
plot_varce(df_distance)

