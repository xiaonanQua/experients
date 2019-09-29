# -*- coding: utf-8 -*-
# @Author: Zhewei Zhang
# @Date: 09/01/2018, 23:39

"""
preditive coding about the visual information analysis
"""

import numpy as np
import pandas as pd
import scipy.stats
import tkinter as tk
import scipy.io as sio
from tkinter import filedialog
from scipy import stats
from task_info import rtshapebrief_config
from toolkits_2 import get_bhvinfo,load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


"""
using the subthreshold activity of the shape output units to predict the 
next presenting shape

fig5
"""


WEIGHTS = (-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9)
prob_R = np.array((0.0056, 0.0166, 0.0480, 0.0835, 0.1771, 0.2229, 0.1665, 0.1520, 0.0834, 0.0444))
prob_G = np.flipud(prob_R)
time_len = 10

input_setting = rtshapebrief_config()
shapes_pos = input_setting['shapes']
reward_pos = input_setting['reward']
choice_pos = input_setting['choice']

def shape_pred(path_files):
    df_pred = pd.DataFrame([], columns = {'label','predict_dist_neuro','predict_dist_neuro_group',
                           'weight_exceptlastone','kl_div'})

    for i, file in enumerate(path_files):
        ## load files
        paod, trial_briefs = load(file)
        trial, _, shape, _, _ = get_bhvinfo(paod,trial_briefs)
        #======================== shape prediction ========================
        predict_dist_neuro, weight_exceptlastone = [], []
        for i, rt in enumerate(shape.rt.values):
            _, _, rd, _= paod.get_neuron_behavior_pair(index = trial.num.iloc[i])
            for j in range(rt):
                #  the shapes in all the epoches are included
                weight_exceptlastone.append(np.sum(shape.tempweight.iloc[i][:j]))
                predict_dist_neuro.append(rd[2 + j * 5, shapes_pos])            
                
        weight_exceptlastone = np.asarray(weight_exceptlastone)
        predict_dist_neuro = np.asarray(predict_dist_neuro)
        
        sega = np.linspace(weight_exceptlastone.min(),weight_exceptlastone.max(),num=11, endpoint=True)        
        kl_div = []
        predict_dist_neuro_group = []
        for n_group in range(len(sega)-1):
            curr_index = (weight_exceptlastone>=sega[n_group]) & (weight_exceptlastone<sega[n_group+1])
            if n_group == len(sega)-1-1:
                curr_index = (weight_exceptlastone>=sega[n_group]) & (weight_exceptlastone<=sega[n_group+1])
            curr_pred = np.mean(predict_dist_neuro[curr_index], axis=0)
            curr_pred = curr_pred/np.sum(curr_pred)
            predict_dist_neuro_group.append(curr_pred)
            kl_div.append([scipy.stats.entropy(curr_pred,qk = prob_R),
                                 scipy.stats.entropy(curr_pred,qk = prob_G)])
#            print(curr_pred.round(3), prob_R.round(3), prob_G.round(3))
#            print(kl_div[-1])
        df_pred.loc[i] = {'label': file,
                          'predict_dist_neuro':   predict_dist_neuro,
                          'predict_dist_neuro_group': np.array(predict_dist_neuro_group),
                          'weight_exceptlastone': weight_exceptlastone,
                          'kl_div':    np.array(kl_div)
                          }
    return df_pred



            

def plot_shapepred(df_pred):
    kl_div = np.hstack(df_pred.kl_div.values)

    fig = plt.figure()
    plt.errorbar(range(kl_div.shape[0]), np.nanmean(kl_div[:,0::2], axis = 1),yerr = stats.sem(kl_div[:,0::2], axis = 1), label = 'red')
    plt.errorbar(range(kl_div.shape[0]), np.nanmean(kl_div[:,1::2], axis = 1),yerr = stats.sem(kl_div[:,1::2], axis = 1), label = 'green')
    plt.title('KL divergence')
    plt.legend()
    plt.xticks(np.linspace(0,kl_div.shape[0]-1,10),('0~10%','10~20%','20~30%','30~40%','40~50%',
               '50~60%','60~70%','70~80%','80~90%','90~100%'))
    plt.xlabel('sorted by the summed weight of first n-1 shapes')
    plt.ylabel('weight on left distribution')
    fig.savefig('../figs/kl divergence.eps', format='eps', dpi=1000)
    
    
    weight_exceptlastone = np.hstack(df_pred.weight_exceptlastone.values)
    predist_neuro = np.vstack(df_pred.predict_dist_neuro.values)
    sort_index = np.argsort(weight_exceptlastone)
    
    fig2 = plt.figure()
    ax = fig2.gca(projection='3d') #    ax = fig2.add_subplot(2, 1, 2, projection='3d')
    predist_neuro = predist_neuro[sort_index,:]
    X = np.linspace(0, 1, endpoint=True, num = predist_neuro.shape[1])
    Y = np.arange(1, predist_neuro.shape[0]+1, 1)
    X, Y = np.meshgrid(X, Y)
    Z = predist_neuro
    sio.savemat('predictive coding.mat',{'X':X,'Y':Y,'Z':Z})# for matlab plotting
    X = np.hstack((X, (2*X[:,-1] - X[:,-2]).reshape(-1,1)))
    Y = np.hstack((Y, Y[:,-1].reshape(-1,1)))
    Z = np.hstack((Z, Z[:,-1].reshape(-1,1)))

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=True)
    ax.view_init(azim=1, elev=270)
    fig2.colorbar(surf, shrink=0.5, aspect=5)# Add a color bar which maps values to colors.
    plt.xlabel('shape weight')
    plt.ylabel('predicted distribution')
    plt.title('neuronal response')
    plt.xticks(np.linspace(0,1,num=10,endpoint = True),np.arange(-0.9,1,0.2).round(2))
    fig2.savefig('../figs/shape prediction.eps', format='eps', dpi=1000)
    plt.show()


    predict_dist_neuro_group = np.dstack(df_pred.predict_dist_neuro_group.values)
    sio.savemat('predictive coding2.mat',{'resp':predict_dist_neuro_group})# for matlab plotting
    fig_w, fig_h = (6, 5)
    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
    fig3 = plt.figure()
    colormap = np.vstack([np.linspace(1,0,10),np.linspace(0,1,10),np.zeros(10,)])
    ax = fig3.gca(projection='3d')
    x_, y_, z_ = [], [], []
    for i, resp in enumerate(predict_dist_neuro_group):
        x = np.arange(0, 10, 1)
        y = np.zeros((10,)) + i
        z = resp.mean(axis=1)
        x_.append(x)
        y_.append(y)
        z_.append(z)
        zerr = stats.sem(resp,axis=1)
#        for ii, zerr_curr in enumerate(zerr):
#            ax.plot([x[ii],x[ii],x[ii]], [y[ii],y[ii],y[ii]], [z[ii]-zerr_curr,z[ii],z[ii]+zerr_curr], color = colormap[:,i])
#        ax.plot(x, y, z, color = colormap[:,i])
        plt.plot(x+i,z)
        
    colors = np.zeros((10, 10, 3))
    for ind_x in range(len(x)):
        for ind_y in range(len(y)):
            colors[ind_x, ind_y] = colormap[:,ind_x]
    x_, y_, z_ = np.array(x_), np.array(y_), np.array(z_)
    surf = ax.plot_surface(x_, y_, z_, facecolors=colors,alpha=0.2)

    ax.view_init(azim=300, elev=15)
    fig3.savefig('../figs/shape prediction-group.eps', format='eps', dpi=1000)
    plt.show()

def main():
    print("start")
    print("select the files")
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(parent = root,
                                            title = 'Choose a file',
                                            filetypes = [("HDF5 files", "*.hdf5")]
                                            )
    ##
    fig_w, fig_h = (10, 4)
    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
    # =========================================================================
    df_pred = shape_pred(file_paths)
    plot_shapepred(df_pred)



if __name__ == '__main__':
    main()

