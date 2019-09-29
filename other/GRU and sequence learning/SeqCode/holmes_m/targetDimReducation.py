# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 19:35:01 2018

@author: YangLab_ZZW
"""
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from scipy import stats
from toolkits_2 import pca, load, get_hidden_resp_all, get_bhvinfo
from lr_step_TDR import regress_tdr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

"""
identify the choice direction, and show what affect the projection on the choice direction

fig3 c/d
"""

def denoisemat(all_resp_cell, threshold):
        ## pca
    pca_data = np.concatenate(all_resp_cell, axis=0).T
    eigvects, eigvals, mean_value = pca(pca_data)
    explained_var = np.cumsum(eigvals/np.sum(eigvals))
    n_dimension = np.where(explained_var > threshold)[0][0]
    ## denoise matrix
    D = np.zeros((pca_data.shape[0],pca_data.shape[0]))
    for i in range(n_dimension):
        D += np.matmul(eigvects[:,i].reshape([-1,1]), eigvects[:,i].reshape([1,-1]))
    
    return D

def TDR(path_files, threshold = 0.8):
    df_tdr = pd.DataFrame([], columns = {'label','params','X_std','Y_std'})

    for i, file in enumerate(path_files):
        ## load files
        paod, trial_briefs = load(file)

        ## neuron response
        all_resp_cell = get_hidden_resp_all(paod, trial_briefs)
        trials, choices, shapes, rewards, finish_rate = get_bhvinfo(paod,trial_briefs)
        D = denoisemat(all_resp_cell, threshold)
        #  linear regression 
        result_all = regress_tdr(all_resp_cell, trials, choices, shapes, rewards)
        
        params = np.zeros([128,5])
        for i in range(128): 
            for j in range(5):
                params[i,j] = result_all['params'][i][j][1]
                
        beta_pca = np.matmul(D, params)
        beta_norm = np.linalg.norm(beta_pca,axis=0)
        largest_t = np.argmax(beta_norm)
        direction = beta_pca[:,largest_t]  # the direction capture the largest variance vaused by the choice    
        # test whether the summed weight/urgency affect the projections in the choice direction
        summed_weight, nth_shape, dist = [], [], []
        for i in range(trials.count().length):
            resp =  all_resp_cell[i]
            for j in range(shapes.rt.iloc[i]):
                nth_shape.append(j+1)
                summed_weight.append(shapes.tempweight.iloc[i][:j+1].sum())
                dist.append(np.matmul(resp[j*5+1,:],direction)) #TODO: pay attention to this time step
        nth_shape = np.asarray(nth_shape)
        dist = np.asarray(dist)
        summed_weight = np.asarray(summed_weight)
        
        X = np.asarray([summed_weight, nth_shape])
        X = np.vstack((X,np.multiply(np.sign(X[0,:]),X[1,:]).reshape(1,-1))) # the sign urgency signal

        Y = dist
        
        lm = LinearRegression(fit_intercept=True)
        scaler = StandardScaler()
        Y_std = scaler.fit_transform(Y.reshape(-1,1))
        X_std = scaler.fit_transform(X.T)
        
        lm.fit(X_std, Y_std.reshape(-1,))
        df_tdr.loc[i] = {'label': file,
                         'params':np.append(lm.intercept_, lm.coef_),
                         'X_std' :X_std,
                         'Y_std': Y_std
                         }

#        Rsquare = lm.score(X_std, Y)
#
#        newX_std = np.append(np.ones((len(X_std), 1)), X_std, axis=1)
#        predictions = lm.predict(X_std)
#        SSresidual = ((Y - predictions) ** 2opy.deepcopy(y_grid).sum()
#        
#        MSE = SSresidual / (len(newX_std) - len(newX_std[0]))
#        var_b = MSE * (np.linalg.inv(np.dot(newX_std.T, newX_std)).diagonal())
#        sd_b = np.sqrt(var_b)
#        ts_b = params / sd_b
#        
#        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX_std) - 1))) for i in ts_b]
    return df_tdr
    
def plot_TDR(df_tdr):
    fig_w, fig_h = (10, 7)
    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})

    coef_tdr = np.vstack(df_tdr.params.values)
    p_values = []
    for i in range(coef_tdr.shape[1]):
        t, p = stats.ttest_1samp(coef_tdr[:,i], 0)
        p_values.append(p)
    fig = plt.figure()
#    plt.bar(range(4), coef_tdr.mean(axis = 0),label = p_values)
#    plt.errorbar(range(4), coef_tdr.mean(axis = 0), yerr = stats.sem(coef_tdr, axis=0), fmt = 'o')
#    plt.legend()
#    plt.xticks(range(4)+1, ('bias', 'logLR', 'shape num', 'signed shape num'))
    plt.boxplot(coef_tdr)
    plt.xticks(np.arange(4)+1, ('bias', 'logLR', 'shape num', 'signed shape num'))
    fig.savefig('../figs/target dimension reducetion-boxplot.eps', format='eps', dpi=1000)
    plt.show()
    
    X_std = np.vstack(df_tdr.X_std.values)
    Y_std = np.vstack(df_tdr.Y_std.values)
    
    num_grid = 20
    x1 = X_std[:,0]
    x2 = X_std[:,2]
    x1_min = x1.min()
    x1_max = x1.max()
    x2_min = x2.min()
    x2_max = x2.max()
    step_x1 = (x1_max-x1_min)/num_grid
    step_x2 = (x2_max-x2_min)/num_grid
    
    y_grid  = np.zeros((num_grid,num_grid))
    X1_plot = np.zeros((num_grid,num_grid))
    X2_plot = np.zeros((num_grid,num_grid))
    for i in range(num_grid):
        x1_lower = x1_min + i*step_x1;
        x1_upper = x1_min + (i+1)*step_x1;
        for j in range(num_grid):
            x2_lower = x2_min + j*step_x2;
            x2_upper = x2_min + (j+1)*step_x2;
            X1_plot[i,j] = (x1_lower+x1_upper)/2
            X2_plot[i,j] = (x2_lower+x2_upper)/2
            
            condition1 = np.intersect1d(np.where(x1>x1_lower)[0], np.where(x1<x1_upper)[0])
            condition2 = np.intersect1d(np.where(x2>x2_lower)[0], np.where(x2<x2_upper)[0])
            
            y_grid[i,j] = np.mean(Y_std[np.intersect1d(condition1, condition2)])
    
    fig2 = plt.figure()
    ax = fig2.gca(projection='3d') #    ax = fig2.add_subplot(2, 1, 2, projection='3d')
    surf = ax.plot_surface(X1_plot, X2_plot, y_grid, cmap=cm.coolwarm,linewidth=0, antialiased=True)
    fig2.colorbar(surf, shrink=0.5, aspect=5)# Add a color bar which maps values to colors.
    plt.xlabel('shape weight')
    plt.ylabel('signed urgency')
    fig2.savefig('../figs/target dimension reducetion-visualization.eps', format='eps', dpi=1000)
    plt.show()
    sio.savemat('target dimension reduction.mat',{'X':X_std,'Y':Y_std}) # for matlab plot

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
    df_tdr = TDR(file_paths)
    plot_TDR(df_tdr)

if __name__ == '__main__':
    main()
