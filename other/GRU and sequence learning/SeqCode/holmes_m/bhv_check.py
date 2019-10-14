# -*- coding: utf-8 -*-
# @Author: Zhewei Zhang
# @Date: 07/27/2018, 23:39

import numpy as np
import pandas as pd
import tkinter as tk
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tkinter import filedialog
from toolkits_2 import get_bhvinfo, load
"""
representing the relation between the logRT and rt; 

fit the psychmetric curve

fig1 a/c
"""

num_shape = 18
psycv_range = 2.
fig_saving = True

def sigmoid(x, x0, k):
    return -1 + 2/(1+np.exp(-k*(x-x0)))


def log_rt(shapes, choices):
    """
    transfer it into category to get the logLR in all rt values, 
        and it is convenient to save and plot in matlab
    retrun: right_rt_log: logLR varies with rt when the right is choosing
            left_rt_log: logLR varies with rt when the left is choosing
    """
    shape_num_category = pd.Categorical(shapes['rt'],ordered = True,
                                        categories = np.arange(1.,num_shape+.1,1)
                                        )
    left_rt_log = shapes['sumweight'][choices['left'] == 1].groupby(
            shape_num_category[choices['left'] == 1])
    right_rt_log = shapes['sumweight'][choices['left'] == -1].groupby(
            shape_num_category[choices['left'] == -1])
    return right_rt_log.describe()[['count','mean','std']], left_rt_log.describe()[['count','mean','std']]

def psych_curve(shapes, choices, groups = np.linspace(-psycv_range,psycv_range,num=40)):
    """
    return: psychmetric curve
    """
    psy_curve = choices.left.groupby(pd.cut(shapes['sumweight'],groups))
    if shapes['sumweight'].shape[0] > 2:
        prpt = np.array([np.nan, np.nan])    
        prpt, pcov = curve_fit(sigmoid, shapes['sumweight'], choices.left)
    else:
        prpt = np.array([np.nan, np.nan])    
    return psy_curve.mean(), prpt



def bhv_extract(file_paths):
    """
    
    """
    df_basic = pd.DataFrame([], columns = {'rt_mean','rt_sem','choice_prop','cr','cr_log','fr','label'})
    df_logRT = pd.DataFrame([], columns = {'right_rt_log','left_rt_log','rt', 'choice'})
    df_psycv = pd.DataFrame([], columns = {'cr','fr','psy_curve','fitting_x0', 'fitting_k'})
    for i, file in enumerate(file_paths):

        paod, trial_briefs = load(file)
        _, choice, shape, _, finish_rate = get_bhvinfo(paod,trial_briefs)
        right_rt_log , left_rt_log = log_rt(shape, choice)
        psy_curve, prpt = psych_curve(shape, choice, groups = np.linspace(-psycv_range,psycv_range,num=20))

        # keep summary data of each file for plotting
        df_basic.loc[i] = {'rt_mean':     np.round(shape.rt.mean(),3),
                           'rt_sem':      np.round(shape.rt.sem(),3),
                           'cr':          np.round(choice.correct.mean(),3),
                           'fr':          np.round(finish_rate,3),
                           'choice_prop': np.round((choice.left.mean()+1)/2,3),
                           'cr_log':      np.round(choice.correct_logLR.mean(),3),
                           'label': file
                           }
        
        df_logRT.loc[i] = {'right_rt_log': right_rt_log['mean'].values, 
                           'left_rt_log':   left_rt_log['mean'].values,
                           'rt':           shape.rt,
                           'choice':       choice.left,
                       }
        
        df_psycv.loc[i] = {'psy_curve': psy_curve, 
                           'cr':          np.round(choice.correct.mean(),3),
                           'fr':          np.round(finish_rate,3),
                           'fitting_x0':  prpt[0], 
                           'fitting_k':   prpt[1]
                       }
#        if np.isnan(df['cr']):
#            continue
    return df_basic, df_logRT, df_psycv

def plot_bhvBasic(df_logRT, df_psycv):

    rt_left, rt_right = [], []
    for i in range(df_logRT.rt.count()):
        rt_left.extend( df_logRT.rt[i][df_logRT.choice[i]==1])
        rt_right.extend(df_logRT.rt[i][df_logRT.choice[i]==-1])
    left_rt_dist = np.histogram(rt_left,bins = np.arange(0.5,25.6,1),density = True)[0]
    right_rt_dist = np.histogram(rt_right,bins = np.arange(0.5,25.6,1),density = True)[0]

    tar_Point = np.intersect1d(np.where(left_rt_dist>1e-3), np.where(right_rt_dist>1e-3)) # ignore the rt happens less than 0.1%
    left_rt_dist = left_rt_dist[tar_Point]
    right_rt_dist = right_rt_dist[tar_Point]
    
#    fig_w, fig_h = (10, 7)
#    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})

    fig = plt.figure('logLR vs rt')
    fig, ax1 = plt.subplots()
    ## right_rt_log/left_rt_log/shape.rt/choice for plot
    x = np.arange(1,df_logRT.right_rt_log[0].shape[0]+1)[tar_Point]
    
    ax1.errorbar(x, np.nanmean(np.vstack(df_logRT.left_rt_log.values),axis=0)[tar_Point],
                    yerr = stats.sem(np.vstack(df_logRT.left_rt_log.values),axis=0,nan_policy = 'omit')[tar_Point],
                    color = 'r')

    ax1.errorbar(x, np.nanmean(np.vstack(df_logRT.right_rt_log.values),axis=0)[tar_Point],
                    yerr = stats.sem(np.vstack(df_logRT.right_rt_log.values),axis=0,nan_policy = 'omit')[tar_Point],
                    color = 'g')
    plt.ylim(-1.7,1.7)

    ax2 = ax1.twinx()
    left_rt_dist = np.append(0, np.append(np.array([0]*tar_Point[0]),left_rt_dist))
    right_rt_dist= np.append(0, np.append(np.array([0]*tar_Point[0]),right_rt_dist))
    
    rt_dist = pd.DataFrame({'left': left_rt_dist,'right':-right_rt_dist})
    rt_dist.left.plot(kind='bar', color='red', ax=ax2)
    rt_dist.right.plot(kind='bar', color='green', ax=ax2)
    plt.xticks(np.arange(0,num_shape,5), ('0', '5', '10', '15','20'))
    plt.ylim(-0.28,0.28)
    plt.xlim(0,num_shape)
    if fig_saving:
        fig.savefig('../figs/logLR vs rt.eps', format='eps', dpi=1000)
    plt.show()
    
    #### psychometric curve
    x_ = np.linspace(-psycv_range,psycv_range, num = len(df_psycv.psy_curve[0].index.values), endpoint = True)
    x_ = np.tile(x_, len(df_psycv))
    y_  = np.hstack(df_psycv.psy_curve.values)

    prpt, pcov = curve_fit(sigmoid, x_, y_)
#    prpt, pcov = curve_fit(sigmoid, x_[np.logical_not(np.isnan(y_))], y_[np.logical_not(np.isnan(y_))])

#    prpt_x0 = df_psycv.fitting_x0.mean()
#    prpt_k = df_psycv.fitting_k.mean()
    prpt_x0, prpt_k = prpt[0], prpt[1]
    plt.figure('psychmetric curve')
    ## psy_curve for plot
    x = range(len(df_psycv.psy_curve[0].index.values))
    psy_values = []
    for i in df_psycv.psy_curve.values:
        psy_values.append(i.values)
    psy_values = np.array(psy_values)
    
    plt.errorbar(x, np.nanmean(psy_values, axis=0), yerr = stats.sem(psy_values, axis=0, nan_policy = 'omit'),fmt = 'o', 
                 label = 'correct rate:{}||completion rate:{}'.format(np.round(df_psycv['cr'].mean(),3), np.round(df_psycv['fr'].mean(),3)))
#    plt.errorbar(x, df_psycv.psy_curve.values.mean(), yerr = df_psycv.psy_curve.values.std()/np.sqrt(df_psycv.psy_curve.values.shape[0]),
#                 fmt = 'o', 
#                 label = 'correct rate:{}||completion rate:{}'.format(np.round(df_psycv['cr'].mean(),3), np.round(df_psycv['fr'].mean(),3)))
    
    plt.xticks(np.array([0,(len(df_psycv.psy_curve[0].index.values)-1)/2,len(df_psycv.psy_curve[0].index.values)-1]), (-psycv_range, '0.0', psycv_range))# 
    x = np.linspace(-psycv_range,psycv_range,100,endpoint = True)
    y = sigmoid(x, prpt_x0, prpt_k) 
    x = x*len(df_psycv.psy_curve[0].index.values)/(psycv_range*2) + (len(df_psycv.psy_curve[0].index.values)-1)/2
    plt.plot(x,y)
    plt.xlabel('')
    plt.legend(loc=2,fontsize = 'large')
    foo_fig = plt.gcf()
    if fig_saving:
        foo_fig.savefig('../figs/psychmetric curve.eps', format='eps', dpi=1000)
        print('figures are saved')
    plt.show()
    a=1
#        sio.savemat('psycurve.mat',{'df': df_curr,
#                                    'logLR':df_curr['logLR'].values,
#                                    'choice':df_curr['choice'].values})

def main(file_paths=None):
    print("start")
    print("select the files")
    root = tk.Tk()
    root.withdraw()
    if file_paths == None:
        file_paths = filedialog.askopenfilenames(parent = root,
                                                title = 'Choose a file',
                                                filetypes = [("HDF5 files", "*.hdf5")]
                                                )
    ##
    df_basic, df_logRT, df_psycv = bhv_extract(file_paths)
    plot_bhvBasic(df_logRT, df_psycv)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_basic)
    print(df_basic.rt_mean.mean(), df_basic.rt_mean.sem())
    return df_basic, df_logRT, df_psycv

if __name__ == '__main__':
    main()
