# -*- coding: utf-8 -*-
# @Author: Zhewei Zhang
# @Date: 07/27/2018, 23:39

import numpy as np
import pandas as pd
import tkinter as tk
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tkinter import filedialog
from toolkits_2 import get_bhvinfo, load
from sklearn.linear_model import LogisticRegressionCV


"""
with logistci regression, we show the leverage of each shape on the choice

fig1 b/d 
"""

max_rt = 25
def shape_extract(path_files):
    df_epoch = pd.DataFrame([], columns = {'label','bais','coef','lastshape_consistency'})
    df_subweight = pd.DataFrame([], columns = {'label','bais','coef'})

    for i, file in enumerate(path_files):
        ## load files
        paod, trial_briefs = load(file)
        _, choice, shape, _, finish_rate = get_bhvinfo(paod,trial_briefs)
        reg_epoch, reg_subweight, lastshape_consistency = regressions(shape,choice)
        df_epoch.loc[i] = {'label': file,
                           'bais':reg_epoch['bais'],
                           'coef':reg_epoch['coef'], 
                           'lastshape_consistency': reg_epoch['lastshape_consistency'],
                           }
        df_subweight.loc[i] = {'label': file,
                           'bais':reg_subweight['bais'],
                           'coef':reg_subweight['coef'], 
                           }

    return df_epoch, df_subweight

def regressions(shape, choice, shapes_inaccount = 3):
    
    # =============================================================================
    first_three, last_three, inter_sum, choice_epoch = [], [], [], []
    temp_weight = np.concatenate(shape.tempweight.values).reshape(-1,shape.tempweight[0].shape[0])

    for i in range(shape.count().rt):
        if shape.rt.iloc[i]<shapes_inaccount*2+0.5:
            continue
        first_three.append(temp_weight[i,:shapes_inaccount])
        last_three.append(temp_weight[i, shape.rt.iloc[i]-shapes_inaccount : shape.rt.iloc[i]])
        inter_sum.append(np.sum(temp_weight[i, shapes_inaccount : shape.rt.iloc[i]-shapes_inaccount]))
        choice_epoch.append(choice.left.iloc[i])
    first_three = np.array(first_three)
    last_three = np.array(last_three)
    inter_sum = np.array(inter_sum)
    choice_epoch = np.array(choice_epoch)
    X = np.hstack((first_three,inter_sum.reshape(-1,1), last_three))

    lastshape_consistency = np.mean(np.sign(X[:,-1]) == choice_epoch)
    choice_epoch[choice_epoch==-1] = 0
    
    lm_epoch = LogisticRegressionCV(cv=10, fit_intercept=True)
    lm_epoch.fit(X[:,:-1], choice_epoch)
    reg_epoch = {'bais':lm_epoch.intercept_[0],'coef': lm_epoch.coef_[0],'lastshape_consistency':lastshape_consistency}
    
#    X_ = sm.add_constant(X)
#    logit_model = sm.GLM(choice_epoch,X_, family= sm.families.Binomial())
#    result = logit_model.fit()
#    coef = result.params
#    reg_epoch = {'bais':coef[0],'coef': coef[1:]}

                
# =============================================================================
# rt = 6 , plot the shape distribution in the first/ last second/ last epoch
#    f1 = plt.figure()
#    plt.subplot(311);plt.hist(first_three[:,0]);plt.title('first shape')
#    plt.subplot(312);plt.hist(last_three[:,-2]);plt.title('last second shape')
#    plt.subplot(313);plt.hist(last_three[:,-1]);plt.title('last shape')
#    f1.savefig('../figs/shape distribution in different epoch.eps', format='eps', dpi=1000)
#    plt.show()
# =============================================================================
    ### subjective weight
    shape_order = np.concatenate(shape.order.values).reshape(-1,shape.tempweight[0].shape[0])
    shapeNum_percon = np.zeros((shape_order.shape[0],10))

    for numT in range(shape_order.shape[0]):
        for i in range(10):
            shapeNum_percon[numT,i] = np.sum(shape_order[numT,:]==i+1)
    choice_left = choice.left.values
    choice_left[choice_left==-1]=0
    
    lm_subweight = LogisticRegressionCV(cv=10,fit_intercept=True)
    lm_subweight.fit(shapeNum_percon, choice_left)
    reg_subweight = {'bais':lm_subweight .intercept_[0],'coef': lm_subweight .coef_[0]}


    return reg_epoch, reg_subweight, lastshape_consistency

def plot_subweight(df_epoch, df_subweight):
    plt.figure('regression weight on each epoch') 
    bais_epoch = df_epoch.bais.values
    coef_epoch = np.vstack(df_epoch.coef.values)
    coefs_epoch = np.hstack((bais_epoch.reshape(-1,1),coef_epoch))
    plt.errorbar([0,1,2,3,5,7,8], coefs_epoch.mean(axis=0), yerr = stats.sem(coefs_epoch, axis=0), 
                 label = [np.round(df_epoch.lastshape_consistency.mean(),5), np.round(stats.sem(df_epoch.lastshape_consistency.values), 5)])
    plt.plot([0,1,2,3,5,7,8], coefs_epoch.mean(axis=0), 'o-')
    plt.legend()
    plt.ylabel('coef')
    plt.xlabel('nth shape')
    foo_fig = plt.gcf()
    foo_fig.savefig('../figs/regression_on_epochs-1000.eps', format='eps', dpi=1000)

    plt.figure('subjective value') 
    coefs_subweight = np.vstack(df_subweight.coef.values)
    f2 = plt.figure()
    plt.errorbar(range(10),coefs_subweight.mean(axis=0),yerr = stats.sem(coefs_subweight,axis=0))
    plt.plot(range(10), coefs_subweight.mean(axis=0), 'o-')
    f2.savefig('../figs/subjective_weight.eps', format='eps', dpi=1000)
    print('figurs are saved')


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
    fig_w, fig_h = (6, 4)
    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})

    plt.figure('shape effect on choice')
      
    df_epoch, df_subweight = shape_extract(file_paths)
    plot_subweight(df_epoch, df_subweight)
    
    
if __name__ == '__main__':
    main()
