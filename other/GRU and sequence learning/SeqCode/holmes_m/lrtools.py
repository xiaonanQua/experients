# -*- coding: utf-8 -*-
# @Author: Zhewei Zhang
# @Date: 04/22/2017


import numpy as np
from scipy import stats
from functools import reduce
from scipy.optimize import curve_fit
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler

def linearReg(x,resp,loc = None):
    
    x = np.asarray(x,dtype = 'float64')
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)
#    x_std = x
    if loc is None:
        y = resp
    else:
        y = resp[loc]
    result = sim_linreg(x_std,y)
#    result = ridge_reg(x_std,y) #lasso_reg
    return result


def ridge_reg(X_std,y):
    newX_std = np.append(np.ones((len(X_std), 1)), X_std, axis=1)

    lm = linear_model.RidgeCV(alphas = [1e-4, 0.001, 0.01,0.1,1], fit_intercept=True)
#    lm = linear_model.RidgeCV(alphas = [1e-4, 0.001, 0.01,0.1,1], fit_intercept=True)
    lm.fit(X_std, y)
    alpha_ = lm.alpha_
    lm = linear_model.Ridge(alpha=alpha_, fit_intercept=True)
    lm.fit(X_std, y)

    params = np.append(lm.intercept_, lm.coef_)
    predictions = lm.predict(X_std)
    SSresidual = ((y - predictions) ** 2).sum()
    SStotal = ((y - y.mean()) ** 2).sum()

    Rsquare = lm.score(X_std, y)
    theata2 = SSresidual / (y.shape[0] - 2)

    temp = np.linalg.inv(np.dot(newX_std.T, newX_std) + alpha_ * np.eye(newX_std.shape[1]))
    var_b = theata2 * reduce(np.dot, [temp, newX_std.T, newX_std, temp]).diagonal()
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b
    
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX_std) - 1))) for i in ts_b]
    result = {
            "sd_b":np.round(sd_b, 3),
              "ts_b": np.round(ts_b, 3),
              "p_values": np.round(p_values, 4),
              "params": np.round(params, 4),
              "R_square": np.round(Rsquare, 4),
              "alpha": alpha_
              }
    return result

def sim_linreg(X_std,y):
    newX_std = np.append(np.ones((len(X_std), 1)), X_std, axis=1)

    lm = linear_model.LinearRegression(fit_intercept=True)
#    lm = linear_model.LogisticRegression(fit_intercept=True,
#                                         random_state=0,
#                                         solver='lbfgs',
#                                         multi_class='multinomial')
    lm.fit(X_std, y)

    params = np.append(lm.intercept_, lm.coef_)
    predictions = lm.predict(X_std)
    SSresidual = ((y - predictions) ** 2).sum()
    Rsquare = lm.score(X_std, y)
    Rsquare_adjusted = 1-(X_std.shape[0]-1)/(X_std.shape[0]-X_std.shape[1])*(1-Rsquare)
    MSE = SSresidual / (len(newX_std) - len(newX_std[0]))
    var_b = MSE * (np.linalg.inv(np.dot(newX_std.T, newX_std)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX_std) - 1))) for i in ts_b]
    
    if np.any(np.intersect1d(np.where(np.abs(params)<0.001), np.where(np.array(p_values)<0.0001))):
        c=1
        
    result = {"sd_b":np.round(sd_b, 3),
              "ts_b": np.round(ts_b, 3),
              "p_values": np.round(p_values, 10),
              "params": np.round(params, 4),
              "R_square": np.round(Rsquare, 4),
              "Rsquare_adjusted":np.round(Rsquare_adjusted, 4)
              }
    return result

def lasso_reg(X_std,y):
    newX_std = np.append(np.ones((len(X_std), 1)), X_std, axis=1)

    lm = linear_model.LassoCV(alphas = [0,1e-5,1e-4,1e-3,0.01], fit_intercept=True)
    lm.fit(X_std, y)
    alpha_ = lm.alpha_
    lm = linear_model.Lasso(alpha=alpha_, fit_intercept=True)
    lm.fit(X_std, y)

    params = np.append(lm.intercept_, lm.coef_)
    predictions = lm.predict(X_std)
    SSresidual = ((y - predictions) ** 2).sum()
    SStotal = ((y - y.mean()) ** 2).sum()

    Rsquare = lm.score(X_std, y)
    theata2 = SSresidual / (y.shape[0] - 2)

    temp = np.linalg.inv(np.dot(newX_std.T, newX_std) + alpha_ * np.eye(newX_std.shape[1]))
    var_b = theata2 * reduce(np.dot, [temp, newX_std.T, newX_std, temp]).diagonal()
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b
    # TODO: p value for lasso regression
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX_std) - 1))) for i in ts_b]

    result = {"sd_b":np.round(sd_b, 3),
              "ts_b": np.round(ts_b, 3),
              "p_values": np.round(p_values, 4),
              "params": np.round(params, 4),
              "R_square": np.round(Rsquare, 4),
              "alpha": alpha_
              }
    return result


