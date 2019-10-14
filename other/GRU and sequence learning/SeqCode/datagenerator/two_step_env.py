# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:16:00 2019

@author: Zhewei Zhang
"""
import os
import datetime
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


class world(object):
    def __init__(self):
        return

    def get_outcome(self):
        return
    
    def get_all_outcomes(self):
        outcomes = {}
        for state in range(self.n_states):
            for action in range(1 if self.n_actions == 0 else self.n_actions):
                next_state, reward = self.get_outcome(state, action)
                outcomes[state, action] = [(1, next_state, reward)]
        return outcomes
    
class drifting_probabilitic_bandit(world):
    """
    World: 2-Armed bandit.
    Each arm returns reward with a different probability.
    The probability of returning rewards for all arms follow Gaussian random walks.
    """
    
    def __init__(self, arm_number, drift):
        self.name = "n_armed_bandit"
        self.n_states = 1
        self.n_actions = arm_number
        self.dim_x = 1
        self.dim_y = 1
        
        self.mu_min = 0.25
        self.mu_max = 0.75
        self.drift = drift
        
        self.reward_mag = 1
        
        self.mu = [np.random.uniform(self.mu_min, self.mu_max) for a in range(self.n_actions)]
        
    def update_mu(self):
        self.mu += np.random.normal(0, self.drift, self.n_actions)
        self.mu[self.mu > self.mu_max] = self.mu_max
        self.mu[self.mu < self.mu_min] = self.mu_min
            
    def get_outcome(self, state, action):
        
        self.update_mu()
        self.rewards = [self.reward_mag if np.random.uniform(0,1) < self.mu[a] else 0 for a in range(self.n_actions) ]
        next_state = None
        
        reward = self.rewards[action]
        return int(next_state) if next_state is not None else None, reward
    
class Daw_two_step_task(drifting_probabilitic_bandit):
  
    def __init__(self, trans_prob, drift = 0.025):
        
        self.name = "two_step"
        self.n_states = 3
        self.n_actions = 2
        self.dim_x = 1
        self.dim_y = 1
        
        self.n_arms = self.n_actions
        self.n_of_bandits = 2
        self.drift = drift
        
        self.context_transition_prob = trans_prob
        self.bandits = [drifting_probabilitic_bandit(self.n_arms, self.drift) for n in range(self.n_of_bandits)]
        
    def get_outcome(self, state, action):
        
        if state == 0:
            reward = 0
            prob = np.random.uniform(0,1)
            if action == 0:
                next_state = 1 if prob< self.context_transition_prob else 2
            elif action == 1:
                next_state = 2 if prob< self.context_transition_prob else 1
            else:
                print('No valid action specified')
                
        if state == 1:
            _, reward = self.bandits[0].get_outcome(0, action)
            next_state = 0
            
        if state == 2:
            _, reward = self.bandits[1].get_outcome(0, action)
            next_state = 0
        
        return int(next_state) if next_state is not None else None, reward

def gassuain_walk(num = 200, upperB = 0.75, lowerB = 0.25, SD = 0.025):
    prob = np.zeros((num,))
    prob[0] = 0.5;
    for i in range(num-1):
        prob[i+1] = np.random.randn()*SD + prob[i]
        prob[i+1] = upperB if prob[i+1] > upperB else prob[i+1] 
        prob[i+1] = lowerB if prob[i+1] < lowerB else prob[i+1] 
    return prob

def softmax(state, q, beta):
    """
    Softmax policy: selects action probabilistically depending on the value.
    Args:
        state: an integer corresponding to the current state.
        q: a matrix indexed by state and action.
        params: a dictionary containing the default parameters.
    Returns:
        an integer corresponding to the action chosen according to the policy.
    """
    
    value = q[state,:]
#     ipdb.set_trace()
    prob = np.exp(value * beta) # beta is the inverse temperature parameter
    prob = prob / np.sum(prob)  # normalize
    cum_prob = np.cumsum(prob)  # cummulation summation
    action = np.where(cum_prob > np.random.rand())[0][0]
    return action

def plot_stayprob(action_all,state_all,reward_all):
    NumTrials = len(reward_all)
    type_num = np.array([0,0,0,0])
    stay_num = np.array([0,0,0,0])
    common_all = []
    for ntrial in range(NumTrials):
        if action_all[ntrial] == 0:
            if state_all[ntrial] == 1:
                common_all.append(1) # common
            elif state_all[ntrial] == 2:
                common_all.append(0) # rare
            else:
                raise('unknown state')
        elif action_all[ntrial] == 1:
            if state_all[ntrial] == 1:
                common_all.append(0) # rare
            elif state_all[ntrial] == 2:
                common_all.append(1) # common
            else:
                raise('unknown state')
        else:
            raise('unknown action')


    for ntrial in range(NumTrials-1):
        if common_all[ntrial]:
            if reward_all[ntrial]:
                type_num[0] += 1
                if action_all[ntrial] == action_all[ntrial+1]:
                    stay_num[0] +=  1
            else:
                type_num[2] += 1
                if action_all[ntrial] == action_all[ntrial+1]:
                    stay_num[2] +=  1
        else:
            if reward_all[ntrial]:
                type_num[1] += 1
                if action_all[ntrial] == action_all[ntrial+1]:
                    stay_num[1] +=  1
            else:
                type_num[3] += 1
                if action_all[ntrial] == action_all[ntrial+1]:
                    stay_num[3] +=  1

#    plt.figure()
#    plt.bar([1,2,4,5],stay_num/type_num)
#    plt.xticks([1,2,4,5],{'cr','rr','cn','rn'})
#    plt.show()
    return stay_num, type_num, common_all

class agent_MF(object):
    
    def __init__(self):
        self.lr = 0.4
        
    def update(self, action, reward):
        pass
    
def data_save(data_TS, data_TS_Brief, info=None, state = 'TrainingSet'):
    # data saving
    pathname = "../data/"
    file_name = datetime.datetime.now().strftime("%Y_%m_%d")
    data_name = 'TwoStep_' + state + '-' + file_name
    n = 0
    while 1: # save the model
        n += 1
        if not os.path.isfile(pathname+data_name+'-'+str(n)+'.mat'):
            sio.savemat(pathname+data_name+'-'+str(n)+'.mat',
                    {'data_TS':data_TS,
                     'data_TS_Brief':data_TS_Brief,
                     'info':info})
            print("_" * 36)
            print("training file for two steps task is saved")
            print("file name:" + pathname+data_name+'-'+str(n)+'.mat')
            break
    filename = pathname + data_name + '-' + str(n) + '.mat'
    return filename
        
        
