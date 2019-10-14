# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:36:40 2018

@author: YangLab_ZZW
"""
import copy
import numpy as np
import tkinter as tk
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
from tkinter import filedialog
from task_info import rtshapebrief_config
from netools import cal_mat_thresholded, wh_loading

input_setting = rtshapebrief_config()
choice_pos = input_setting['choice']
numNeuron = 128

"""
visualize the connection pattern of the when/which neuron

construct a strutral network based on the recurrent connection; the calculate 
the gesdistic distance and maximum flow of differet types of neurons;

fig 4 a/d
"""


def others(when, which, zero_node):
    other_nodes = []
    for i in range(numNeuron):
        if not(any(i == when) or any(i == which) or any(i == zero_node)):#
            other_nodes.append(i)
            
    return other_nodes


def balabala(type1,type2,distance_all):
    distance = []
    for i in type1:
        for j in type2:
            if i!=j:
                distance.append(distance_all[i,j])
    return distance

def interaction(when, which, others, value):
    
    when_which_distance = balabala(when,which,value)
    when_when_distance = balabala(when,when,value)
    which_which_distance = balabala(which,which,value)
    control_distance = balabala(which,others,value)
    control_distance.extend(balabala(when,others,value))
    control_distance = np.array(control_distance)
    
    value_mean = {'which_which':np.mean(which_which_distance),
                     'which_when':np.mean(when_which_distance),
                     'when_when':np.mean(when_when_distance),
                     'control':np.mean(control_distance),
                     }
    
    value_sem = {'which_which':stats.sem(which_which_distance),
                     'which_when':stats.sem(when_which_distance),
                     'when_when':stats.sem(when_when_distance),
                     'control':stats.sem(control_distance),
                     }
    num = {'which_which':len(which_which_distance),
                     'which_when':len(when_which_distance),
                     'when_when':len(when_when_distance),
                     'control':control_distance.shape[0],
                     }

    return value_mean, value_sem, num

def gesDistance(stru_graph, zero_node, when, which):
    """
    get the shortest path length
    """
    g = copy.deepcopy(stru_graph)
    path_length = nx.all_pairs_dijkstra_path_length(g) # for weighted network
#    path_length = nx.all_pairs_shortest_path_length(g)# for unweighted network
#    distance_all = np.zeros([numNeuron,numNeuron])
    distance_all = 1e10 + np.zeros([numNeuron,numNeuron])
    for i in path_length:
        for key, value in i[1].items():
            distance_all[i[0],key] = value
    
    distance_all = 1/distance_all
    other_nodes = others(when, which, zero_node)
    
    distance_mean, distance_sem, distance_num = interaction(when, which, other_nodes, distance_all)
    
    return distance_mean, distance_sem, distance_num
 


def maxFlow(stru_graph, zero_node, when, which):
    """
    maximun flow between two nodes
    """
    g = copy.deepcopy(stru_graph)
    max_flow = np.zeros([numNeuron,numNeuron])
    for i in range(numNeuron):
        for j in range(numNeuron):
            if i==j or np.any(zero_node==i) or np.any(zero_node==j):
                continue
            max_flow[i,j] = nx.maximum_flow_value(g,i,j)
        
    other_nodes = others(when, which, zero_node)
    
    flow_mean, flow_sem, flow_num = interaction(when, which, other_nodes, max_flow)
    
    return flow_mean, flow_sem, flow_num

def gesdes_plot(df_gesdis):
    distance = np.zeros((df_gesdis.label.count(), 4))
    for i in range(df_gesdis.label.count()):
        distance[i,:] = np.array([df_gesdis['mean'][i]['control'], df_gesdis['mean'][i]['when_when'], 
                 df_gesdis['mean'][i]['which_which'], df_gesdis['mean'][i]['which_when']])
    
    fig = plt.figure()
    plt.boxplot(distance)
    plt.xticks(1+np.arange(4),('all nodes(largest component)','when-when','which-which','when-which'))
    plt.ylabel('shortest path length ')
    plt.title('zeros nodes are not included, 0.1, 1/geodesic distance, 1/connectivity')
    fig.savefig('../figs/shortest path length.eps', format='eps', dpi=1000)
    for i in range(distance.shape[1]):
        for j in range(distance.shape[1]):
            if i >= j:
                continue
            t, p = stats.ttest_ind(distance[:,i], distance[:,j])
            print('gesdes',i,j,p)
    plt.show()

def maxflow_plot(df_maxflow, title = ''):
    maxflow = np.zeros((df_maxflow.label.count(), 4))
    for i in range(df_maxflow.label.count()):
        maxflow[i,:] = np.array([df_maxflow['mean'][i]['control'], df_maxflow['mean'][i]['when_when'], 
                 df_maxflow['mean'][i]['which_which'], df_maxflow['mean'][i]['which_when']])
    fig2 = plt.figure()
    plt.boxplot(maxflow)
    plt.xticks(np.arange(4)+1,('all nodes(largest component)','when-when','which-which','when-which'))
    plt.ylabel('maximun flow ')
    fig2.savefig('../figs/max flow'+title+'.eps', format='eps', dpi=1000)
    for i in range(maxflow.shape[1]):
        for j in range(maxflow.shape[1]):
            if i >= j:
                continue
            t, p = stats.ttest_ind(maxflow[:,i], maxflow[:,j])
            print('maxflow',i,j,p)
    plt.show()

def conpattern_plot(df_contion):
    """
    connection pattern
    """
    
    fig_w, fig_h = (10, 7)
    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})

    pos_when_con, neg_when_con = np.zeros((df_contion.label.count(), 4)), np.zeros((df_contion.label.count(), 4))
    pos_which_con, neg_which_con = np.zeros((df_contion.label.count(), 4)), np.zeros((df_contion.label.count(), 4))
    for i in range(df_contion.label.count()):
        when = df_contion.when.iloc[i]
        which = df_contion.which.iloc[i]
        output_weight = df_contion.output.iloc[i]
       
        pos_when_con[i,:] =(output_weight[choice_pos[0]:choice_pos[-1]+1, when.pos].mean(axis=1))
        neg_when_con[i,:] =(output_weight[choice_pos[0]:choice_pos[-1]+1, when.neg].mean(axis=1))
        pos_which_con[i,:] =(output_weight[choice_pos[0]:choice_pos[-1]+1, which.pos].mean(axis=1))
        neg_which_con[i,:] =(output_weight[choice_pos[0]:choice_pos[-1]+1, which.neg].mean(axis=1))
    
    
    f = plt.figure()
    plt.subplot(211)
    plt.errorbar(range(4), y = np.mean(pos_when_con, axis=0),yerr = stats.sem(pos_when_con,axis=0),label = 'when pos')
    plt.errorbar(range(4), y = np.mean(neg_when_con, axis=0),yerr = stats.sem(neg_when_con,axis=0),label = 'when neg')
    plt.legend()
    plt.xticks(range(4),('fixtaion', 'left target','right target','break'))
    plt.subplot(212)
    plt.errorbar(range(4), y = np.mean(pos_which_con, axis=0),yerr = stats.sem(pos_which_con,axis=0),label = 'which pos')
    plt.errorbar(range(4), y = np.mean(neg_which_con, axis=0),yerr = stats.sem(neg_which_con,axis=0),label = 'which neg')
    plt.legend()
    plt.xticks(range(4),('fixtaion', 'left target','right target','break'))
    f.savefig('../figs/weight pattern.eps', format='eps', dpi=1000)
    plt.show()

def graph_construct(rnn_weight, threshold = 0.3):
    connectivity = copy.deepcopy(rnn_weight)
    connectivity = np.abs(connectivity)
    net_mat = cal_mat_thresholded(connectivity,threshold = threshold)
    stru_graph = nx.DiGraph()
    norms = np.mean(connectivity[net_mat!=0])
    for i in range(net_mat.shape[0]):
        for j in range(net_mat.shape[1]):
            if net_mat[i,j] != 0:
                stru_graph.add_edge(i, j, weight = norms/np.abs(connectivity[i,j]), capacity= np.abs(connectivity[i,j]))
                net_mat[i,j] = norms/np.abs(connectivity[i,j])
    # find the largest component in the newgate weight
    zero_node = np.intersect1d(np.where(net_mat.sum(axis=0)==0),np.where(net_mat.sum(axis=1)==0))
    return stru_graph, zero_node



def data_extract(file_paths):
    df_contion = pd.DataFrame([], columns = {'label','when','which','connectivity','output'})
    df_gesdis = pd.DataFrame([], columns = {'label','mean','num'})
    df_maxflow = pd.DataFrame([], columns = {'label','mean','num'})
    neuron_num = pd.DataFrame([], columns = {'when_pos','when_neg','which_pos','which_neg','zero_node'})
    for i, file in enumerate(file_paths):
        print(file)
        y = wh_loading(file)
        when = y['when']
        which = y['which']
        when_neuron = np.append(when.pos,when.neg)
        which_neuron = np.append(which.pos,which.neg)
        
        print(np.intersect1d(when_neuron,which_neuron).shape)

        rnn_weight = y['rnn_weight'].detach().numpy()
        stru_graph, zero_node = graph_construct(rnn_weight)
        distance_mean, distance_sem, distance_num = gesDistance(stru_graph, zero_node, when_neuron, which_neuron)

        neuron_num.loc[i] = {'when_pos':  when.pos.shape[0], 'when_neg' : when.neg.shape[0],
                             'which_pos': which.pos.shape[0],'which_neg': which.neg.shape[0],
                             'zero_node': zero_node.shape[0]} 
        
        df_contion.loc[i] = {'label': file,'when':when,'which':which,'connectivity':rnn_weight, 'output':y['output_weight']}
        df_gesdis.loc[i] = {'label': file, 'mean': distance_mean,'num':distance_num}

        print('calcuating maximum flow takes a long time, please be patient')
        flow_mean, distance_sem, flow_num = maxFlow(stru_graph, zero_node, when_neuron, which_neuron)
        df_maxflow.loc[i] = {'label': file, 'mean': flow_mean,'num':flow_num}
    
    return df_contion, df_gesdis, df_maxflow, neuron_num


def main():
    print("start")
    print("select the model files")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilenames(
                parent=root,title='Choose the when/which file',
                filetypes=[("model files", "*old_threshold0.5.yaml")]
                )
    
    print('*'*49)
#    df_contion, df_gesdis, df_maxflow, neuron_num, df_maxflow_includeZeros = data_extract(file_path)
    df_contion, df_gesdis, df_maxflow, neuron_num = data_extract(file_path)
#    for neuron_type in neuron_num:
#        print(neuron_type, neuron_num[neuron_type].mean(), stats.sem(neuron_num[neuron_type]))
    
    conpattern_plot(df_contion)
    gesdes_plot(df_gesdis)
    maxflow_plot(df_maxflow)




if __name__ == '__main__':
    main()

