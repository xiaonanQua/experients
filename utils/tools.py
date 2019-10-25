# -*- coding:utf-8 -*-
from __future__ import division, print_function, absolute_import
import math
import sys
import os
import time
import torch
from torch.utils.data import random_split
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from config.cifar10_config import Cifar10Config


def view_bar(message, num, total):
    """
    进度条工具
    :param message: 进度条信息
    :param num: 当前的值,从1开始..
    :param total: 整体的值
    :return:
    """
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">"*rate_num, ""*(40-rate_num), rate_nums, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()


def mkdir(dir_path):
    """
    判断文件夹是否存在,创建文件夹
    :param dir_path: 文件夹路径
    :return:
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def imshow(images, title=None):
    """
    显示一些PIL格式的图像
    :param images: 图像
    :param title: 标题
    :return:
    """
    # 将PIL格式的图像转化成numpy形式，再将图像维度转化成（高度，宽度，颜色通道）
    images = images.numpy().transpose([1, 2, 0])
    # 设置平均值和标准差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # 将进行归一化后的图像还原
    images = images * std + mean
    images = np.clip(images, 0, 1)
    plt.imshow(images)
    if title is not None:
        plt.title(title)

def show_image(images, num_rows, num_cols, scale=2):
    """
    显示多个图片
    :param images: 多个图片
    :param num_rows: 行数量
    :param num_cols: 列数量
    :param scale: 尺度
    :return:
    """
    # 图像大小
    figsize = (num_cols*scale, num_rows*scale)


def one_hot_embedding(labels, num_classes):
    """
    将标签嵌入成one-hot形式
    :param labels: 标签,（LongTensor）类别标签,形状[N,]
    :param num_classes: 类别数,
    :return:(tensor)被编码的标签,形状（N,类别数）
    """
    # 返回2维张量，对角线全是1，其余全是0
    y = torch.eye(num_classes)
    return y[labels]  # 使用按行广播机制


def split_valid_set(dataset, save_coef):
    """
    从原始数据集中划分出一定比例的验证集
    :param dataset: 原始数据集，一般是训练集。这里的数据集是经过pytorch中DataSet读取出来的数据集对象。
    :param save_coef: 保存原始数据集的系数
    :return: 划分后的数据集。格式类似于：train_dataset, valid_dataset
    """
    # 训练集的长度
    train_length = int(save_coef*len(dataset))
    # 验证集的长度
    valid_length = len(dataset) - train_length
    # 使用pytorch中的随机划分成数据集来划分
    train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length])

    return train_dataset, valid_dataset


def show_label_distribute(data_loader):
    """
    显示数据集标签的分布情况
    :param data_loader: 数据集加载器（pytorch加载器对象）
    :return:
    """
    print('label distribution ..')
    figure, axes = plt.subplots()
    labels = [label.numpy().tolist() for _, label in data_loader]
    print(labels)
    class_labels, counts = np.unique(labels, return_counts=True)
    axes.bar(class_labels, counts)
    axes.set_xticks(class_labels)
    plt.show()


def vis(test_accs, confusion_mtxes, labels, figsize=(20, 8)):
    cm = confusion_mtxes[np.argmax(test_accs)]
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    fig = plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(test_accs, 'g')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sn.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    plt.show()

if __name__ == "__main__":
    # labels = torch.tensor([1, 2, 3, 1])
    # # print(labels.squeeze(1))
    # print(one_hot_embedding(labels, 4))
    cfg = Cifar10Config()
    test_loader = cfg.dataset_loader(cfg.cifar_10_dir, train=False, shuffle=False)
    show_label_distribute(test_loader)