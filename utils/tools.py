# -*- coding:utf-8 -*-
from __future__ import division, print_function, absolute_import
import math
import sys
import os
import time
import torch
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt


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



if __name__ == "__main__":
    labels = torch.tensor([1, 2, 3, 1])
    # print(labels.squeeze(1))
    print(one_hot_embedding(labels, 4))