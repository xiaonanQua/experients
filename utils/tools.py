# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import
import math
import sys
import os
import time
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

if __name__ == "__main__":
    for i in range(1000):
        view_bar('test', i+1, 1000)
        time.sleep(0.1)
