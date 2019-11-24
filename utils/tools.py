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
import cv2


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


def confusion_matrix(targets, preds):
    """
    生成混淆矩阵
    :param targets: 真实标签数据，数据格式list
    :param preds: 与真实标签对应的预测标签数据，数据格式list
    :return: 混淆矩阵
    """
    # 统计真实标签中的类别数量
    num_class = len(set(targets))
    # 初始化相应类别数量大小的混淆矩阵
    conf_matrix = np.zeros(shape=[num_class, num_class])
    print(conf_matrix)
    # 判断真实标签与预测标签的数量是否相等
    if len(targets) != len(preds):
        raise Exception('The number of real and predicted labels is inconsistent')
    # 进行标签的统计
    for i in range(len(targets)):
        true_i = np.array(targets[i])
        pred_i = np.array(preds[i])
        conf_matrix[true_i, pred_i] += 1.0

    return conf_matrix


def visiual_confusion_matrix(confusion_mat, classes_name, graph_name=None, out_path=None):
    """
    可视化混淆矩阵
    :param confusion_mat: 统计好的混淆矩阵
    :param classes_name: 混淆矩阵对应的类别名称
    :param graph_name: 当前图的名称
    :param out_path: 以png的图像格式保存混淆矩阵
    :return:
    """
    # 判断混淆矩阵中的类别与类别名称中类别数量是否一致
    if confusion_mat.shape[0] != len(classes_name):
        raise Exception('Inconsistent number of categories')
    # 对混淆矩阵逐行进行数值归一化
    confusion_mat_normal = confusion_mat.copy()
    for i in range(len(classes_name)):
        confusion_mat_normal[i, :] = confusion_mat[i, :] /confusion_mat_normal[i, :].sum()
    print(confusion_mat_normal)

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')
    plt.imshow(confusion_mat_normal, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes_name)))
    plt.xticks(xlocations, classes_name, rotation=60)
    plt.yticks(xlocations, classes_name)
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix ' + graph_name)

    # 打印数字
    for i in range(confusion_mat_normal.shape[0]):
        for j in range(confusion_mat_normal.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    if out_path is not None:
        plt.savefig(os.path.join(out_path, 'Confusion_Matrix_' + graph_name + '.png'))
    plt.show()
    plt.close()


def read_and_write_videos(video_files=None, out_files=None):
    """
    通过OpenCV中的VideoCapture函数调用系统摄像头读取视频图像，或者读取特定视频文件
    :param video_files: 读取的视频文件地址，若为Ｎｏｎｅ则读取摄像头文件
    :param out_files: 输出文件
    :return:
    """
    # 创建VideoCapture进行一帧一帧视频读取
    if video_files is None:
        # 调用系统单个摄像头作为视频输入
        cap = cv2.VideoCapture(0)
    else:
        # 读取特定视频文件
        cap = cv2.VideoCapture(video_files)

    # 判断摄像头是否打开
    if cap.isOpened() is False:
        print('Error opening video stream or file')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(out_files,
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          10, (frame_width, frame_height))

    # 读取视频，直到读取所有时间段视频
    while cap.isOpened():
        # 一帧一帧的读取视频
        ret, frame = cap.read()
        if ret == True:
            out.write(frame)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, 'xiaonan', (30, 30), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'xiaoshuai', (30, 90), font, 1, (0, 0, 255), 2)

            # 显示帧结果
            cv2.imshow('frame', frame)
            # 播放每一帧时等待25秒或者按ｑ结束
            if cv2.waitKey(1)&0xFF==ord('q'):
                print('结束..')
                break
        else:  # 结束循环
            break

    # 当视频读取结束时，释放视频捕捉的输出Ｏ
    cap.release()
    # 关闭所有帧窗口
    cv2.destroyAllWindows()




if __name__ == "__main__":
    # labels = torch.tensor([1, 2, 3, 1])
    # # print(labels.squeeze(1))
    # print(one_hot_embedding(labels, 4))
    # cfg = Cifar10Config()
    # test_loader = cfg.dataset_loader(cfg.cifar_10_dir, train=False, shuffle=False)
    # show_label_distribute(test_loader)
    video_file = '/home/xiaonan/sf6_1.avi'
    read_and_write_videos()