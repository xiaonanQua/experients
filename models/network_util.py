import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models import resnet18, resnet50
from torch.nn import init
import torch
import torch.nn as nn
from torchvision.datasets import ImageNet


def init_weight(net, zero_gamma=False, init_type='normal', gain=0.02):
    def init_func(m):
        # print('m:', m)
        classname = m.__class__.__name__
        # print('class name',classname)
        # print(classname.find)
        if zero_gamma:
            if hasattr(m, 'bn2'):
                init.constant_(m.bn2.weight.data, 0.0)
                init.constant_(m.bn2.bias.data, 0.0)
                print(1)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
                print(2)
        elif classname.find('BatchNorm2d') != -1:
            if zero_gamma:
                init.constant_(m.weight.data, 0.0)
                init.constant_(m.bias.data, 0.0)
                print(3)
            else:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
                print(4)

    net.apply(init_func)


def show_network_param(net,  data_and_grad=False):
    """
    显示网络中的参数
    :param net: 网络结构
    :param data_and_grad: 是否显示数据和梯度
    :return:
    """
    print('print network parameters...\n')
    for name, param in net.named_parameters():
        print(name, param.size())
        if data_and_grad is True:
            print(name, param.data, param.grad)


def parameter_initial(net):
    """
    对网络结构的参数进行初始化
    :param net: 需要进行参数初始化的网络
    :return:
    """
    for name, param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.01)
            print(name, param.data)
        if 'bias' in name:
            init.constant_(param, val=0.0)
    return net


def conv2d(x, kernel):
    """
    二维卷积运算（互相关）
    :param x: 二维数据
    :param kernel: 二维卷积核
    :return: 经过卷积运算后的数据
    """
    # 获得卷积核的高度和宽度
    height, width = kernel.size()
    # 初始化经过卷积后的二维数组
    y = torch.zeros(size=(x.size(0) - height + 1, x.size(1) - width + 1))
    # print('卷积后的形状：{}'.format(y.size()))
    for i in range(y.size(0)):
        for j in range(y.size(1)):
            # 进行卷积运算，并更新
            y[i, j] = (x[i:height+i, j:width+j]*kernel).sum()
    return y


class Conv2D(nn.Module):
    """
    自定义二维卷积层，包括权重和偏差参数
    """
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
        print(self.weight, self.bias)

    def forward(self, x):
        return conv2d(x, self.weight) + self.bias


def simple_example():
    """
    简单实现卷积的前向传播和反向传播
    :return:
    """
    x = torch.tensor([[1, 1, 1, 1, 1], [-1, 0, -3, 0, 1],
                      [2, 1, 1, -1, 0], [0, -1, 1, 2, 1],
                      [1, 2, 1, 1, 1]])
    conv = Conv2D(kernel_size=(3, 3))

    step = 50
    lr = 0.01
    y = torch.ones(3, 3)
    y[:, 1:3] = 0
    print(y)

    for i in range(step):
        y_pred = conv(x.float())
        loss = ((y - y_pred)**2).sum()
        loss.backward()

        # 梯度下降
        conv.weight.data = conv.weight.data - lr*conv.weight.grad
        conv.bias.data = conv.bias.data - lr*conv.bias.grad

        # 梯度清0
        conv.weight.grad.fill_(0)
        conv.bias.grad.fill_(0)
        print('{},{}'.format(i, loss))

if __name__ == '__main__':
    # net = resnet50()
    # classname = net.__class__.__name__
    # print(classname.find('BatchNorm2d'))
    # print(hasattr(net, 'bn2'))
    # init_weight(net, zero_gamma=True)
    # x = torch.tensor([[1, 1, 1, 1, 1], [-1, 0, -3, 0, 1],
    #                   [2, 1, 1, -1, 0], [0, -1, 1, 2, 1],
    #                   [1, 2, 1, 1, 1]])
    # kernel = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    # y = conv2d(x, kernel)
    # print(y)
    # print(x.float())
    # conv = Conv2D(kernel_size=(3,3))
    # print(conv(x.float()))
    # x = torch.ones(size=(6, 8))
    # x[:, 2:6] = 0
    # print(x)
    # k = torch.tensor([[1, -1]])
    # y = conv2d(x, k.float())
    # print(y)
    # simple_example()
    x = torch.randn(1, 3, 12, 12)
    y = nn.Conv2d(3, 4, kernel_size=1)
    y = y(x)
    print(y.size())





