"""
实现VGGNet网络结构
"""
import torch.nn as nn


class VGGNet(nn.Module):

    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv_list = ((1, 3, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))