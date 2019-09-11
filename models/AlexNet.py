from models.BasicModule import BasicModule
import torch.nn as nn
import torch


class AlexNet(BasicModule):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.model_name = 'AlexNet'  # 模型名称
        # 初始化参数
        # self.input_width = input_width
        # self.input_height = input_height
        # self.input_channels = input_channels
        # self.num_classes = num_classes
        # self.learning_rate = learning_rate
        # self.momentum = momentum
        # self.keep_prob = keep_prob

        # 定义特征序列
        self.features = nn.Sequential(
            # 卷积层1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),  # (227->55)
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 55->27
            # 卷积层2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # 27->27
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27->13
            # 卷积层3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # 13->13
            nn.ReLU(inplace=True),
            # 卷积层4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # 13->13
            nn.ReLU(inplace=True),
            # 卷积层5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 13->13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 13->6
        )

        # 定义全连接序列
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256*6*6), out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self,x):
        """
        前向传播
        :return:
        """
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 将特征的结果形状相乘
        logit = self.classifier(x)
        return logit


if __name__ == '__main__':
    net = AlexNet()
