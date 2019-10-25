import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from config.cifar10_config import Cifar10Config
from config.test_config import TestConfig
from train_and_test.train_and_valid import train_and_valid, train_and_valid_, test
from models.AlexNet import AlexNet
from utils.tools import vis


# ----------------配置数据--------------------------
# 配置实例化
cfg = Cifar10Config()
# cfg = TestConfig()

# 数据预处理
train_data_preprocess = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            # transforms.RandomResizedCrop(224, 224),
                                            transforms.ColorJitter(brightness=0.4, saturation=0.4,
                                                                   hue=0.4, contrast=0.4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.49139961, 0.48215843, 0.44653216],
                                                                 std=[0.24703216, 0.2434851 , 0.26158745])])
valid_data_preprocess = transforms.Compose([# transforms.Resize(256),
                                           # transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.49139961, 0.48215843, 0.44653216],
                                                                std=[0.24703216, 0.2434851 , 0.26158745])])

# 获取训练集、测试集的加载器
train_loader, valid_loader = cfg.dataset_loader(root=cfg.cifar_10_dir, train=True,
                                                data_preprocess=[train_data_preprocess, valid_data_preprocess],
                                                valid_coef=0.1)

test_loader = cfg.dataset_loader(root=cfg.cifar_10_dir, train=False, shuffle=False,
                                 data_preprocess=valid_data_preprocess)
# train_loader = cfg.dataset_loader_test(root=cfg.test_dir)

# ---------------构建网络--------------------------
# 构建网络结构
# net = resnet18()
# net = AlexNet(num_classes=cfg.num_classes)
net = resnet50()
# 重写网络最后一层
fc_in_features = net.fc.in_features  # 网络最后一层的输入通道
net.fc = nn.Linear(in_features=fc_in_features, out_features=cfg.num_classes)

# --------------进行训练-----------------
print('进行训练....')
train_and_valid_(net, criterion=nn.CrossEntropyLoss(),
                 optimizer=optim.SGD,
                 train_loader=train_loader,
                 valid_loader=valid_loader, cfg=cfg,
                 is_lr_warmup=False, is_lr_adjust=False)

# -------------进行测试-----------------
print('进行测试.....')
test_accs, confusion_mat = test(net, test_loader, cfg)

# -------------可视化-------------------
# print(test_accs, confusion_mat)
# vis(test_accs, confusion_mat, cfg.classes)

