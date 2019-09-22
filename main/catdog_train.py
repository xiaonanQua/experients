#  -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision.models as model
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
# import ResNet as resnet
import time
import copy
import os
from config.config import Config

# 训练、测试数据集路径
cfg = Config()
train_dataset_path = cfg.catdog_train_dir
test_dataset_path = cfg.catdog_test_dir

# 设置实验超参数
num_classes = 2
num_epoch = 50
batch_size = 32
learning_rate = 0.0001
learning_rate_decay = 0.95
learning_rate_decay_step = 7  # 学习率衰减周期
momentum = 0.9
keep_prob = 0.5

# 类别名称和设备
class_name = None  # 类别名称
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# 平均值和标准差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 保存模型路径
model_path = '../checkpoints/catdog.pth'

# 对数据进行预处理
data_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# 加载数据集
image_datasets = ImageFolder(root=train_dataset_path, transform=data_preprocess)
class_name = image_datasets.classes
# print(image_datasets.imgs[:30])
print(image_datasets.class_to_idx)
print(image_datasets[0][0].size(), image_datasets[1][0].size())
print(class_name)

# 数据加载器
train_data_loader = DataLoader(dataset=image_datasets, batch_size=batch_size, shuffle=True)

# for index, data in enumerate(train_data_loader):
#     images, labels = data
#     print(images.size(), labels)
#     break

# 定义模型
# 获取ResNet50的网络结构
net = model.resnet50(pretrained=True, progress=True)

# 重写网络的最后一层
fc_in_features = net.fc.in_features
net.fc = nn.Linear(fc_in_features, num_classes)

# 若模型已经训练存在，则加载参数继续进行训练
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path))
# 将网络结构放置在gpu上
net.to(device)

# net = resnet.ResNet(resnet.ResidualBlock, [3, 3, 3])
# net.to(device)

# 显示网络结构参数
# for name, child in net.named_children():
#     for name2, params in child.named_parameters():
#         print(name,name2)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=momentum)
# 通过一个因子gamma每7次进行一次学习率衰减
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=learning_rate_decay_step, gamma=0.1)

# ----------------------------进行训练

# 训练的开始时间
start_time = time.time()

# 深层复制模型的状态字典（模型的参数）， 定义最好的精确度
best_model_wts = copy.deepcopy(net.state_dict())
best_acc = 0.0

# 开始周期训练
for epoch in range(num_epoch):
    print('Epoch {}/{}'.format(epoch, num_epoch - 1))
    print('-' * 10)

    # 定义运行时训练的损失和正确率
    running_loss = 0.0
    running_corrects = 0
    running_corrects2 = 0.5
    # 统计数据数量
    num_data = 0

    # 迭代整个数据集
    for index, data in enumerate(train_data_loader):
        # 获取图像和标签数据
        images, labels = data
        # 若gpu存在，将图像和标签数据放入gpu上
        images = images.to(device)
        labels = labels.to(device)
        # print(index)
        # 将梯度参数设置为0
        optimizer.zero_grad()

        # 前向传播
        outputs = net(images)
        # print('预测的值的维度：{}'.format(outputs.size()))
        # 两中预测方法
        # _, preds = torch.max(outputs, 1)
        preds = torch.argmax(outputs, 1)
        loss = criterion(outputs, labels)

        # 仅仅在训练的情况下，进行反向传播，更新权重参数
        loss.backward()
        optimizer.step()

        # print('loss:{}, accuracy{}'.format(loss, torch.sum(preds == labels.data).double() / images.size(0)))
        # 统计损失,准确值,数据数量
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)
        running_corrects2 += accuracy_score(labels.cpu(), preds.cpu())  # 使用sklearn中的正确率函数
        num_data += images.size(0)

    # # 经过一定周期对学习率进行衰减
    # exp_lr_scheduler.step()

    # 计算每周期的损失函数和正确率
    epoch_loss = running_loss / num_data
    epoch_acc = running_corrects.double() / num_data
    print('Loss: {}, Acc: {}'.format(epoch_loss, epoch_acc))

    # 选出最好的模型参数
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(net.state_dict())
        # 保存最好的模型参数
        torch.save(best_model_wts, model_path)
        print('epoch:{}, update model...'.format(epoch))
    print()

# 训练结束时间
end_time = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(
    end_time // 60, end_time % 60))
print('Best val Acc: {:4f}'.format(best_acc))


