#  -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as model
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import time
from sklearn.metrics import f1_score
from config.config import Config

# 训练、测试数据集路径
cfg = Config()
test_dataset_path = cfg.catdog_test_dir

# 类别数量
num_classes = 2

# 类别名称和设备
class_name = None  # 类别名称
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 平均值和标准差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 保存模型路径
model_path = '../checkpoints/catdog.pth'
result_file = 'catdog.w_result.txt'

# 对数据进行预处理
data_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# 加载数据集
image_datasets = ImageFolder(root=test_dataset_path, transform=data_preprocess)
class_name = image_datasets.classes
# print(image_datasets.imgs[:30])
# print(image_datasets.class_to_idx)
# print(image_datasets[0][0].size(), image_datasets[1][0].size())
# print(class_name)

# 数据加载器
train_data_loader = DataLoader(dataset=image_datasets)

# for index, data in enumerate(train_data_loader):
#     images, labels = data
#     print(images.size(), labels)
#     break

# 定义模型
# 获取ResNet50的网络结构
net = model.resnet50(pretrained=False, progress=True)

# # 重写网络的最后一层
fc_in_features = net.fc.in_features
net.fc = nn.Linear(fc_in_features, num_classes)

# 加载模型参数
net.load_state_dict(torch.load(model_path))
# 将网络结构放置在gpu上
net.to(device)

# net = resnet.ResNet(resnet.ResidualBlock, [3, 3, 3])
# net.to(device)

# 显示网络结构参数
# for name, child in net.named_children():
#     for name2, params in child.named_parameters():
#         print(name,name2)

# 训练的开始时间
since = time.time()

# 通过上下文管理器禁用梯度计算，减少运行内存
with torch.no_grad():
    # 迭代整个数据集
    for index, data in enumerate(train_data_loader):
        # 获取图像和标签数据
        images, labels = data
        # 若gpu存在，将图像和标签数据放入gpu上
        images = images.to(device)
        labels = labels.to(device)

        # 预测结果
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        # 微平均，宏平均
        micro_f1 = f1_score(labels, preds, average='micro')
        macro_f1 = f1_score(labels, preds, average='macro')

        # 将结果写入结果文件中
        with open(result_file, mode='w+') as file:
            for i in range(images.size(0)):
                content = '{} {}\n'.format(image_datasets[i][0], class_name[preds[i]])
                file.write(content)
        print('结果保存完成...')

print()
print('micro_f1_score:{}, macro_f1_score:{}'.format(micro_f1, macro_f1))







