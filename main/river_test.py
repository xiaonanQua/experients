#  -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torch.nn.functional as F

# 训练、测试数据集路径
test_dataset_path = '/home/data/V1.0/test/'

# 类别数量
num_classes = 4

# 类别名称和设备
class_name = ['garbage', 'health', 'others', 'waterpollute'] # 类别名称
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 平均值和标准差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 保存模型路径
model_path = '/root/notebook/model/river.pth'
result_file = '/root/notebook/river_result.txt'

# 文件列表
file_list = os.listdir(test_dataset_path)

# 对数据进行预处理
data_preprocess = transforms.Compose([
    # transforms.Resize(size=(256, 256)),
    # transforms.CenterCrop(size=(112, 112)),
    transforms.Resize(size=(112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


class RiverData(Dataset):
    """
    继承父类数据集
    """
    def __init__(self, root, transform):
        super(RiverData, self).__init__()
        # 文件路径列表
        self.images_path_list = [os.path.join(root, image_name) for image_name in file_list]
        print(len(self.images_path_list))
        self.image_preprocess = transform

    def __getitem__(self, index):
        # 获取单个图像路径
        image_path = self.images_path_list[index]
        # 读取图像
        image = Image.open(image_path)
        image = image.convert('RGB')
        # 预处理图像
        image = self.image_preprocess(image)
        return image

    def __len__(self):
        return len(self.images_path_list)

# 加载数据集
image_datasets = RiverData(root=test_dataset_path, transform=data_preprocess)

# 数据加载器
test_data_loader = DataLoader(dataset=image_datasets, batch_size=32)


# 定义模型
# 获取ResNet50的网络结构
net = model.resnet18(pretrained=False)

# # 重写网络的最后一层
fc_in_features = net.fc.in_features
net.fc = nn.Linear(fc_in_features, num_classes)

# 加载模型参数
net.load_state_dict(torch.load(model_path))
# 通过上下文管理器禁用梯度计算，减少运行内存
with torch.no_grad():
    j = 0
    with open(result_file, 'w+') as file:
        # 迭代整个数据集
        for images in test_data_loader:
            # 预测结果
            outputs = net(images)
            outputs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, 1)
            predict_result = preds.numpy().tolist()

            # 将结果写入结果文件中
            for i in range(images.size(0)):
                content = '{} {}\n'.format(file_list[j], class_name[predict_result[i]])
                file.write(content)
                j = j + 1
            print('{}/{}'.format(j, len(file_list)))








