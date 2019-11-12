from config.config import Config
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import numpy as np
import os


class CatDogConfig(Config):
    """
    CatDog数据集的配置文件，继承父类配置文件。
    """

    def __init__(self):
        super(CatDogConfig, self).__init__()
        # 图像宽度、高度、通道
        self.image_width = 32
        self.image_height = 32
        self.image_channels = 3
        # 类别数量,类别名称
        self.num_classes = 10
        self.name_classes = ['cat', 'dog']
        # 实验的超参数配置
        self.epochs = 200
        self.batch_size = 128
        self.learning_rate = 0.1  # 原始是0.01
        self.linear_scale_lr = 0.1*(self.batch_size/256)
        self.lr_decay_step = 50
        self.lr_warmup_type = ['step', 'epoch', None]
        self.lr_warmup_step = 5
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.keep_prob = 0.5

        # 模型的名称
        self.model_name = 'cat_dog_resnet18'
        # 模型检查点地址；日志保存路径
        self.checkpoints = self.model_dir + self.model_name + '.pth'
        self.log_dir = self.log_dir + self.model_name
        if os.path.exists(self.log_dir) is False:
            os.mkdir(self.log_dir)
        # 梯度累积
        self.grad_accuml = False
        self.batch_accumulate_size = 4

        # 训练集、验证集、测试集预处理操作
        self.train_preprocess = transforms.Compose([
            # transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.valid_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.test_preprocess = transforms.Compose([
            # transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # cat dog数据集目录、文件名称
        self.cat_dog_dir = self.root_dataset + 'cat_dog/'
        self.cat_dog_train = self.cat_dog_dir+'train/'
        self.cat_dog_test = self.cat_dog_dir+'test/'

    def dataset_loader(self, root, train=True, shuffle=True, data_preprocess=None, valid_coef=None):
        """
        加载cat_dog数据集
        :param root: 数据存在路径
        :param train: True:则只获得训练集，否则获取测试集
        :param shuffle: True:对小批次数据打乱顺序
        :param data_preprocess: 数据预处理操作,若没有指定，则使用默认的处理操作。若进行验证集的划分，则给出训练集和验证集的字典。
        :param valid_coef: 划分验证集的比例
        :return: 数据集加载器
        """
        # 若预处理为空，则使用默认的
        if data_preprocess is None and train is True:
            data_preprocess = self.train_preprocess
        elif data_preprocess is None and train is True and valid_coef is not None:
            data_preprocess = {'train': self.train_preprocess, 'valid': self.valid_preprocess}
        elif data_preprocess is None and train is False:
            data_preprocess = self.test_preprocess

        # 如果验证比例不为空，则进行验证集的划分
        if valid_coef is not None:
            # 获取不同预处理的训练集和验证集
            train_dataset = ImageFolder(root=root, transform=data_preprocess[0])
            valid_dataset = ImageFolder(root=root, transform=data_preprocess[1])
            # 获取训练集的长度
            num_samples = len(os.listdir(self.cat_dog_train))

            # 计算样本数量的下标；计算划分出训练集的长度
            indices = list(range(num_samples))
            split = num_samples - int(np.floor(valid_coef*num_samples))

            # True：打乱索引下标的顺序
            if shuffle:
                np.random.seed(self.random_seed)
                np.random.shuffle(indices)

            # 划分出训练和验证集的采样
            train_idx, valid_idx = indices[:split], indices[split:]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            # 获取训练加载器和验证加载器(若定制特定的采样操作，则不能使用shuffle)
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers, sampler=train_sampler)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers, sampler=valid_sampler)
            return (train_loader, valid_loader)
        else:
            dataset = ImageFolder(root=root, transform=data_preprocess)

        # 获得数据集加载器
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader



