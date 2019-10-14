from config.config import Config


class CatDogConfig(Config):
    """
    CatDog数据集的配置文件，继承父类配置文件。
    """

    def __init__(self):
        super(CatDogConfig, self).__init__()
        # 图像宽度、高度、通道
        self.image_width = 70
        self.image_height = 70
        self.image_channels = 3
        # 类别数量
        self.num_classes = 2
        # 实验的超参数配置
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 0.01  # 原始是0.01
        self.learning_rate_decay = 0.95
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.keep_prob = 0.5

        # 模型的名称
        self.model_name = 'catdog'
        # 模型检查点地址
        self.checkpoints = '../checkpoints/' + self.model_name + '.pth'

        # cat和dog数据集根目录，训练集目录，测试集目录,验证集
        self.catdog_root_dir = self.root_dataset + 'cat_dog/'
        self.catdog_train_dir = self.root_dataset + 'cat_dog/train/'
        self.catdog_val_dir = self.root_dataset + 'cat_dog/val/'
        self.catdog_test_dir = self.root_dataset + 'cat_dog/test/'