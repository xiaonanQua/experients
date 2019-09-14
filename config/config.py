"""
保存实验的设置
"""
import os
import warnings
import torch


class Config(object):
    """
    配置实验参数
    """
    def __init__(self):
        # -----------------------------------------------------------文件目录配置
        # 数据集根目录、项目根目录、训练数据保存目录（实验室）
        self.root_dataset = '/home/team/xiaonan/Dataset/'
        self.root_project = '/home/team/xiaonan/personal_experiments/'
        self.root_data_save = '/home/team/xiaonan/data_save/'
        # 数据集根目录、项目根目录、训练数据保存目录（本机）
        # self.root_dataset = '/home/xiaonan/Dataset/'
        # self.root_project = '/home/xiaonan/personal_experiments/'
        # self.root_data_save = '/home/xiaonan/data_save/'


        # cifar-10数据集目录、文件名称
        self.cifar_10_dir = self.root_dataset + 'cifar-10/'
        self.cifar_file_name = {'meta': 'batches.meta',
                                'train': ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'],
                                'test': 'test_batch'}

        # svhn数据集目录、文件名称
        self.svhn_dir = self.root_dataset + 'svhn/'
        self.svhn_file_name = ['train_32.mat', 'test_32.mat', 'extra_32.mat']

        # mnist数据集目录,文件名称
        self.mnist_dir = self.root_dataset + 'mnist/'
        self.mnist_file_name = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                                't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

        # cat和dog数据集目录，训练集目录，测试集目录
        self.catdog_train_dir = self.root_dataset + 'cat_dog/train/'
        self.catdog_test_dir = self.root_dataset + 'cat_dog/test/'

        # 模型保存目录、日志文件保存目录、实验结果保存目录
        self.model_dir = self.root_data_save + 'model_train/'
        self.log_dir = self.root_data_save + 'log/'
        self.results = self.root_data_save + 'results/'

        # 初始化文件夹
        self._init()

        # ---------------------------------------------------实验参数配置
        # visdom环境
        self.env = 'default'
        # 图像宽度、高度、通道
        self.image_width = None
        self.image_height = None
        self.image_channels = None
        # 类别数量
        self.num_classes = None
        # 实验的超参数配置
        self.epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.learning_rate_decay = None
        self.weight_decay = None
        self.momentum = None
        self.keep_prob = None

        self.model = None  # 使用模型，名字必须和models/__init__.py中的名字一致
        self.load_model_path = None  # 模型检查点地址
        if torch.cuda.is_available():
            self.use_gpu = True  # 使用gpu
        else:
            self.use_gpu = False
        self.num_workers = 4  # 使用的线程数，用于加速文件的读取
        self.print_rate = 20  # 每批次打印信息的频率
        self.debug_file = '/tmp/debug'  # debug文件
        self.result_file = 'result.csv'  # 结果文件

    def _init(self):
        # 若文件夹不存在，则创建
        if os.path.exists(self.root_data_save) is False:
            os.mkdir(self.root_data_save)
        if os.path.exists(self.model_dir) is False:
            os.mkdir(self.model_dir)
        if os.path.exists(self.log_dir) is False:
            os.mkdir(self.log_dir)
        if os.path.exists(self.results) is False:
            os.mkdir(self.results)

    def update(self, kwargs):
        """
        根据字典kwargs更新config参数
        :param kwargs: 更新的参数，字典形式
        :return:
        """
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning:config has not attribut %s"%k)
            setattr(self, k, v)

        # 打印配置信息
        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


if __name__ == '__main__':
    cfg = Config()
    new_config = {'epochs':10}
    cfg.update(new_config)
    print(cfg.epochs)
    print(cfg.use_gpu)
