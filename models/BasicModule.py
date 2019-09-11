"""
提供快速加载和保存模型的接口
"""
import torch.nn as nn
import torch as t
import time


class BasicModule(nn.Module):
    """
    封装了nn.Module,主要提供save和load两个方法
    """
    def __init__(self, name='model'):
        super(BasicModule, self).__init__()
        self.model_name = name

    def load(self, path):
        """
        可加载指定路径的模型
        :param path: 模型路径
        :return:
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用‘模型名字+时间’作为文件名
        :param name: 模型名称
        :return:
        """
        if name is None:
            prefix = '../checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')  # 按照特定格式将时间转化成str
        t.save(self.state_dict(), name)
        return name


if __name__ == '__main__':
    model = BasicModule('ss')
    print(model.save())
    model.load(model.save())