import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models import resnet18
from torch.nn import


def init_weight(net, zero_gamma=False, init_type='normal', gain=0.02):
    def init_func(m):
        print('m:', m)
        classname = m.__class__.__name__
        print('class name',classname)
        if zero_gamma:
            if hasattr(m, 'bn2'):
                init

    net.apply(init_func)


if __name__ == '__main__':
    net = resnet18()
    init_weight(net)





