import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models import resnet18, resnet50
from torch.nn import init


def init_weight(net, zero_gamma=False, init_type='normal', gain=0.02):
    def init_func(m):
        # print('m:', m)
        classname = m.__class__.__name__
        # print('class name',classname)
        # print(classname.find)
        if zero_gamma:
            if hasattr(m, 'bn2'):
                init.constant_(m.bn2.weight.data, 0.0)
                init.constant_(m.bn2.bias.data, 0.0)
                print(1)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
                print(2)
        elif classname.find('BatchNorm2d') != -1:
            if zero_gamma:
                init.constant_(m.weight.data, 0.0)
                init.constant_(m.bias.data, 0.0)
                print(3)
            else:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
                print(4)

    net.apply(init_func)


if __name__ == '__main__':
    net = resnet50()
    # classname = net.__class__.__name__
    # print(classname.find('BatchNorm2d'))
    # print(hasattr(net, 'bn2'))
    init_weight(net, zero_gamma=True)





