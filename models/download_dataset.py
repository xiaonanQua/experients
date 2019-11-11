from torchvision.datasets import ImageNet
import os

imagenet_path = '/home/team/Dataset/imagenet'
if os.path.exists(imagenet_path) is False:
    os.mkdir(imagenet_path)

ImageNet(root=imagenet_path, download=True)