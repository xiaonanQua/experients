"""
实现将pytorch模型转成onnx
"""
import torch
import torchvision

input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()

torch.onnx.export(model, input, 'alexnet.onnx', verbose=True)
