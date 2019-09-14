import torch

a = torch.randn(3, 10, 11, 12)
print(a)
b = torch.argmax(a, 1)
c = torch.max(a, 1)
print((b == c).sum())