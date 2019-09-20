import torch
from sklearn.metrics import f1_score, accuracy_score

preds = torch.randn(3, 4)
print(preds)
labels = torch.tensor([3, 1, 0])
preds = torch.argmax(preds, 1)
print(preds)
print(accuracy_score(labels, preds))
print(f1_score(labels, preds, average='micro'))
print(f1_score(labels, preds, average='macro'))