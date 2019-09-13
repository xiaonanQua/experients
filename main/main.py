import os
import models
import torch
import torchnet.meter as meter
from torch.utils.data import DataLoader
from config.alexnet_config import AlexNetConf
from data.cat_dog_dataset import CatDog

cfg = AlexNetConf()


def train():
    # 模型实例化
    model = getattr(models, cfg.model)()
    # 若模型存在，则导入模型
    if os.path.exists(cfg.load_model_path):
        model.load(cfg.load_model_path)
    # 让模型使用gpu
    if cfg.use_gpu:
        model.cuda()

    # 获取数据集
    train_dataset = CatDog(root=cfg.catdog_train_dir, train=True, low_memory=False)
    val_dataset = CatDog(root=cfg.catdog_train_dir, low_memory=False)

    # 通过数据加载器加载数据
    train_data_loader = DataLoader(train_dataset, cfg.batch_size,
                                   shuffle=True, num_workers=cfg.num_workers,)
    val_data_loader = DataLoader(val_dataset, cfg.batch_size,
                                 shuffle=True, num_workers=cfg.num_workers)

    # 通过数据加载器加载数据（低内存版本）
    # train_data_loader = DataLoader(train_dataset, num_workers=cfg.num_workers)
    # val_data_loader = DataLoader(val_dataset, num_workers=cfg.num_workers)

    # 定义目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)

    # 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # 训练
    for epoch in range(cfg.epochs):
        # 汇总运行损失
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, start=0):
            # 获得训练图像和标签，data是一个列表[images,labels]
            images, labels = data
            # 使用gpu
            if cfg.use_gpu:
                images = images.cuda()
                labels = labels.cuda()

            # 将参数梯度设置为0
            optimizer.zero_grad()

            # 进行前向传播，反向传播，优化参数
            logit = model(images)
            loss = criterion(logit, labels)
            loss.backward()
            optimizer.step()  # 更新参数

            # 更新统计指标及可视化
            # loss_meter.add(loss)
            # confusion_matrix.add(logit, labels)

            # 输出统计值
            running_loss += loss
            print('epoch:{},step:{},loss:{}'.format(epoch + 1, i + 1, loss))

    print('训练完成...\n保存模型...')
    model.save()


if __name__ == '__main__':
    train()





