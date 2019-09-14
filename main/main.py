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
    # if os.path.exists(model.model_path):
    #     model.load(model.model_path)
    # 让模型使用gpu
    if cfg.use_gpu:
        model.cuda()

    # 获取数据集
    train_dataset = CatDog(root=cfg.catdog_train_dir, train=True, low_memory=False)
    val_dataset = CatDog(root=cfg.catdog_train_dir, low_memory=False)

    # 获取数据集（低内存版）
    # train_dataset = CatDog(root=cfg.catdog_train_dir, train=True)
    # val_dataset = CatDog(root=cfg.catdog_train_dir)

    # 通过数据加载器加载数据
    train_data_loader = DataLoader(train_dataset, cfg.batch_size,
                                   shuffle=True, num_workers=cfg.num_workers,)
    val_data_loader = DataLoader(val_dataset, batch_size=32,
                                 shuffle=True, num_workers=cfg.num_workers)

    # 通过数据加载器加载数据（低内存版本）
    # train_data_loader = DataLoader(train_dataset, num_workers=cfg.num_workers)
    # val_data_loader = DataLoader(val_dataset, num_workers=cfg.num_workers)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)

    # 训练
    for epoch in range(cfg.epochs):
        # 对每个批次的数据进行处理
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
            batch_loss = criterion(logit, labels)
            batch_loss.backward()
            optimizer.step()  # 更新参数

            # 返回预测样本中的最大值的索引
            predict = torch.argmax(logit, dim=1)
            # 计算预测样本类别与真实标签的正确值数量
            num_correct = (predict == labels).sum().item()
            # 计算准确率
            batch_accuracy = num_correct/labels.size(0)

            # 获得验证集结果
            valid_accuracy, valid_loss = val(model=model, dataloader=val_data_loader,
                                             criterion=criterion)

            # 输出训练结果
            print('epoch:{},step:{}, batch_loss:{}, batch_accuracy:{}, valid_loss:{}, valid_accuracy:{}'.
                  format(epoch + 1, i + 1, batch_loss, batch_accuracy, valid_loss, valid_accuracy))
    print('训练完成...\n保存模型...')
    return model.save()


def val(model, dataloader, criterion=None):
    """
    计算模型在验证集上的准确率等信息
    :param model: 定义的网络模型对象
    :param dataloader: 数据加载器
    :param criterion: 损失函数
    :return:
    """
    # 将训练模式切换成验证模式，因为在验证时对于使用dropout和BatchNorm不需要设置
    model.eval()
    model.cuda()
    # 定义预测样本正确数,整体损失函数值,平均损失值和样本数
    num_correct = 0
    total_loss = 0
    average_loss = 0
    num_total = 0

    # 进行样本验证
    for index, (images, labels) in enumerate(dataloader, start=0):
        # 使用gpu
        if cfg.use_gpu:
            images = images.cuda()
            labels = labels.cuda()
        # 获得神经网络的预测值
        logits = model(images)
        # 返回一个张量在特定维度上的最大值的索引
        predicted = torch.argmax(logits, dim=1)
        # 统计批次样本的数量
        num_total += labels.size(0)
        # 统计预测正确样本的值
        num_correct += (predicted == labels).sum().item()

        if criterion is not None:
            # 计算验证样本的损失值并加入整体损失中
            loss = criterion(logits, labels)
            total_loss += loss

    # 计算验证样本的准确率,平均损失
    accuracy = num_correct/num_total
    if criterion is not None:
        average_loss = total_loss/num_total
    # 将训练模式改成训练模式
    model.train()

    return accuracy, average_loss


def test(model_path, test_dataset_loader, train_dataset_loader=None):
    """
    使用测试集测试模型的准确率
    :param model_path: 加载模型的路径，检查点位置
    :param test_dataset_loader: 测试集加载器
    :param train_dataset_loader: 训练集加载器
    :return:
    """
    # 模型实例化
    model = getattr(models, cfg.model)()
    # 若检查点存在，则加载
    if os.path.exists(model_path):
        # 加载模型的
        model.load(model_path)
        # 将模型模式调整为验证模式，来约束dropout和batch Norm
        model.eval()
        # 定义预测样本正确数,整体损失函数值,平均损失值和样本数
        num_correct = 0
        total_loss = 0
        average_loss = 0
        num_total = 0

        # 进行样本验证
        for index, data in enumerate(test_dataset_loader):
            images, labels = data
            print(images.size(), labels.size())
            # 获得神经网络的预测值
            logits = model(images)
            # 返回一个张量在特定维度上的最大值的索引
            predicted = torch.argmax(logits, dim=1)
            # 统计批次样本的数量
            num_total += labels.size(0)
            # 统计预测正确样本的值
            num_correct += (predicted == labels).sum().item()

        # 计算验证样本的准确率,平均损失
        test_accuracy = num_correct / num_total

        # 进行样本验证
        for index, data in enumerate(train_dataset_loader):
            images, labels = data
            # 获得神经网络的预测值
            logits = model(images)
            # 返回一个张量在特定维度上的最大值的索引
            predicted = torch.argmax(logits, dim=1)
            # 统计批次样本的数量
            num_total += labels.size(0)
            # 统计预测正确样本的值
            num_correct += (predicted == labels).sum().item()

        # 计算验证样本的准确率,平均损失
        train_accuracy = num_correct / num_total

        print("test accuracy:{}, train accuracy:{}".format(test_accuracy, train_accuracy))


if __name__ == '__main__':
    model_path = train()
    # 测试
    model_path = '../checkpoints/AlexNet_0913_15:41:50.pth'
    print('model save path:{}'.format(model_path))

    # 加载训练集、测试集
    train_dataset = CatDog(root=cfg.catdog_train_dir, train=True)
    test_dataset = CatDog(root=cfg.catdog_test_dir, test=True)
    # 数据加载器
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size)
    test(model_path, test_dataset_loader=test_data_loader, train_dataset_loader=train_data_loader)







