import torch
import torchvision
import time
import copy


def train(model, train_data_loader, criterion, optimizer, cfg, valid_data_loader=None,):
    """
    训练器
    :param model: 网络结构（模型）
    :param train_data_loader: 训练数据集加载器， 加载批次数据
    :param valid_data_loader: 验证数据集，加载整体数据
    :param criterion: 评估器，实例化评估对象
    :param optimizer: 优化器，实例化优化对象
    :param cfg: 配置文件
    :return:
    """
    # 记录训练的开始时间
    start_time = time.time()

    # 若模型存在，则导入模型
    # if os.path.exists(model.model_path):
    #     model.load(model.model_path)
    # 让模型使用gpu
    if cfg.use_gpu:
        model.cuda()

    print('_'*5 + '进行训练...' + '\n')
    # 训练
    for epoch in range(cfg.epochs):
        # 对每个批次的数据进行处理
        for i, data in enumerate(train_data_loader, start=0):
            # 获得训练图像和标签，data是一个列表[images,labels]
            images, labels = data
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)
            # # 使用gpu
            # if cfg.use_gpu:
            #     images = images.cuda()
            #     labels = labels.cuda()

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
            # valid_accuracy, valid_loss = val(model=model, dataloader=val_data_loader,
            #                                  criterion=criterion)

            # 输出训练结果
            print('epoch:{},step:{}, batch_loss:{}, batch_accuracy:{}, valid_loss:{}, valid_accuracy:{}'.
                  format(epoch + 1, i + 1, batch_loss, batch_accuracy, None, None))
    end_time = time.time() - start_time
    print('训练完成,训练时间：{:.0f}m {:.0f}s'.format(start_time//60, end_time%60))
    return model.save()  # 保存模型


def train_model(model, dataloaders, criterion, optimizer, scheduler, cfg):
    """
    训练模型，包含验证集
    :param model: 网络结构
    :param dataloaders: 数据加载器，包含训练集和验证集
    :param criterion: 评估器
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param cfg: 配置文件
    :return:
    """
    # 训练的开始时间
    since = time.time()

    # 深层复制模型的状态字典（模型的参数）， 定义最好的精确度
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 开始周期训练
    for epoch in range(cfg.epochs):
        print('Epoch {}/{}'.format(epoch, cfg.epochs - 1))
        print('-' * 10)

        # 每个周期要进行训练和验证两个任务
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为验证模式

            # 定义运行时训练的损失和正确率
            running_loss = 0.0
            running_corrects = 0
            # 统计数据数量
            num_data = 0

            # 迭代整个数据集
            for index, data in dataloaders[phase]:
                # 获取图像和标签数据
                images, labels = data
                # 若gpu存在，将图像和标签数据放入gpu上
                images = images.to(cfg.device)
                labels = labels.to(cfg.device)

                # 将梯度参数设置为0
                optimizer.zero_grad()

                # 前向传播
                # 追踪训练的历史,通过上下文管理器设置计算梯度的开关
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 仅仅在训练的情况下，进行反向传播，更新权重参数
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失,准确值,数据数量
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_data += images[0]

            epoch_loss = running_loss / num_data
            epoch_acc = running_corrects.double() / num_data

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 选出最好的模型参数
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 保存最好的模型参数
    model.state_dict = best_model_wts
    return model.save()  # 保存模型的路径


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

    # 将模型切换到cpu上
    device = torch.device('cpu')
    model.to(device)
    # model.cuda()
    # 定义预测样本正确数,整体损失函数值,平均损失值和样本数
    num_correct = 0
    total_loss = 0
    average_loss = 0
    num_total = 0

    # 进行样本验证
    for index, (images, labels) in enumerate(dataloader, start=0):
        # 使用gpu
        # if cfg.use_gpu:
        #     images = images.cuda()
        #     labels = labels.cuda()
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
