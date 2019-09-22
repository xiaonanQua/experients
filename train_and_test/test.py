import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F


def test(model, model_path, test_data_loader, train_data_loader=None):
    """
    使用测试集测试模型的准确率
    :param model: 网络结构（模型）对象
    :param model_path: 加载模型的路径，检查点位置
    :param test_data_loader: 测试集加载器
    :param train_data_loader: 训练集加载器
    :return:
    """

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
        for index, data in enumerate(test_data_loader):
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
        for index, data in enumerate(train_data_loader):
            images, labels = data
            # 获得神经网络的预测值
            logits = model(images)
            preds = F.softmax(logits, dim=1)
            # 返回一个张量在特定维度上的最大值的索引
            predicted = torch.argmax(preds, dim=1)
            # 统计批次样本的数量
            num_total += labels.size(0)
            # 统计预测正确样本的值
            num_correct += (predicted == labels).sum().item()

        # 计算验证样本的准确率,平均损失
        test_accuracy = num_correct / num_total

        print("test accuracy:{}, train accuracy:{}".format(test_accuracy, test_accuracy))


def test_v2(model, model_path, test_data_loader, train_data_loader=None, class_name=None):
    """
    使用测试集测试模型的准确率
    :param model: 网络结构（模型）对象
    :param model_path: 加载模型的路径，检查点位置
    :param test_data_loader: 测试集加载器
    :param train_data_loader: 训练集加载器
    :param class_name: 类别名称
    :return:
    """

    # 若检查点存在，则加载
    if os.path.exists(model_path):
        # 加载模型的
        # model.load(model_path)
        model.load_state_dict(torch.load(model_path))
        # 将模型模式调整为验证模式，来约束dropout和batch Norm
        model.eval()
        # 定义预测样本正确数,整体损失函数值,平均损失值和样本数
        num_correct = 0
        total_loss = 0
        average_loss = 0
        num_total = 0

        # 进行样本验证
        for index, data in enumerate(test_data_loader):
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
        for index, data in enumerate(train_data_loader):
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


def visualize_model(model, dataloaders, cfg, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format([preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
