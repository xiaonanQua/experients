import argparse
import os

class Scheduler:
    def __init__(self, device_ids, model_path):




def run(image_path, num_device, name_ie, model_path, input_shape):
    # 扫描图片路径下的所有图片
    image_list = list()
    for image_name in os.listdir(image_path):
        image_list.append(os.path.join(image_path, image_name))

    # 初始化计划
    x = Scheduler(device_ids, model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', help='需要预处理图片的路径')
    parser.add_argument('--num_device', type=int, help='推理设备的数量', default=1)
    parser.add_argument('--name_ie', help='推理设备的名称')
    parser.add_argument('--model_path', help='已训练好模型的.xml文件路径', required=False, type=str)
    parser.add_argument('--input_shape', help='模型输入的形状', required=True, type=str)

    args = parser.parse_args()

    device_ids = [int(x) for x in range(args.num_device)]
    input_shapes = [int(x) for x in args.input_shape.split(',')]
    print(device_ids, input_shapes, args.image_path)

    run(args.image_path, args.num_device, args.name_ie, args.model_path, args.input_shape)