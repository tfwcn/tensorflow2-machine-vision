import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import random
import json
import argparse
import random

sys.path.append(os.getcwd())
import ai_api.utils.image_helper as ImageHelper
import ai_api.utils.file_helper as FileHelper
import ai_api.yolo_v4.data_helper as DataHelper
from ai_api.yolo_v4.model import Yolov4Model

# 把模型的变量分布在哪个GPU上给打印出来
# tf.debugging.set_log_device_placement(True)

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default='labels')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
args = parser.parse_args()

# 下标转名称
classes_name = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    }
# 名称转下标
classes_index={}
for key,val in classes_name.items():
    classes_index[val]=key

classes_num = len(classes_name)

layers_size = np.int32([[13, 13], [26, 26], [52, 52]])
anchors_wh = np.int32([
    [[116, 90], [156, 198], [373, 326]],
    [[30, 61], [62, 45], [59, 119]],
    [[10, 13], [16, 30], [33, 23]],
])
image_size = np.int32([416, 416])

def generator(file_list, batch_size):
    # 记录存在分类
    class_list = set()
    # 图片对于分类列表
    image_class_list = {}
    # 读取素材标签
    for file_path in file_list:
        # 读取json文件
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        # 读取目标框信息
        image_class_list[file_path]=set()
        for i in range(len(json_data['shapes'])):
            shapes_label = json_data['shapes'][i]["label"].split('_')
            if shapes_label[0] not in classes_index:
                print("标签异常：", file_path)
                continue
            class_list.add(classes_index[shapes_label[0]])
            image_class_list[file_path].add(classes_index[shapes_label[0]])
        image_class_list[file_path]=list(image_class_list[file_path])
    class_list=list(class_list)
    print('图片标签：', class_list)

    X = []
    Y = []
    class_index = 0
    while True:
        # 打乱列表顺序
        random_list = np.array(file_list)
        np.random.shuffle(random_list)
        for file_path in random_list:
            # 数据平均
            if len(class_list)>0:
                if image_class_list[file_path].count(class_list[class_index])==0:
                    continue
                
                # 找下一个类型
                if class_index < len(class_list)-1:
                    class_index += 1
                else:
                    class_index = 0
            # 读取json文件
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            # json文件目录
            json_dir = os.path.dirname(file_path)
            # json文件名
            json_name = os.path.basename(file_path)
            # 图片路径
            image_path = os.path.join(
                json_dir, json_data['imagePath'].replace('\\', '/'))
            # 图片文件名
            image_name = os.path.basename(image_path)
            # 读取目标框信息
            boxes = []
            classes = []
            for i in range(len(json_data['shapes'])):
                # 原始点列表
                shapes_points = np.float32(json_data['shapes'][i]['points'])
                point_x1 = min(shapes_points[:, 0])
                point_y1 = min(shapes_points[:, 1])
                point_x2 = max(shapes_points[:, 0])
                point_y2 = max(shapes_points[:, 1])
                boxes.append([point_x1, point_y1, point_x2, point_y2])
                shapes_label = json_data['shapes'][i]["label"].split('_')
                if shapes_label[0] not in classes_index:
                    print("标签异常：", file_path)
                    continue
                classes.append([classes_index[shapes_label[0]]])
            boxes = np.array(boxes, dtype=np.float32)
            classes = np.array(classes, dtype=np.float32)
            # print('boxes:', boxes.shape)
            # print('classes:', classes.shape)
            # 转换boxes成点列表
            boxes = boxes.reshape((-1,2))
            # 读取图片
            img = ImageHelper.fileToOpencvImage(image_path)
            if np.max(img)>255 or np.min(img)<0:
                print("图片异常", image_path)
                continue
            # 缩放图片
            img, boxes, _ = ImageHelper.opencvProportionalResize(
                img, image_size, points=boxes)
            # 随机变换图片
            random_img, boxes = DataHelper.GetRandomImage(img, points=boxes)
            # 读取json文件
            bg_json_path = random_list[int(random.random() * (len(random_list) - 1))]
            with open(bg_json_path, 'r', encoding='utf-8') as f:
                bg_json_data = json.load(f)
            # json文件目录
            bg_json_dir = os.path.dirname(bg_json_path)
            # 图片路径
            bg_image_path = os.path.join(
                bg_json_dir, bg_json_data['imagePath'].replace('\\', '/'))
            # 读取反光图片
            bg_img = ImageHelper.fileToOpencvImage(bg_image_path)
            # 随机反光
            random_img = ImageHelper.opencvReflective(random_img, bg_img, 0.85 + (0.15* random.random()))
            # 最后输出图片
            random_img = cv2.cvtColor(random_img, cv2.COLOR_BGR2RGB)
            # 调整参数范围
            random_img = random_img.astype(np.float32)
            random_img = random_img / 255
            # 转换boxes成框列表
            boxes = boxes.reshape((-1,4))
            # 截取框超出图片部分
            boxes[:,0][boxes[:,0]<0] = 0
            boxes[:,1][boxes[:,1]<0] = 0
            boxes[:,2][boxes[:,2]>image_size[0]] = image_size[0]
            boxes[:,3][boxes[:,3]>image_size[1]] = image_size[1]
            # 去掉无效框
            mask = np.logical_and(boxes[:,2]-boxes[:,0]>=2, boxes[:,3]-boxes[:,1]>=2)
            boxes = boxes[mask]
            classes = classes[mask]
            if classes.shape[0]<=0:
                # print('空数据')
                continue
            # print("读取文件：", file_path)
            # (boxes_size, 6)
            batch_indexes = np.float32([[len(X)] for _ in range(classes.shape[0])])
            target_data = np.concatenate([batch_indexes, boxes, classes], axis=-1)
            X.append(random_img)
            for i in range(target_data.shape[0]):
                Y.append(target_data[i, :])
            if len(X) == batch_size:
                result_x = np.array(X)
                result_y = np.array(Y)
                yield result_x, result_y
                X = []
                Y = []


def train():
    '''训练'''
    train_path = args.file_path
    model = Yolov4Model(classes_num=classes_num, anchors_wh=anchors_wh, 
                                 image_size=image_size, layers_size=layers_size)
    file_list = FileHelper.ReadFileList(train_path, r'.json$')
    print('图片数：', len(file_list))
    # 训练参数
    batch_size = args.batch_size
    steps_per_epoch = 1000
    epochs = 500
    model.FitGenerator(generator(file_list, batch_size),
                        steps_per_epoch, epochs, auto_save=True, learning_rate=args.learning_rate)


def main():
    train()


if __name__ == '__main__':
    main()
