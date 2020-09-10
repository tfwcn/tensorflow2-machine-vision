import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import random
import json
import argparse
import random
import xml.dom.minidom as xml

sys.path.append(os.getcwd())
import ai_api.ai_models.utils.image_helper as ImageHelper
import ai_api.ai_models.utils.file_helper as FileHelper
import ai_api.ai_models.yolo_v4.data_helper_xray as DataHelper
from ai_api.ai_models.yolo_v4.model2 import ObjectDetectionModel

# 把模型的变量分布在哪个GPU上给打印出来
# tf.debugging.set_log_device_placement(True)

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default='labels')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
args = parser.parse_args()

# 下标转名称
classes_name = {
    0: 'knife',
    1: 'scissors',
    2: 'lighter',
    3: 'zippooil',
    4: 'pressure',
    5: 'slingshot',
    6: 'handcuffs',
    7: 'nailpolish',
    8: 'powerbank',
    9: 'firecrackers',
    }
# 名称转下标
classes_index={}
for key,val in classes_name.items():
    classes_index[val]=key

classes_num = len(classes_name)

layers_size = np.int32([[13, 13], [26, 26], [52, 52]])
anchors_wh = np.int32([
    [[210, 76], [128, 128], [171, 239]],
    [[81, 77], [128, 49], [58, 132]],
    [[27, 29], [65, 32], [36, 69]],
])
image_size = np.int32([416, 416])

class DataGenerator():
    def __init__(self, file_list, batch_size):
        self.file_list = file_list
        self.batch_size = batch_size
    
    def generate(self):# 记录存在分类
        class_list = set()
        # 图片对于分类列表
        image_class_list = {}
        # 读取素材标签
        for file_path in self.file_list:
            # 读取xml文件
            xml_path = os.path.join(os.path.dirname(file_path), 'XML', os.path.basename(file_path)[:-4] + '.xml')
            # print('xml_path:', xml_path)
            xml_data = xml.parse(xml_path)
            collection = xml_data.documentElement
            object_list = collection.getElementsByTagName("object")
            # 读取目标框信息
            image_class_list[file_path]=set()
            for object_item in object_list:
                shapes_label = object_item.getElementsByTagName("name")[0].childNodes[0].data
                if shapes_label not in classes_index:
                    print("标签异常：", file_path)
                    continue
                class_list.add(classes_index[shapes_label])
                image_class_list[file_path].add(classes_index[shapes_label])
            image_class_list[file_path]=list(image_class_list[file_path])
        class_list=list(class_list)
        print('图片标签：', class_list)

        X = []
        Y = np.zeros((0,6), dtype=np.float32)
        # Y1 = []
        # Y2 = []
        # Y3 = []
        class_index = 0
        while True:
            # 打乱列表顺序
            random_list = np.array(self.file_list)
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
                
                # 读取xml文件
                xml_path = os.path.join(os.path.dirname(file_path), 'XML', os.path.basename(file_path)[:-4] + '.xml')
                # print('xml_path:', xml_path)
                xml_data = xml.parse(xml_path)
                collection = xml_data.documentElement
                object_list = collection.getElementsByTagName("object")
                # 图片路径
                image_path = file_path
                # 图片文件名
                image_name = os.path.basename(image_path)
                # 读取目标框信息
                boxes = []
                classes = []
                for object_item in object_list:
                    # 原始点列表
                    object_bndbox = object_item.getElementsByTagName("bndbox")[0]
                    point_x1 = float(object_bndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
                    point_y1 = float(object_bndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
                    point_x2 = float(object_bndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
                    point_y2 = float(object_bndbox.getElementsByTagName("ymax")[0].childNodes[0].data)
                    # 补全4角坐标，用于透视变换后，目标框纠正
                    boxes.append([point_x1, point_y1, point_x2, point_y2, point_x1, point_y2, point_x2, point_y1])
                    shapes_label = object_item.getElementsByTagName("name")[0].childNodes[0].data
                    if shapes_label not in classes_index:
                        print("标签异常：", file_path)
                        continue
                    classes.append([classes_index[shapes_label]])
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
                # 随机变换图片
                # random_img = img
                random_img, boxes = DataHelper.GetRandomImage(img, points=boxes)
                # # 读取json文件
                # bg_json_path = random_list[int(random.random() * (len(random_list) - 1))]
                # with open(bg_json_path, 'r', encoding='utf-8') as f:
                #     bg_json_data = json.load(f)
                # # json文件目录
                # bg_json_dir = os.path.dirname(bg_json_path)
                # # 图片路径
                # bg_image_path = os.path.join(
                #     bg_json_dir, bg_json_data['imagePath'].replace('\\', '/'))
                # # 读取反光图片
                # bg_img = ImageHelper.fileToOpencvImage(bg_image_path)
                # # 随机反光
                # random_img = ImageHelper.opencvReflective(random_img, bg_img, 0.85 + (0.15* random.random()))
                # 缩放图片,最后再缩放，尽量保留信息
                random_img, boxes, _ = ImageHelper.opencvProportionalResize(
                    random_img, image_size, points=boxes, bg_color=None, bg_mode=None)
                # 转换boxes成框列表
                # boxes = boxes.reshape((-1, 8))
                # 将4角坐标转换成矩形坐标
                boxes = np.reshape(boxes, (-1, 4, 2))
                boxes_min = np.min(boxes, axis=1)
                boxes_max = np.max(boxes, axis=1)
                boxes = np.concatenate([boxes_min, boxes_max], axis=-1)
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
                # 显示图片
                # show_img, show_boxes, _ = ImageHelper.opencvProportionalResize(
                #     random_img, image_size * 2, points=boxes.reshape((-1, 2)), bg_color=None, bg_mode=None)
                # show_boxes = show_boxes.reshape((-1, 4))
                # show_img = ImageHelper.opencvDrowBoxes(show_img, 
                #     boxes=show_boxes, classes_id=classes.reshape([-1,]), classes_name=classes_name)
                # ImageHelper.showOpencvImage(show_img)
                # ImageHelper.opencvImageToFile('test.jpg', random_img)
                # 最后输出图片
                random_img = cv2.cvtColor(random_img, cv2.COLOR_BGR2RGB)
                # 调整参数范围
                random_img = random_img.astype(np.float32)
                random_img = random_img / 255
                # print("读取文件：", file_path)
                # print('boxes：', boxes)
                # (boxes_size, 6)
                batch_indexes = np.float32([[len(X)] for _ in range(classes.shape[0])])
                # [batch_size*boxes_size, 1+4+1]
                target_data = np.concatenate([batch_indexes, boxes, classes], axis=-1)
                X.append(random_img)
                Y = np.concatenate([Y, target_data], axis=0)
                if len(X) == self.batch_size:
                    result_x = np.array(X)
                    result_y = Y
                    # print('generator:', result_x.shape, result_y.shape)
                    yield result_x, result_y
                    X = []
                    Y = np.zeros((0,6), dtype=np.float32)

def train():
    '''训练'''
    train_path = args.file_path
    model = ObjectDetectionModel(classes_num=classes_num, anchors_wh=anchors_wh, 
                                 image_size=image_size, layers_size=layers_size, model_path='./data/yolov4_xray_model.h5')
    file_list = FileHelper.ReadFileList(train_path, r'.jpg$')
    print('图片数：', len(file_list))
    # 训练参数
    batch_size = args.batch_size
    steps_per_epoch = 1000
    epochs = 500
    data_generator = DataGenerator(file_list, batch_size)
    # 数据预处理
    dataset = tf.data.Dataset.from_generator(data_generator.generate,(tf.float32, tf.float32), (tf.TensorShape([None,416,416,3]),tf.TensorShape([None,6])))
    def func1(x, y):
        return (x, model.GetTarget(y))
    dataset = dataset.map(func1)
    for x, y in dataset.take(2):
        print(x.shape, y[0].shape, y[1].shape, y[2].shape)
    model.Fit(dataset,steps_per_epoch, epochs, learning_rate=args.learning_rate)


def main():
    train()


if __name__ == '__main__':
    main()
