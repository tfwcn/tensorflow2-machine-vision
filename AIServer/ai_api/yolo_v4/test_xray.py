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
import ai_api.utils.image_helper as ImageHelper
import ai_api.utils.file_helper as FileHelper
import ai_api.yolo_v4.data_helper_xray as DataHelper
from ai_api.yolo_v4.model2 import ObjectDetectionModel

# 把模型的变量分布在哪个GPU上给打印出来
# tf.debugging.set_log_device_placement(True)

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default='labels')
parser.add_argument('--batch_size', default=3, type=int)
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
    [[116, 90], [156, 198], [373, 326]],
    [[30, 61], [62, 45], [59, 119]],
    [[10, 13], [16, 30], [33, 23]],
])
# 图片大小
image_size = np.int32([416, 416])

def test():
    '''识别'''
    train_path = args.file_path
    model = ObjectDetectionModel(classes_num=classes_num, anchors_wh=anchors_wh, 
                                 image_size=image_size, layers_size=layers_size, model_path='./data/yolov4_xray_model.h5')
    file_list = FileHelper.ReadFileList(train_path, r'.jpg$')
    print('图片数：', len(file_list))
    json_data = []
    for file_path in file_list:
        print('file_path:', file_path)
        img_old = ImageHelper.fileToOpencvImage(file_path)
        # 缩放图片
        img, _, padding = ImageHelper.opencvProportionalResize(img_old, image_size, bg_color=(255, 255, 255))

        # print('imgType:', type(img))
        width, height = ImageHelper.opencvGetImageSize(img_old)
        image_size_old = np.int32([width, height])
        print('imgSize:', width, height)
        # 最后输出图片
        predict_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 调整参数范围
        predict_img = predict_img.astype(np.float32)
        predict_img = predict_img / 255
        # 增加一个维度
        predict_img = np.expand_dims(predict_img, 0)
        y_boxes, y_classes_id, y_scores, y_classes, y_confidence = model.Predict(predict_img, scores_thresh=0.5, iou_thresh=0.5)
        # 结果转物体框列表
        y_boxes = y_boxes.numpy()
        y_classes_id = y_classes_id.numpy()
        y_scores = y_scores.numpy()
        y_classes = y_classes.numpy()
        y_confidence = y_confidence.numpy()
        y_boxes[:,[0,2]] = (y_boxes[:,[0,2]] * image_size[0] - padding[2]) / (image_size[0] - padding[2] - padding[3]) * image_size_old[0]
        y_boxes[:,[1,3]] = (y_boxes[:,[1,3]] * image_size[1] - padding[0]) / (image_size[1] - padding[0] - padding[1]) * image_size_old[1]
        # 截取框超出图片部分
        y_boxes[:,0][y_boxes[:,0]<0] = 0
        y_boxes[:,1][y_boxes[:,1]<0] = 0
        y_boxes[:,2][y_boxes[:,2]>image_size_old[0]] = image_size_old[0]
        y_boxes[:,3][y_boxes[:,3]>image_size_old[1]] = image_size_old[1]
        # 去掉无效框
        y_mask = np.logical_and(y_boxes[:,2]-y_boxes[:,0]>2, y_boxes[:,3]-y_boxes[:,1]>2)
        y_boxes = y_boxes[y_mask]
        y_classes_id = y_classes_id[y_mask]
        y_scores = y_scores[y_mask]
        y_classes = y_classes[y_mask]
        y_confidence = y_confidence[y_mask]
        y_boxes = y_boxes.astype(np.int32)
        print('y_boxes:', y_boxes.shape)
        print('y_classes_id:', y_classes_id.shape)
        print('y_scores:', y_scores.shape)
        print('y_classes:', y_classes.shape)
        print('y_confidence:', y_confidence.shape)
        knife = []
        scissors = []
        lighter = []
        zippooil = []
        pressure = []
        slingshot = []
        handcuffs = []
        nailpolish = []
        powerbank = []
        firecrackers = []
        for i in range(len(y_boxes)):
            if classes_name[y_classes_id[i]] == 'knife':
                knife.append(y_boxes[i].tolist()+[y_scores[i].tolist()])
            elif classes_name[y_classes_id[i]] == 'scissors':
                scissors.append(y_boxes[i].tolist()+[y_scores[i].tolist()])
            elif classes_name[y_classes_id[i]] == 'lighter':
                lighter.append(y_boxes[i].tolist()+[y_scores[i].tolist()])
            elif classes_name[y_classes_id[i]] == 'zippooil':
                zippooil.append(y_boxes[i].tolist()+[y_scores[i].tolist()])
            elif classes_name[y_classes_id[i]] == 'pressure':
                pressure.append(y_boxes[i].tolist()+[y_scores[i].tolist()])
            elif classes_name[y_classes_id[i]] == 'slingshot':
                slingshot.append(y_boxes[i].tolist()+[y_scores[i].tolist()])
            elif classes_name[y_classes_id[i]] == 'handcuffs':
                handcuffs.append(y_boxes[i].tolist()+[y_scores[i].tolist()])
            elif classes_name[y_classes_id[i]] == 'nailpolish':
                nailpolish.append(y_boxes[i].tolist()+[y_scores[i].tolist()])
            elif classes_name[y_classes_id[i]] == 'powerbank':
                powerbank.append(y_boxes[i].tolist()+[y_scores[i].tolist()])
            elif classes_name[y_classes_id[i]] == 'firecrackers':
                firecrackers.append(y_boxes[i].tolist()+[y_scores[i].tolist()])
        json_data.append([knife, scissors, lighter, zippooil, pressure, slingshot, handcuffs, nailpolish, powerbank, firecrackers])
    
    print('json_data:', json_data)
    with open('./data/xray.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_data,indent=4))


def main():
    test()


if __name__ == '__main__':
    main()
