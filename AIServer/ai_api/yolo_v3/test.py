import numpy as np
import tensorflow as tf
import cv2
import time

import sys
import os
sys.path.append(os.getcwd())
from ai_api.yolo_v3.model import YoloV3Model
from ai_api.yolo_v3.dataset_coco import GetDataSet
from ai_api.utils.radam import RAdam
import ai_api.utils.image_helper as ImageHelper

@tf.function
def Predict(model, input_image, scores_thresh=0.5, iou_thresh=0.5):
    '''
    预测(编译模式)
    input_image:图片(416,416,3)
    return:两个指针值(2)
    '''
    # 预测
    start = time.process_time()
    output = model(input_image, training=False)
    tf.print('output[0]:', tf.math.reduce_max(output[0]), tf.math.reduce_min(output[0]), tf.shape(output[0]))
    tf.print('output[1]:', tf.math.reduce_max(output[1]), tf.math.reduce_min(output[1]), tf.shape(output[1]))
    tf.print('output[2]:', tf.math.reduce_max(output[2]), tf.math.reduce_min(output[2]), tf.shape(output[2]))
    selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence = model.GetNMSBoxes(
        output[0], output[1], output[2], scores_thresh, iou_thresh)
    end = time.process_time()
    tf.print('predict time: %f' % ((end - start)))
    return selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence


def test():
    '''训练'''
    # 加载数据
    anchors = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
    anchors = np.array(anchors).reshape(-1, 2)
    # 构建模型
    model = YoloV3Model(anchors_num=len(anchors)//3, classes_num=80, anchors=anchors, image_size=(416, 416))

    # 编译模型
    print('编译模型')
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=Yolov4Loss(anchors=anchors,classes_num=data_generator_train.classes_num)) # recompile to apply the change
    model.compile(optimizer=RAdam(lr=1e-4))

    # 日志
    log_dir = './data/'
    model_path = log_dir + 'trained_weights_final.h5'
    _ = model(tf.ones((1, 416, 416, 3)))
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print('加载模型:{}'.format(model_path))
    model.summary()


    img_old = ImageHelper.fileToOpencvImage('C:\\Users\\yongj\\Pictures\\e.jpg')
    # 图片大小
    image_size = np.int32([416, 416])
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
    y_boxes, y_classes_id, y_scores, y_classes, y_confidence = Predict(model, predict_img, scores_thresh=0.5, iou_thresh=0.3)
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
    # 画框
    classes_name = [
        'person',
        'bicycle',
        'car',
        'motorcycle',
        'airplane',
        'bus',
        'train',
        'truck',
        'boat',
        'traffic light',
        'fire hydrant',
        'stop sign',
        'parking meter',
        'bench',
        'bird',
        'cat',
        'dog',
        'horse',
        'sheep',
        'cow',
        'elephant',
        'bear',
        'zebra',
        'giraffe',
        'backpack',
        'umbrella',
        'handbag',
        'tie',
        'suitcase',
        'frisbee',
        'skis',
        'snowboard',
        'sports ball',
        'kite',
        'baseball bat',
        'baseball glove',
        'skateboard',
        'surfboard',
        'tennis racket',
        'bottle',
        'wine glass',
        'cup',
        'fork',
        'knife',
        'spoon',
        'bowl',
        'banana',
        'apple',
        'sandwich',
        'orange',
        'broccoli',
        'carrot',
        'hot dog',
        'pizza',
        'donut',
        'cake',
        'chair',
        'couch',
        'potted plant',
        'bed',
        'dining table',
        'toilet',
        'tv',
        'laptop',
        'mouse',
        'remote',
        'keyboard',
        'cell phone',
        'microwave',
        'oven',
        'toaster',
        'sink',
        'refrigerator',
        'book',
        'clock',
        'vase',
        'scissors',
        'teddy bear',
        'hair drier',
        'toothbrush',
    ]
    result_img = img_old.copy()
    for i in range(y_boxes.shape[0]):
        print('y_boxes:', y_boxes[i,:])
        print('y_classes_id:', y_classes_id[i])
        print('y_scores:', y_scores[i])
        print('y_classes:', y_classes[i])
        print('y_confidence:', y_confidence[i])
        cv2.rectangle(result_img, tuple(y_boxes[i,0:2]), tuple(y_boxes[i,2:4]), (0,0,255), thickness=1)
        cv2.putText(result_img, classes_name[y_classes_id[i]], tuple(y_boxes[i,0:2]), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 100, 0), 1)
        cv2.putText(result_img, str(y_scores[i]), tuple(y_boxes[i,0:2]+(0, 20)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 100), 1)
    ImageHelper.showOpencvImage(result_img)

def main():
    test()


if __name__ == '__main__':
    main()