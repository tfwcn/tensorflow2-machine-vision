from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import base64
import io
import cv2
import random
import numpy as np
import math
import time
import os
import tensorflow as tf
from ai_api.ai_models.yolo_v4.model import YoloV4Model
import ai_api.ai_models.utils.image_helper as ImageHelper
from ai_api.ai_models.utils.load_object_detection_data import LoadClasses, LoadAnchors

modelPath = './data/yolo_v3_weights/tf2_weights/'
classesFile = './data/coco_classes.txt'
anchorsFile = './data/coco_anchors.txt'
# 加载数据
image_wh = (416, 416)
# image_wh = (608, 608)
anchors = LoadAnchors(anchorsFile)
classes_name, classes_num = LoadClasses(classesFile)
# 构建模型
model = YoloV4Model(classes_num=classes_num, anchors=anchors, image_wh=image_wh)

# 编译模型
print('编译模型')
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4))

# 加载模型
_ = model(tf.ones((1, image_wh[1], image_wh[0], 3)), training=False)
if os.path.exists(modelPath):
  last_model_path = tf.train.latest_checkpoint(modelPath)
  model.load_weights(last_model_path).expect_partial()
  print('加载模型:{}'.format(last_model_path))
  # model.summary()

def predict(request):
  '''识别'''
  request_data = json.loads(request.body)
  # print('request_data:', request_data)
  read = request_data['read']
  img_data = request_data['img_data'].split(',')[1]
  img_data = ImageHelper.base64ToBytes(img_data)
  img_old = ImageHelper.bytesToOpencvImage(img_data)
  # 图片大小
  image_size = np.int32([416, 416])
  # 缩放图片
  img, _, padding = ImageHelper.opencvProportionalResize(img_old, image_size, bg_color=(0, 0, 0))

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
  y_boxes, y_classes_id, y_scores, y_classes, y_confidence = model.Predict(predict_img, confidence_thresh=0.5, scores_thresh=0.2, iou_thresh=0.5)
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
  jsonObj = {
    'boxes': y_boxes.tolist(),
    'classes': y_classes.tolist(),
    'random_img': ImageHelper.bytesTobase64(ImageHelper.opencvImageToBytes(img)),
    'result_img': ImageHelper.bytesTobase64(ImageHelper.opencvImageToBytes(result_img)),
  }
  # print('jsonObj:',jsonObj)
  return HttpResponse(json.dumps(jsonObj), content_type="application/json")
