import argparse

import tensorflow as tf
import numpy as np
import cv2
import sys
import os

sys.path.append(os.getcwd())
from ai_api.ai_models.unsupervised_learning.model import YoloV3Model
from ai_api.ai_models.utils.load_object_detection_data import LoadClasses, LoadAnchors
import ai_api.ai_models.utils.image_helper as ImageHelper
from ai_api.ai_models.utils.file_helper import ReadFileList

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', default='./coco2017/train2017')
parser.add_argument('--outFile', default='./data/coco_train2017_teacher_labels.txt')
parser.add_argument('--modelPath', default='./data/unsupervised_learning_weights/teacher_weights/')
parser.add_argument('--classesFile', default='./data/coco_classes.txt')
parser.add_argument('--anchorsFile', default='./data/coco_anchors.txt')
args = parser.parse_args()

dataDir = args.dataDir
outFile = args.outFile
modelPath = args.modelPath
classesFile = args.classesFile
anchorsFile = args.anchorsFile


# 加载数据
image_wh = (416, 416)
# image_wh = (608, 608)
anchors = LoadAnchors(anchorsFile)
classes_name, classes_num = LoadClasses(classesFile)
# 构建模型
model = YoloV3Model(classes_num=classes_num, anchors=anchors, image_wh=image_wh)

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

file_list = ReadFileList(dataDir, pattern=r'.*\.jpg', select_sub_path=True, is_full_path=False)

with open(outFile, 'w', encoding='utf-8') as f:
  for image_path in file_list:
    # 图片路径
    full_path=os.path.join(dataDir, image_path)
    img_old = ImageHelper.fileToOpencvImage(full_path)
    # 缩放图片
    img, _, padding = ImageHelper.opencvProportionalResize(img_old, image_wh, bg_color=(0, 0, 0))

    # print('imgType:', type(img))
    width, height = ImageHelper.opencvGetImageSize(img_old)
    image_size_old = np.int32([width, height])
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
    y_boxes[:,[0,2]] = (y_boxes[:,[0,2]] * image_wh[0] - padding[2]) / (image_wh[0] - padding[2] - padding[3]) * image_size_old[0]
    y_boxes[:,[1,3]] = (y_boxes[:,[1,3]] * image_wh[1] - padding[0]) / (image_wh[1] - padding[0] - padding[1]) * image_size_old[1]
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
    # 图片文件名
    image_name = os.path.basename(image_path)
    # 读取目标框信息
    line=''
    for i in range(y_boxes.shape[0]):
      # 原始点列表
      point_x1 = float(y_boxes[i,0])
      point_y1 = float(y_boxes[i,1])
      point_x2 = float(y_boxes[i,2])
      point_y2 = float(y_boxes[i,3])
      tmp_label = classes_name[y_classes_id[i]]
      tmp_box=str(int(point_x1))+','+str(int(point_y1))+','+str(int(point_x2))+','+str(int(point_y2))
      line+=tmp_label+','+tmp_box+'|'
    if len(line)==0:
      continue
    # set转list
    line=image_path+'|'+line[:-1]
    line += '\n'
    print('label_text:', line)
    f.write(line)