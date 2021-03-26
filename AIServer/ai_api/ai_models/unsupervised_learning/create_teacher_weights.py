import numpy as np
import tensorflow as tf
import cv2

import sys
import os

sys.path.append(os.getcwd())
from ai_api.ai_models.unsupervised_learning.model import YoloV3Model
import ai_api.ai_models.utils.image_helper as ImageHelper
from ai_api.ai_models.utils.load_object_detection_data import LoadClasses, LoadAnchors

import argparse

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('--modelPath', default='./data/unsupervised_learning_weights/train_weights/')
parser.add_argument('--teacherModelPath', default='./data/unsupervised_learning_weights/teacher_weights/teacher_weights.ckpt')
parser.add_argument('--classesFile', default='./data/coco_classes.txt')
parser.add_argument('--anchorsFile', default='./data/coco_anchors.txt')
args = parser.parse_args()

modelPath = args.modelPath
teacherModelPath = args.teacherModelPath
classesFile = args.classesFile
anchorsFile = args.anchorsFile

def main():
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
  # 保存模型
  model.save_weights(teacherModelPath)
  print('保存模型:{}'.format(teacherModelPath))

if __name__ == '__main__':
  main()