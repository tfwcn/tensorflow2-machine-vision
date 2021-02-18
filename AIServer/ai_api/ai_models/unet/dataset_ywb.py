from typing import Tuple
import numpy as np
import tensorflow as tf
import random
import json
import cv2

import sys
import os
sys.path.append(os.getcwd())
import ai_api.ai_models.utils.image_helper as image_helper
import ai_api.ai_models.utils.file_helper as file_helper
from ai_api.ai_models.unet.soft_label import SoftLabel
import ai_api.ai_models.utils.image_helper as ImageHelper


class DataGenerator():
  def __init__(self, label_path, input_shape: Tuple[int, int]):
      self.label_path = label_path
      self.input_shape = input_shape

      # 加载标签
      self.LoadLabels()

  def _get_random_data(self, image, target_points=None):
    '''生成随机图片与标签，用于训练'''
    # 画矩形
    # cv2.rectangle(image, (20, 20), (380, 380), tuple(np.random.randint(0, 30, (3), dtype=np.int32)), thickness=8)
    # 变换图像
    random_offset_x = random.random()*90-45
    random_offset_y = random.random()*90-45
    random_angle_x = random.random()*60-30
    random_angle_y = random.random()*60-30
    random_angle_z = random.random()*40-20
    random_scale_x = random.random()*0.9+0.5
    random_scale_y = random_scale_x
    # random_offset_x = 0
    # random_offset_y = 0
    # random_angle_x = 0
    # random_angle_y = 0
    # random_scale = 1
    image, org, dst, perspective_points = image_helper.opencvPerspective(image, offset=(random_offset_x, random_offset_y, 0),
                                                    angle=(random_angle_x, random_angle_y, random_angle_z), scale=(random_scale_x, random_scale_y, 1),
                                                    points=target_points, bg_color=None, bg_mode=None)
    # 增加模糊
    ksize = random.randint(0, 4)
    if ksize>0:
        image= image_helper.opencvBlur(image, (ksize, ksize))
    # 增加噪声
    # image = ImageHelper.opencvRandomLines(image, 8)
    image = image_helper.opencvNoise(image)
    # 颜色抖动
    image = image_helper.opencvRandomColor(image, random_h=False)

    # cv2.imwrite(path, image)
    random_img = image
    # # 最后输出图片
    # random_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # # 调整参数范围
    # random_img = random_img.astype(np.float32)
    # random_img = random_img / 255
    # print('random_img:', random_img.shape)
    # print('target_data:', target_data.shape)
    return random_img, perspective_points

  def LoadLabels(self):
    '''加载标签'''
    print('加载标签：', self.label_path)
    file_list = file_helper.ReadFileList(self.label_path, r'.json$')
    self.labels = []
    for file_path in file_list:
      # 读取json文件
      with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
      # json文件目录
      json_dir = os.path.dirname(file_path)
      # json文件名
      json_name = os.path.basename(file_path)
      # 图片路径
      image_path = os.path.join(json_dir, json_data['imagePath'].replace('\\', '/'))
      # 图片文件名
      image_name = os.path.basename(image_path)
      # 原始点列表
      json_points = np.float32(json_data['shapes'][0]['points'])
      if len(json_data['shapes'])!=1:
        continue
      # 点匹配
      point_center_x = (min(json_points[:, 0]) + max(json_points[:, 0])) / 2
      point_center_y = (min(json_points[:, 1]) + max(json_points[:, 1])) / 2
      pointLT=None
      pointLB=None
      pointRT=None
      pointRB=None
      for p in json_points:
        if p[0] < point_center_x and p[1] < point_center_y:
          pointLT = p
        elif p[0] > point_center_x and p[1] < point_center_y:
          pointRT = p
        elif p[0] < point_center_x and p[1] > point_center_y:
          pointLB = p
        elif p[0] > point_center_x and p[1] > point_center_y:
          pointRB = p
      points = np.float32([pointLT, pointLB, pointRT, pointRB])
      # print('points:', points)
    
      self.labels.append({
        'image_path': image_path,
        'points': points
        })
    self.labels_num = len(self.labels)
    print('已加载%d个标签' % (self.labels_num))
    
  def Generate(self):
    n = len(self.labels)
    i = 0
    clone_labels = self.labels.copy()
    while True:
      if i==0:
        random.shuffle(clone_labels)
      # 数据平均
      label = clone_labels[i]
      # print('image_path:', label['image_path'])
      # 读取图片
      image_path = label['image_path']
      img = image_helper.fileToOpencvImage(image_path)
      # print('imgType:', type(img))
      # width, height = ImageHelper.opencvGetImageSize(img)
      # print('imgSize:', width, height)
      # 获取随机变换图片及标签
      points = label['points']
      img, points = self._get_random_data(
          img, target_points=points)
      # 缩放图片
      img, points, _ = image_helper.opencvProportionalResize(
          img, self.input_shape, points=points, bg_color=None, bg_mode=None)
      # 最后输出图片
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      # 调整参数范围
      img =img.astype(np.float32)
      img = img / 255
      
      points = points/self.input_shape
      points = points[...,::-1]
      i = (i+1) % n
      if (points<0).any() or (points>1).any():
        continue
      yield img, points


def GetDataSet(label_path: str, batch_size: int, points_num: int, 
               input_size: Tuple[int, int], output_size: Tuple[int, int], kernel_size: Tuple[int, int]=(11, 11)):
    '''获取数据集'''
    data_generator = DataGenerator(label_path, input_size)
    # 数据预处理
    dataset = tf.data.Dataset.from_generator(
      data_generator.Generate, 
      (tf.float32, tf.float32),
      output_shapes=(tf.TensorShape([input_size[0], input_size[1], 3]), tf.TensorShape([points_num, 2])))
    soft_label = SoftLabel(image_size=output_size, points_num=points_num, kernel_size=kernel_size)
    dataset = dataset.map(lambda image, points: (image, soft_label.get_target(points*output_size)))
    dataset = dataset.batch(batch_size)
    for x, y in dataset.take(1):
      tf.print(tf.shape(x), tf.shape(y))
      x_one = x[0].numpy() * 255
      x_one.astype(np.int32)
      print(x_one.shape)
      ImageHelper.opencvImageToFile('a.jpg',x_one)
      for i in range(tf.shape(y)[-1]):
        y_one = y[0,...,i:i+1].numpy() * 255
        y_one.astype(np.int32)
        print(y_one.shape)
        ImageHelper.opencvImageToFile('b'+str(i)+'.jpg',y_one)
    return dataset, data_generator

