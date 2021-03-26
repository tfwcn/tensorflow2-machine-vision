import numpy as np
import tensorflow as tf
import random

import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.utils.load_object_detection_data import LoadClasses, LoadLabels, LoadAnchors
from ai_api.ai_models.utils.tf_image_utils import LoadImage, PadOrCropToBoundingBox, ResizeWithPad
import ai_api.ai_models.utils.file_helper as FileHelper


class DataGenerator():
  def __init__(self,
    image_path,
    batch_size,
    image_wh=(416, 416),
    label_mean=True,
    image_random=True,
    jitter=.3,
    hue=.1,
    sat=1.5,
    val=1.5,
    flip=True):
    '''
    数据生成

    Args:
      anchors: (layers_num, anchors_num, 2)
      label_mean: 数据标签均衡
      image_random: 数据数据增强
    '''
    self.image_path = image_path
    self.batch_size = batch_size
    self.image_wh = image_wh
    self.label_mean = label_mean
    self.image_random = image_random
    # 随机参数
    self.jitter = jitter
    self.hue = hue
    self.sat = sat
    self.val = val
    self.flip = flip

    # 加载图片
    self.images_path = FileHelper.ReadFileList(self.image_path, r'.jpg$')
    self.images_num = len(self.images_path)
  
  @tf.function
  def Rand(self, a=0, b=1):
    return tf.random.uniform([], minval=a, maxval=b)

  @tf.function
  def GetRandomData(self, image_path):
    '''随机数据扩展
    jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True, flip=False
    '''
    img = LoadImage(image_path)
    w, h = self.image_wh
    img_hw = tf.shape(img)[0:2]
    ih = img_hw[0]
    iw = img_hw[1]
    # 不随机
    if not random:
      img, dy, dx, nh, nw, _ = ResizeWithPad(img, h, w)
      img = img / 255.
      return img

    jitter=self.jitter
    hue=self.hue
    sat=self.sat
    val=self.val
    flip=self.flip
    # resize image
    # 随机缩放
    new_ar = w/h * self.Rand(1-jitter,1+jitter)/self.Rand(1-jitter,1+jitter)
    scale = self.Rand(.25, 2)
    if new_ar < 1:
        nh = tf.math.floor(scale*h)
        nw = tf.math.floor(nh*new_ar)
    else:
        nw = tf.math.floor(scale*w)
        nh = tf.math.floor(nw/new_ar)
    img = tf.image.resize(
      img,
      (nh, nw),
      method=tf.image.ResizeMethod.BILINEAR,
      antialias=False
    )

    # place image
    # 偏移图片
    dx = tf.cast(tf.math.floor(self.Rand(0, w-nw)), tf.int32)
    dy = tf.cast(tf.math.floor(self.Rand(0, h-nh)), tf.int32)
    img = PadOrCropToBoundingBox(img, dy, dx, h, w)

    # flip image or not
    # 随机翻转图片
    if flip:
      flip = self.Rand()<.5
      if flip:
        img = tf.image.flip_left_right(img)

    # distort image
    # 颜色偏移
    ch = self.Rand(-hue, hue)
    cs = self.Rand(1, sat) if self.Rand()<.5 else 1/self.Rand(1, sat)
    cv = self.Rand(1, val) if self.Rand()<.5 else 1/self.Rand(1, val)
    x = tf.image.rgb_to_hsv(img/255.)
    x_h = x[..., 0:1] + ch
    x_h = tf.where(x_h>1.0, x_h-1, x_h)
    x_h = tf.where(x_h<0.0, x_h+1, x_h)
    x_s = x[..., 1:2] * cs
    x_v = x[..., 2:3] * cv
    x = tf.concat([x_h, x_s, x_v], axis=-1)
    x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0)
    img = tf.image.hsv_to_rgb(x) # numpy array, 0 to 1
    return img
  
  @tf.function
  def GetRandomData2(self, image_path):
    img_q = self.GetRandomData(image_path)
    img_k = self.GetRandomData(image_path)
    return img_q, img_k

  def Generate(self):
    i = 0
    clone_images_path = self.images_path.copy()
    while True:
      if i==0:
        random.shuffle(clone_images_path)
      image_path = clone_images_path[i]
      # print('image_path:', label['image_path'])
      i = (i+1) % self.images_num
      # tf.print('boxes:', tf.shape(boxes))
      yield image_path

  def GetDataSet(self):
    '''获取数据集'''
    # 数据预处理
    dataset = tf.data.Dataset.from_generator(self.Generate,
      (tf.string),
      (tf.TensorShape([])))
    dataset = dataset.map(self.GetRandomData2)
    # for x_q, x_k in dataset.take(1):
    #   tf.print(tf.shape(x_q), tf.shape(x_k))
    #   # ImageHelper.opencvImageToFile('a.jpg', tf.cast(image*255.0, tf.int32).numpy())
    #   # tf.print(boxes)
    dataset = dataset.batch(self.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # for x_q, x_k in dataset.take(1):
    #   tf.print(tf.shape(x_q), tf.shape(x_k))
    return dataset

def main():
  anchors = LoadAnchors('./data/coco_anchors.txt')
  data_generator = DataGenerator(image_path='Z:/Labels/coco2017/train2017',
    label_path='.\\data\\coco_train2017_labels.txt',
    classes_path='.\\data\\coco_classes.txt', batch_size=3, anchors=anchors,
    image_wh=(416, 416), label_mean=True, flip=True)
  dataset = data_generator.GetDataSet()
  # for x, y in dataset.take(1):
  #   print(x.shape, y[0].shape, y[1].shape, y[2].shape)
  

if __name__ == '__main__':
  main()
