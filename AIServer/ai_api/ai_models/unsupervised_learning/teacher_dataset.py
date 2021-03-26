import numpy as np
import tensorflow as tf
import random
from PIL import Image, ImageFilter
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2

import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.utils.load_object_detection_data import LoadClasses, LoadLabels, LoadAnchors
from ai_api.ai_models.utils.tf_image_utils import LoadImage, PadOrCropToBoundingBox, ResizeWithPad
from ai_api.ai_models.utils.tf_iou_utils import GetIOU
from ai_api.ai_models.utils.file_helper import ReadFileList
import ai_api.ai_models.utils.image_helper as ImageHelper
from ai_api.ai_models.unsupervised_learning.model import YoloV3Model


class DataGenerator():
  def __init__(self,
    image_path,
    classes_path,
    batch_size,
    anchors,
    image_wh=(416, 416),
    model_path='./data/unsupervised_learning_weights/teacher_weights/',
    image_random=True,
    jitter=.3,
    hue=.1,
    sat=1.5,
    val=1.5,
    scale=(.25, 2),
    flip=True):
    '''
    数据生成

    Args:
      anchors: (layers_num, anchors_num, 2)
      label_mean: 数据标签均衡
      image_random: 数据数据增强
    '''
    self.image_path = image_path
    self.classes_path = classes_path
    self.batch_size = batch_size
    self.image_wh = image_wh
    self.anchors_wh = anchors
    self.model_path = model_path
    self.image_random = image_random
    # 随机参数
    self.jitter = jitter
    self.hue = hue
    self.sat = sat
    self.val = val
    self.scale = scale
    self.flip = flip

    self.layers_hw = [[self.image_wh[1] // i, self.image_wh[0] // i] for i in [32, 16, 8]]
    print('layers_hw:', self.layers_hw)

    # 加载类型
    self.classes, self.classes_num = LoadClasses(self.classes_path)
    # 读取图片路径
    self.image_list = ReadFileList(self.image_path, pattern=r'.*\.jpg', select_sub_path=True, is_full_path=True)
    self.image_num = len(self.image_list)
    # 初始化模型
    self.InitModel()

  def InitModel(self):
    # 构建模型
    self.model = YoloV3Model(classes_num=self.classes_num, anchors=self.anchors_wh, image_wh=self.image_wh)

    # 编译模型
    print('编译模型')
    self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4))

    # 加载模型
    _ = self.model(tf.ones((1, self.image_wh[1], self.image_wh[0], 3)), training=False)
    if os.path.exists(self.model_path):
      last_model_path = tf.train.latest_checkpoint(self.model_path)
      self.model.load_weights(last_model_path).expect_partial()
      print('加载模型:{}'.format(last_model_path))
    # self.model.summary()
  
  @tf.function
  def Rand(self, a=0, b=1):
    '''生成[a,b)随机数'''
    return tf.random.uniform([], minval=a, maxval=b)
    # return tf.random.uniform([])*(b-a) + a

  @tf.function
  def GetRandomImage(self, image_path):
    '''随机数据扩展
    jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True, flip=False
    '''
    img = LoadImage(image_path)
    w, h = self.image_wh
    img_hw = tf.shape(img)[0:2]
    ih = img_hw[0]
    iw = img_hw[1]

    jitter=self.jitter
    hue=self.hue
    sat=1.0
    val=self.val
    flip=self.flip
    # resize image
    # 随机缩放
    new_ar = w/h * self.Rand(1-jitter,1+jitter)/self.Rand(1-jitter,1+jitter)
    scale = self.Rand(*self.scale)
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
    cs = 1.0
    if sat!=1:
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
    
    # 增加一个维度
    predict_img = tf.expand_dims(img, 0)
    boxes, classes, _, _, _ = self.model.Predict(predict_img, confidence_thresh=self.Rand(0.3,0.5), scores_thresh=0.2, iou_thresh=0.5)

    
    # 调整box位置
    boxes = tf.reshape(boxes, (-1, 2, 2))
    # 缩放、偏移
    # boxes = boxes*(nw/tf.cast(iw, tf.float32),nh/tf.cast(ih, tf.float32)) + (dx,dy)
    dx_f = tf.cast(dx, tf.float32)
    dy_f = tf.cast(dy, tf.float32)
    w_f = tf.cast(w, tf.float32)
    h_f = tf.cast(h, tf.float32)
    boxes = (boxes * (w_f,h_f) - (dx_f,dy_f))*(tf.cast(iw, tf.float32)/nw,tf.cast(ih, tf.float32)/nh)
    # 裁剪
    boxes = tf.clip_by_value(boxes,
      clip_value_min=0.0,
      clip_value_max=(iw, ih))
    boxes = tf.reshape(boxes, (-1, 4))
    # 翻转
    if flip:
      boxes = tf.concat([tf.cast(iw, tf.float32) - boxes[..., 2:3], boxes[..., 1:2],
                         tf.cast(iw, tf.float32) - boxes[..., 0:1], boxes[..., 3:4]], axis=-1)
    # 过滤不及格box
    # tf.print('image_path:', image_path)
    # tf.print('boxes_num:', tf.shape(boxes)[0])
    # tf.print('boxes:', boxes)
    boxes_wh = boxes[..., 2:4] - boxes[..., 0:2]
    boxes_mask = tf.math.logical_and(boxes_wh[..., 0]>1, boxes_wh[..., 1]>1)
    boxes = tf.boolean_mask(boxes, boxes_mask)
    classes = tf.boolean_mask(classes, boxes_mask)
    # tf.print('boxes_num2:', tf.shape(boxes)[0])
    return image_path, classes, boxes

  @tf.function
  def GetRandomData(self, image_path, classes, boxes):
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
      # 调整box位置
      boxes = tf.reshape(boxes, (-1, 2, 2))
      # 缩放、偏移
      boxes = boxes*(nw/tf.cast(iw, tf.float32),nh/tf.cast(ih, tf.float32)) + (dx,dy)
      # 裁剪
      boxes = tf.clip_by_value(boxes,
        clip_value_min=0.0,
        clip_value_max=(w, h))
      boxes = tf.reshape(boxes, (-1, 4))
      # 过滤不及格box
      boxes_wh = boxes[..., 2:4] - boxes[..., 0:2]
      boxes_mask = tf.math.logical_and(tf.math.greater(boxes_wh[..., 0], 1), tf.math.greater(boxes_wh[..., 1], 1))
      boxes = tf.boolean_mask(boxes, boxes_mask)
      classes = tf.boolean_mask(classes, boxes_mask)
      return img, classes, boxes

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
    cs = 1.0
    if sat!=1:
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
    
    # 调整box位置
    boxes = tf.reshape(boxes, (-1, 2, 2))
    # 缩放、偏移
    boxes = boxes*(nw/tf.cast(iw, tf.float32),nh/tf.cast(ih, tf.float32)) + (dx,dy)
    # 裁剪
    boxes = tf.clip_by_value(boxes,
      clip_value_min=0.0,
      clip_value_max=(w, h))
    boxes = tf.reshape(boxes, (-1, 4))
    # 翻转
    if flip:
      boxes = tf.concat([w - boxes[..., 2:3], boxes[..., 1:2],
                         w - boxes[..., 0:1], boxes[..., 3:4]], axis=-1)
    # 过滤不及格box
    # tf.print('image_path:', image_path)
    # tf.print('boxes_num:', tf.shape(boxes)[0])
    # tf.print('boxes:', boxes)
    boxes_wh = boxes[..., 2:4] - boxes[..., 0:2]
    boxes_mask = tf.math.logical_and(boxes_wh[..., 0]>1, boxes_wh[..., 1]>1)
    boxes = tf.boolean_mask(boxes, boxes_mask)
    classes = tf.boolean_mask(classes, boxes_mask)
    # tf.print('boxes_num2:', tf.shape(boxes)[0])
    return img, classes, boxes

  @tf.function
  def GetTargets(self, img, classes, boxes):
    '''
    获取训练Target
    labels:(boxes_num, 6=(batch_index, x1, y1, x2, y2, class_index))
    '''
    layers_num = tf.shape(self.anchors_wh)[0]
    # 计算中心坐标，并换算成层坐标
    boxes_xy = (boxes[..., 2:4] + boxes[..., 0:2]) // 2
    boxes_wh = boxes[..., 2:4] - boxes[..., 0:2]
    # 归一化
    boxes_xy = boxes_xy / self.image_wh
    boxes_wh = boxes_wh / self.image_wh
    # 计算框(x1,y1,x2,y2),并把中心移到(0,0)
    # (boxes_num, 2)
    boxes_maxes = boxes_wh / 2.0
    boxes_mins = -boxes_maxes
    # (boxes_num, 4)
    boxes_mins_maxes = tf.concat([boxes_mins, boxes_maxes], axis=-1)
    # (boxes_num, 4) => (boxes_num, 1, 4)
    boxes_mins_maxes = tf.expand_dims(boxes_mins_maxes, axis=-2)

    # 计算anchors(x1,y1,x2,y2),并把中心移到(0,0)
    # (9, 2)
    anchors_wh = tf.reshape(self.anchors_wh, (-1, 2))
    # 计算iou
    # (9, 2)
    anchors_maxes = tf.cast(anchors_wh, tf.float32) / 2.0
    anchors_mins = -anchors_maxes
    # (9, 4)
    anchors_boxes = tf.concat([anchors_mins, anchors_maxes], axis=-1)
    # (9, 4) => (1, 9, 4)
    anchors_boxes = tf.expand_dims(anchors_boxes, axis=0)
    # (boxes_num, 9)
    iou = GetIOU(boxes_mins_maxes, anchors_boxes, iou_type='iou')
    # 最优anchor下标
    # (boxes_num, )
    anchors_idx = tf.cast(tf.argmax(iou, axis=-1), tf.int32)
    # tf.print('anchors_idx:', anchors_idx)
    # tf.print('anchors_idx:', anchors_idx.shape)

    anchors_num = tf.shape(self.anchors_wh)[1]
    layers_hw = tf.constant(self.layers_hw, dtype=tf.int32)
    boxes_num = tf.shape(boxes)[0]
    # tf.print('boxes_num:', boxes_num)
    # 更新值对应输出的下标
    target_indexes = tf.TensorArray(tf.int32, 0, dynamic_size=True)
    # 更新值对应输出的下标的值
    target_updates = tf.TensorArray(tf.float32, 0, dynamic_size=True)
    # 遍历boxes
    def boxes_foreach(boxes_index, target_indexes, target_updates):
      # 最优层下标
      layer_index = anchors_idx[boxes_index] // layers_num
      # tf.print('layer_index:', layer_index, layer_index.dtype)
      # tf.print('batch_index:', batch_index, batch_index.dtype)
      # 最优候选框下标
      anchor_index = anchors_idx[boxes_index] % layers_num
      # tf.print('anchor_index:', anchor_index, anchor_index.dtype)
      # tf.print('boxes_wh:', boxes_wh)
      layer_yx = tf.cast(tf.math.floor(boxes_xy[boxes_index][::-1] * tf.cast(layers_hw[layer_index], dtype=tf.float32)), dtype=tf.int32)
      # tf.print('layer_xy:', layer_xy, layer_index.dtype)
      # xy下标是反的，因为输入是高宽
      target_indexes = target_indexes.write(boxes_index, [layer_index, layer_yx[0], layer_yx[1], anchor_index])
      # 传入的是原始坐标数据
      target_update = tf.concat([boxes_xy[boxes_index], boxes_wh[boxes_index], [1], tf.one_hot(classes[boxes_index], self.classes_num)], axis=-1)
      # tf.print('target_update:', target_update, target_update.dtype)
      target_updates = target_updates.write(boxes_index, target_update)
      return boxes_index+1, target_indexes, target_updates
    
    _, target_indexes, target_updates = tf.while_loop(lambda x, *args: x<boxes_num, boxes_foreach, [0, target_indexes, target_updates])
    
    # if boxes_num == 0:
    #   target1 = tf.zeros((layers_hw[0, 0], layers_hw[0, 1], anchors_num, 5+self.classes_num), dtype=tf.float32)
    #   target2 = tf.zeros((layers_hw[1, 0], layers_hw[1, 1], anchors_num, 5+self.classes_num), dtype=tf.float32)
    #   target3 = tf.zeros((layers_hw[2, 0], layers_hw[2, 1], anchors_num, 5+self.classes_num), dtype=tf.float32)
    #   return img, (target1, target2, target3)

    target_indexes = target_indexes.stack()
    target_updates = target_updates.stack()
    # 创建0张量，并根据索引赋值
    target_mask = tf.equal(target_indexes[:, 0], 0)
    target1 = tf.scatter_nd(tf.boolean_mask(target_indexes[:, 1:], target_mask),
      tf.boolean_mask(target_updates, target_mask),
      (layers_hw[0, 0], layers_hw[0, 1], anchors_num, 5+self.classes_num))
    target_mask = tf.equal(target_indexes[:, 0], 1)
    target2 = tf.scatter_nd(tf.boolean_mask(target_indexes[:, 1:], target_mask),
      tf.boolean_mask(target_updates, target_mask),
      (layers_hw[1, 0], layers_hw[1, 1], anchors_num, 5+self.classes_num))
    target_mask = tf.equal(target_indexes[:, 0], 2)
    target3 = tf.scatter_nd(tf.boolean_mask(target_indexes[:, 1:], target_mask),
      tf.boolean_mask(target_updates, target_mask),
      (layers_hw[2, 0], layers_hw[2, 1], anchors_num, 5+self.classes_num))
    
    # 去掉因压缩导致的目标重叠
    target1_mask = tf.math.less_equal(target1[...,4:5], 1)
    target1 = target1 * tf.cast(target1_mask, dtype=tf.float32)
    target2_mask = tf.math.less_equal(target2[...,4:5], 1)
    target2 = target2 * tf.cast(target2_mask, dtype=tf.float32)
    target3_mask = tf.math.less_equal(target3[...,4:5], 1)
    target3 = target3 * tf.cast(target3_mask, dtype=tf.float32)
    return img, (target1, target2, target3)
  
  def Generate(self):
    n = len(self.image_list)
    i = 0
    clone_image_list = self.image_list.copy()
    while True:
      if i==0:
        random.shuffle(clone_image_list)
      image_path = clone_image_list[i]
      # print('image_path:', label['image_path'])
      i = (i+1) % n
      yield image_path

  def GetDataSet(self):
    '''获取数据集'''
    # 数据预处理
    dataset = tf.data.Dataset.from_generator(self.Generate,
      (tf.string),
      (tf.TensorShape([])))
    dataset = dataset.map(self.GetRandomImage)
    dataset = dataset.map(self.GetRandomData)
    # for image, classes, boxes in dataset.take(1):
    #   tf.print(tf.shape(image), tf.shape(classes), tf.shape(boxes))
    #   # ImageHelper.opencvImageToFile('a.jpg', tf.cast(image*255.0, tf.int32).numpy())
    #   # tf.print(boxes)
    dataset = dataset.map(self.GetTargets)
    dataset = dataset.batch(self.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
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
