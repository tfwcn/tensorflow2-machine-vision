import numpy as np
import tensorflow as tf
import random
import cv2
from typing import Tuple, List, Union

import sys
import os
sys.path.append(os.getcwd())
import ai_api.ai_models.utils.image_helper as ImageHelper
from ai_api.ai_models.efficientnet.utils.anchors import Anchors


class DataGenerator(object):
  def __init__(self, image_path, label_path, classes_path, anchors: Anchors, is_train=True):
    self.image_path = image_path
    self.label_path = label_path
    self.classes_path = classes_path
    self.image_path = image_path
    self.anchors = anchors
    self.image_size = anchors.image_size
    self.is_train = is_train

    # 加载类型
    self.LoadClasses()
    # 加载标签
    self.LoadLabels()

  def LoadClasses(self):
    '''加载类型'''
    print('加载类型：', self.classes_path)
    with open(self.classes_path, 'r', encoding='utf-8') as f:
        self.classes = f.readlines()
    # 0:BG
    self.classes = ['BG'] + [c.strip() for c in self.classes]
    self.classes_num = len(self.classes)
    print('已加载%d个类型' % (self.classes_num-1))

  def LoadLabels(self):
    '''加载标签'''
    print('加载标签：', self.label_path)
    self.labels = []
    with open(self.label_path, 'r', encoding='utf-8') as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip().split('|')
        image_full_path = os.path.join(self.image_path, line[0])
        # print('image_full_path:', image_full_path)
        classes = []
        boxes = []
        for i in range(1, len(line)):
          if line[i] == '':
            continue
          info = line[i].split(',')
          if info[0] not in self.classes:
            print('标签异常：', info[0], image_full_path)
            continue
          classes.append(self.classes.index(info[0]))
          x1 = float(info[1])
          y1 = float(info[2])
          x2 = float(info[3])
          y2 = float(info[4])
          # print('boxes:', [y1, x1, y2, x2])
          boxes.append([y1, x1, y2, x2])
        self.labels.append({
          'image_path': image_full_path,
          'classes': classes,
          'boxes': boxes
          })
    self.labels_num = len(self.labels)
    print('已加载%d个标签' % (self.labels_num))

  def get_random_data(self, label):
    '''
    随机数据扩展

    Args:
      label: {
        'image_path': 图片路径,
        'classes': [obj_num,],
        'boxes': [obj_num,[x1,y1,x2,y2]]
        }

    Returns:
      random_img: [h,w,3]
      random_boxes: [obj_num,[y1,x1,y2,x2]]
      classes: [obj_num,]
    '''
    random_img = ImageHelper.fileToOpencvImage(label['image_path'])
    random_boxes = label['boxes']
    random_boxes = np.array(random_boxes)
    # 转成点集
    random_boxes = random_boxes.reshape((-1,2))
    # 图片大小
    # width, height = ImageHelpler.opencvGetImageSize(random_img)
    # 画矩形
    # cv2.rectangle(image, (20, 20), (380, 380), tuple(np.random.randint(0, 30, (3), dtype=np.int32)), thickness=8)
    # 增加模糊
    ksize = random.randint(0, 4)
    if ksize>0:
        random_img = ImageHelper.opencvBlur(random_img, (ksize, ksize))
    # 变换图像，旋转会导致框不准
    # random_offset_x = random.random()*width-width/2
    # random_offset_y = random.random()*height-height/2
    random_offset_x = random.random()*90-45
    random_offset_y = random.random()*90-45
    # random_angle_x = random.random()*60-30
    # random_angle_y = random.random()*60-30
    # random_angle_z = random.random()*40-20
    random_scale_x = random.random()*1.5+0.5
    random_scale_y = random.random()*1.5+0.5
    random_scale_z = 1
    # random_offset_x = 0
    # random_offset_y = 0
    random_angle_x = 0
    random_angle_y = 0
    random_angle_z = 0
    # random_scale = 1
    random_img, org, dst, random_boxes = ImageHelper.opencvPerspective(random_img, offset=(random_offset_x, random_offset_y, 0),
                                                                              angle=(random_angle_x, random_angle_y, random_angle_z),
                                                                              scale=(random_scale_x, random_scale_y, random_scale_z), points=random_boxes,
                                                                              bg_color=None, bg_mode=None)
    # 增加线条
    # random_img = image_helper.opencvRandomLines(random_img, 8)
    # 增加噪声
    random_img = ImageHelper.opencvNoise(random_img)
    # 颜色抖动
    # random_img = ImageHelper.opencvRandomColor(random_img)
    # 调整图片大小
    random_img, random_boxes, padding = ImageHelper.opencvProportionalResize(
                random_img, self.image_size, points=random_boxes, bg_color=None, bg_mode=None)
    # 最后输出图片
    random_img = cv2.cvtColor(random_img, cv2.COLOR_BGR2RGB)
    # 调整参数范围
    random_img = random_img.astype(np.float32)
    random_img = random_img / 255
    # 转换boxes成框列表
    random_boxes = random_boxes.reshape((-1,4))
    # # 将4角坐标转换成矩形坐标
    # random_boxes = np.reshape(random_boxes, (-1, 4, 2))
    # boxes_min = np.min(random_boxes, axis=1)
    # boxes_max = np.max(random_boxes, axis=1)
    # random_boxes = np.concatenate([boxes_min, boxes_max], axis=-1)
    # 截取框超出图片部分
    random_boxes[:,0][random_boxes[:,0]<0] = 0
    random_boxes[:,1][random_boxes[:,1]<0] = 0
    random_boxes[:,2][random_boxes[:,2]>self.image_size[0]] = self.image_size[0]
    random_boxes[:,3][random_boxes[:,3]>self.image_size[1]] = self.image_size[1]
    # 去掉无效框
    mask = np.logical_and(random_boxes[:,2]-random_boxes[:,0]>=2, random_boxes[:,3]-random_boxes[:,1]>=2)
    random_boxes = random_boxes[mask][:,[1,0,3,2]]
    classes = np.array(label['classes'], dtype=np.int32)
    classes = classes[mask]

    # cv2.imwrite(path, image)
    return random_img, random_boxes, classes

  def generate(self):
    '''
    生成训练数据

    Returns:
      image: [h,w,3]
      boxes: [obj_num,[y1,x1,y2,x2]]
      classes: [obj_num,]
    '''
    # 记录存在分类
    class_list = set()
    # 图片对于分类列表
    image_class_list = {}
    # 读取素材标签，用于素材均衡
    if self.is_train:
      for label in self.labels:
        image_path = label['image_path']
        image_class_list[image_path]=set()
        for c in label['classes']:
          class_list.add(c)
          image_class_list[image_path].add(c)
        image_class_list[image_path]=list(image_class_list[image_path])
      class_list = list(class_list)
      print('存在标签：', class_list)

    n = len(self.labels)
    i = 0
    class_index = 0
    clone_labels = self.labels.copy()
    while True:
      if i==0:
        random.shuffle(clone_labels)
      # 数据平均
      label = clone_labels[i]
      if len(class_list)>0 and self.is_train:
        if class_list[class_index] not in image_class_list[label['image_path']]:
          i = (i+1) % n
          continue

        # 找下一个类型
        if class_index < len(class_list)-1:
          class_index += 1
        else:
          class_index = 0
      # print('image_path:', label['image_path'])
      # input_shape：(416, 416)
      image, boxes, classes = self.get_random_data(label)
      i = (i+1) % n
      if len(classes) == 0:
        continue
      
      # tf.print('boxes:', boxes)
      yield image, boxes, classes


def GetDataSet(image_path, label_path, classes_path, batch_size, anchors: Anchors, is_train=True):
  '''获取数据集'''
  data_generator = DataGenerator(image_path, label_path, classes_path, anchors, is_train)
  # 数据预处理
  # dataset = tf.data.Dataset.from_generator(data_generator.generate, 
  #   (tf.float32, 
  #   (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
  #   (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
  #   (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)),
  #   (tf.TensorShape([None, None, 3]), 
  #   (tf.TensorShape([None, None, None, 4]), tf.TensorShape([None, None, None, 4]), tf.TensorShape([None, None, None, 4]), tf.TensorShape([None, None, None, 4]), tf.TensorShape([None, None, None, 4])),
  #   (tf.TensorShape([None, None, None, data_generator.classes_num]), tf.TensorShape([None, None, None, data_generator.classes_num]), tf.TensorShape([None, None, None, data_generator.classes_num]), tf.TensorShape([None, None, None, data_generator.classes_num]), tf.TensorShape([None, None, None, data_generator.classes_num])),
  #   (tf.TensorShape([None, None, None, 1]), tf.TensorShape([None, None, None, 1]), tf.TensorShape([None, None, None, 1]), tf.TensorShape([None, None, None, 1]), tf.TensorShape([None, None, None, 1]))))
  dataset = tf.data.Dataset.from_generator(data_generator.generate, 
    (tf.float32, tf.float32, tf.float32),
    (tf.TensorShape([None, None, 3]), tf.TensorShape([None, 4]), tf.TensorShape([None,])))
  if is_train:
    def map_fun(image, boxes, classes):
      y_true_boxes, y_true_classes, y_true_masks = anchors.generate_targets(boxes, classes, data_generator.classes_num)
      return image, y_true_boxes, y_true_classes, y_true_masks
    dataset = dataset.map(map_fun)
    dataset = dataset.batch(batch_size)
    # for x, y_true_boxes, y_true_classes, y_true_masks in dataset.take(1):
    #   # y_true_boxes = anchors.convert_outputs_boxes(y_true_boxes)
    #   # convert_boxes,convert_classes_id,convert_scores=anchors.convert_outputs_one(0, y_true_boxes, y_true_classes)
    #   # tf.print('outputs boxes:', convert_boxes)
    #   print(x.shape, 
    #     y_true_boxes[0].shape, y_true_boxes[1].shape, y_true_boxes[2].shape, y_true_boxes[3].shape, y_true_boxes[4].shape,
    #     y_true_classes[0].shape, y_true_classes[1].shape, y_true_classes[2].shape, y_true_classes[3].shape, y_true_classes[4].shape,
    #     y_true_masks[0].shape, y_true_masks[1].shape, y_true_masks[2].shape, y_true_masks[3].shape, y_true_masks[4].shape)
  else:
    def map_fun(image, boxes, classes):
      # 返回维度不同，不能用batch_size>1
      y_true_boxes, y_true_classes, y_true_masks = anchors.generate_targets(boxes, classes, data_generator.classes_num)
      return image, boxes, classes, y_true_boxes, y_true_classes, y_true_masks
    dataset = dataset.map(map_fun)
    dataset = dataset.batch(batch_size)
    # for x, _, _, y, z, m in dataset.take(1):
    #   print(x.shape, 
    #     y[0].shape, y[1].shape, y[2].shape, y[3].shape, y[4].shape,
    #     z[0].shape, z[1].shape, z[2].shape, z[3].shape, z[4].shape,
    #     m[0].shape, m[1].shape, m[2].shape, m[3].shape, m[4].shape)
  return dataset, data_generator

def main():
  anchors = Anchors(3, 7, (512, 512), 3, [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)], 3)
  data_set = GetDataSet(image_path='E:\\MyFiles\\labels\\coco2017\\train2017',
    label_path='E:\\MyFiles\\git\\tensorflow2-yolov4\\AIServer\\data\\coco_train2017_labels.txt',
    classes_path='E:\\MyFiles\\git\\tensorflow2-yolov4\\AIServer\\data\\coco_classes.txt', batch_size=3, anchors=anchors)


if __name__ == '__main__':
  main()
