import os
import numpy as np

def LoadClasses(classes_path):
  '''加载类型'''
  print('加载类型：', classes_path)
  with open(classes_path, 'r', encoding='utf-8') as f:
      classes_name = f.readlines()
  classes_name = [c.strip() for c in classes_name]
  classes_num = len(classes_name)
  print('已加载%d个类型' % (classes_num))
  return classes_name, classes_num

def LoadLabels(labels_file, images_path, classes_name):
  '''
  加载标签

  Args:
    labels_file：标签文件路径
    images_path：图片文件跟目录
  '''
  print('加载标签：', labels_file)
  labels = []
  with open(labels_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip().split('|')
      image_full_path = os.path.join(images_path, line[0])
      # print('image_full_path:', image_full_path)
      classes = []
      boxes = []
      for i in range(1, len(line)):
        if line[i] == '':
          continue
        info = line[i].split(',')
        if info[0] not in classes_name:
          print('标签异常：', info[0], image_full_path)
          continue
        x1 = float(info[1])
        y1 = float(info[2])
        x2 = float(info[3])
        y2 = float(info[4])
        # print('boxes:', [x1, y1, x2, y2])
        if x2<=x1 or y2<=y1:
          print('标签异常boxes：', [x1, y1, x2, y2])
          continue
        classes.append(classes_name.index(info[0]))
        boxes.append([x1, y1, x2, y2])
      labels.append({
        'image_path': image_full_path,
        'classes': classes,
        'boxes': np.array(boxes, np.float).reshape([-1,4])
        })
  labels_num = len(labels)
  print('已加载%d个标签' % (labels_num))
  return labels, labels_num

def LoadAnchors(anchors_path):
  '''加载Anchors'''
  print('加载Anchors：', anchors_path)
  with open(anchors_path, 'r', encoding='utf-8') as f:
    anchors = f.readline()
  anchors = [float(x) for x in anchors.split(',')]
  anchors = np.array(anchors, dtype=np.int).reshape(3, -1, 2)
  anchors = anchors[[2,1,0]]
  print('已加载Anchors: ', anchors)
  return anchors