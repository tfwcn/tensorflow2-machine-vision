import numpy as np
import tensorflow as tf

import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.momentum_contrast.model import YoloV3Model
from ai_api.ai_models.datasets.coco_dataset import DataGenerator
from ai_api.ai_models.utils.radam import RAdam
from ai_api.ai_models.utils.load_object_detection_data import LoadAnchors

import argparse

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('--trainData', default='./train2017')
parser.add_argument('--trainLabels', default='./data/coco_train2017_labels.txt')
parser.add_argument('--valData', default='./val2017')
parser.add_argument('--valLabels', default='./data/coco_val2017_labels.txt')
parser.add_argument('--classesFile', default='./data/coco_classes.txt')
parser.add_argument('--anchorsFile', default='./data/coco_anchors.txt')
parser.add_argument('--batchSize', default=4, type=int)
args = parser.parse_args()

trainData = args.trainData
trainLabels = args.trainLabels
valData = args.valData
valLabels = args.valLabels
classesFile = args.classesFile
anchorsFile = args.anchorsFile
batchSize = args.batchSize


def train():
  '''训练'''
  # 设置GPU显存自适应
  gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
  cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
  print(gpus, cpus)
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  # if len(gpus) > 1:
  #     tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
  # elif len(cpus) > 0:
  #     tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
  # 加载数据
  anchors = LoadAnchors(anchorsFile)
  data_generator_train = DataGenerator(image_path=trainData,
    label_path=trainLabels,
    classes_path=classesFile, batch_size=batchSize, anchors=anchors, image_wh=(416, 416), label_mean=True, image_random=True, flip=True)
  data_set_train = data_generator_train.GetDataSet()
  data_generator_val = DataGenerator(image_path=valData,
    label_path=valLabels,
    classes_path=classesFile, batch_size=1, anchors=anchors, image_wh=(416, 416), label_mean=False, image_random=False, flip=False)
  data_set_val = data_generator_val.GetDataSet()

  # 构建模型
  model = YoloV3Model(classes_num=data_generator_train.classes_num, anchors=anchors, image_wh=(416, 416), backbone_weights=None)

  # 编译模型
  print('编译模型')
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4))
  # model.compile(optimizer=RAdam(lr=1e-4))

  # 日志
  log_dir = './data/momentum_contrast_weights/'
  model_path = log_dir + 'train_object_detection_weights/'
  old_model_path = log_dir + 'tf2_weights/'
  _ = model(tf.ones((1, 416, 416, 3)))
  is_old_model = False
  if os.path.exists(model_path):
    last_model_path = tf.train.latest_checkpoint(model_path)
    model.load_weights(last_model_path)
    print('加载模型:{}'.format(last_model_path))
  elif os.path.exists(old_model_path):
    last_model_path = tf.train.latest_checkpoint(old_model_path)
    model.load_weights(last_model_path)
    print('加载模型:{}'.format(last_model_path))
    is_old_model = True
  model.summary()

  # 训练回调方法
  # logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_path, 'ep{epoch:04d}-loss{loss:.3f}-val_mAP{val_mAP:.3f}.ckpt'),
    monitor='loss', mode='min', save_weights_only=True, save_best_only=False, verbose=1)
  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1)
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)

  if is_old_model:
    print('旧权重，开始预训练')
    model.FreeLayer(['darknet_conv2d_6','darknet_conv2d_14', 'darknet_conv2d_22'])
    model.fit(data_set_train,
      steps_per_epoch=1000,
      epochs=1,
      initial_epoch=0
      )
    model.save_weights(os.path.join(model_path,'start_weights.ckpt'))

  print('Train on {} samples, val on {} samples, with batch size {}.'.format(data_generator_train.labels_num, data_generator_val.labels_num, batchSize))
  print('开始训练')
  
  steps_per_epoch=5000
  model.FreeLayer([''])
  model.fit(data_set_train,
    # steps_per_epoch=max(1, data_generator_train.labels_num//batchSize),
    steps_per_epoch=steps_per_epoch,
    validation_data=data_set_val,
    validation_steps=max(1, data_generator_val.labels_num),
    # validation_steps=10,
    epochs=300,
    # initial_epoch=(model.global_step.numpy()//steps_per_epoch),
    initial_epoch=(model.optimizer.iterations.numpy()//steps_per_epoch),
    callbacks=[reduce_lr, early_stopping, checkpoint]
    )
  model.save_weights(os.path.join(model_path,'last_weights.ckpt'))


def main():
  train()


if __name__ == '__main__':
  main()