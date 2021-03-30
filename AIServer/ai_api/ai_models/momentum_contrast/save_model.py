import numpy as np
import tensorflow as tf

import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.momentum_contrast.model import MoCoModel
from ai_api.ai_models.momentum_contrast.moco_dataset import DataGenerator
from ai_api.ai_models.utils.radam import RAdam

import argparse

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('--output_path', default='./data/momentum_contrast_weights/tf2_weights/tf2_weights.ckpt')
parser.add_argument(
  '--classes_num', default=80, type=int,
  help='类别数量')
args = parser.parse_args()

output_path = args.output_path
classes_num = args.classes_num


def main():
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
  anchors_num = 3
  out_filters = anchors_num*(classes_num+5)

  # 构建模型
  model = MoCoModel(
    out_filters=out_filters,
    image_wh=(416, 416),
    K=100)

  # 编译模型
  print('编译模型')
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4))
  # model.compile(optimizer=RAdam(lr=1e-4))

  # 日志
  log_dir = './data/momentum_contrast_weights/'
  model_path = log_dir + 'train_weights/'
  _ = model((tf.ones([1, 416, 416, 3]), tf.ones([1, 416, 416, 3])))
  if os.path.exists(model_path):
    last_model_path = tf.train.latest_checkpoint(model_path)
    model.load_weights(last_model_path)
    print('加载模型:{}'.format(last_model_path))
  model.summary()
  print('保存模型:{}'.format(output_path))
  model.save_weights_k(output_path)


if __name__ == '__main__':
  main()