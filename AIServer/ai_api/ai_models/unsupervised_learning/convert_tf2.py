import numpy as np
import tensorflow as tf
import shutil
import h5py
import argparse
import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.unsupervised_learning.model import YoloV3ModelBase
from ai_api.ai_models.utils.load_object_detection_data import LoadClasses, LoadAnchors


parser = argparse.ArgumentParser()
parser.add_argument('--output_path', default='./data/unsupervised_learning_weights/tf2_weights/tf2_weights.ckpt')
parser.add_argument(
  '--classes_num', default=80, type=int,
  help='类别数量')

args = parser.parse_args()

tf.random.set_seed(991)
anchors = LoadAnchors('./data/coco_anchors.txt')

model = YoloV3ModelBase(classes_num=args.classes_num,
                        anchors_num=anchors.shape[1],
                        backbone_weights='imagenet')
y = model(tf.zeros([1,416,416,3], tf.float32), training=True)
model.summary()
model.save_weights(args.output_path)