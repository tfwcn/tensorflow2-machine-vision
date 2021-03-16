import numpy as np
import tensorflow as tf
import shutil
import h5py
import argparse
import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.yolo_v3.model import YoloV3ModelBase
from ai_api.ai_models.utils.load_object_detection_data import LoadClasses, LoadAnchors


parser = argparse.ArgumentParser()
parser.add_argument('--old_weights_path', default='./data/yolo_v3_weights/yolo.h5')
parser.add_argument('--output_path', default='./data/yolo_v3_weights/tf2_weights/tf2_weights.ckpt')
parser.add_argument(
  '--classes_num', default=80, type=int,
  help='类别数量')

args = parser.parse_args()

tf.random.set_seed(991)
anchors = LoadAnchors('./data/coco_anchors.txt')

model = YoloV3ModelBase(classes_num=args.classes_num,
                    anchors_num=anchors.shape[1])

y = model(tf.zeros([1,416,416,3], tf.float32), training=True)

weight_list = {}
with h5py.File(args.old_weights_path, 'r') as f:
  for layer_name in f.attrs['layer_names']:
    for weight_name in f[layer_name].attrs['weight_names']:
      weight_name = weight_name.decode('utf8')
      weight_list[weight_name]=np.asarray(f[layer_name][weight_name])
      
model.summary()
# tf.print(model.trainable_variables[0])
for v in model.variables:
    # print(v.name, v.shape)
    weight_value = weight_list[v.name]
    if weight_value.shape == v.shape:
      v.assign(weight_value)
    else:
      print('stip: %s, old: %s, new: %s' % (v.name, v.shape, weight_value.shape))
print(len(model.variables))
# tf.print(model.trainable_variables[0])
model.save_weights(args.output_path)