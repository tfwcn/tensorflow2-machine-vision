import tensorflow as tf
import functools
import sys
import os
sys.path.append(os.getcwd())

from ai_api.ai_models.layers.stem_layer import Stem
from ai_api.ai_models.utils.global_params import get_efficientdet_config


global_params = get_efficientdet_config()

m = Stem(10, global_params)
x = tf.ones([1, 28, 28, 3], dtype=tf.float32)
x = m(x)
print(x.shape)
for v in m.trainable_weights:
  print('trainable_weight:', v.name, v.shape)