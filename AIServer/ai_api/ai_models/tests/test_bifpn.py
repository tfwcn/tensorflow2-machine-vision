import tensorflow as tf
import functools
import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.layers.bifpn import BiFPN
from ai_api.ai_models.utils.global_params import get_efficientdet_config


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



global_params = get_efficientdet_config()
print('level_size:', global_params.levels_size)
m = BiFPN(88, levels_size=global_params.levels_size[3:])

@tf.function
def test(x):
  x = m(x, True)
  tf.print('m:',len(x), type(x))
  for i in range(len(x)):
    tf.print('m:',tf.shape(x[i]), x[i].shape, type(x[i]))
  for v in m.trainable_weights:
    tf.print('m1 trainable_weight:', v.name, tf.shape(v))

p3 = tf.ones([1, global_params.levels_size[3], global_params.levels_size[3], 320], dtype=tf.float32)
p4 = tf.ones([1, global_params.levels_size[4], global_params.levels_size[4], 320], dtype=tf.float32)
p5 = tf.ones([1, global_params.levels_size[5], global_params.levels_size[5], 320], dtype=tf.float32)
p6 = tf.ones([1, global_params.levels_size[6], global_params.levels_size[6], 320], dtype=tf.float32)
p7 = tf.ones([1, global_params.levels_size[7], global_params.levels_size[7], 320], dtype=tf.float32)
x = (p3,p4,p5,p6,p7)
test(x)
