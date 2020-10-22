import tensorflow as tf
import functools
import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.layers.resample_feature_map import ResampleFeatureMap
from ai_api.ai_models.utils.global_params import get_efficientdet_config
from ai_api.ai_models.utils.block_args import EfficientDetBlockArgs


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



global_params = get_efficientdet_config()
print('level_size:', global_params.levels_size)
m1 = ResampleFeatureMap(88, global_params.levels_size[6])
m2 = ResampleFeatureMap(88, global_params.levels_size[7])
m3 = ResampleFeatureMap(88, global_params.levels_size[6])

@tf.function
def test(x):
  tf.print(x.shape, type(x))
  x = m1(x, True)
  tf.print('m1:',tf.shape(x), x.shape, type(x))
  x = m2(x, True)
  tf.print('m2:',tf.shape(x), x.shape, type(x))
  x = m3(x, True)
  tf.print('m3:',tf.shape(x), x.shape, type(x))
  for v in m1.trainable_weights:
    tf.print('m1 trainable_weight:', v.name, tf.shape(v))
  for v in m2.trainable_weights:
    tf.print('m2 trainable_weight:', v.name, tf.shape(v))
  for v in m3.trainable_weights:
    tf.print('m3 trainable_weight:', v.name, tf.shape(v))

x = tf.ones([1, global_params.levels_size[5], global_params.levels_size[5], 320], dtype=tf.float32)
test(x)
