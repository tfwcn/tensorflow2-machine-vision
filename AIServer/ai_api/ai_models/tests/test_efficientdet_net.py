import tensorflow as tf
import functools
import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.efficientnet.efficientdet_net import EfficientDetNet
from ai_api.ai_models.utils.global_params import get_efficientdet_config
from ai_api.ai_models.utils.block_args import EfficientDetBlockArgs


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



global_params = get_efficientdet_config()
blocks_args = [
  EfficientDetBlockArgs(1,3,(1,1),1,32,16,0.25),
  EfficientDetBlockArgs(2,3,(2,2),6,16,24,0.25),
  EfficientDetBlockArgs(2,5,(2,2),6,24,40,0.25),
  EfficientDetBlockArgs(3,3,(2,2),6,40,80,0.25),
  EfficientDetBlockArgs(3,5,(1,1),6,80,112,0.25),
  EfficientDetBlockArgs(4,5,(2,2),6,112,192,0.25),
  EfficientDetBlockArgs(1,3,(1,1),6,192,320,0.25),
]

m = EfficientDetNet(blocks_args, global_params)

# @tf.function
def test(x):
  x = m(x, training=global_params.is_training_bn)
  tf.print(type(x), len(x), len(x[0]), len(x[1]))
  tf.print('classes', tf.shape(x[0][0]), tf.shape(x[0][1]), tf.shape(x[0][2]), tf.shape(x[0][3]), tf.shape(x[0][4]))
  tf.print('boxes', tf.shape(x[1][0]), tf.shape(x[1][1]), tf.shape(x[1][2]), tf.shape(x[1][3]), tf.shape(x[1][4]))
  for v in m.trainable_weights:
    print(v.name, tf.shape(v))

x = tf.ones([1, global_params.image_size, global_params.image_size, 3], dtype=tf.float32)
test(x)
