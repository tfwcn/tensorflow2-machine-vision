import tensorflow as tf
import functools
import sys
import os
sys.path.append(os.getcwd())

from ai_api.ai_models.layers.mb_conv_block import MBConvBlock
from ai_api.ai_models.utils.global_params import get_efficientdet_config
from ai_api.ai_models.utils.block_args import EfficientDetBlockArgs

@tf.function
def test():
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

  x = tf.ones([1, 224, 224, 32], dtype=tf.float32)
  layers = []
  for block_args in blocks_args:
    layers.append(MBConvBlock(block_args, global_params))

  for m in layers:
    x = m(x)
    tf.print(tf.shape(x))
    for v in m.trainable_weights:
      tf.print('trainable_weight:', v.name, tf.shape(v))


test()