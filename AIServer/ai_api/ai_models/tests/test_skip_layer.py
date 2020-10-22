import tensorflow as tf
import functools
import sys
import os
sys.path.append(os.getcwd())

from ai_api.ai_models.layers.skip import SkipLayer

l = [
  tf.keras.layers.Conv2D(10, 3, padding='same'),
  tf.keras.layers.Conv2D(5, 1, padding='same'),
  tf.keras.layers.Conv2D(3, 3, padding='same'),
  ]

m = SkipLayer(l)
# tf.add_n需要输入维度和输出维度一样
# skip_layer = SkipLayer(l, merger_opr=tf.add_n)
x = tf.ones([1, 28, 28, 3], dtype=tf.float32)
x = m(x)
print(x.shape)
for v in m.trainable_weights:
  print('trainable_weight:', v.name, v.shape)