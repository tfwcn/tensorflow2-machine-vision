import functools
import tensorflow as tf

class SkipLayer(tf.keras.layers.Layer):
  '''跳过中间层，将输入值与最后输出只合并'''
  def __init__(self, layers, merger_fn=None, name=None):
    '''
    layers：层列表
    merger_fn：用于合并的方法，参数为[x, inputs]
    '''
    super().__init__(name=name)
    self._layers = layers
    if merger_fn:
      self._merger_fn = merger_fn
    else:
      self._merger_fn = functools.partial(tf.concat, axis=-1)

  def build(self, input_shape):
    super().build(input_shape)

  def call(self, inputs, training):
    x = inputs
    for l in self._layers:
      x = l(x, training=training)
    outputs = self._merger_fn([x, inputs])
    return outputs