import tensorflow as tf
from absl import logging
from ai_api.ai_models.utils.conv_kernel_initializer import conv_kernel_initializer

class SE(tf.keras.layers.Layer):
  """Squeeze-and-excitation layer."""

  def __init__(self, se_filters, output_filters, global_params, name=None):
    super().__init__(name=name)

    self._se_filters = se_filters
    self._output_filters = output_filters
    self._global_params = global_params

  def build(self, input_shape):
    # super().build(input_shape)
    # Squeeze and Excitation layer.
    self._se_reduce = tf.keras.layers.Conv2D(
        self._se_filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=True,
        name='conv2d')
    self._se_expand = tf.keras.layers.Conv2D(
        self._output_filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=True,
        name='conv2d_1')

  def call(self, inputs):
    se_tensor = tf.reduce_mean(inputs, [1, 2], keepdims=True)
    se_tensor = self._se_expand(tf.nn.swish(self._se_reduce(se_tensor)))
    # logging.info('Built SE %s : %s', self.name, se_tensor.shape)
    return tf.sigmoid(se_tensor) * inputs