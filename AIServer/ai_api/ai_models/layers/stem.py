import tensorflow as tf
from ai_api.ai_models.utils.round_filters import round_filters
from ai_api.ai_models.utils.conv_kernel_initializer import conv_kernel_initializer
from ai_api.ai_models.layers.attention_conv import AttentionConv2D

class Stem(tf.keras.layers.Layer):
  '''Stem层'''
  def __init__(self, stem_filters, global_params, name=None):
    super().__init__(name=name)
    self._stem_filters = stem_filters
    self._global_params = global_params

  def build(self, input_shape):
    # super().build(input_shape)
    self._conv_stem = tf.keras.layers.Conv2D(
        # 根据缩放参数调整
        filters=round_filters(self._stem_filters, self._global_params.width_coefficient, self._global_params.depth_divisor),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False)
    # self._conv_stem = AttentionConv2D(
    #     # 根据缩放参数调整
    #     filters=round_filters(self._stem_filters, self._global_params.width_coefficient, self._global_params.depth_divisor),
    #     kernel_size=[3, 3],
    #     strides=[2, 2],
    #     kernel_initializer=conv_kernel_initializer,
    #     padding='same',
    #     use_bias=False)
    self._bn = tf.keras.layers.BatchNormalization(
        name='tpu_batch_normalization',
        momentum=self._global_params.batch_norm_momentum,
        epsilon=self._global_params.batch_norm_epsilon)
    self._relu_fn = tf.nn.swish

  def call(self, inputs, training):
    return self._relu_fn(self._bn(self._conv_stem(inputs), training=training))