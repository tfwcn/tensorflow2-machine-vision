import tensorflow as tf
import functools
import numpy as np
from ai_api.ai_models.utils.drop_connect import drop_connect

class ClassNet(tf.keras.layers.Layer):
  """Object class prediction network."""

  def __init__(self,
               num_classes=90,
               num_anchors=9,
               num_filters=32,
               min_level=3,
               max_level=7,
               is_training_bn=False,
               repeats=4,
               survival_prob=None,
               name='class_net',
               **kwargs):
    """Initialize the ClassNet.

    Args:
      num_classes: number of classes.
      num_anchors: number of anchors.
      num_filters: number of filters for "intermediate" layers.
      min_level: minimum level for features.
      max_level: maximum level for features.
      is_training_bn: True if we train the BatchNorm.
      repeats: number of intermediate layers.
      survival_prob: if a value is set then drop connect will be used.
      name: the name of this layerl.
      **kwargs: other parameters.
    """

    super().__init__(name=name, **kwargs)
    self.num_classes = num_classes
    self.num_anchors = num_anchors
    self.num_filters = num_filters
    self.min_level = min_level
    self.max_level = max_level
    self.repeats = repeats
    self.is_training_bn = is_training_bn
    self.survival_prob = survival_prob
    self.conv_ops = []
    self.bns = []
    conv2d_layer = functools.partial(
        tf.keras.layers.SeparableConv2D,
        depth_multiplier=1,
        pointwise_initializer=tf.initializers.VarianceScaling(),
        depthwise_initializer=tf.initializers.VarianceScaling())
    # conv2d_layer = functools.partial(
    #     tf.keras.layers.Conv2D,
    #     kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    for i in range(self.repeats):
      # If using SeparableConv2D
      self.conv_ops.append(
          conv2d_layer(
              self.num_filters,
              kernel_size=3,
              bias_initializer=tf.zeros_initializer(),
              activation=None,
              padding='same',
              name='class-%d' % i))

      bn_per_level = []
      for level in range(self.min_level, self.max_level + 1):
        bn_per_level.append(
            tf.keras.layers.BatchNormalization(name='class-%d-bn-%d' % (i, level)))
      self.bns.append(bn_per_level)

    self.classes = conv2d_layer(
        num_classes * num_anchors,
        kernel_size=3,
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        padding='same',
        name='class-predict')

  # @tf.function
  def call(self, inputs, training, **kwargs):
    """Call ClassNet."""

    class_outputs = []
    for level_id in range(0, self.max_level - self.min_level + 1):
      image = inputs[level_id]
      for i in range(self.repeats):
        # 这里有跳层
        original_image = image
        # 所有层经过同一个卷积
        image = self.conv_ops[i](image)
        # 每层单独对应BN
        image = self.bns[i][level_id](image, training=training)
        image = tf.nn.swish(image)
        if i > 0 and self.survival_prob:
          # 多于1层时，drop掉后面几层
          image = drop_connect(image, training, self.survival_prob)
          image = image + original_image
      c = self.classes(image)
      c_shape = tf.shape(c)
      c = tf.reshape(c,(c_shape[0],c_shape[1],c_shape[2],self.num_anchors,self.num_classes))
      # 最后输出结果
      class_outputs.append(c)

    return tuple(class_outputs)

