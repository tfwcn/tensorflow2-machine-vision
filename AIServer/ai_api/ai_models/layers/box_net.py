import tensorflow as tf
from ai_api.ai_models.utils.drop_connect import drop_connect


class BoxNet(tf.keras.layers.Layer):
  """Box regression network."""

  def __init__(self,
               num_anchors=9,
               num_filters=32,
               min_level=3,
               max_level=7,
               is_training_bn=False,
               repeats=4,
               survival_prob=None,
               name='box_net',
               **kwargs):
    """Initialize BoxNet.

    Args:
      num_anchors: number of  anchors used.
      num_filters: number of filters for "intermediate" layers.
      min_level: minimum level for features.
      max_level: maximum level for features.
      is_training_bn: True if we train the BatchNorm.
      act_type: String of the activation used.
      repeats: number of "intermediate" layers.
      separable_conv: True to use separable_conv instead of conv2D.
      survival_prob: if a value is set then drop connect will be used.
      strategy: string to specify training strategy for TPU/GPU/CPU.
      data_format: string of 'channel_first' or 'channels_last'.
      name: Name of the layer.
      **kwargs: other parameters.
    """

    super().__init__(name=name, **kwargs)

    self.num_anchors = num_anchors
    self.num_filters = num_filters
    self.min_level = min_level
    self.max_level = max_level
    self.repeats = repeats
    self.is_training_bn = is_training_bn
    self.survival_prob = survival_prob

    self.conv_ops = []
    self.bns = []

    for i in range(self.repeats):
      # using SeparableConv2D
      self.conv_ops.append(
          tf.keras.layers.SeparableConv2D(
              filters=self.num_filters,
              depth_multiplier=1,
              pointwise_initializer=tf.initializers.VarianceScaling(),
              depthwise_initializer=tf.initializers.VarianceScaling(),
              kernel_size=3,
              activation=None,
              bias_initializer=tf.zeros_initializer(),
              padding='same',
              name='box-%d' % i))

      bn_per_level = []
      for level in range(self.min_level, self.max_level + 1):
        bn_per_level.append(
            tf.keras.layers.BatchNormalization(name='box-%d-bn-%d' % (i, level)))
      self.bns.append(bn_per_level)

    self.boxes = tf.keras.layers.SeparableConv2D(
        filters=4 * self.num_anchors,
        depth_multiplier=1,
        pointwise_initializer=tf.initializers.VarianceScaling(),
        depthwise_initializer=tf.initializers.VarianceScaling(),
        kernel_size=3,
        activation=None,
        bias_initializer=tf.zeros_initializer(),
        padding='same',
        name='box-predict')

  # @tf.function
  def call(self, inputs, training):
    """Call boxnet."""
    # 和分类网络一样
    box_outputs = []
    for level_id in range(0, self.max_level - self.min_level + 1):
      image = inputs[level_id]
      for i in range(self.repeats):
        # 这里有跳层
        original_image = image
        image = self.conv_ops[i](image)
        image = self.bns[i][level_id](image, training=training)
        image = tf.nn.swish(image)
        if i > 0 and self.survival_prob:
          image = drop_connect(image, training, self.survival_prob)
          image = image + original_image

      b = self.boxes(image)
      b_shape = tf.shape(b)
      b = tf.reshape(b,(b_shape[0],b_shape[1],b_shape[2],self.num_anchors,4))
      box_outputs.append(b)

    return tuple(box_outputs)
