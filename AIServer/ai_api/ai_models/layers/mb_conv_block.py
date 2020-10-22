import tensorflow as tf
from absl import logging
import itertools
from ai_api.ai_models.utils.conv_kernel_initializer import conv_kernel_initializer
from ai_api.ai_models.layers.se import SE

class MBConvBlock(tf.keras.layers.Layer):
  """A class of MBConv: Mobile Inverted Residual Bottleneck.

  Attributes:
    endpoints: dict. A list of internal tensors.
  """

  def __init__(self, block_args, global_params, name=None):
    """Initializes a MBConv block.

    Args:
      block_args: BlockArgs, arguments to create a Block.
      global_params: GlobalParams, a set of global parameters.
      name: layer name.
    """
    super().__init__(name=name)

    self._block_args = block_args
    self._global_params = global_params
    # local_pooling默认为False
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    # 节点
    self.endpoints = None

    # Builds the block accordings to arguments.
    self._build()

  @property
  def block_args(self):
    return self._block_args


  def _build(self):
    """Builds block according to the arguments."""
    # super().build(input_shape)
    # pylint: disable=g-long-lambda
    bid = itertools.count(0)
    # 这里一个方法调用了两次next(bid)，因此要除以2
    get_bn_name = lambda: 'tpu_batch_normalization' + ('' if not next(
        bid) else '_' + str(next(bid) // 2))
    cid = itertools.count(0)
    get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
        next(cid) // 2))
    # pylint: enable=g-long-lambda

    # self._block_args.expand_ratio扩大倍率
    filters = self._block_args.input_filters * self._block_args.expand_ratio
    kernel_size = self._block_args.kernel_size

    # Expansion phase. Called if not using fused convolutions and expansion
    # phase is necessary.
    # 第一层不执行
    if self._block_args.expand_ratio != 1:
      self._expand_conv = tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=False,
          name=get_conv_name())
      self._bn0 = tf.keras.layers.BatchNormalization(
          momentum=self._batch_norm_momentum,
          epsilon=self._batch_norm_epsilon,
          name=get_bn_name())

    # Depth-wise convolution phase. Called if not using fused convolutions.
    # 深度卷积输出维度与输入维度一样，每个输入经过一个卷积核得到一个输出
    self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=[kernel_size, kernel_size],
        strides=self._block_args.strides,
        depthwise_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        name='depthwise_conv2d')

    self._bn1 = tf.keras.layers.BatchNormalization(
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        name=get_bn_name())

    num_reduced_filters = max(
        1, int(self._block_args.input_filters * self._block_args.se_ratio))
    self._se = SE(num_reduced_filters, filters, 
        self._global_params, name='se')

    # Output phase.
    filters = self._block_args.output_filters
    self._project_conv = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        name=get_conv_name())
    self._bn2 = tf.keras.layers.BatchNormalization(
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        name=get_bn_name())

  # @tf.function
  def call(self, inputs, training):
    """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.

    Returns:
      A output tensor.
    """
    # logging.info('Block %s input shape: %s', self.name, inputs.shape)
    x = inputs

    # creates conv 2x2 kernel
    # Otherwise, first apply expansion and then apply depthwise conv.
    # 第一层不执行
    if self._block_args.expand_ratio != 1:
      x = tf.nn.swish(self._bn0(self._expand_conv(x), training=training))
      # logging.info('Expand shape: %s', x.shape)

    x = tf.nn.swish(self._bn1(self._depthwise_conv(x), training=training))
    # logging.info('DWConv shape: %s', x.shape)

    x = self._se(x)

    self.endpoints = {'expansion_output': x}

    x = self._bn2(self._project_conv(x), training=training)
    # Add identity so that quantization-aware training can insert quantization
    # ops correctly.
    # tf.identity这里相当与克隆
    # x = tf.identity(x)
    # logging.info('Project shape: %s', x.shape)
    return x
