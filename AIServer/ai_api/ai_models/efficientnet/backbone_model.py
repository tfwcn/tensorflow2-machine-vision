import tensorflow as tf
import itertools
from absl import logging
from six.moves import xrange
import six
from ai_api.ai_models.layers.mb_conv_block import MBConvBlock
from ai_api.ai_models.layers.stem import Stem
from ai_api.ai_models.utils.round_filters import round_filters
from ai_api.ai_models.utils.round_repeats import round_repeats


class BackboneModel(tf.keras.Model):
  """A class implements tf.keras.Model.

    Reference: https://arxiv.org/abs/1807.11626
  """

  def __init__(self, blocks_args=None, global_params=None, name=None):
    """Initializes an `Model` instance.

    Args:
      blocks_args: A list of BlockArgs to construct block modules.
      global_params: GlobalParams, a set of global parameters.
      name: A string of layer name.

    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super().__init__(name=name)

    if not isinstance(blocks_args, list):
      raise ValueError('blocks_args should be a list.')
    self._global_params = global_params
    self._blocks_args = blocks_args

    self.endpoints = None

    self._build()

  def _build(self):
    """Builds a model."""
    self._blocks = []

    # Stem part.
    # input_filters:32
    self._stem = Stem(self._blocks_args[0].input_filters, self._global_params)

    # Builds blocks.
    # 迭代器，从0开始，返回结果不断加1
    block_id = itertools.count(0)
    block_name = lambda: 'blocks_%d' % next(block_id)
    # _blocks_args默认值
    # _blocks_args = [
    #     'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
    #     'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
    #     'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
    #     'r1_k3_s11_e6_i192_o320_se0.25',
    # ]
    for i, block_args in enumerate(self._blocks_args):
      assert block_args.num_repeat > 0
      # Update block input and output filters based on depth multiplier.
      # 维度缩放
      input_filters = round_filters(block_args.input_filters,
                                    self._global_params.width_coefficient, 
                                    self._global_params.depth_divisor)

      output_filters = round_filters(block_args.output_filters,
                                     self._global_params.width_coefficient, 
                                     self._global_params.depth_divisor)
      kernel_size = block_args.kernel_size
      # 默认采用，按深度系数*默认重复次数，计算出实际重复次数
      repeats = round_repeats(block_args.num_repeat, self._global_params)
      # 更新每个block的参数，为实际数
      block_args = block_args._replace(
          input_filters=input_filters,
          output_filters=output_filters,
          num_repeat=repeats)

      # The first block needs to take care of stride and filter size increase.
      # if superpixel, adjust filters, kernels, and strides.
      self._blocks.append(
          MBConvBlock(block_args, self._global_params, name=block_name()))
      
      # 重复次数大于1，后面的重复层输入维度等于输出维度，strides=[1, 1]
      if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
        # pylint: disable=protected-access
        block_args = block_args._replace(
            input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
      # xrange在数据较大时效率高
      for _ in xrange(block_args.num_repeat - 1):
        self._blocks.append(
            MBConvBlock(block_args, self._global_params, name=block_name()))

  # @tf.function
  def call(self,
           inputs,
           training):
    """Implementation of call().

    Args:
      inputs: input tensors.
      training: boolean, whether the model is constructed for training.

    Returns:
      output tensors.
    """
    outputs = None
    self.endpoints = {}
    reduction_idx = 0

    # Calls Stem layers
    outputs = self._stem(inputs, training)
    # logging.info('Built stem %s : %s', self._stem.name, outputs.shape)
    self.endpoints['stem'] = outputs

    # Calls blocks.
    for idx, block in enumerate(self._blocks):
      is_reduction = False  # reduction flag for blocks after the stem layer
      # If the first block has super-pixel (space-to-depth) layer, then stem is
      # the first reduction point.
      if ((idx == len(self._blocks) - 1) or
            self._blocks[idx + 1].block_args.strides[0] > 1):
        # 最后一层或strides>1时
        is_reduction = True
        reduction_idx += 1
      # 调用块
      outputs = block(outputs, training=training)
      # 记录每个block结果
      self.endpoints['block_%s' % idx] = outputs
      if is_reduction:
        # 记录每个block维度下降后的结果
        self.endpoints['reduction_%s' % reduction_idx] = outputs
      if block.endpoints:
        # 记录每个block的子结果
        for k, v in six.iteritems(block.endpoints):
          self.endpoints['block_%s/%s' % (idx, k)] = v
          if is_reduction:
            self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
    self.endpoints['features'] = outputs

    return [outputs] + list(filter(lambda endpoint: endpoint is not None, [
      self.endpoints.get('reduction_1'),
      self.endpoints.get('reduction_2'),
      self.endpoints.get('reduction_3'),
      self.endpoints.get('reduction_4'),
      self.endpoints.get('reduction_5'),
    ]))
