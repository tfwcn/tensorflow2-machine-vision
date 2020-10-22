import tensorflow as tf
import itertools
import functools
from ai_api.ai_models.utils.config_class import Config
from ai_api.ai_models.layers.resample_feature_map import ResampleFeatureMap


def bifpn_config(min_level, max_level, weight_method):
  """A dynamic bifpn config that can adapt to different min/max levels."""
  p = Config()
  p.weight_method = weight_method

  # Node id starts from the input features and monotonically increase whenever
  # a new node is added. Here is an example for level P3 - P7:
  #     P7 (4)              P7" (12)
  #     P6 (3)    P6' (5)   P6" (11)
  #     P5 (2)    P5' (6)   P5" (10)
  #     P4 (1)    P4' (7)   P4" (9)
  #     P3 (0)              P3" (8)
  # So output would be like:
  # [
  #   {'feat_level': 6, 'inputs_offsets': [3, 4]},  # for P6'
  #   {'feat_level': 5, 'inputs_offsets': [2, 5]},  # for P5'
  #   {'feat_level': 4, 'inputs_offsets': [1, 6]},  # for P4'
  #   {'feat_level': 3, 'inputs_offsets': [0, 7]},  # for P3"
  #   {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},  # for P4"
  #   {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},  # for P5"
  #   {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},  # for P6"
  #   {'feat_level': 7, 'inputs_offsets': [4, 11]},  # for P7"
  # ]
  num_levels = max_level - min_level + 1
  node_ids = {min_level + i: [i] for i in range(num_levels)}

  level_last_id = lambda level: node_ids[level][-1]
  level_all_ids = lambda level: node_ids[level]
  id_cnt = itertools.count(num_levels)

  p.nodes = []
  # 6 -> 3
  # 这里第3层实际是第3列，最后一个节点
  for i in range(max_level - 1, min_level - 1, -1):
    # top-down path.
    p.nodes.append({
        # 特征层级，对应不同大小的特征，共7层，这里是后面5层，6-3
        'feat_level': i,
        'inputs_offsets': [level_last_id(i),
                           level_last_id(i + 1)]
    })
    node_ids[i].append(next(id_cnt))

  for i in range(min_level + 1, max_level + 1):
    # bottom-up path.
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)]
    })
    node_ids[i].append(next(id_cnt))

  return p

class FNode(tf.keras.layers.Layer):
  """A Keras Layer implementing BiFPN Node."""

  def __init__(self,
               feat_level,
               inputs_offsets,
               fpn_num_filters,
               is_training_bn,
               weight_method,
               levels_size,
               name='fnode'):
    super().__init__(name=name)
    self.feat_level = feat_level
    self.inputs_offsets = inputs_offsets
    self.fpn_num_filters = fpn_num_filters
    self.is_training_bn = is_training_bn
    self.weight_method = weight_method
    self.levels_size = levels_size
    self.resample_layers = []
    self.vars = []

  def fuse_features(self, nodes):
    """Fuse features from different resolutions and return a weighted sum.

    Args:
      nodes: a list of tensorflow features at different levels

    Returns:
      A tensor denoting the fused feature.
    """
    dtype = nodes[0].dtype

    if self.weight_method == 'attn':
      edge_weights = []
      for var in self.vars:
        var = tf.cast(var, dtype=dtype)
        edge_weights.append(var)
      normalized_weights = tf.nn.softmax(tf.stack(edge_weights))
      nodes = tf.stack(nodes, axis=-1)
      new_node = tf.reduce_sum(nodes * normalized_weights, -1)
    elif self.weight_method == 'fastattn':
      edge_weights = []
      for var in self.vars:
        var = tf.cast(var, dtype=dtype)
        edge_weights.append(var)
      weights_sum = tf.add_n(edge_weights)
      nodes = [
          nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
          for i in range(len(nodes))
      ]
      new_node = tf.add_n(nodes)
    elif self.weight_method == 'channel_attn':
      edge_weights = []
      for var in self.vars:
        var = tf.cast(var, dtype=dtype)
        edge_weights.append(var)
      normalized_weights = tf.nn.softmax(tf.stack(edge_weights, -1), axis=-1)
      nodes = tf.stack(nodes, axis=-1)
      new_node = tf.reduce_sum(nodes * normalized_weights, -1)
    elif self.weight_method == 'channel_fastattn':
      edge_weights = []
      for var in self.vars:
        var = tf.cast(var, dtype=dtype)
        edge_weights.append(var)

      weights_sum = tf.add_n(edge_weights)
      nodes = [
          nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
          for i in range(len(nodes))
      ]
      new_node = tf.add_n(nodes)
    elif self.weight_method == 'sum':
      new_node = tf.add_n(nodes)
    else:
      new_node = tf.add_n(nodes)

    return new_node

  def _add_wsm(self, initializer):
    for i, _ in enumerate(self.inputs_offsets):
      name = 'WSM' + ('' if i == 0 else '_' + str(i))
      self.vars.append(
          # 这里权重是一个标量
          self.add_weight(
              initializer=initializer, name=name,
              trainable=self.is_training_bn))

  def build(self, feats_shape):
    for i, input_offset in enumerate(self.inputs_offsets):
      name = 'resample_{}_{}_{}'.format(i, input_offset, len(feats_shape))
      self.resample_layers.append(
          ResampleFeatureMap(
              self.feat_level,
              self.fpn_num_filters,
              self.levels_size,
              name=name))
    self._add_wsm('ones')
    self.op_after_combine = OpAfterCombine(
        self.fpn_num_filters,
        name='op_after_combine{}'.format(len(feats_shape)))
    self.built = True
    super().build(feats_shape)

  # @tf.function
  def call(self, feats, training):
    nodes = []
    append_feats = []
    for i, input_offset in enumerate(self.inputs_offsets):
      input_node = feats[input_offset]
      input_node = self.resample_layers[i](input_node, training, feats)
      nodes.append(input_node)
    # 这里把所有结果加起来
    new_node = self.fuse_features(nodes)
    # 这里先经过激活函数，再卷积+BN
    new_node = self.op_after_combine(new_node)
    append_feats.append(new_node)
    # 原特征 + 新特征，用于特征累加
    return feats + append_feats


class OpAfterCombine(tf.keras.layers.Layer):
  """Operation after combining input features during feature fusiong."""

  def __init__(self,
               fpn_num_filters,
               name='op_after_combine'):
    super().__init__(name=name)
    self.fpn_num_filters = fpn_num_filters
    # SeparableConv2D：等于先经过DepthwiseConv2D，在经过卷积核为1*1的卷积。
    conv2d_layer = functools.partial(
        tf.keras.layers.SeparableConv2D, depth_multiplier=1)

    self.conv_op = conv2d_layer(
        filters=fpn_num_filters,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        name='conv')
    self.bn = tf.keras.layers.BatchNormalization(name='bn')

  # @tf.function
  def call(self, new_node, training):
    new_node = tf.nn.swish(new_node)
    new_node = self.conv_op(new_node)
    new_node = self.bn(new_node, training=training)
    return new_node


class FPNCells(tf.keras.layers.Layer):
  """FPN cells."""

  def __init__(self, config, name='fpn_cells'):
    super().__init__(name=name)
    self.config = config

    self.fpn_config = bifpn_config(config.min_level, config.max_level, config.fpn_weight_method)

    self.cells = [
        FPNCell(self.config, name='cell_%d' % rep)
        for rep in range(self.config.fpn_cell_repeats)
    ]

  # @tf.function
  def call(self, feats, training):
    for cell in self.cells:
      # feats会在里面累加
      cell_feats = cell(feats, training)
      min_level = self.config.min_level
      max_level = self.config.max_level
      # 每次会清空
      feats = []
      # 3 -> 7
      for level in range(min_level, max_level + 1):
        # reversed：序列反转
        for i, fnode in enumerate(reversed(self.fpn_config.nodes)):
          if fnode['feat_level'] == level:
            feats.append(cell_feats[-1 - i])
            break
    # 这里输出的是最后FPNCell结果
    return feats


class FPNCell(tf.keras.layers.Layer):
  """A single FPN cell."""

  def __init__(self, config, name='fpn_cell'):
    super().__init__(name=name)
    self.config = config
    self.fpn_config = bifpn_config(config.min_level, config.max_level, config.fpn_weight_method)
    self.fnodes = []
    for i, fnode_cfg in enumerate(self.fpn_config.nodes):
      # logging.info('fnode %d : %s', i, fnode_cfg)
      fnode = FNode(
          fnode_cfg['feat_level'] - self.config.min_level,
          fnode_cfg['inputs_offsets'],
          config.fpn_num_filters,
          # True
          config.is_training_bn,
          # fastattn
          weight_method=self.fpn_config.weight_method,
          levels_size=config.levels_size[self.config.min_level:],
          name='fnode%d' % i)
      self.fnodes.append(fnode)

  # @tf.function
  def call(self, feats, training):
    for fnode in self.fnodes:
      feats = fnode(feats, training)
    return feats
