import itertools
from typing import List
import tensorflow as tf

from ai_api.ai_models.layers.resample_feature_map import ResampleFeatureMap

class OpAfterCombine(tf.keras.layers.Layer):
  def __init__(self,
               filters,
               *args, **kwargs):
    super(OpAfterCombine, self).__init__(*args, **kwargs)
    self.filters = filters

  def build(self, input_shape):
    # tf.print(input_shape)
    self.conv1 = tf.keras.layers.SeparableConv2D(
      self.filters,
      depth_multiplier=1,
      kernel_size=(3, 3),
      padding='same',
      use_bias=True)
    self.bn1 = tf.keras.layers.BatchNormalization()

  def call(self, inputs, training):
    x = inputs
    x = tf.nn.swish(x)
    x = self.conv1(x)
    x = self.bn1(x, training=training)
    return x

class BiFPNNode(tf.keras.layers.Layer):
  def __init__(self,
               filters,
               level_size: int,
               *args, **kwargs):
    super(BiFPNNode, self).__init__(*args, **kwargs)
    self.filters = filters
    self.level_size = level_size

  def build(self, input_shape):
    # tf.print(input_shape)
    self.input_num = len(input_shape)
    self.wsms = []
    self.resample_layers = []
    wsm_id = itertools.count(0)
    wsm_name = lambda: 'WSM_%d' % next(wsm_id)
    for _ in range(self.input_num):
      # 创建权重
      self.wsms.append(
        # 这里权重是一个标量
        self.add_weight(
          initializer='ones',
          name=wsm_name(),
          trainable=True))
      # 创建ResampleFeatureMap
      self.resample_layers.append(ResampleFeatureMap(self.filters, self.level_size))
    self.op_after_combine1 = OpAfterCombine(self.filters)

  def call(self, inputs, training):
    x = inputs
    wsms_sum = tf.add_n(self.wsms)
    nodes = []
    for i in range(self.input_num):
      nodes.append(self.resample_layers[i](x[i], training=training)*self.wsms[i]/(wsms_sum+0.0001))
    x = tf.add_n(nodes)
    x = self.op_after_combine1(x, training=training)
    return x

class BiFPN(tf.keras.layers.Layer):
  def __init__(self,
               filters: int,
               levels_size: List[int],
               name=None, *args, **kwargs):
    super(BiFPN, self).__init__(name=name, *args, **kwargs)
    self.filters = filters
    self.levels_size = levels_size

  def build(self, input_shape):
    # tf.print(input_shape)
    self.input_num = len(input_shape)
    self.nodes = []
    # p4-p6
    for i in range(self.input_num-2,0,-1):
      self.nodes.append(BiFPNNode(self.filters, self.levels_size[i]))
    # p3-p7
    for i in range(self.input_num):
      self.nodes.append(BiFPNNode(self.filters, self.levels_size[i]))

  def call(self, inputs, training):
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
    p3_0, p4_0, p5_0, p6_0, p7_0 = inputs
    p6_1 = self.nodes[0]([p6_0,p7_0], training=training)
    p5_1 = self.nodes[1]([p5_0,p6_1], training=training)
    p4_1 = self.nodes[2]([p4_0,p5_1], training=training)
    p3_2 = self.nodes[3]([p3_0,p4_1], training=training)
    p4_2 = self.nodes[4]([p4_0,p4_1,p3_2], training=training)
    p5_2 = self.nodes[5]([p5_0,p5_1,p4_2], training=training)
    p6_2 = self.nodes[6]([p6_0,p6_1,p5_2], training=training)
    p7_2 = self.nodes[7]([p7_0,p6_2], training=training)

    return (p3_2, p4_2, p5_2, p6_2, p7_2)