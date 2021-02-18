import tensorflow as tf
from ai_api.ai_models.efficientnet.backbone_model import BackboneModel
from ai_api.ai_models.layers.resample_feature_map import ResampleFeatureMap
from ai_api.ai_models.layers.bifpn import BiFPN
from ai_api.ai_models.layers.class_net import ClassNet
from ai_api.ai_models.layers.box_net import BoxNet



class EfficientDetNet(tf.keras.Model):
  """EfficientDet keras network without pre/post-processing."""
  '''EfficientDet模型主体'''

  def __init__(self, blocks_args=None, global_params=None, name=''):
    """Initialize model."""
    super().__init__(name=name)
    self._blocks_args = blocks_args
    self._global_params = global_params

    # Backbone.
    # 构建主体
    self.backbone = BackboneModel(self._blocks_args, self._global_params, name=self._global_params.backbone_name)

    # Feature network.
    # 特征网络
    self.resample_layers = []  # additional resampling layers.
    # 共7层，主体只有5层，补了两层ResampleFeatureMap
    for level in range(6, self._global_params.max_level + 1):
      # Adds a coarser level by downsampling the last feature map.
      self.resample_layers.append(
          ResampleFeatureMap(
              target_num_channels=self._global_params.fpn_num_filters,
              level_size=self._global_params.levels_size[level],
              name='resample_p%d' % level,
          ))
    # 这里是
    self.fpn_cells = []
    for _ in range(self._global_params.fpn_cell_repeats):
      self.fpn_cells.append(BiFPN(
        filters=self._global_params.fpn_num_filters,
        levels_size=self._global_params.levels_size[self._global_params.min_level:]))

    # class/box output prediction network.
    # aspect_ratios：3种形状，3种大小，共9个候选框
    num_anchors = len(self._global_params.aspect_ratios) * self._global_params.num_scales
    # fpn通道数
    num_filters = self._global_params.fpn_num_filters
    # 目标检测
    self.class_net = ClassNet(
        num_classes=self._global_params.num_classes,
        num_anchors=num_anchors,
        num_filters=num_filters,
        min_level=self._global_params.min_level,
        max_level=self._global_params.max_level,
        is_training_bn=self._global_params.is_training_bn,
        repeats=self._global_params.box_class_repeats,
        survival_prob=self._global_params.survival_prob)

    self.box_net = BoxNet(
        num_anchors=num_anchors,
        num_filters=num_filters,
        min_level=self._global_params.min_level,
        max_level=self._global_params.max_level,
        is_training_bn=self._global_params.is_training_bn,
        repeats=self._global_params.box_class_repeats,
        survival_prob=self._global_params.survival_prob,)

  def _init_set_name(self, name, zero_based=True):
    """A hack to allow empty model name for legacy checkpoint compitability."""
    if name == '':  # pylint: disable=g-explicit-bool-comparison
      self._name = name
    else:
      self._name = super().__init__(name, zero_based)

  # @tf.function
  def call(self, inputs, training):
    # call backbone network.
    all_feats = self.backbone(inputs, training=training)
    # 5层特征,这里只有3层
    feats = all_feats[self._global_params.min_level:self._global_params.max_level + 1]

    # Build additional input features that are not from backbone.
    # 5层特征,这里补2层
    for resample_layer in self.resample_layers:
      feats.append(resample_layer(feats[-1], training))

    # call feature network.
    fpn_feats = feats
    for fpn_cell in self.fpn_cells:
      fpn_feats = fpn_cell(fpn_feats, training)

    # call class/box/seg output network.
    classes_outputs = self.class_net(fpn_feats, training)
    boxes_outputs = self.box_net(fpn_feats, training)
    return boxes_outputs, classes_outputs


