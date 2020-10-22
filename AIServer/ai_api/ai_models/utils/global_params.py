from ai_api.ai_models.utils.config_class import Config


efficientdet_model_param_dict = {
  'efficientdet-d0':
    dict(
      name='efficientdet-d0',
      backbone_name='efficientnet-b0',
      image_size=512,
      fpn_num_filters=64,
      fpn_cell_repeats=3,
      box_class_repeats=3,
      width_coefficient=1.0,
      depth_coefficient=1.0,
      resolution=224,
      dropout_rate=0.2,
    ),
  'efficientdet-d1':
    dict(
      name='efficientdet-d1',
      backbone_name='efficientnet-b1',
      image_size=640,
      fpn_num_filters=88,
      fpn_cell_repeats=4,
      box_class_repeats=3,
      width_coefficient=1.0,
      depth_coefficient=1.1,
      resolution=240,
      dropout_rate=0.2,
    ),
  'efficientdet-d2':
    dict(
      name='efficientdet-d2',
      backbone_name='efficientnet-b2',
      image_size=768,
      fpn_num_filters=112,
      fpn_cell_repeats=5,
      box_class_repeats=3,
      width_coefficient=1.1,
      depth_coefficient=1.2,
      resolution=260,
      dropout_rate=0.3,
    ),
  'efficientdet-d3':
    dict(
      name='efficientdet-d3',
      backbone_name='efficientnet-b3',
      image_size=896,
      fpn_num_filters=160,
      fpn_cell_repeats=6,
      box_class_repeats=4,
      width_coefficient=1.2,
      depth_coefficient=1.4,
      resolution=300,
      dropout_rate=0.3,
    ),
  'efficientdet-d4':
    dict(
      name='efficientdet-d4',
      backbone_name='efficientnet-b4',
      image_size=1024,
      fpn_num_filters=224,
      fpn_cell_repeats=7,
      box_class_repeats=4,
      width_coefficient=1.4,
      depth_coefficient=1.8,
      resolution=380,
      dropout_rate=0.4,
    ),
  'efficientdet-d5':
    dict(
      name='efficientdet-d5',
      backbone_name='efficientnet-b5',
      image_size=1280,
      fpn_num_filters=288,
      fpn_cell_repeats=7,
      box_class_repeats=4,
      width_coefficient=1.6,
      depth_coefficient=2.2,
      resolution=456,
      dropout_rate=0.4,
    ),
  'efficientdet-d6':
    dict(
      name='efficientdet-d6',
      backbone_name='efficientnet-b6',
      image_size=1280,
      fpn_num_filters=384,
      fpn_cell_repeats=8,
      box_class_repeats=5,
      fpn_weight_method='sum',  # Use unweighted sum for stability.
      width_coefficient=1.8,
      depth_coefficient=2.6,
      resolution=528,
      dropout_rate=0.5,
    ),
  'efficientdet-d7':
    dict(
      name='efficientdet-d7',
      backbone_name='efficientnet-b6',
      image_size=1536,
      fpn_num_filters=384,
      fpn_cell_repeats=8,
      box_class_repeats=5,
      anchor_scale=5.0,
      fpn_weight_method='sum',  # Use unweighted sum for stability.
      width_coefficient=1.8,
      depth_coefficient=2.6,
      resolution=528,
      dropout_rate=0.5,
    ),
  'efficientdet-d7x':
    dict(
      name='efficientdet-d7x',
      backbone_name='efficientnet-b7',
      image_size=1536,
      fpn_num_filters=384,
      fpn_cell_repeats=8,
      box_class_repeats=5,
      anchor_scale=4.0,
      max_level=8,
      fpn_weight_method='sum',  # Use unweighted sum for stability.
      width_coefficient=2.0,
      depth_coefficient=3.1,
      resolution=600,
      dropout_rate=0.5,
    ),
}


def default_detection_configs():
  h = Config()
  h.name='',
  h.backbone_name='',
  h.batch_norm_momentum = 0.99
  h.batch_norm_epsilon = 1e-3

  # # (width_coefficient, depth_coefficient, resolution, dropout_rate)
  # # 宽度系数，深度系数，分辨率，dropout比例
  # 'efficientnet-b0': (1.0, 1.0, 224, 0.2),
  # 'efficientnet-b1': (1.0, 1.1, 240, 0.2),
  # 'efficientnet-b2': (1.1, 1.2, 260, 0.3),
  # 'efficientnet-b3': (1.2, 1.4, 300, 0.3),
  # 'efficientnet-b4': (1.4, 1.8, 380, 0.4),
  # 'efficientnet-b5': (1.6, 2.2, 456, 0.4),
  # 'efficientnet-b6': (1.8, 2.6, 528, 0.5),
  # 'efficientnet-b7': (2.0, 3.1, 600, 0.5),
  # 'efficientnet-b8': (2.2, 3.6, 672, 0.5),
  # 'efficientnet-l2': (4.3, 5.3, 800, 0.5),
  h.width_coefficient = 1.0
  h.depth_coefficient = 1.0
  h.resolution = 240
  h.dropout_rate = 0.2
  h.depth_divisor = 8
  # FPN取层数范围
  h.min_level = 3
  h.max_level = 7
  h.image_size = 512
  h.fpn_num_filters = 88
  h.fpn_cell_repeats = 4
  h.fpn_weight_method = 'fastattn'
  h.box_class_repeats = 3
  # 是否训练模式
  h.is_training_bn = True
  h.num_scales = 3
  # aspect ratio with format (w, h). Can be computed with k-mean per dataset.
  h.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
  h.anchor_scale = 3.0
  # 1+ actual classes, 0 is reserved for background.
  h.num_classes = 81
  h.survival_prob = 0.8
  h.alpha = 0.25
  h.gamma = 1.5

  # For post-processing nms, must be a dict.
  h.nms_configs = {
      'method': 'gaussian',
      'iou_thresh': None,  # use the default value based on method.
      'score_thresh': None,
      'sigma': None,
      'max_nms_inputs': 0,
      'max_output_size': 1000,
  }
  return h


def get_efficientdet_config(model_name='efficientdet-d4'):
  """Get the default config for EfficientDet based on model name."""
  h = default_detection_configs()
  if model_name in efficientdet_model_param_dict:
    h.override(efficientdet_model_param_dict[model_name])
    # h.image_size = 1920
    h.levels_size = [(h.image_size)]
    for _ in range(h.max_level):
      h.levels_size.append((h.levels_size[-1]+1)//2)
  else:
    raise ValueError('Unknown model name: {}'.format(model_name))

  return h
