from inspect import stack
import tensorflow as tf
from ai_api.ai_models.efficientnet.utils.get_feat_sizes import get_feat_sizes
from ai_api.ai_models.efficientnet.utils.iou import get_iou
from ai_api.ai_models.efficientnet.utils.nms import get_nms
from typing import Tuple, List, Union
import numpy as np


EPSILON = 1e-8

class Anchors(object):
  '''
  Anchors计算与转换
  '''
  def __init__(self, min_level: int, max_level: int, image_size: Tuple[int, int],
               num_scales: int, aspect_ratios: List[Tuple[float, float]],
               anchor_scale: Union[float, List[float]]):
    '''
    Args:
      image_size: (H, W)
      min_level: minimum feature level.
      max_level: maximum feature level.
      num_scales: 缩放大小数量
      aspect_ratios: 缩放比例
      anchor_scale: anchor缩放倍率调整

    '''
    self.min_level = min_level
    self.max_level = max_level
    self.image_size = image_size
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    self.anchor_scale = anchor_scale
    if isinstance(anchor_scale, (list, tuple)):
      assert len(anchor_scale) == max_level - min_level + 1
      self.anchor_scales = anchor_scale
    else:
      self.anchor_scales = [anchor_scale] * (max_level - min_level + 1)
    # 计算每层特征大小
    self.feat_sizes = get_feat_sizes(self.image_size, self.max_level)
    # print('feat_sizes: ', self.feat_sizes)
    self.boxes = self._generate_boxes()
    # print('boxes: ', self.boxes)

  @tf.function
  def _generate_boxes(self):
    boxes_all = []
    feat_sizes = self.feat_sizes
    for level in range(self.min_level, self.max_level + 1):
      boxes_level = []
      # num_scales：3
      for scale_octave in range(self.num_scales):
        # aspect_ratios：[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        # anchor_scales：[4, 4, 4, 4]
        for aspect in self.aspect_ratios:
          # stride, octave_scale, aspect, anchor_scale
          stride = (feat_sizes[0][0] / float(feat_sizes[level][0]), feat_sizes[0][1] / float(feat_sizes[level][1]))
          octave_scale = scale_octave / float(self.num_scales)
          anchor_scale = self.anchor_scales[level - self.min_level]
          # anchor_scale:固定倍率，stride：特征大小相对应图片大小的步长，octave_scale：随着层数增加，这个值会越来越大(0,1]
          # 固定倍率*步长*2^(0,1],2^(0,1]范围是(1,2]
          base_anchor_size_x = anchor_scale * stride[1] * 2**octave_scale
          base_anchor_size_y = anchor_scale * stride[0] * 2**octave_scale
          # 基础大小*高宽比例*1/2，实际范围是(0.5,1.0]
          anchor_size_x_2 = base_anchor_size_x * aspect[1] / 2.0
          anchor_size_y_2 = base_anchor_size_y * aspect[0] / 2.0

          # 用实际图片大小，特征步长，计算中心坐标
          x = tf.range(stride[1] / 2, self.image_size[1], stride[1])
          y = tf.range(stride[0] / 2, self.image_size[0], stride[0])
          xv, yv = tf.meshgrid(x, y)

          # 计算boxes左上和右下坐标，(h,w,(y1,x1,y2,x2))# 
          yv = tf.expand_dims(yv, axis=-1)
          xv = tf.expand_dims(xv, axis=-1)
          boxes = tf.concat([yv - anchor_size_y_2, xv - anchor_size_x_2,
                              yv + anchor_size_y_2, xv + anchor_size_x_2], axis=-1)
                              
          boxes = tf.expand_dims(boxes, axis=-2)
          # 变换维度
          boxes_level.append(boxes)
      boxes_all.append(tf.concat(boxes_level, axis=-2))
    return boxes_all

  def get_anchors_per_location(self):
    '''获得每层anchor数量'''
    return self.num_scales * len(self.aspect_ratios)

  @tf.function
  def generate_targets(self, boxes, classes, classes_num, iou_threshold=0.5):
    '''
    boxes转成targets(已验证)

    Args:
      boxes: [boxes_size, 4]
      classes: [boxes_size,]

    Returns:
      boxes: [level,[h,w,anchors,[ty, tx, th, tw]]]
      classes: [level,[h,w,anchors,classes_num]]
    '''
    output_boxes = []
    output_classes = []
    output_mask = []
    # 增加一维，用于计算iou
    # boxes_reshape = tf.reshape(boxes, (1, 1, -1, 4))
    classes_reshape = tf.reshape(classes, (-1, 1))
    for anchor_level in self.boxes:
      # 计算IOU
      # [h,w,1,[y1,x1,y2,x2]]
      anchor_reshape = tf.expand_dims(anchor_level, axis=-2)
      # iou: [h,w,anchors,boxes_num]
      iou = get_iou(anchor_reshape, boxes)
      # tf.print('iou:', iou)
      # iou_index: [h,w,anchors]
      iou_index = tf.math.argmax(iou, axis=-1)
      # iou_max: [h,w,anchors]
      iou_max = tf.math.reduce_max(iou, axis=-1)
      # iou_mask: [h,w,anchors]
      iou_mask = tf.math.greater_equal(iou_max, iou_threshold)
      # tf.print('iou_mask:', iou_mask)
      # iou_mask: [h,w,anchors,1]
      iou_mask = tf.expand_dims(iou_mask, axis=-1)
      # boxes_level: [h,w,anchors,[y1,x1,y2,x2]]
      boxes_level = tf.gather(boxes, iou_index, axis=0)
      # classes_level: [h,w,anchors,class_id]
      classes_level = tf.gather(classes_reshape, iou_index, axis=0)
      # 转编码
      boxes_level = self._boxes_encoder(anchor_level, boxes_level)
      boxes_level = tf.where(iou_mask,boxes_level,tf.zeros_like(boxes_level))
      classes_level = tf.where(iou_mask,classes_level,tf.zeros_like(classes_level))
      classes_level = tf.one_hot(tf.cast(tf.reshape(classes_level, tf.shape(classes_level)[:-1]), tf.int32), classes_num, dtype=tf.float32)
      output_boxes.append(boxes_level)
      output_classes.append(classes_level)
      output_mask.append(iou_mask)

    return tuple(output_boxes), tuple(output_classes), tuple(output_mask)

  @tf.function
  def convert_outputs_boxes(self, outputs_boxes):
    '''
    outputs转成boxes

    Args:
      outputs_boxes: [level,[batch_size,h,w,anchors,[ty, tx, th, tw]]]

    Returns:
      boxes: [level,[batch_size,h,w,anchors,[y1, x1, y2, x2]]]
    '''
    convert_boxes = []
    for level in range(len(self.boxes)):
      boxes_level = outputs_boxes[level]
      anchor_level = self.boxes[level]
      # tf.print('boxes_level:', tf.shape(boxes_level))
      # tf.print('anchor_level:', tf.shape(anchor_level))
      convert_boxes.append(self._boxes_decoder(anchor_level,boxes_level))
    return tuple(convert_boxes)

  @tf.function
  def convert_outputs_one(self, batch_index, outputs_boxes, outputs_classes):
    # tf.print('boxes_outputs:', boxes_outputs)
    # tf.print('outputs_classes:', outputs_classes)
    # for batch in tf.range(batch_size):
    nms_boxes = []
    nms_classes_id = []
    nms_scores = []
    for level in range(len(outputs_classes)):
      # 转换classes结果(boxes_num,)
      classes_outputs_item = outputs_classes[level][batch_index]
      # tf.print('classes_outputs_item:', classes_outputs_item)
      classes_id = tf.math.argmax(classes_outputs_item, axis=-1)
      # tf.print('classes_id:', classes_id)
      classes_scores = tf.math.reduce_max(classes_outputs_item, axis=-1)
      # tf.print('classes_scores:', classes_scores)
      # 转换boxes结果(boxes_num,4)
      boxes_outputs_item = outputs_boxes[level][batch_index]
      # 选出有效目标，去除背景
      classes_mask = tf.math.not_equal(classes_id,0)
      # tf.print('classes_mask:', classes_mask)
      nms_boxes.append(tf.boolean_mask(boxes_outputs_item,classes_mask))
      # tf.print('nms_boxes:', nms_boxes)
      nms_classes_id.append(tf.boolean_mask(classes_id,classes_mask))
      # tf.print('nms_classes_id:', nms_classes_id)
      nms_scores.append(tf.boolean_mask(classes_scores,classes_mask))
      # tf.print('nms_scores:', nms_scores)
    nms_boxes = tf.concat(nms_boxes,axis=0)
    nms_classes_id = tf.concat(nms_classes_id,axis=0)
    nms_scores = tf.concat(nms_scores,axis=0)
    # NMS去重
    nms_indexes = get_nms(nms_boxes,nms_scores,max_output_size=200,iou_threshold=0.5,score_threshold=0.0001,iou_type='diou')
    # tf.print('nms_indexes:', nms_indexes)
    # [目标数,4]
    nms_boxes = tf.gather(nms_boxes,nms_indexes)
    # tf.print('nms_boxes:', nms_boxes)
    # [目标数,]
    nms_classes_id = tf.gather(nms_classes_id,nms_indexes)
    # tf.print('nms_classes_id:', nms_classes_id)
    # [目标数,]
    nms_scores = tf.math.sigmoid(tf.gather(nms_scores,nms_indexes))
    # tf.print('nms_scores:', nms_scores)
    return nms_boxes, nms_classes_id, nms_scores

  def _get_center_coordinates_and_sizes(self, boxes):
    '''
    [y1, x1, y2, x2]转[y, x, h, w]
    '''
    ycenter = (boxes[..., 2] + boxes[..., 0]) / 2.0
    xcenter = (boxes[..., 3] + boxes[..., 1]) / 2.0
    h = boxes[..., 2] - boxes[..., 0]
    w = boxes[..., 3] - boxes[..., 1]
    ycenter = tf.expand_dims(ycenter, axis=-1)
    xcenter = tf.expand_dims(xcenter, axis=-1)
    h = tf.expand_dims(h, axis=-1)
    w = tf.expand_dims(w, axis=-1)

    return ycenter, xcenter, h, w
  
  def _boxes_encoder(self, anchors, boxes):
    """
    将框坐标根据anchors坐标编码

    Args:
      anchors: [h,w,anchors_num,[y1, x1, y2, x2]]
      boxes: [h,w,anchors_num,[y1, x1, y2, x2]]

    Returns:
      rel_codes: [h,w,anchors_num,[ty, tx, th, tw]].
    """
    # [y1, x1, y2, x2]转[y, x, h, w]
    ycenter_a, xcenter_a, ha, wa = self._get_center_coordinates_and_sizes(anchors)
    ycenter, xcenter, h, w = self._get_center_coordinates_and_sizes(boxes)
    # 防止除0
    ha = tf.maximum(EPSILON, ha)
    wa = tf.maximum(EPSILON, wa)
    h = tf.maximum(EPSILON, h)
    w = tf.maximum(EPSILON, w)

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.math.log(w / wa)
    th = tf.math.log(h / ha)
    return tf.concat([ty, tx, th, tw], axis=-1)

  def _boxes_decoder(self, anchors, rel_codes):
    """
    将框坐标根据anchors坐标编码

    Args:
      anchors: [h,w,anchors_num,[y1, x1, y2, x2]]
      rel_codes: [h,w,anchors_num,[ty, tx, th, tw]]

    Returns:
      boxes: [h,w,anchors_num,[y1, x1, y2, x2]].
    """
    ycenter_a, xcenter_a, ha, wa = self._get_center_coordinates_and_sizes(anchors)

    ty = rel_codes[..., 0]
    tx = rel_codes[..., 1]
    th = rel_codes[..., 2]
    tw = rel_codes[..., 3]
    ty = tf.expand_dims(ty, axis=-1)
    tx = tf.expand_dims(tx, axis=-1)
    th = tf.expand_dims(th, axis=-1)
    tw = tf.expand_dims(tw, axis=-1)
    w = tf.math.exp(tw) * wa
    h = tf.math.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return tf.concat([ymin, xmin, ymax, xmax], axis=-1)
