import tensorflow as tf
import math


@tf.function
def GetIOU(b1, b2, iou_type='iou'):
  '''
  计算IOU,DIOU,CIOU
  b1与b2前面维度一样或缺少也可以，返回结果会自动把1维度长度补全

  Args:
    b1:(..., b1_num, 1, (x1, y1, x2, y2))
    b2:(1, b2_num, (x1, y1, x2, y2))
    iou_type:iou、diou、ciou
  Results:
    return:(..., b1_num, b2_num)
  '''
  assert iou_type in ['iou','diou','ciou']
  # (..., b2_num, b1_num, 2)
  intersect_mins = tf.math.maximum(b1[..., 0:2], b2[..., 0:2])
  intersect_maxes = tf.math.minimum(b1[..., 2:4], b2[..., 2:4])
  intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins, 0.)
  # (..., b2_num, b1_num)
  intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
  # (1, b1_num, 2)
  b1_wh = b1[..., 2:4] - b1[..., 0:2]
  # (..., b2_num, 1, 2)
  b2_wh = b2[..., 2:4] - b2[..., 0:2]
  # (1, b1_num)
  b1_area = b1_wh[..., 0] * b1_wh[..., 1]
  # (..., b2_num, 1)
  b2_area = b2_wh[..., 0] * b2_wh[..., 1]
  # (h, w, anchors_num, boxes_num)
  iou = intersect_area / (b1_area + b2_area - intersect_area)
  # tf.print('iou:', tf.math.reduce_max(iou), tf.math.reduce_min(iou))
  if iou_type == 'iou':
    return iou
  # 最小外界矩形
  ub_mins = tf.math.minimum(b1[..., 0:2], b2[..., 0:2])
  ub_maxes = tf.math.maximum(b1[..., 2:4], b2[..., 2:4])
  ub_wh = ub_maxes - ub_mins
  c = tf.math.square(ub_wh[..., 0]) + tf.math.square(ub_wh[..., 1])
  # 计算中心距离
  b1_xy = (b1[..., 2:4] + b1[..., 0:2]) / 2
  b2_xy = (b2[..., 2:4] + b2[..., 0:2]) / 2
  u = tf.math.reduce_sum(tf.math.square(b1_xy - b2_xy), axis=-1)
  # 中心距离越近，d值越小
  d = u / c
  # tf.print('d:', tf.math.reduce_max(d), tf.math.reduce_min(d))
  diou = iou - pow(d, 0.6)
  diou = tf.where(c==0.0, iou, diou)
  if iou_type == 'diou':
    return diou
  # 两个框宽高比越接近，v值越小
  v = 4 / tf.math.square(math.pi) * tf.math.square(tf.math.atan(b1_wh[..., 0] / b1_wh[..., 1]) - tf.math.atan(b2_wh[..., 0] / b2_wh[..., 1]))
  # tf.print('v:', tf.math.reduce_max(v), tf.math.reduce_min(v))
  alpha = v / (1 - iou + v + 1e-8)
  # tf.print('alpha:', tf.math.reduce_max(alpha), tf.math.reduce_min(alpha))
  # 目标不相交时，为负值。目标重叠时为0。
  ciou = iou - (d + alpha * v)
  ciou = tf.where(c==0.0, iou, ciou)
  # tf.print('iou:', tf.math.reduce_max(iou), tf.math.reduce_min(iou))
  # tf.print('diou:', tf.math.reduce_max(diou), tf.math.reduce_min(diou))
  # tf.print('ciou:', tf.math.reduce_max(ciou), tf.math.reduce_min(ciou))
  return ciou

@tf.function
def GetIOUNMS(
  boxes,
  scores,
  max_output_size,
  iou_threshold=0.5,
  iou_type='iou'):
  '''通过NMS筛选框'''
  # 分数倒序下标
  scores_sort_indexes = tf.argsort(scores, direction='DESCENDING')
  # 排序后的框
  boxes_sort = tf.gather(boxes, scores_sort_indexes)
  # NMS后的下标
  result_indexes = tf.TensorArray(tf.int32, 0, dynamic_size=True)
  def boxes_foreach(idx, result_indexes, boxes_sort, scores_sort_indexes):
    # 已达到最大输出框数量
    if idx >= max_output_size:
      return -1, result_indexes, boxes_sort, scores_sort_indexes
    boxes_num = tf.shape(boxes_sort)[0]
    # 已遍历所有框
    if boxes_num == 0:
      return -1, result_indexes, boxes_sort, scores_sort_indexes
    # 取最高分的box
    boxes_top = boxes_sort[0:1, :]
    indexes_top = scores_sort_indexes[0]
    if boxes_num > 1:
      # 计算IOU
      boxes_other = boxes_sort[1:, :]
      indexes_other = scores_sort_indexes[1:]
      iou = GetIOU(boxes_top, boxes_other, iou_type)
      # 过滤掉小于阈值的数据
      iou_mask = iou < iou_threshold
      boxes_sort = tf.boolean_mask(boxes_other, iou_mask)
      scores_sort_indexes = tf.boolean_mask(indexes_other, iou_mask)
      result_indexes = result_indexes.write(idx, indexes_top)
      return idx + 1, result_indexes, boxes_sort, scores_sort_indexes
    else:
      result_indexes = result_indexes.write(idx, indexes_top)
      return -1, result_indexes, boxes_sort, scores_sort_indexes
  _, result_indexes, _, _ = tf.while_loop(lambda i, *args: tf.math.not_equal(i, -1), boxes_foreach, [0, result_indexes, boxes_sort, scores_sort_indexes])
  result_indexes = result_indexes.stack()
  return result_indexes

@tf.function
def GetIOUNMSByClasses(
  boxes,
  scores,
  classes,
  max_output_size,
  iou_threshold=0.5,
  iou_type='iou'):
  '''分类计算NMS'''
  # 分数倒序下标
  scores_sort_indexes = tf.argsort(scores, direction='DESCENDING')
  # 排序后的框
  boxes_sort = tf.gather(boxes, scores_sort_indexes)
  # 排序后的类型ID
  classes_sort = tf.gather(classes, scores_sort_indexes)
  # NMS后的下标
  result_indexes = tf.TensorArray(tf.int32, 0, dynamic_size=True)
  def boxes_foreach(idx, result_indexes, boxes_sort, classes_sort, scores_sort_indexes):
    # 已达到最大输出框数量
    if idx >= max_output_size:
      return -1, result_indexes, boxes_sort, classes_sort, scores_sort_indexes
    boxes_num = tf.shape(boxes_sort)[0]
    # 已遍历所有框
    if boxes_num == 0:
      return -1, result_indexes, boxes_sort, classes_sort, scores_sort_indexes
    # 取最高分的box
    boxes_top = boxes_sort[0:1, :]
    indexes_top = scores_sort_indexes[0]
    classes_top = classes_sort[0]
    if boxes_num > 1:
      # 计算IOU
      boxes_other = boxes_sort[1:, :]
      indexes_other = scores_sort_indexes[1:]
      classes_other = classes_sort[1:]
      iou = GetIOU(boxes_top, boxes_other, iou_type)
      # 过滤掉小于阈值的数据
      iou_mask = tf.math.logical_not(tf.math.logical_and(iou >= iou_threshold, classes_other==classes_top))
      boxes_sort = tf.boolean_mask(boxes_other, iou_mask)
      scores_sort_indexes = tf.boolean_mask(indexes_other, iou_mask)
      classes_sort = tf.boolean_mask(classes_other, iou_mask)
      result_indexes = result_indexes.write(idx, indexes_top)
      return idx + 1, result_indexes, boxes_sort, classes_sort, scores_sort_indexes
    else:
      result_indexes = result_indexes.write(idx, indexes_top)
      return -1, result_indexes, boxes_sort, classes_sort, scores_sort_indexes
  _, result_indexes, _, _, _ = tf.while_loop(lambda i, *args: tf.math.not_equal(i, -1), boxes_foreach, [0, result_indexes, boxes_sort, classes_sort, scores_sort_indexes])
  result_indexes = result_indexes.stack()
  return result_indexes