import tensorflow as tf
from ai_api.ai_models.efficientnet.utils.iou import get_iou

@tf.function
def get_nms(boxes,
            scores,
            max_output_size,
            iou_threshold=0.5,
            score_threshold=float('-inf'),
            iou_type='diou'):
  '''
  计算DIOU NMS

  Args:
    boxes: [boxes_num, 4]
    scores: [boxes_num,]
    max_output_size: 最大输出数
    iou_threshold: IOU阈值
    score_threshold: 分数阈值
    iou_type: ['iou', 'ciou', 'diou', 'giou']其中一个

  Returns:
    boxes_indexes: 适合的box下标列表
  '''
  # 分数倒序下标
  scores_sort_indexes = tf.argsort(scores, direction='DESCENDING')
  # 排序后的框
  boxes_sort = tf.gather(boxes, scores_sort_indexes)
  # NMS后的下标
  result_indexes = tf.TensorArray(tf.int32, 0, dynamic_size=True)
  def boxes_foreach(idx, result_indexes, boxes_sort, scores_sort_indexes):
    '''取最高分的box'''
    # 达到最大输出数量，退出搜索
    if idx >= max_output_size:
      return -1, result_indexes, boxes_sort, scores_sort_indexes
    # 已搜索所有框，退出搜索
    boxes_num = tf.shape(boxes_sort)[0]
    if boxes_num == 0:
      return -1, result_indexes, boxes_sort, scores_sort_indexes
    boxes_top = boxes_sort[0:1, :]
    indexes_top = scores_sort_indexes[0]
    scores_top = scores[indexes_top]
    # 过滤分数
    if scores_top < score_threshold:
      return -1, result_indexes, boxes_sort, scores_sort_indexes
    if boxes_num > 1:
      # 计算IOU
      boxes_other = boxes_sort[1:, :]
      indexes_other = scores_sort_indexes[1:]
      iou = get_iou(boxes_top, boxes_other, iou_type=iou_type)
      iou_mask = iou < iou_threshold
      boxes_sort = tf.boolean_mask(boxes_other, iou_mask)
      scores_sort_indexes = tf.boolean_mask(indexes_other, iou_mask)
      result_indexes = result_indexes.write(idx, indexes_top)
      return idx + 1, result_indexes, boxes_sort, scores_sort_indexes
    else:
      result_indexes = result_indexes.write(idx, indexes_top)
      return -1, result_indexes, boxes_sort, scores_sort_indexes
  _, result_indexes, _, _ = tf.while_loop(lambda i, x1, x2, x3: tf.math.not_equal(i, -1), boxes_foreach, [0, result_indexes, boxes_sort, scores_sort_indexes])
  result_indexes = result_indexes.stack()
  return result_indexes
