import tensorflow as tf
import math


def _get_v(b1_height, b1_width, b2_height, b2_width):
  """Get the consistency measurement of aspect ratio for ciou."""

  # 梯度为NaN时，执行后面函数
  @tf.custom_gradient
  def _get_grad_v(height, width):
    """backpropogate gradient."""
    arctan = tf.math.atan(tf.math.divide_no_nan(b1_width, b1_height)) - tf.atan(
        tf.math.divide_no_nan(width, height))
    v = 4 * ((arctan / math.pi)**2)

    def _grad_v_graph(dv, variables):
      """Grad for graph mode."""
      gdw = dv * 8 * arctan * height / (math.pi**2)
      gdh = -dv * 8 * arctan * width / (math.pi**2)
      return [gdh, gdw], tf.gradients(v, variables, grad_ys=dv)

    return v, _grad_v_graph

  return _get_grad_v(b2_height, b2_width)

@tf.function
def get_iou(boxes1, boxes2, iou_type = 'iou'):
  """
  计算IoU
  注意这里yx是反的，对应维度位置

  Args:
    boxes1: [..., [y_min, x_min, y_max, x_max]]
    boxes2: [..., [y_min, x_min, y_max, x_max]]
    iou_type: ['iou', 'ciou', 'diou', 'giou']其中一个

  Returns:
    IoU: [...,], 会将boxes1与boxes2相同维度不变，把各自为1的维度扩展到其中的最大值。
  """
  # t_ denotes target boxes and p_ denotes predicted boxes.
  b1_ymin = boxes1[..., 0]
  b1_xmin = boxes1[..., 1]
  b1_ymax = boxes1[..., 2]
  b1_xmax = boxes1[..., 3]
  b2_ymin = boxes2[..., 0]
  b2_xmin = boxes2[..., 1]
  b2_ymax = boxes2[..., 2]
  b2_xmax = boxes2[..., 3]

  zero = tf.zeros_like(b1_xmin, b1_xmin.dtype)
  b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
  b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
  b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
  b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
  b1_area = b1_width * b1_height
  b2_area = b2_width * b2_height

  intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
  intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
  intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
  intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
  intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
  intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
  intersect_area = intersect_width * intersect_height

  union_area = b1_area + b2_area - intersect_area
  iou_v = tf.math.divide_no_nan(intersect_area, union_area)
  if iou_type == 'iou':
    return iou_v  # iou is the simplest form.

  enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
  enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
  enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
  enclose_xmax = tf.maximum(b1_xmax, b2_xmax)

  assert iou_type in ('giou', 'diou', 'ciou')
  if iou_type == 'giou':  # giou is the generalized iou.
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    giou_v = iou_v - tf.math.divide_no_nan(
        (enclose_area - union_area), enclose_area)
    return giou_v

  assert iou_type in ('diou', 'ciou')
  b1_center = tf.stack([(b1_ymin + b1_ymax) / 2, (b1_xmin + b1_xmax) / 2], axis=-1)
  b2_center = tf.stack([(b2_ymin + b2_ymax) / 2, (b2_xmin + b2_xmax) / 2], axis=-1)
  euclidean = tf.linalg.norm(b2_center - b1_center, axis=-1)
  diag_length = tf.linalg.norm(
      tf.stack([enclose_ymax - enclose_ymin, enclose_xmax - enclose_xmin],
               axis=-1),
      axis=-1)
  diou_v = iou_v - tf.math.divide_no_nan(euclidean**2, diag_length**2)
  if iou_type == 'diou':  # diou is the distance iou.
    return diou_v

  assert iou_type == 'ciou'
  v = _get_v(b1_height, b1_width, b2_height, b2_width)
  alpha = tf.math.divide_no_nan(v, ((1 - iou_v) + v))
  return diou_v - alpha * v  # the last one is ciou.


def main():
  boxes1 = tf.constant([[10,10,11,12]], dtype=tf.float32)
  # boxes2 = tf.constant([[10.5,10,13,12]], dtype=tf.float32)
  boxes2 = tf.constant([[10,10,11,12]], dtype=tf.float32)
  iou = get_iou(boxes1, boxes2, iou_type='ciou')
  print(iou)

if __name__ == '__main__':
  main()