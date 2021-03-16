import unittest
import tensorflow as tf
import math
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())

from ai_api.ai_models.losses.yolo_loss import Yolov4Loss
from ai_api.ai_models.utils.tf_iou_utils import GetIOU

class LossTest(unittest.TestCase):
  """用于测试loss"""
        
  @tf.function
  def GetLoss(self, y_true, y_pred):
    '''
    获取损失值
    y_true:坐标还没归一化，[(batch_size, 13, 13, 3, 5+num_classes), (batch_size, 26, 26, 3, 5+num_classes), (batch_size, 52, 52, 3, 5+num_classes)]
    y_pred:[(batch_size, 13, 13, 3, 5+num_classes), (batch_size, 26, 26, 3, 5+num_classes), (batch_size, 52, 52, 3, 5+num_classes)]
    '''
    print('loss_fun:', type(y_true), type(y_pred))
    layers_size = [[13, 13], [26, 26], [52, 52]]
    anchors_wh = [
        [[116, 90], [156, 198], [373, 326]],
        [[30, 61], [62, 45], [59, 119]],
        [[10, 13], [16, 30], [33, 23]],
    ]
    classes_num=80
    train_iou_thresh=0.5
    image_size = tf.constant((416,416), dtype=tf.float32)
    # (layers_num, anchors_num, 2)
    anchors_wh = tf.constant(anchors_wh, dtype=tf.float32)
    # anchors_wh = anchors_wh / image_size
    anchors_num = tf.shape(anchors_wh)[1]
    layers_size = tf.constant(layers_size, dtype=tf.int32)
    layers_num = tf.shape(layers_size)[0]
    classes_num = tf.constant(classes_num, dtype=tf.int32)
    batch_size = tf.shape(y_true[0])[0]
    batch_size_float = tf.cast(batch_size, dtype=tf.float32)
    loss = 0.0
    layer_index = 0
    for layer_index in range(3):
      y_true_read = y_true[layer_index]
      y_pred_raw = y_pred[layer_index]
      y_pred_raw = tf.reshape(y_pred_raw, tf.shape(y_true_read))
      # 特征网格对应实际图片的坐标
      grid_shape = tf.shape(y_pred_raw)[1:3] # height, width
      grid_y = tf.range(0, grid_shape[0], dtype=tf.float32)
      grid_x = tf.range(0, grid_shape[1], dtype=tf.float32)
      grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
      grid_x = tf.reshape(grid_x, (grid_shape[0], grid_shape[1], 1, 1))
      grid_y = tf.reshape(grid_y, (grid_shape[0], grid_shape[1], 1, 1))
      grid_xy = tf.concat([grid_x, grid_y], axis=-1)
      # 计算真实坐标与相对坐标
      # y_true
      y_true_object = y_true_read[..., 4:5]
      y_true_classes = y_true_read[..., 5:]
      y_true_read_xy = y_true_read[..., 0:2]
      # tf.print('grid_xy:', tf.math.reduce_max(grid_xy), tf.math.reduce_min(grid_xy))
      # tf.print('grid_shape:', grid_shape[::-1])
      y_true_raw_xy = y_true_read_xy * tf.cast(grid_shape[::-1], dtype=tf.float32) - grid_xy
      # tf.print('y_true_raw_xy:', tf.math.reduce_max(y_true_raw_xy), tf.math.reduce_min(y_true_raw_xy))
      # tf.print('y_true_object:', tf.math.reduce_max(y_true_object), tf.math.reduce_min(y_true_object))
      # y_true_raw_xy = y_true_object * y_true_raw_xy
      # tf.print('y_true_raw_xy:', tf.math.reduce_max(y_true_raw_xy), tf.math.reduce_min(y_true_raw_xy))
      
      y_true_read_wh = y_true_read[..., 2:4]
      y_true_raw_wh = tf.math.log(y_true_read_wh * image_size[::-1] / anchors_wh[layer_index, ...])
      y_true_raw_wh = tf.where(tf.cast(y_true_object, dtype=tf.bool), y_true_raw_wh, tf.zeros_like(y_true_raw_wh))
      # tf.print('y_true_raw_wh:', tf.math.reduce_max(y_true_raw_wh), tf.math.reduce_min(y_true_raw_wh))
      
      # y_pred
      y_pred_object = y_pred_raw[..., 4:5]
      y_pred_classes = y_pred_raw[..., 5:]
      y_pred_raw_xy = y_pred_raw[..., 0:2]
      # tf.print('y_pred_raw_xy:', tf.math.reduce_max(y_pred_raw_xy), tf.math.reduce_min(y_pred_raw_xy))
      y_pred_read_xy = (tf.math.sigmoid(y_pred_raw_xy) + grid_xy) / tf.cast(grid_shape[::-1], dtype=tf.float32)
      
      y_pred_raw_wh = y_pred_raw[..., 2:4]
      # tf.print('y_pred_raw_wh:', tf.math.reduce_max(y_pred_raw_wh), tf.math.reduce_min(y_pred_raw_wh))
      y_pred_read_wh = tf.math.exp(y_pred_raw_wh) * anchors_wh[layer_index, ...] / image_size[::-1]
      # y_pred_read_wh = tf.where(tf.math.is_inf(y_pred_read_wh), tf.zeros_like(y_pred_read_wh), y_pred_read_wh)
      
      # y_pred_object = tf.math.sigmoid(y_pred_object)
      # y_pred_classes = tf.math.sigmoid(y_pred_classes)

      # 框坐标(batch_size, h, w, anchors_num, (x1, y1, x2, y2))
      y_true_read_wh_half = y_true_read_wh / 2
      y_true_read_mins = y_true_read_xy - y_true_read_wh_half
      y_true_read_maxes = y_true_read_xy + y_true_read_wh_half
      y_true_boxes = tf.concat([y_true_read_mins, y_true_read_maxes], axis=-1)
      y_pred_read_wh_half = y_pred_read_wh / 2
      y_pred_read_mins = y_pred_read_xy - y_pred_read_wh_half
      y_pred_read_maxes = y_pred_read_xy + y_pred_read_wh_half
      y_pred_boxes = tf.concat([y_pred_read_mins, y_pred_read_maxes], axis=-1)
      
      ignore_mask = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
      def foreach_batch(batch_index, ignore_mask):
        y_true_boxes_one = y_true_boxes[batch_index, ...]
        y_pred_boxes_one = y_pred_boxes[batch_index, ...]
        y_true_object_one = y_true_object[batch_index, ...]
        y_true_boxes_tmp = tf.boolean_mask(y_true_boxes_one, tf.cast(y_true_object_one[..., 0], dtype=tf.bool))
        # 计算IOU
        # (boxes_num, 4) => (1, boxes_num, 4)
        y_true_boxes_tmp = tf.expand_dims(y_true_boxes_tmp, axis=0)
        y_pred_boxes_tmp = y_pred_boxes_one
        # (h, w, anchors_num, 4) => (h, w, anchors_num, 1, 4)
        y_pred_boxes_tmp = tf.expand_dims(y_pred_boxes_tmp, axis=-2)
        # (h, w, anchors_num, boxes_num)
        iou = GetIOU(y_pred_boxes_tmp, y_true_boxes_tmp, 'iou')
        # (h, w, anchors_num)
        best_iou = tf.math.reduce_max(iou, axis=-1)
        # 把IOU<0.5的认为是背景
        ignore_mask = ignore_mask.write(batch_index, tf.cast(best_iou < train_iou_thresh, dtype=tf.float32))
        return batch_index + 1, ignore_mask
      # (batch_size, h, w, anchors_num, y_true_boxes_num)
      _, ignore_mask = tf.while_loop(lambda b,*args: b<batch_size, foreach_batch, [0, ignore_mask])
      ignore_mask = ignore_mask.stack()
      # (batch_size, h, w, anchors_num)
      ignore_mask = tf.expand_dims(ignore_mask, axis=-1)
      # ignore_mask = tf.where(tf.math.is_nan(ignore_mask), tf.zeros_like(ignore_mask), ignore_mask)
      # tf.print('ignore_mask:', tf.math.reduce_max(ignore_mask), tf.math.reduce_min(ignore_mask))
      # 计算loss
      boxes_loss_scale = 2 - y_true_read_wh[..., 0:1] * y_true_read_wh[..., 1:2]
      # tf.print('boxes_loss_scale:', tf.math.reduce_max(boxes_loss_scale), tf.math.reduce_min(boxes_loss_scale))
      
      xy_loss_bc = tf.keras.losses.binary_crossentropy(tf.expand_dims(y_true_raw_xy, axis=-1),
              tf.expand_dims(y_pred_raw_xy, axis=-1), from_logits=True)
      xy_loss = y_true_object * boxes_loss_scale * xy_loss_bc
      wh_loss = y_true_object * boxes_loss_scale * 0.5 * tf.math.square(y_true_raw_wh - y_pred_raw_wh)
      object_loss_bc = tf.keras.losses.binary_crossentropy(tf.expand_dims(y_true_object, axis=-1),
              tf.expand_dims(y_pred_object, axis=-1), from_logits=True)
      # tf.print('object_loss_bc:', tf.math.reduce_max(object_loss_bc), tf.math.reduce_min(object_loss_bc))
      object_loss = y_true_object * object_loss_bc + (1 - y_true_object) * object_loss_bc * ignore_mask
      classes_loss_bc = tf.keras.losses.binary_crossentropy(tf.expand_dims(y_true_classes, axis=-1),
              tf.expand_dims(y_pred_classes, axis=-1), from_logits=True)
      # tf.print('classes_loss_bc:', tf.math.reduce_max(classes_loss_bc), tf.math.reduce_min(classes_loss_bc))
      classes_loss = y_true_object * classes_loss_bc

      xy_loss = tf.math.reduce_sum(xy_loss) / batch_size_float
      wh_loss = tf.math.reduce_sum(wh_loss) / batch_size_float
      object_loss = tf.math.reduce_sum(object_loss) / batch_size_float
      classes_loss = tf.math.reduce_sum(classes_loss) / batch_size_float
      # tf.print('loss:', xy_loss, wh_loss, object_loss, classes_loss)
      loss += xy_loss + wh_loss + object_loss + classes_loss
    # tf.print('loss:', loss)
    return loss

  def test_loss(self):
    """测试loss"""
    anchors = [10,13,  16,30,  33,23,
               30,61,  62,45,  59,119,
               116,90,  156,198,  373,326]
    anchors = np.array(anchors).reshape(-1, 2)
    yolo_v4_loss = Yolov4Loss(anchors,80)
    y_true=[
      tf.random.uniform((1,13,13,3,85),dtype=tf.float32),
      tf.random.uniform((1,26,26,3,85),dtype=tf.float32),
      tf.random.uniform((1,52,52,3,85),dtype=tf.float32)
    ]
    y_pred=[
      tf.random.uniform((1,13,13,255),dtype=tf.float32),
      tf.random.uniform((1,26,26,255),dtype=tf.float32),
      tf.random.uniform((1,52,52,255),dtype=tf.float32)
    ]
    out1 = self.GetLoss(y_true,y_pred).numpy()
    out2 = yolo_v4_loss(y_true,y_pred).numpy()
    print(out1.shape,out2.shape)
    self.assertEqual(out1,out2)

unittest.main()