import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.efficientnet.utils.anchors import Anchors
from ai_api.ai_models.efficientnet.utils.iou import get_iou
import tensorflow as tf

def main():
  # a = Anchors(0, 7, (64, 64), 3, [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)], 3.0)
  a = Anchors(0, 0, (10, 10), 3, [(1.0, 1.0)], 3.0)
  # print(a.boxes[0][11][2])
  # boxes = tf.constant([[10,10,20,20]],dtype=tf.float32)
  boxes = tf.constant([[3,3,6,6],[5,5,9,9]],dtype=tf.float32)
  classes = tf.constant([1,2])
  output_boxes, output_classes, output_mask = a.generate_targets(boxes, classes, 3, iou_threshold=0.5)
  output_boxes = list(output_boxes)
  output_classes = list(output_classes)
  output_mask = list(output_mask)
  for i in range(len(output_boxes)):
    tf.print(i)
    output_boxes[i] = tf.expand_dims(output_boxes[i], axis=0)
    output_classes[i] = tf.expand_dims(output_classes[i], axis=0)
    output_mask[i] = tf.expand_dims(output_mask[i], axis=0)
    tf.print('output_shape:', tf.shape(output_boxes[i]), tf.shape(output_classes[i]), tf.shape(output_mask[i]))
    # tf.print('output_boxes:', output_boxes[i])
    # tf.print('output_classes:', output_classes[i])
    # tf.print('output_mask:', output_mask)

  output_boxes = a.convert_outputs_boxes(output_boxes)
  # a.convert_outputs(output_boxes, output_classes)
  convert_boxes, convert_classes_id, convert_scores = a.convert_outputs_one(0, output_boxes, output_classes)
  tf.print('convert_boxes:', convert_boxes)
  tf.print('convert_classes_id:', convert_classes_id)
  tf.print('convert_scores:', convert_scores)

if __name__ == '__main__':
  main()
  
  # # 用实际图片大小，特征步长，计算中心坐标
  # stride = 2
  # x = tf.range(stride / 2, 10, stride)
  # y = tf.range(stride / 2, 10, stride)
  # xv, yv = tf.meshgrid(x, y)
  # print('xv:', xv)
  # print('yv:', yv)

  # # 计算boxes左上和右下坐标，(h,w,(y1,x1,y2,x2))
  # yv = tf.expand_dims(yv, axis=-1)
  # xv = tf.expand_dims(xv, axis=-1)
  # boxes = tf.concat([yv - stride / 2, xv - stride / 2,
  #                     yv + stride / 2, xv + stride / 2], axis=-1)
  # # print('boxes:', ((boxes[...,2]-boxes[...,0])*(boxes[...,3]-boxes[...,1])))
  # print('boxes:', boxes)

  # # (h,w,1,(y1,x1,y2,x2))
  # boxes = tf.expand_dims(boxes, axis=-2)
  # # (boxes_num,(y1,x1,y2,x2))
  # boxes2 = tf.constant([[0,0,2,2],[1,1,3,3]], dtype=tf.float32)
  # iou = get_iou(boxes,boxes2)
  # print('iou:', iou)
  # iou_index = tf.math.argmax(iou, axis=-1)
  # print('iou_index:', iou_index)
  # iou_max = tf.math.reduce_max(iou, axis=-1)
  # print('iou_max:', iou_max)
  # # iou_mask = tf.expand_dims(tf.math.greater_equal(iou_max, 0.5), axis=-1)
  # iou_mask = tf.math.greater_equal(iou_max, 0.5)
  # print('iou_mask:', iou_mask)
  # print('iou_mask:', tf.where(iou_mask,iou_max,tf.zeros_like(iou_max)))
