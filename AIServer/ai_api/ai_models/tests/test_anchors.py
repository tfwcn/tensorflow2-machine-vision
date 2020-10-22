import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.efficientnet.utils.anchors import Anchors
import tensorflow as tf

def main():
  a = Anchors(3, 7, (512, 512), 3, [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)], 3)
  # print(a.boxes[0][11][2])
  boxes = tf.constant([[10,10,31,32],[20,22,41,45]],dtype=tf.float32)
  classes = tf.constant([1,2])
  output_boxes, output_classes = a.generate_targets(boxes, classes)
  for i in range(len(output_boxes)):
    tf.print(tf.shape(output_boxes[i]), tf.shape(output_classes[i]))
    tf.print(tf.math.reduce_sum(output_classes[i]))
  # tf.print(output_boxes[4], output_classes[4])

if __name__ == '__main__':
  main()