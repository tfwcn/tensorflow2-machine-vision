import re
import tensorflow as tf

from ai_api.ai_models.efficientnet.efficientdet_net import EfficientDetNet
from ai_api.ai_models.efficientnet.utils.anchors import Anchors
from ai_api.ai_models.efficientnet.utils.nms import get_nms
from ai_api.ai_models.losses.box_loss import BoxLoss
from ai_api.ai_models.losses.focal_loss import FocalLoss
from ai_api.ai_models.utils.mAP import Get_mAP_one

class EfficientDetNetTrain(EfficientDetNet):

  def __init__(self, blocks_args, global_params, anchors: Anchors, name=''):
    """Initialize model."""
    super().__init__(blocks_args=blocks_args, global_params=global_params, name=name)
    self.anchors = anchors
    self.box_loss = BoxLoss()
    self.focal_loss = FocalLoss(self._global_params.alpha, self._global_params.gamma, label_smoothing=0.0)

  def _reg_l2_loss(self, weight_decay, regex=r'.*(kernel|weight):0$'):
    """Return regularization l2 loss loss."""
    var_match = re.compile(regex)
    return weight_decay * tf.add_n([
        tf.nn.l2_loss(v)
        for v in self.trainable_variables
        if var_match.match(v.name)
    ])

  @tf.function
  def train_step(self, data):
    '''训练'''
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    # print('data:', data)
    x, y_true_boxes, y_true_classes, y_true_masks = data
    loss = self._reg_l2_loss(4e-5)
    with tf.GradientTape() as tape:
        y_pred_classes, y_pred_boxes = self(x, training=True)
        loss_b = self.box_loss(y_true_boxes, (y_pred_boxes, y_true_masks))
        loss_c = self.focal_loss(y_true_classes, (y_pred_classes, y_true_masks))
        # tf.print('loss:', loss_b, loss_c)
        loss += loss_b * 50.0 + loss_c

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # tf.clip_by_global_norm，将梯度限制在10.0内，防止loss异常导致梯度爆炸
    gradients, gnorm = tf.clip_by_global_norm(gradients,
                                              10.0)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    return {'loss': loss, 'gnorm': gnorm}

  @tf.function
  def test_step(self, data):
    '''评估'''
    # x: [batch_size,h,w,3]
    # y_true_boxes: [batch_size,obj_num,[y1,x1,y2,x2]]
    # y_true_classes: [batch_size,obj_num,]
    x, y_real_boxes, y_real_classes, y_true_boxes, y_true_classes, y_true_masks = data
    batch_size = tf.shape(x)[0]
    y_pred_classes, y_pred_boxes = self(x, training=False)
    loss = self._reg_l2_loss(4e-5)
    loss_b = self.box_loss(y_true_boxes, (y_pred_boxes, y_true_masks))
    loss_c = self.focal_loss(y_true_classes, (y_pred_classes, y_true_masks))
    # tf.print('loss:', loss_b, loss_c)
    loss += loss_b * 50.0 + loss_c
    boxes_outputs = self.anchors.convert_outputs_boxes(y_pred_boxes)
    mAP = tf.constant(0.0, dtype=tf.float32)
    for batch in tf.range(batch_size):
      nms_boxes = []
      nms_classes_id = []
      nms_scores = []
      for level in range(len(y_pred_classes)):
        # 转换classes结果(boxes_num,)
        classes_outputs_item = y_pred_classes[level][batch]
        classes_id = tf.math.argmax(classes_outputs_item, axis=-1)
        classes_scores = tf.math.reduce_max(classes_outputs_item, axis=-1)
        # 转换boxes结果(boxes_num,4)
        boxes_outputs_item = boxes_outputs[level][batch]
        # 选出有效目标，去除背景
        classes_mask = tf.math.not_equal(classes_id,0)
        nms_boxes.append(tf.boolean_mask(boxes_outputs_item,classes_mask))
        nms_classes_id.append(tf.boolean_mask(classes_id,classes_mask))
        nms_scores.append(tf.boolean_mask(classes_scores,classes_mask))
      nms_boxes = tf.concat(nms_boxes,axis=0)
      nms_classes_id = tf.concat(nms_classes_id,axis=0)
      nms_scores = tf.concat(nms_scores,axis=0)
      # NMS去重
      nms_indexes = get_nms(nms_boxes,nms_scores,max_output_size=200,iou_threshold=0.5,score_threshold=0.0001,iou_type='diou')
      # [目标数,4]
      nms_boxes = tf.gather(nms_boxes,nms_indexes)
      # [目标数,]
      nms_classes_id = tf.gather(nms_classes_id,nms_indexes)
      # [目标数,]
      nms_scores = tf.gather(nms_scores,nms_indexes)

      prediction = tf.concat([nms_boxes,
        tf.cast(tf.expand_dims(nms_classes_id, axis=-1), dtype=tf.float32),
        tf.expand_dims(nms_scores, axis=-1)], axis=-1)
      groud_truth = tf.concat([y_real_boxes[batch],
        tf.cast(tf.expand_dims(y_real_classes[batch], axis=-1), dtype=tf.float32)], axis=-1)
      mAP_one = tf.numpy_function(Get_mAP_one, (groud_truth, prediction, self._global_params.num_classes, 0.5), tf.float64)
      mAP += tf.cast(tf.reshape(mAP_one,()), tf.float32)
    mAP /= tf.cast(batch_size, dtype=tf.float32)
    return {'loss': loss, 'mAP': mAP}
    