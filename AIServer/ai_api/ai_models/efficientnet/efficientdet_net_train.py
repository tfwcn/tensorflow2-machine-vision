import re
import tensorflow as tf

from ai_api.ai_models.efficientnet.efficientdet_net import EfficientDetNet
from ai_api.ai_models.efficientnet.utils.anchors import Anchors
from ai_api.ai_models.efficientnet.utils.nms import get_nms
from ai_api.ai_models.losses.box_loss import BoxLoss
from ai_api.ai_models.losses.focal_loss import FocalLoss
from ai_api.ai_models.utils.mAP import Get_mAP_one

class EfficientDetNetTrain(EfficientDetNet):

  def __init__(self, blocks_args, global_params, anchors: Anchors, name='', min_lr=1e-6):
    """Initialize model."""
    super().__init__(blocks_args=blocks_args, global_params=global_params, name=name)
    self.anchors = anchors
    self.box_loss = BoxLoss()
    self.focal_loss = FocalLoss(self._global_params.alpha, self._global_params.gamma, label_smoothing=0.0)
    self.min_lr = min_lr

  def _reg_l2_loss(self, weight_decay, regex=r'.*(kernel|weight):0$'):
    """Return regularization l2 loss loss."""
    var_match = re.compile(regex)
    return weight_decay * tf.add_n([
        tf.nn.l2_loss(v)
        for v in self.trainable_variables
        if var_match.match(v.name)
    ])
    
  def build(self, input_shape):
    result = super().build(input_shape)
    trainable_vars = self.trainable_variables
    self.bak_trainable_vars = []
    for var in trainable_vars:
      self.bak_trainable_vars.append(tf.Variable(var, trainable=False))
    self.bak_trainable_last_vars = []
    for var in trainable_vars:
      self.bak_trainable_last_vars.append(tf.Variable(var, trainable=False))
    return result

  def _get_loss(self, y_true_boxes, y_true_classes, y_true_masks, y_pred_boxes, y_pred_classes):
    loss = self._reg_l2_loss(4e-5)
    num_positives_sum = 0.0
    for level in range(len(y_true_boxes)):
      num_positives_sum += tf.reduce_sum(tf.cast(y_true_masks[level], tf.float32))
    num_positives_sum += 1.0
    for level in range(len(y_true_boxes)):
      loss_b = self.box_loss([num_positives_sum, y_true_boxes[level]], y_pred_boxes[level])
      loss_c = self.focal_loss([num_positives_sum, y_true_classes[level]], y_pred_classes[level])
      # tf.print('loss:', loss_b, loss_c)
      loss += loss_b * 50.0 + loss_c
    return loss

  def train_step(self, data):
    # 正常训练流程
    return self.train_step_normal(data)
    # 动态学习速率训练流程
    # return self.train_step_fast(data)


  def train_step_fast(self, data):
    '''
    动态学习速率
    '''
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    x, y_true_boxes, y_true_classes, y_true_masks = data
    loss = 0.0
    new_loss = 1.0
    # self.optimizer.learning_rate.assign(0.1)
    y_pred = self(x, training=True)  # Forward pass
    def loop_fun(loss, new_loss, y_pred, lr, gnorm):
      with tf.GradientTape() as tape:
        y_pred_boxes, y_pred_classes = self(x, training=True)
        loss = self._get_loss(y_true_boxes, y_true_classes, y_true_masks, y_pred_boxes, y_pred_classes)

      # Compute gradients
      trainable_vars = self.trainable_variables
      # 记录训练前的权重
      for i in range(len(trainable_vars)):
        self.bak_trainable_vars[i].assign(trainable_vars[i])

      # Compute gradients
      trainable_vars = self.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)
      # tf.clip_by_global_norm，将梯度限制在10.0内，防止loss异常导致梯度爆炸
      gradients, gnorm = tf.clip_by_global_norm(gradients,
                                                10.0)
      self.optimizer.learning_rate.adjusted_lr = lr
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))
      
      y_pred_boxes, y_pred_classes = self(x, training=True)
      new_loss = self._get_loss(y_true_boxes, y_true_classes, y_true_masks, y_pred_boxes, y_pred_classes)
      # 记录训练后的权重
      for i in range(len(trainable_vars)):
        self.bak_trainable_last_vars[i].assign(trainable_vars[i])
      # 还原训练前的权重
      for i in range(len(trainable_vars)):
        trainable_vars[i].assign(self.bak_trainable_vars[i])
      # y_pred_boxes, y_pred_classes = self(x, training=True)
      # old_loss = self._get_loss(y_true_boxes, y_true_classes, y_true_masks, y_pred_boxes, y_pred_classes)
      # tf.print('new_loss:', loss, new_loss, old_loss, self.optimizer.learning_rate.adjusted_lr)
      return (loss, new_loss, y_pred, lr * 0.3, gnorm)
    loss, new_loss, y_pred, _, gnorm = tf.while_loop(lambda loss, new_loss, y_pred, lr, gnorm: tf.math.logical_and(loss<=new_loss,lr>self.min_lr), loop_fun, (loss, new_loss, y_pred, 0.05, 0.0))

    # 恢复最佳权重
    trainable_vars = self.trainable_variables
    for i in range(len(trainable_vars)):
      trainable_vars[i].assign(self.bak_trainable_last_vars[i])
    return {'loss': loss, 'gnorm': gnorm}

  def train_step_normal(self, data):
    '''
    正常训练流程
    '''
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    # print('data:', data)
    x, y_true_boxes, y_true_classes, y_true_masks = data
    loss = 0.0
    with tf.GradientTape() as tape:
      y_pred_boxes, y_pred_classes = self(x, training=True)
      loss = self._get_loss(y_true_boxes, y_true_classes, y_true_masks, y_pred_boxes, y_pred_classes)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # tf.clip_by_global_norm，将梯度限制在10.0内，防止loss异常导致梯度爆炸
    gradients, gnorm = tf.clip_by_global_norm(gradients,
                                              10.0)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    return {'loss': loss, 'gnorm': gnorm}

  # @tf.function
  def test_step(self, data):
    '''评估'''
    # x: [batch_size,h,w,3]
    # y_true_boxes: [batch_size,obj_num,[y1,x1,y2,x2]]
    # y_true_classes: [batch_size,obj_num,]
    x, y_real_boxes, y_real_classes, y_true_boxes, y_true_classes, y_true_masks = data
    batch_size = tf.shape(x)[0]
    y_pred_boxes, y_pred_classes = self(x, training=False)
    loss = self._reg_l2_loss(4e-5)
    num_positives_sum = 0.0
    for level in range(len(y_true_boxes)):
      num_positives_sum += tf.reduce_sum(tf.cast(y_true_masks[level], tf.float32))
    num_positives_sum += 1.0
    for level in range(len(y_true_boxes)):
      loss_b = self.box_loss([num_positives_sum, y_true_boxes[level]], y_pred_boxes[level])
      loss_c = self.focal_loss([num_positives_sum, y_true_classes[level]], y_pred_classes[level])
      # tf.print('loss:', loss_b, loss_c)
      loss += loss_b * 50.0 + loss_c
    y_pred_boxes = self.anchors.convert_outputs_boxes(y_pred_boxes)
    # tf.print('boxes_outputs:', boxes_outputs)
    mAP = tf.constant(0.0, dtype=tf.float32)
    for batch in tf.range(batch_size):
      convert_boxes, convert_classes_id, convert_scores = self.anchors.convert_outputs_one(batch, y_pred_boxes, y_pred_classes)

      prediction = tf.concat([convert_boxes,
        tf.cast(tf.expand_dims(convert_classes_id, axis=-1), dtype=tf.float32),
        tf.expand_dims(convert_scores, axis=-1)], axis=-1)
      groud_truth = tf.concat([y_real_boxes[batch],
        tf.cast(tf.expand_dims(y_real_classes[batch], axis=-1), dtype=tf.float32)], axis=-1)
      # tf.print('prediction:', prediction)
      # tf.print('groud_truth:', groud_truth)
      mAP_one = tf.numpy_function(Get_mAP_one, (groud_truth, prediction, self._global_params.num_classes, 0.5), tf.float64)
      mAP += tf.cast(tf.reshape(mAP_one,()), tf.float32)
    mAP /= tf.cast(batch_size, dtype=tf.float32)
    return {'loss': loss, 'mAP': mAP}
    