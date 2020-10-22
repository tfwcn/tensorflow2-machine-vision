import tensorflow as tf

from ai_api.ai_models.efficientnet.demo.model import DemoModel
from ai_api.ai_models.losses.box_loss import BoxLoss
from ai_api.ai_models.losses.focal_loss import FocalLoss
from ai_api.ai_models.utils.mAP import Get_mAP_one

class DemoModelTrain(DemoModel):

  def __init__(self, global_params, *args, **kwargs):
    super(DemoModelTrain, self).__init__(*args, **kwargs)
    # print('args:',args)
    # print('kwargs:',kwargs)
    self.box_loss = BoxLoss()
    self.focal_loss = FocalLoss(global_params.alpha, global_params.gamma)

  @tf.function
  def train_step(self, data):
    '''训练'''
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    # print('data:', data)
    x, y_true_boxes, y_true_classes, y_true_mask = data
    # tf.print('x:', tf.shape(x))
    # tf.print('y:', tf.shape(y[0]), tf.shape(y[1]), tf.shape(y[2]))
    # tmp_y_true_classes=[]
    # for i in range(len(y_true_classes)):
    #   tmp_shape = tf.shape(y_true_classes[i])
    #   tmp_y_true_classes.append(tf.one_hot(tf.cast(tf.reshape(y_true_classes[i], tmp_shape[:-1]), tf.int32), self._global_params.num_classes, dtype=tf.float32))
    # y_true_classes = tuple(tmp_y_true_classes)
    # y_pred_classes, y_pred_boxes = self(x, training=True)  # Forward pass
    # for i in range(len(y_pred_classes)):
    #   tf.print('y_pred_classes:', tf.shape(y_pred_classes[i]))
    #   tf.print('y_pred_boxes:', tf.shape(y_pred_boxes[i]))
    #   tf.print('y_true_classes:', tf.shape(y_true_classes[i]))
    #   tf.print('y_true_boxes:', tf.shape(y_true_boxes[i]))
    # return {'loss': 0}
    loss = 0.0
    with tf.GradientTape() as tape:
        y_pred_classes, y_pred_boxes = self(x, training=True)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        # loss = self.loss(y, y_pred, regularization_losses=self.losses)
        loss_b = self.box_loss(y_true_boxes, (y_pred_boxes, y_true_mask))
        loss_c = self.focal_loss(y_true_classes, (y_pred_classes, y_true_mask))
        # tf.print('loss:', loss_b, loss_c)
        loss += loss_b + loss_c

    # Compute gradients
    trainable_vars = self.trainable_variables
    # for var in trainable_vars:
    #     tf.print('trainable_vars:', var.shape)
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    # self.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    # return {m.name: m.result() for m in self.metrics}
    return {'loss': loss}

  # @tf.function
  # def test_step(self, data):
  #   '''评估'''
  #   x, y_true_boxes, y_true_classes, y_true_mask = data
  #   # 维度丢失，需重置维度
  #   x = tf.reshape(x, (-1,416,416,3))
  #   y = (tf.reshape(y[0], (-1,13,13,3,(5+self.classes_num))),tf.reshape(y[1], (-1,26,26,3,(5+self.classes_num))),tf.reshape(y[2], (-1,52,52,3,(5+self.classes_num))))
  #   y_pred = self(x, training=False)

  #   selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence = self.GetNMSBoxes(
  #     y_pred[0], y_pred[1], y_pred[2], scores_thresh=0.2, iou_thresh=0.5)
  #   # tf.print('selected_boxes:', tf.shape(selected_boxes))
  #   # tf.print('selected_classes_id:', tf.shape(selected_classes_id))
  #   # tf.print('selected_scores:', tf.shape(selected_scores))
  #   # tf.print('selected_classes:', tf.shape(selected_classes))
  #   # tf.print('selected_confidence:', tf.shape(selected_confidence))
  #   prediction = tf.concat([selected_boxes,
  #     tf.cast(tf.expand_dims(selected_classes_id, axis=-1), dtype=tf.float32),
  #     tf.expand_dims(selected_scores, axis=-1)], axis=-1)
  #   # tf.print('prediction:', tf.shape(prediction), prediction)
  #   prediction = prediction
    
  #   groud_truth1 = self.GetGroudTruth(y[0])
  #   groud_truth2 = self.GetGroudTruth(y[1])
  #   groud_truth3 = self.GetGroudTruth(y[2])
  #   groud_truth = tf.concat([groud_truth1,groud_truth2,groud_truth3], axis=0)
  #   mAP = tf.numpy_function(Get_mAP_one, (groud_truth, prediction, self.classes_num, 0.5), tf.float64)
  #   return {'mAP': mAP}
    