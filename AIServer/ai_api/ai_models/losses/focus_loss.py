import tensorflow as tf

class FocusLoss(tf.keras.losses.Loss):

  def __init__(self, threshold=0.5, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.threshold = threshold

  def call(self, y_true, y_pred):
    """Compute focal loss for y and y_pred.

    Args:
      y_true: [batcn, height, width, points_num]
      y_pred: [batcn, height, width, points_num].

    Returns:
      the focus loss.
    """
    batch_size = tf.shape(y_true)[0]
    height = tf.shape(y_true)[1]
    width = tf.shape(y_true)[2]

    y_pred = tf.sigmoid(y_pred)
    
    object_mask = tf.math.not_equal(y_true, 0.0)
    object_mask = tf.cast(object_mask, tf.float32)
    object_num = tf.math.reduce_sum(object_mask)
    other_num = tf.cast(height*width, tf.float32) - object_num
    object_percent = object_num / tf.cast(height*width, tf.float32)
    y_true_object = tf.expand_dims(y_true * object_mask, axis=-1)
    y_pred_object = tf.expand_dims(y_pred * object_mask, axis=-1)
    y_true_other = tf.expand_dims(y_true * (1.0-object_mask), axis=-1)
    y_pred_other = tf.expand_dims(y_pred * (1.0-object_mask), axis=-1)

    loss_object = tf.math.reduce_sum(tf.math.square(y_true_object-y_pred_object)) / object_num / object_percent
    loss_other = tf.math.reduce_sum(tf.math.square(y_true_other-y_pred_other)) / other_num / (1.0-object_percent)
    loss = (loss_object+loss_other) / tf.cast(batch_size, tf.float32)

    return loss
