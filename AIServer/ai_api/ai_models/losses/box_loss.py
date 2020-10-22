import tensorflow as tf

class BoxLoss(tf.keras.losses.Loss):
  """L2 box regression loss."""

  def __init__(self, delta=0.1, **kwargs):
    """Initialize box loss.

    Args:
      delta: `float`, the point where the huber loss function changes from a
        quadratic to linear. It is typically around the mean value of regression
        target. For instances, the regression targets of 512x512 input with 6
        anchors on P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
      **kwargs: other params.
    """
    super().__init__(**kwargs)
    self.huber = tf.keras.losses.Huber(
        delta, reduction=tf.keras.losses.Reduction.NONE)

  @tf.autograph.experimental.do_not_convert
  def call(self, y_true, box_outputs):
    box_targets = y_true
    box_outputs, mask = box_outputs
    total_loss = 0
    for i in range(len(box_targets)):
      mask_one = tf.cast(mask[i], dtype=tf.float32)
      box_targets_one = box_targets[i]
      box_outputs_one = box_outputs[i]
      num_positives = tf.reduce_sum(mask_one) / tf.cast(tf.shape(mask_one)[0], tf.float32)
      normalizer = num_positives * 4.0
      box_targets_one = tf.expand_dims(box_targets_one, axis=-1)
      box_outputs_one = tf.expand_dims(box_outputs_one, axis=-1)
      box_loss = self.huber(box_targets_one, box_outputs_one) * mask_one
      box_loss = tf.reduce_sum(box_loss)
      box_loss = tf.math.divide_no_nan(box_loss,normalizer)
      total_loss+=box_loss
    return total_loss