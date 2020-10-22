import tensorflow as tf

class FocalLoss(tf.keras.losses.Loss):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.
  """

  def __init__(self, alpha, gamma, label_smoothing=0.0, **kwargs):
    """Initialize focal loss.

    Args:
      alpha: A float32 scalar multiplying alpha to the loss from positive
        examples and (1-alpha) to the loss from negative examples.
      gamma: A float32 scalar modulating loss from hard and easy examples.
      label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
      **kwargs: other params.
    """
    super().__init__(**kwargs)
    self.alpha = alpha
    self.gamma = gamma
    self.label_smoothing = label_smoothing

  @tf.autograph.experimental.do_not_convert
  def call(self, y_true, y_pred):
    """Compute focal loss for y and y_pred.

    Args:
      y: A tuple of (normalizer, y_true), where y_true is the target class.
      y_pred: A float32 tensor [batch, height_in, width_in, num_predictions].

    Returns:
      the focal loss.
    """
    class_targets = y_true
    class_outputs, mask = y_pred
    alpha = tf.convert_to_tensor(self.alpha, dtype=y_pred[0][0].dtype)
    gamma = tf.convert_to_tensor(self.gamma, dtype=y_pred[0][0].dtype)

    total_loss = 0
    for i in range(len(class_targets)):
      mask_one = tf.cast(mask[i], dtype=tf.float32)
      class_targets_one = class_targets[i]
      class_outputs_one = class_outputs[i]
      normalizer = tf.reduce_sum(mask_one) / tf.cast(tf.shape(mask_one)[0], tf.float32)
      # compute focal loss multipliers before label smoothing, such that it will
      # not blow up the loss.
      pred_prob = tf.sigmoid(class_outputs_one)
      p_t = (class_targets_one * pred_prob) + ((1 - class_targets_one) * (1 - pred_prob))
      alpha_factor = class_targets_one * alpha + (1 - class_targets_one) * (1 - alpha)
      modulating_factor = (1.0 - p_t)**gamma

      # apply label smoothing for cross_entropy for each entry.
      class_targets_one = class_targets_one * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
      ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=class_targets_one, logits=class_outputs_one)
      # ce = tf.math.log(tf.math.square(class_targets_one-class_outputs_one)*4+1)

      # compute the final loss and return
      total_loss += tf.reduce_sum(tf.math.divide_no_nan(alpha_factor * modulating_factor * ce , normalizer))
    return total_loss
