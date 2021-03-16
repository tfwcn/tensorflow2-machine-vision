import tensorflow as tf
from ai_api.ai_models.utils.tf_iou_utils import GetIOU, GetIOUNMSByClasses
from ai_api.ai_models.utils.mish import Mish


@tf.function
def GetLoss(y_true, y_pred, image_wh, anchors_wh, iou_thresh=0.5, iou_type='iou'):
  '''
  获取损失值

  Args:
    y_true:坐标还没归一化，[(batch_size, 13, 13, 3, 5+num_classes), (batch_size, 26, 26, 3, 5+num_classes), (batch_size, 52, 52, 3, 5+num_classes)]
    y_pred:[(batch_size, 13, 13, 3, 5+num_classes), (batch_size, 26, 26, 3, 5+num_classes), (batch_size, 52, 52, 3, 5+num_classes)]
  '''
  # print('loss_fun:', type(y_true), type(y_pred))
  image_wh_f = tf.cast(image_wh, dtype=tf.float32)
  anchors_wh_f = tf.cast(anchors_wh, dtype=tf.float32)
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
    y_true_raw_xy = y_true_object * y_true_raw_xy
    # tf.print('y_true_raw_xy:', tf.math.reduce_max(y_true_raw_xy), tf.math.reduce_min(y_true_raw_xy))
    
    y_true_read_wh = y_true_read[..., 2:4]
    y_true_raw_wh = tf.math.log((y_true_read_wh * image_wh_f[::-1]+1e-8) / anchors_wh_f[layer_index, ...])
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
    y_pred_read_wh = tf.math.exp(y_pred_raw_wh) * anchors_wh_f[layer_index, ...] / image_wh_f[::-1]
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
      y_true_boxes_tmp = tf.boolean_mask(y_true_boxes_one, y_true_object_one[..., 0])
      # 计算IOU
      # (boxes_num, 4) => (1, boxes_num, 4)
      y_true_boxes_tmp = tf.expand_dims(y_true_boxes_tmp, axis=0)
      y_pred_boxes_tmp = y_pred_boxes_one
      # (h, w, anchors_num, 4) => (h, w, anchors_num, 1, 4)
      y_pred_boxes_tmp = tf.expand_dims(y_pred_boxes_tmp, axis=-2)
      # (h, w, anchors_num, boxes_num)
      iou = GetIOU(y_pred_boxes_tmp, y_true_boxes_tmp, iou_type=iou_type)
      # (h, w, anchors_num)
      best_iou = tf.math.reduce_max(iou, axis=-1)
      # 把IOU<0.5的认为是背景
      ignore_mask = ignore_mask.write(batch_index, tf.cast(best_iou < iou_thresh, dtype=tf.float32))
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

@tf.function
def GetBoxes(y, anchors_wh, classes_num):
  '''将偏移转换成真实值，范围0到1'''
  # 拆分特征
  # box_xy:(batch_size, y_w, y_h, anchors_num, 2)
  # box_wh:(batch_size, y_w, y_h, anchors_num, 2)
  # confidence:(batch_size, y_w, y_h, anchors_num, 1)
  # classes:(batch_size, y_w, y_h, anchors_num, classes_num)
  boxes_xy, boxes_wh, confidence, classes = tf.split(
    y, (2, 2, 1, tf.cast(classes_num, dtype=tf.int32)), axis=-1)
  confidence = tf.math.sigmoid(confidence)
  classes = tf.math.sigmoid(classes)
  y_pred_raw = y
  # 特征网格对应实际图片的坐标
  grid_shape = tf.shape(y_pred_raw)[1:3] # height, width
  grid_x = tf.range(0, grid_shape[1])
  grid_y = tf.range(0, grid_shape[0])
  grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
  grid_x = tf.reshape(grid_x, (grid_shape[0], grid_shape[1], 1, 1))
  grid_y = tf.reshape(grid_y, (grid_shape[0], grid_shape[1], 1, 1))
  grid_xy = tf.concat([grid_x, grid_y], axis=-1)
  # 计算真实坐标与相对坐标
  # y_pred
  y_pred_raw_xy = boxes_xy
  y_pred_read_xy = (tf.math.sigmoid(y_pred_raw_xy) + tf.cast(grid_xy, dtype=tf.float32)) / tf.cast(grid_shape[::-1], dtype=tf.float32)
  y_pred_raw_wh = boxes_wh
  y_pred_read_wh = tf.math.exp(y_pred_raw_wh) * anchors_wh
  y_pred_read_wh = tf.where(tf.math.is_inf(y_pred_read_wh), tf.zeros_like(y_pred_read_wh), y_pred_read_wh)
  # 计算IOU
  y_pred_read_wh_half = y_pred_read_wh / 2
  y_pred_read_mins = y_pred_read_xy - y_pred_read_wh_half
  y_pred_read_maxes = y_pred_read_xy + y_pred_read_wh_half
  y_pred_boxes = tf.concat([y_pred_read_mins, y_pred_read_maxes], axis=-1)
  # 去掉无效框
  mask = tf.math.logical_and(y_pred_boxes[...,2] > y_pred_boxes[...,0], y_pred_boxes[...,3] > y_pred_boxes[...,1])
  y_pred_boxes = tf.boolean_mask(y_pred_boxes, mask)
  confidence = tf.boolean_mask(confidence, mask)
  classes = tf.boolean_mask(classes, mask)
  return y_pred_boxes, confidence, classes

@tf.function
def GetNMSBoxes(y1, y2, y3, anchors_wh, image_wh, classes_num,
  confidence_thresh=0.5, scores_thresh=0.3, iou_thresh=0.5, iou_type='iou'):
  '''经过NMS去重后，转换成框坐标'''
  image_wh_f = tf.cast(image_wh, dtype=tf.float32)
  anchors_wh_f = tf.cast(anchors_wh, dtype=tf.float32)
  # 拆分维度
  anchors_num = tf.shape(anchors_wh)[1]
  y1 = tf.reshape(y1, [tf.shape(y1)[0], tf.shape(
    y1)[1], tf.shape(y1)[2], anchors_num, -1])
  y2 = tf.reshape(y2, [tf.shape(y2)[0], tf.shape(
    y2)[1], tf.shape(y2)[2], anchors_num, -1])
  y3 = tf.reshape(y3, [tf.shape(y3)[0], tf.shape(
    y3)[1], tf.shape(y3)[2], anchors_num, -1])
  
  y1_pred_boxes, y1_pred_confidence, y1_pred_classes = GetBoxes(
    y1, anchors_wh=(anchors_wh_f[0]/image_wh_f), classes_num=classes_num)
  y2_pred_boxes, y2_pred_confidence, y2_pred_classes = GetBoxes(
    y2, anchors_wh=(anchors_wh_f[1]/image_wh_f), classes_num=classes_num)
  y3_pred_boxes, y3_pred_confidence, y3_pred_classes = GetBoxes(
    y3, anchors_wh=(anchors_wh_f[2]/image_wh_f), classes_num=classes_num)

  y1_pred_mask = tf.math.logical_and(y1_pred_confidence > confidence_thresh, 
                      tf.expand_dims(tf.math.reduce_max(y1_pred_classes, axis=-1), axis=-1) > scores_thresh)
  y1_pred_boxes = tf.boolean_mask(y1_pred_boxes, y1_pred_mask[..., 0])
  y1_pred_classes = tf.boolean_mask(y1_pred_classes, y1_pred_mask[..., 0])
  y1_pred_confidence = tf.boolean_mask(y1_pred_confidence, y1_pred_mask[..., 0])
  y1_pred_boxes = tf.reshape(
    y1_pred_boxes, [-1, tf.shape(y1_pred_boxes)[-1]])
  # scores
  y1_pred_scores = tf.expand_dims(tf.math.reduce_max(y1_pred_classes, axis=-1), axis=-1)
  y1_pred_scores = y1_pred_scores
  y1_pred_scores = tf.reshape(y1_pred_scores, [-1, ])
  # classes
  y1_pred_classes_id = tf.math.argmax(y1_pred_classes, axis=-1, output_type=tf.int32)
  y1_pred_classes_id = tf.reshape(y1_pred_classes_id, [-1, ])

  y2_pred_mask = tf.math.logical_and(y2_pred_confidence > confidence_thresh, 
                      tf.expand_dims(tf.math.reduce_max(y2_pred_classes, axis=-1), axis=-1) > scores_thresh)
  y2_pred_boxes = tf.boolean_mask(y2_pred_boxes, y2_pred_mask[..., 0])
  y2_pred_classes = tf.boolean_mask(y2_pred_classes, y2_pred_mask[..., 0])
  y2_pred_confidence = tf.boolean_mask(y2_pred_confidence, y2_pred_mask[..., 0])
  y2_pred_boxes = tf.reshape(
    y2_pred_boxes, [-1, tf.shape(y2_pred_boxes)[-1]])
  # scores
  y2_pred_scores = tf.expand_dims(tf.math.reduce_max(y2_pred_classes, axis=-1), axis=-1)
  y2_pred_scores = y2_pred_scores
  y2_pred_scores = tf.reshape(y2_pred_scores, [-1, ])
  # classes
  y2_pred_classes_id = tf.math.argmax(y2_pred_classes, axis=-1, output_type=tf.int32)
  y2_pred_classes_id = tf.reshape(y2_pred_classes_id, [-1, ])

  y3_pred_mask = tf.math.logical_and(y3_pred_confidence > confidence_thresh, 
                      tf.expand_dims(tf.math.reduce_max(y3_pred_classes, axis=-1), axis=-1) > scores_thresh)
  y3_pred_boxes = tf.boolean_mask(y3_pred_boxes, y3_pred_mask[..., 0])
  y3_pred_classes = tf.boolean_mask(y3_pred_classes, y3_pred_mask[..., 0])
  y3_pred_confidence = tf.boolean_mask(y3_pred_confidence, y3_pred_mask[..., 0])
  y3_pred_boxes = tf.reshape(
    y3_pred_boxes, [-1, tf.shape(y3_pred_boxes)[-1]])
  # scores
  y3_pred_scores = tf.expand_dims(tf.math.reduce_max(y3_pred_classes, axis=-1), axis=-1)
  y3_pred_scores = y3_pred_scores
  y3_pred_scores = tf.reshape(y3_pred_scores, [-1, ])
  # classes
  y3_pred_classes_id = tf.math.argmax(y3_pred_classes, axis=-1, output_type=tf.int32)
  y3_pred_classes_id = tf.reshape(y3_pred_classes_id, [-1, ])

  y_pred_boxes = tf.concat(
    [y1_pred_boxes, y2_pred_boxes, y3_pred_boxes], axis=0)
  y_pred_classes_id = tf.concat(
    [y1_pred_classes_id, y2_pred_classes_id, y3_pred_classes_id], axis=0)
  y_pred_scores = tf.concat(
    [y1_pred_scores, y2_pred_scores, y3_pred_scores], axis=0)
  y_pred_classes = tf.concat(
    [y1_pred_classes, y2_pred_classes, y3_pred_classes], axis=0)
  y_pred_confidence = tf.concat(
    [y1_pred_confidence, y2_pred_confidence, y3_pred_confidence], axis=0)

  # selected_indices = tf.image.non_max_suppression(
  #   y_pred_boxes, y_pred_scores, 500, iou_threshold=iou_thresh)
  # selected_indices = GetIOUNMS(
  #   y_pred_boxes, 
  #   y_pred_scores, 
  #   500, 
  #   iou_threshold=iou_thresh, 
  #   iou_type='diou')
  selected_indices = GetIOUNMSByClasses(
    y_pred_boxes, 
    y_pred_scores, 
    y_pred_classes_id, 
    500, 
    iou_threshold=iou_thresh, 
    iou_type=iou_type)
  selected_boxes = tf.gather(y_pred_boxes, selected_indices)
  selected_classes_id = tf.gather(y_pred_classes_id, selected_indices)
  selected_scores = tf.gather(y_pred_scores, selected_indices)
  selected_classes = tf.gather(y_pred_classes, selected_indices)
  selected_confidence = tf.gather(y_pred_confidence, selected_indices)
  # tf.print('y_pred_boxes:', tf.shape(y_pred_boxes))
  # tf.print('selected_boxes:', tf.shape(selected_boxes))
  return selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence


class DarknetConv2D(tf.keras.layers.Layer):

  def __init__(self, *args, **kwargs):
    '''初始化网络'''
    super(DarknetConv2D, self).__init__()
    darknet_conv_kwargs = {'kernel_regularizer': tf.keras.regularizers.l2(5e-4)}
    darknet_conv_kwargs['kernel_initializer'] = tf.keras.initializers.he_uniform()
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    # print('darknet_conv_kwargs:', darknet_conv_kwargs)
    self.conv1 = tf.keras.layers.Conv2D(*args, **darknet_conv_kwargs)

  @tf.function
  def call(self, x, training):
    '''运算部分'''
    # tf.print('DarknetConv2D:', tf.shape(x))
    x = self.conv1(x)
    return x

class DarknetConv2D_BN_Leaky(tf.keras.layers.Layer):

  def __init__(self, *args, **kwargs):
    '''初始化网络'''
    super(DarknetConv2D_BN_Leaky, self).__init__()
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    self.conv1 = DarknetConv2D(*args, **no_bias_kwargs)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.leaky_relu1 = tf.keras.layers.LeakyReLU(alpha=0.1)
    # self.dorp_block1 = DorpBlock(0.1, block_size=3)

  @tf.function
  def call(self, x, training):
    '''运算部分'''
    # tf.print('DarknetConv2D_BN_Leaky:', tf.shape(x))
    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = self.leaky_relu1(x)
    # x = self.dorp_block1(x, training=training)
    return x

class DarknetConv2D_BN_Mish(tf.keras.layers.Layer):

  def __init__(self, *args, **kwargs):
    '''初始化网络'''
    super(DarknetConv2D_BN_Mish, self).__init__()
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    self.conv1 = DarknetConv2D(*args, **no_bias_kwargs)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.mish1 = Mish()
    # self.dorp_block1 = DorpBlock(0.1, block_size=3)

  @tf.function
  def call(self, x, training):
    '''运算部分'''
    # tf.print('DarknetConv2D_BN_Leaky:', tf.shape(x))
    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = self.mish1(x)
    # x = self.dorp_block1(x, training=training)
    return x

