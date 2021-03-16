import tensorflow as tf
import functools
from ai_api.ai_models.utils.mAP import Get_mAP_one
from ai_api.ai_models.utils.tf_yolo_utils import GetLoss, GetNMSBoxes, DarknetConv2D, DarknetConv2D_BN_Leaky


class ResblockBody(tf.keras.layers.Layer):

  def __init__(self, num_filters, num_blocks, **args):
    '''初始化网络'''
    super(ResblockBody, self).__init__(**args)
    # 参数
    self.num_filters = num_filters
    self.num_blocks = num_blocks
    
    # 层定义
    self.zero_padding1 = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))
    self.darknet_conv_bn_leaky1 = DarknetConv2D_BN_Leaky(self.num_filters, (3,3), strides=(2,2))
    self.blocks = []
    for i in range(self.num_blocks):
      self.blocks.append([
        DarknetConv2D_BN_Leaky(self.num_filters//2, (1,1)),
        DarknetConv2D_BN_Leaky(self.num_filters, (3,3)),
        tf.keras.layers.Add()
        ])

  @tf.function
  def call(self, x, training):
    '''运算部分'''
    # tf.print('ResblockBody:', tf.shape(x))
    x = self.zero_padding1(x)
    x = self.darknet_conv_bn_leaky1(x, training=training)
    for block in self.blocks:
      y = block[0](x, training=training)
      y = block[1](y, training=training)
      x = block[2]([x, y])
    return x
    
  # def get_config(self):
  #   '''获取配置，用于保存模型'''
  #   return {'num_filters': self.num_filters, 'num_blocks': self.num_blocks}


class DarknetBody(tf.keras.layers.Layer):

  def __init__(self, **args):
    '''初始化网络'''
    super(DarknetBody, self).__init__(**args)
    
    self.darknet_conv_bn_leaky1 = DarknetConv2D_BN_Leaky(32, (3,3))
    self.resblock_body1 = ResblockBody(64, 1)
    self.resblock_body2 = ResblockBody(128, 2)
    self.resblock_body3 = ResblockBody(256, 8)
    self.resblock_body4 = ResblockBody(512, 8)
    self.resblock_body5 = ResblockBody(1024, 4)

  @tf.function
  def call(self, x, training):
    '''运算部分'''
    # tf.print('DarknetBody:', tf.shape(x))
    x = self.darknet_conv_bn_leaky1(x, training=training)
    x = self.resblock_body1(x, training=training)
    x = self.resblock_body2(x, training=training)
    x = self.resblock_body3(x, training=training)
    y3 = x
    x = self.resblock_body4(x, training=training)
    y2 = x
    x = self.resblock_body5(x, training=training)
    y1 = x
    return y1, y2, y3


class LastLayers(tf.keras.layers.Layer):

  def __init__(self, num_filters, **args):
    '''初始化网络'''
    super(LastLayers, self).__init__(**args)
    # 参数
    self.num_filters = num_filters
    
    # 层定义
    self.darknet_conv_bn_leaky1 = DarknetConv2D_BN_Leaky(self.num_filters, (1,1))
    self.darknet_conv_bn_leaky2 = DarknetConv2D_BN_Leaky(self.num_filters*2, (3,3))
    self.darknet_conv_bn_leaky3 = DarknetConv2D_BN_Leaky(self.num_filters, (1,1))
    self.darknet_conv_bn_leaky4 = DarknetConv2D_BN_Leaky(self.num_filters*2, (3,3))
    self.darknet_conv_bn_leaky5 = DarknetConv2D_BN_Leaky(self.num_filters, (1,1))
    self.darknet_conv_bn_leaky6 = DarknetConv2D_BN_Leaky(self.num_filters*2, (3,3))

  @tf.function
  def call(self, x, training):
    '''运算部分'''
    # tf.print('LastLayers:', tf.shape(x))
    x = self.darknet_conv_bn_leaky1(x, training=training)
    x = self.darknet_conv_bn_leaky2(x, training=training)
    x = self.darknet_conv_bn_leaky3(x, training=training)
    x = self.darknet_conv_bn_leaky4(x, training=training)
    x = self.darknet_conv_bn_leaky5(x, training=training)
    y = self.darknet_conv_bn_leaky6(x, training=training)
    return x, y
    
  # def get_config(self):
  #   '''获取配置，用于保存模型'''
  #   return {'num_filters': self.num_filters, 'out_filters': self.out_filters}

class YoloV3ModelBase(tf.keras.Model):

  def __init__(self, classes_num, anchors_num, **args):
    '''初始化网络'''
    super(YoloV3ModelBase, self).__init__(**args)
    # 参数
    self.anchors_num = anchors_num
    self.classes_num = classes_num
    self.out_filters = self.anchors_num*(self.classes_num+5)
    
    # 层定义
    self.darknet_body1 = DarknetBody()
    self.last_layers1 = LastLayers(512)
    self.darknetConv2D1 = DarknetConv2D(self.out_filters, (1,1))
    
    self.darknet_conv_bn_leaky1 = DarknetConv2D_BN_Leaky(256, (1,1))
    self.up_sampling1 = tf.keras.layers.UpSampling2D(2)
    self.concatenate1 = tf.keras.layers.Concatenate()
    self.last_layers2 = LastLayers(256)
    self.darknetConv2D2 = DarknetConv2D(self.out_filters, (1,1))

    self.darknet_conv_bn_leaky2 = DarknetConv2D_BN_Leaky(128, (1,1))
    self.up_sampling2 = tf.keras.layers.UpSampling2D(2)
    self.concatenate2 = tf.keras.layers.Concatenate()
    self.last_layers3 = LastLayers(128)
    self.darknetConv2D3 = DarknetConv2D(self.out_filters, (1,1))

  @tf.function
  def call(self, x, training):
    '''运算部分'''
    y1, y2, y3 = self.darknet_body1(x, training=training)
    # tf.print('darknet_body1:', tf.shape(x), tf.shape(y1), tf.shape(y2), tf.shape(y3))
    
    x, y1 = self.last_layers1(y1, training=training)
    y1 = self.darknetConv2D1(y1, training=training)
    # tf.print('last_layers1:', tf.shape(x), tf.shape(y1))

    x = self.darknet_conv_bn_leaky1(x, training=training)
    x = self.up_sampling1(x)
    x = self.concatenate1([x, y2])
    x, y2 = self.last_layers2(x, training=training)
    y2 = self.darknetConv2D2(y2, training=training)
    # tf.print('last_layers2:', tf.shape(x), tf.shape(y2))

    x = self.darknet_conv_bn_leaky2(x, training=training)
    x = self.up_sampling2(x)
    x = self.concatenate2([x, y3])
    x, y3 = self.last_layers3(x, training=training)
    y3 = self.darknetConv2D3(y3, training=training)
    # tf.print('last_layers3:', tf.shape(x), tf.shape(y3))
    return y1, y2, y3
    
  # def get_config(self):
  #   '''获取配置，用于保存模型'''
  #   return {'anchors_num': self.anchors_num, 'classes_num': self.classes_num}

class YoloV3Model(YoloV3ModelBase):

  def __init__(self, classes_num, anchors, image_wh, **args):
    '''初始化网络'''
    super(YoloV3Model, self).__init__(classes_num=classes_num, anchors_num=anchors.shape[1], **args)
    # 参数
    self.anchors_num = anchors.shape[1]
    self.classes_num = classes_num
    self.image_wh = image_wh
    self.anchors_wh = anchors
    self.train_iou_thresh = 0.5
    self.loss_decay = 0.9

    self.loss_obj = functools.partial(GetLoss, 
      image_wh=self.image_wh,
      anchors_wh=self.anchors_wh,
      iou_thresh=0.5,
      iou_type='iou')

  def build(self, input_shape):
    super(YoloV3Model, self).build(input_shape)
    self.shadow_loss = self.add_weight(name='shadow_loss',
                                      shape=(), 
                                      dtype=tf.float32, 
                                      trainable=False,
                                      initializer=tf.keras.initializers.Zeros())

  @tf.function
  def train_step(self, data):
    '''训练'''
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    # print('data:', data)
    x, y = data
    # tf.print('x:', tf.shape(x))
    # tf.print('y:', tf.shape(y[0]), tf.shape(y[1]), tf.shape(y[2]))

    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)  # Forward pass
      # print('y_pred:', y_pred)
      # Compute the loss value
      # (the loss function is configured in `compile()`)
      # loss = self.loss(y, y_pred, regularization_losses=self.losses)
      loss = self.loss_obj(y, y_pred)
      # 平均移动loss
      if self.optimizer.iterations > 1:
        global_step = tf.cast(self.optimizer.iterations, tf.float32)
        decay = tf.math.minimum(self.loss_decay, (1 + global_step) / (1000 + global_step))
        # tf.print('\ndecay:', decay)
        loss = decay * self.shadow_loss + (1 - decay) * loss
    # tf.print('\nloss:', loss)
    # tf.print('\nglobal_step:', self.optimizer.iterations)

    # Compute gradients
    trainable_vars = self.trainable_variables
    # for var in trainable_vars:
    #   tf.print('trainable_vars:', var.shape)
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    # self.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    # return {m.name: m.result() for m in self.metrics}
    self.shadow_loss.assign(loss)
    return {'loss': loss}

  @tf.function
  def test_step(self, data):
    '''评估'''
    x, y = data
    # 维度丢失，需重置维度
    x = tf.reshape(x, (-1,416,416,3))
    y = (tf.reshape(y[0], (-1,13,13,3,(5+self.classes_num))),tf.reshape(y[1], (-1,26,26,3,(5+self.classes_num))),tf.reshape(y[2], (-1,52,52,3,(5+self.classes_num))))
    y_pred = self(x, training=False)

    selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence = GetNMSBoxes(
      y_pred[0], y_pred[1], y_pred[2],
      anchors_wh=self.anchors_wh, image_wh=self.image_wh, classes_num=self.classes_num,
      confidence_thresh=0.5, scores_thresh=0.3, iou_thresh=0.5, iou_type='iou')
    # tf.print('selected_boxes:', tf.shape(selected_boxes))
    # tf.print('selected_classes_id:', tf.shape(selected_classes_id))
    # tf.print('selected_scores:', tf.shape(selected_scores))
    # tf.print('selected_classes:', tf.shape(selected_classes))
    # tf.print('selected_confidence:', tf.shape(selected_confidence))
    prediction = tf.concat([selected_boxes,
      tf.cast(tf.expand_dims(selected_classes_id, axis=-1), dtype=tf.float32),
      tf.expand_dims(selected_scores, axis=-1)], axis=-1)
    # tf.print('prediction:', tf.shape(prediction), prediction)
    prediction = prediction
    
    groud_truth1 = self.GetGroudTruth(y[0])
    groud_truth2 = self.GetGroudTruth(y[1])
    groud_truth3 = self.GetGroudTruth(y[2])
    groud_truth = tf.concat([groud_truth1,groud_truth2,groud_truth3], axis=0)
    mAP = tf.numpy_function(Get_mAP_one, (groud_truth, prediction, self.classes_num, 0.5), tf.float64)
    return {'mAP': mAP}
  
  @tf.function
  def GetGroudTruth(self, y):
    '''计算框真实纵坐标，用于计算mAP'''
    boxes_xy, boxes_wh, confidence, classes = tf.split(
      y, (2, 2, 1, self.classes_num), axis=-1)
    confidence = confidence[..., 0]
    boxes_wh_half = boxes_wh / 2
    boxes_mins = boxes_xy - boxes_wh_half
    boxes_maxes = boxes_xy + boxes_wh_half
    boxes = tf.concat([boxes_mins, boxes_maxes], axis=-1)
    boxes = tf.boolean_mask(boxes, confidence)
    classes = tf.boolean_mask(classes, confidence)
    groud_truth = tf.concat([boxes, 
      tf.cast(tf.expand_dims(tf.math.argmax(classes, axis=-1), axis=-1), dtype=tf.float32)], axis=-1)
    # tf.print('groud_truth:', tf.shape(groud_truth))
    return groud_truth

  @tf.function
  def Predict(self, input_image, confidence_thresh=0.5, scores_thresh=0.2, iou_thresh=0.5):
    '''
    预测(编译模式)
    input_image:图片(416,416,3)
    return:两个指针值(2)
    '''
    # 预测
    # start = time.process_time()
    output = self(input_image, training=False)
    # tf.print('output[0]:', tf.math.reduce_max(output[0]), tf.math.reduce_min(output[0]), tf.shape(output[0]))
    # tf.print('output[1]:', tf.math.reduce_max(output[1]), tf.math.reduce_min(output[1]), tf.shape(output[1]))
    # tf.print('output[2]:', tf.math.reduce_max(output[2]), tf.math.reduce_min(output[2]), tf.shape(output[2]))
    selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence = GetNMSBoxes(
      output[0], output[1], output[2], 
      anchors_wh=self.anchors_wh, image_wh=self.image_wh, classes_num=self.classes_num,
      confidence_thresh=confidence_thresh, scores_thresh=scores_thresh,
      iou_thresh=iou_thresh, iou_type='iou')
    # end = time.process_time()
    # tf.print('%s predict time: %f' % (self.__class__, (end - start)))
    return selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence

  def FreeLayer(self, types):
    for l in self.layers:
      # print('FreeLayer:', l.name)
      l.trainable=False
      for t in types:
        if l.name.startswith(t):
          l.trainable=True
          print('free:', l.name)
