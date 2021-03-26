import tensorflow as tf
import os
import sys
import functools

sys.path.append(os.getcwd())
from ai_api.ai_models.utils.mish import Mish
from ai_api.ai_models.utils.tf_yolo_utils import GetLoss, GetNMSBoxes, DarknetConv2D, DarknetConv2D_BN_Leaky, DarknetConv2D_BN_Mish
from ai_api.ai_models.utils.mAP import Get_mAP_one


class BlocksLayer(tf.keras.layers.Layer):
  '''自定义层'''

  def __init__(self, filters, blocks_num, **args):
    '''初始化网络'''
    super(BlocksLayer, self).__init__(**args)
    self.filters = filters
    self.blocks_num = blocks_num
    # 下采样
    self.zero_padding1 = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))
    self.conv1 = DarknetConv2D_BN_Mish(self.filters, (3, 3), strides=(2, 2), padding='valid')
    # 分支1
    self.conv2 = DarknetConv2D_BN_Mish(self.filters, (1, 1))
    # 分支2_1
    self.conv3 = DarknetConv2D_BN_Mish(self.filters, (1, 1))
    # 分支2_2
    self.conv4 = DarknetConv2D_BN_Mish(self.filters // 2, (1, 1))
    self.conv5 = DarknetConv2D_BN_Mish(self.filters, (3, 3))
    # 分支2_1和分支2_2相加
    self.add1 = tf.keras.layers.Add()
    self.conv6 = DarknetConv2D_BN_Mish(self.filters, (1, 1))
    # 分支2和分支1合并
    self.concat1 = tf.keras.layers.Concatenate()
    self.conv7 = DarknetConv2D_BN_Mish(self.filters, (1, 1))

  @tf.function
  def call(self, x, training=False):
    '''运算部分'''
    x = self.zero_padding1(x)
    x = self.conv1(x, training=training)
    x1 = self.conv2(x, training=training)
    x2_1 = self.conv3(x, training=training)
    x2_2 = self.conv4(x2_1, training=training)
    x2_2 = self.conv5(x2_2, training=training)
    x2 = self.add1([x2_1, x2_2])
    x2 = self.conv6(x2, training=training)
    x = self.concat1([x2, x1])
    x = self.conv7(x, training=training)
    return x

class BlocksLayer2(tf.keras.layers.Layer):
  '''自定义层'''

  def __init__(self, filters, blocks_num, **args):
    '''初始化网络'''
    super(BlocksLayer2, self).__init__(**args)
    self.filters = filters
    self.blocks_num = blocks_num
    # 下采样
    self.zero_padding1 = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))
    self.conv1 = DarknetConv2D_BN_Mish(self.filters, (3, 3), strides=(2, 2), padding='valid')
    # 分支1
    self.conv2 = DarknetConv2D_BN_Mish(self.filters // 2, (1, 1))
    # 分支2_1
    self.conv3 = DarknetConv2D_BN_Mish(self.filters // 2, (1, 1))
    self.layer_list = []
    for _ in range(self.blocks_num):
      self.layer_list.append([
        # 分支2_2
        DarknetConv2D_BN_Mish(self.filters // 2, (1, 1)),
        DarknetConv2D_BN_Mish(self.filters // 2, (3, 3)),
        # 分支2_1和分支2_2相加,作为下次的2_1
        tf.keras.layers.Add(),
      ])
    self.conv4 = DarknetConv2D_BN_Mish(self.filters // 2, (1, 1))
    # 分支2和分支1合并
    self.concat1 = tf.keras.layers.Concatenate()
    self.conv5 = DarknetConv2D_BN_Mish(self.filters, (1, 1))

  @tf.function
  def call(self, x, training=False):
    '''运算部分'''
    x = self.zero_padding1(x)
    x = self.conv1(x, training=training)
    x1 = self.conv2(x, training=training)
    x2_1 = self.conv3(x, training=training)
    for i in range(self.blocks_num):
      x2_2 = self.layer_list[i][0](x2_1, training=training)
      x2_2 = self.layer_list[i][1](x2_2, training=training)
      x2_1 = self.layer_list[i][2]([x2_1, x2_2])
    x2 = self.conv4(x2_1, training=training)
    x = self.concat1([x2, x1])
    x = self.conv5(x, training=training)
    return x

class LastLayer(tf.keras.layers.Layer):
  '''自定义层'''

  def __init__(self, filters, **args):
    '''初始化网络'''
    super(LastLayer, self).__init__(**args)
    self.filters = filters

    self.conv3 = DarknetConv2D_BN_Leaky(self.filters, (1, 1))
    self.conv4 = DarknetConv2D_BN_Leaky(self.filters * 2, (3, 3))
    self.conv5 = DarknetConv2D_BN_Leaky(self.filters, (1, 1))
    # SPP
    self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')
    self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')
    self.max_pool3 = tf.keras.layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')
    self.concat2 = tf.keras.layers.Concatenate()
    self.conv6 = DarknetConv2D_BN_Leaky(self.filters, (1, 1))
    self.conv7 = DarknetConv2D_BN_Leaky(self.filters * 2, (3, 3))
    self.conv8 = DarknetConv2D_BN_Leaky(self.filters, (1, 1))

  @tf.function
  def call(self, x, training=False):
    '''运算部分'''
    x = self.conv3(x, training=training)
    x = self.conv4(x, training=training)
    x = self.conv5(x, training=training)
    # SPP
    x2 = self.max_pool1(x)
    x3 = self.max_pool2(x)
    x4 = self.max_pool3(x)
    x = self.concat2([x4, x3, x2, x])
    x = self.conv6(x, training=training)
    x = self.conv7(x, training=training)
    x = self.conv8(x, training=training)
    return x

class LastLayer2(tf.keras.layers.Layer):
  '''自定义层'''

  def __init__(self, filters, **args):
    '''初始化网络'''
    super(LastLayer2, self).__init__(**args)
    self.filters = filters
    # 上采样
    self.conv1 = DarknetConv2D_BN_Leaky(self.filters, (1, 1))
    self.up1 = tf.keras.layers.UpSampling2D((2, 2))
    # 上一维度输出做一次卷积
    self.conv2 = DarknetConv2D_BN_Leaky(self.filters, (1, 1))
    # 合并
    self.concat1 = tf.keras.layers.Concatenate()
    self.conv3 = DarknetConv2D_BN_Leaky(self.filters, (1, 1))
    self.conv4 = DarknetConv2D_BN_Leaky(self.filters * 2, (3, 3))
    self.conv5 = DarknetConv2D_BN_Leaky(self.filters, (1, 1))
    self.conv7 = DarknetConv2D_BN_Leaky(self.filters * 2, (3, 3))
    self.conv8 = DarknetConv2D_BN_Leaky(self.filters, (1, 1))

  @tf.function
  def call(self, x1, x2, training=False):
    '''运算部分'''
    x1 = self.conv1(x1, training=training)
    x1 = self.up1(x1)
    x2 = self.conv2(x2, training=training)
    x = self.concat1([x2, x1])
    x = self.conv3(x, training=training)
    x = self.conv4(x, training=training)
    x = self.conv5(x, training=training)
    x = self.conv7(x, training=training)
    x = self.conv8(x, training=training)
    return x

class OutputLayer(tf.keras.layers.Layer):
  '''自定义层'''

  def __init__(self, filters, **args):
    '''初始化网络'''
    super(OutputLayer, self).__init__(**args)
    self.filters = filters

    self.conv1 = DarknetConv2D_BN_Leaky(self.filters * 2, (3, 3))

  @tf.function
  def call(self, x, training=False):
    '''运算部分'''
    y = self.conv1(x, training=training)
    return y

class OutputLayer2(tf.keras.layers.Layer):
  '''自定义层'''

  def __init__(self, filters, **args):
    '''初始化网络'''
    super(OutputLayer2, self).__init__(**args)
    self.filters = filters
    
    # 下采样
    self.zero_padding1 = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))
    self.conv1 = DarknetConv2D_BN_Leaky(self.filters, (3, 3), strides=(2, 2), padding='valid')
    # 与下一维度结果合并
    self.concat1 = tf.keras.layers.Concatenate()
    self.conv2 = DarknetConv2D_BN_Leaky(self.filters, (1, 1))
    self.conv3 = DarknetConv2D_BN_Leaky(self.filters * 2, (3, 3))
    self.conv4 = DarknetConv2D_BN_Leaky(self.filters, (1, 1))
    self.conv5 = DarknetConv2D_BN_Leaky(self.filters * 2, (3, 3))
    self.conv6 = DarknetConv2D_BN_Leaky(self.filters, (1, 1))
    # 处理输出
    self.conv7 = DarknetConv2D_BN_Leaky(self.filters * 2, (3, 3))

  @tf.function
  def call(self, x, y, training=False):
    '''运算部分'''
    # 下采样
    x = self.zero_padding1(x)
    x = self.conv1(x, training=training)
    # 与下一维度结果合并
    x = self.concat1([x, y])
    x = self.conv2(x, training=training)
    x = self.conv3(x, training=training)
    x = self.conv4(x, training=training)
    x = self.conv5(x, training=training)
    x = self.conv6(x, training=training)
    # 处理输出
    y = self.conv7(x, training=training)
    return y, x
    
class YoloV4ModelBase(tf.keras.Model):
  '''Yolov4模型'''

  def __init__(self, classes_num, anchors_num, **args):
    '''初始化模型层'''
    super(YoloV4ModelBase, self).__init__(**args)
    self.anchors_num = anchors_num
    self.classes_num = classes_num
    self.conv1 = DarknetConv2D_BN_Mish(32, (3, 3))
    self.blocks1 = BlocksLayer(64, 1)
    self.blocks2 = BlocksLayer2(128, 2)
    self.blocks3 = BlocksLayer2(256, 8)
    self.blocks4 = BlocksLayer2(512, 8)
    self.blocks5 = BlocksLayer2(1024, 4)
    self.last1 = LastLayer(512)
    self.last2 = LastLayer2(256)
    self.last3 = LastLayer2(128)
    output_num = self.anchors_num * (5 + self.classes_num)
    self.output1 = OutputLayer(128)
    self.conv2 = DarknetConv2D(output_num, (1, 1), use_bias=True)
    self.output2 = OutputLayer2(256)
    self.conv3 = DarknetConv2D(output_num, (1, 1), use_bias=True)
    self.output3 = OutputLayer2(512)
    self.conv4 = DarknetConv2D(output_num, (1, 1), use_bias=True)

  @tf.function
  def call(self, x, training=False):
    '''运算部分'''
    # (416 * 416)
    x = self.conv1(x, training=training)
    # (208 * 208)
    x = self.blocks1(x, training=training)
    # (104 * 104)
    x = self.blocks2(x, training=training)
    # (52 * 52)
    x = self.blocks3(x, training=training)
    y3 = x
    # (26 * 26)
    x = self.blocks4(x, training=training)
    y2 = x
    # (13 * 13)
    x = self.blocks5(x, training=training)
    y1 = x
    # 计算y1,(13 * 13)
    y1 = self.last1(y1, training=training)
    # 计算y2,(26 * 26)
    y2 = self.last2(y1, y2, training=training)
    # 计算y3,(52 * 52)
    y3 = self.last3(y2, y3, training=training)
    # 计算y3,(52 * 52)
    z3 = self.output1(y3, training=training)
    z3 = self.conv2(z3, training=training)
    # 计算y2,(26 * 26)
    z2, y2 = self.output2(y3, y2, training=training)
    z2 = self.conv3(z2, training=training)
    # 计算y1,(13 * 13)
    z1, y1 = self.output3(y2, y1, training=training)
    z1 = self.conv4(z1, training=training)
    return (z1, z2, z3)

class YoloV4Model(YoloV4ModelBase):

  def __init__(self, classes_num, anchors, image_wh, **args):
    '''初始化网络'''
    super(YoloV4Model, self).__init__(classes_num=classes_num, anchors_num=anchors.shape[1], **args)
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
      iou_type='ciou')

  def build(self, input_shape):
    super(YoloV4Model, self).build(input_shape)
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
      confidence_thresh=0.5, scores_thresh=0.3,
      iou_thresh=self.train_iou_thresh, iou_type='diou')
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
      iou_thresh=iou_thresh, iou_type='diou')
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

