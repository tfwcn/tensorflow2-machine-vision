import tensorflow as tf
import functools
from ai_api.ai_models.utils.mAP import Get_mAP_one
from ai_api.ai_models.utils.tf_yolo_utils import GetLoss, GetNMSBoxes, DarknetConv2D, DarknetConv2D_BN_Leaky


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

  def __init__(self, out_filters, backbone_weights='imagenet', **args):
    '''初始化网络'''
    super(YoloV3ModelBase, self).__init__(**args)
    # 参数
    self.backbone_weights = backbone_weights
    self.out_filters = out_filters
    
    # 层定义
    self.backbone1 = tf.keras.applications.ResNet50V2(
      include_top=False, weights=self.backbone_weights)
    self.body1 = tf.keras.models.Model(inputs=self.backbone1.inputs,
      outputs=[self.backbone1.get_layer(name='conv5_block3_out').output,
        self.backbone1.get_layer(name='conv4_block5_out').output,
        self.backbone1.get_layer(name='conv3_block3_out').output])
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
    y1, y2, y3 = self.body1(x, training=training)
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
    y_pred = self(x, training=False)
    loss = self.loss_obj(y, y_pred)

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
    return {'loss': loss, 'mAP': mAP}
  
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


class MoCoModel(tf.keras.Model):

  def __init__(self, out_filters, image_wh, K=65536, m=0.999, T=0.07, **args):
    '''初始化网络'''
    super(MoCoModel, self).__init__()
    # 参数
    self.image_wh = image_wh
    self.out_filters = out_filters
    # MoCo队列长度
    self.K = K
    # 动量，平滑系数
    self.m = m
    self.T = T
    
    self.model_q = YoloV3ModelBase(
      out_filters=self.out_filters,
      backbone_weights=None,
      **args)
    
    self.model_k = YoloV3ModelBase(
      out_filters=self.out_filters,
      backbone_weights=None,
      trainable=False,
      **args)

    self.loss_obj = self.GetLoss

  def build(self, input_shape):
    # super(MoCoModel, self).build(input_shape)
    self.model_q.build(input_shape[0])
    self.model_k.build(input_shape[1])
    tf.print('model_q.trainable_variables:', len(self.model_q.trainable_variables))
    tf.print('model_k.trainable_variables:', len(self.model_k.trainable_variables))
    # 克隆k到q
    for i in range(len(self.model_q.variables)):
      var_q = self.model_q.variables[i]
      var_k = self.model_k.variables[i]
      var_q.assign(var_k)
    # 队列
    queue_list = tf.random.uniform([self.K,13*13*self.out_filters+26*26*self.out_filters+52*52*self.out_filters])
    queue_list = tf.nn.l2_normalize(queue_list, axis=1)
    self.queue = self.add_weight(
      name='queue',
      shape=[self.K,13*13*self.out_filters+26*26*self.out_filters+52*52*self.out_filters],
      dtype=tf.float32,
      trainable=False,
      initializer=tf.constant_initializer(queue_list.numpy()))
    self.queue_ptr = self.add_weight(
      name='queue_ptr',
      shape=[],
      dtype=tf.int32,
      trainable=False,
      initializer=tf.zeros_initializer())

  def call(self, inputs, training, mask=None):
    x_q, x_k = inputs
    out_q = self.model_q(x_q, training=training)
    out_k = self.model_k(x_k, training=training)
    return out_q, out_k

  def push_queue(self, items):
    # queue: KxC
    # items: NxC
    batch_size = tf.shape(items)[0]
    end_queue_ptr = self.queue_ptr + batch_size

    inds = tf.range(self.queue_ptr, end_queue_ptr, dtype=tf.int32)
    inds = inds % self.K
    self.queue.scatter_nd_update(inds, items)
    self.queue_ptr.assign(end_queue_ptr % self.K)

  @tf.function
  def GetLoss(self, y_q, y_k):
    N = tf.shape(y_q[0])[0]
    y_q = tf.concat([
      tf.reshape(y_q[0], [N, -1]),
      tf.reshape(y_q[1], [N, -1]),
      tf.reshape(y_q[2], [N, -1]),
      ], axis=-1)
    y_q = tf.nn.l2_normalize(y_q, axis=1)
    # tf.print('y_q:', tf.math.reduce_max(y_q), tf.math.reduce_min(y_q))
    y_k = tf.concat([
      tf.reshape(y_k[0], [N, -1]),
      tf.reshape(y_k[1], [N, -1]),
      tf.reshape(y_k[2], [N, -1]),
      ], axis=-1)
    y_k = tf.nn.l2_normalize(y_k, axis=1)
    # tf.print('y_k:', tf.math.reduce_max(y_k), tf.math.reduce_min(y_k))
    # positive logits: Nx1
    l_pos = tf.matmul(tf.reshape(y_q, [N,1,-1]), tf.reshape(y_k, [N,-1,1]))
    # l_pos = tf.einsum('ai,ai->a', y_q, y_k) # 功能同上，效率低
    l_pos = tf.reshape(l_pos, [N,-1])
    # negative logits: NxK
    l_neg = tf.matmul(y_q, tf.transpose(self.queue, perm=[1,0]))
    # l_neg = tf.einsum('ij,kj->ik', y_q, self.queue) # 功能同上，效率低
    # logits: Nx(1+K)
    logits = tf.concat([l_pos, l_neg], axis=1)
    # contrastive loss, Eqn.(1)
    # labels = tf.zeros(N,dtype=tf.int32)
    # positives are the 0-th
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits/self.T)
    loss = -tf.math.log(tf.math.softmax(logits/self.T)[:,0])
    loss = tf.reduce_mean(loss)
    return loss

  @tf.function
  def train_step(self, data):
    '''训练'''
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    # print('data:', data)
    x_q, x_k = data
    # tf.print('x_q:', tf.shape(x_q))
    # tf.print('x_k:', tf.shape(x_k))
    y_k = self.model_k(x_k, training=False)
    loss = 0.0
    global_step = tf.cast(self.optimizer.iterations, tf.float32)
    with tf.GradientTape() as tape:
      y_q = self.model_q(x_q, training=True)  # Forward pass
      # print('y_pred:', y_pred)
      # Compute the loss value
      # (the loss function is configured in `compile()`)
      # loss = self.loss(y, y_pred, regularization_losses=self.losses)
      loss = self.loss_obj(y_q, y_k)
    # tf.print('\nloss:', loss)
    # tf.print('\nglobal_step:', self.optimizer.iterations)

    # Compute gradients
    trainable_vars = self.model_q.trainable_variables
    # for var in trainable_vars:
    #   tf.print('trainable_vars:', var.shape)
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    # self.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    # return {m.name: m.result() for m in self.metrics}
    decay = tf.math.minimum(self.m, (1 + global_step) / (1000 + global_step))
    for i in range(len(self.model_q.variables)):
      var_q = self.model_q.variables[i]
      var_k = self.model_k.variables[i]
      var_q.assign(decay * var_k + (1 - decay) * var_q)
      var_k.assign(var_q)
    # 加入队列
    N = tf.shape(x_q)[0]
    y_k = tf.concat([
      tf.reshape(y_k[0], [N, -1]),
      tf.reshape(y_k[1], [N, -1]),
      tf.reshape(y_k[2], [N, -1]),
      ], axis=-1)
    y_k = tf.nn.l2_normalize(y_k, axis=1)
    self.push_queue(y_k)
    return {'loss': loss}

  @tf.function
  def test_step(self, data):
    '''评估'''
    x_q, x_k = data
    y_k = self.model_k(x_k, training=False)
    y_q = self.model_q(x_q, training=False)  # Forward pass
    loss = self.loss_obj(y_q, y_k)
    return {'loss': loss}

  def save_k(self, filepath, overwrite, include_optimizer, save_format, signatures, options, save_traces):
    return self.model_k.save(filepath, overwrite=overwrite, include_optimizer=include_optimizer, save_format=save_format, signatures=signatures, options=options, save_traces=save_traces)

  def save_weights_k(self, filepath, overwrite, save_format, options):
    return self.model_k.save_weights(filepath, overwrite=overwrite, save_format=save_format, options=options)