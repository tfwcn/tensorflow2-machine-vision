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
    self.queue = tf.queue.FIFOQueue(self.K, dtypes=tf.float32, shapes=[13*13*self.out_filters+26*26*self.out_filters+52*52*self.out_filters])
    self.queue.enqueue_many(queue_list)

  def call(self, inputs, training, mask=None):
    x_q, x_k = inputs
    out_q = self.model_q(x_q, training=training)
    out_k = self.model_k(x_k, training=training)
    return out_q, out_k

  @tf.function
  def GetLoss(self, y_q, y_k, queue):
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
    l_pos = tf.tensordot(tf.reshape(y_q, [N,1,-1]), tf.reshape(y_k, [N,-1,1]), axes=[[1,2],[2,1]])
    # negative logits: NxK
    l_neg = tf.matmul(tf.reshape(y_q, [N,-1]), queue)
    # logits: Nx(1+K)
    logits = tf.concat([l_pos, l_neg], axis=1)
    # contrastive loss, Eqn.(1)
    # labels = tf.zeros_like(logits,dtype=tf.float32)
    labels = tf.zeros(N,dtype=tf.int32)
    # positives are the 0-th
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits/self.T)
    # loss = tf.reduce_sum(tf.math.abs(loss))
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
    queue_size = self.queue.size()
    queue = self.queue.dequeue_many(queue_size)
    self.queue.enqueue_many(queue)
    queue = tf.reshape(queue,[-1,queue_size])
    with tf.GradientTape() as tape:
      y_q = self.model_q(x_q, training=True)  # Forward pass
      # print('y_pred:', y_pred)
      # Compute the loss value
      # (the loss function is configured in `compile()`)
      # loss = self.loss(y, y_pred, regularization_losses=self.losses)
      loss = self.loss_obj(y_q, y_k, queue)
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
    self.queue.dequeue_many(N)
    self.queue.enqueue_many(y_k)
    return {'loss': loss}

  @tf.function
  def test_step(self, data):
    '''评估'''
    x_q, x_k = data
    y_k = self.model_k(x_k, training=False)
    y_q = self.model_q(x_q, training=True)  # Forward pass
    queue_size = self.queue.size()
    queue = self.queue.dequeue_many(queue_size)
    self.queue.enqueue_many(queue)
    queue = tf.reshape(queue,[-1,queue_size])
    loss = self.loss_obj(y_q, y_k, queue)
    return {'loss': loss}

  def save_k(self, filepath, overwrite, include_optimizer, save_format, signatures, options, save_traces):
    return self.model_k.save(filepath, overwrite=overwrite, include_optimizer=include_optimizer, save_format=save_format, signatures=signatures, options=options, save_traces=save_traces)

  def save_weights_k(self, filepath, overwrite, save_format, options):
    return self.model_k.save_weights(filepath, overwrite=overwrite, save_format=save_format, options=options)