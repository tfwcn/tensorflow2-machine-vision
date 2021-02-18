import tensorflow as tf
import numpy as np

# import sys
import os
# sys.path.append(os.getcwd())
from ai_api.ai_models.backbones.inception_resnet_v1 import InceptionResNetV1
from ai_api.ai_models.backbones.inception_resnet_v2 import InceptionResNetV2
from ai_api.ai_models.backbones.inception_v4 import InceptionV4
from ai_api.ai_models.backbones.repvgg import get_RepVGG_func_by_name
import ai_api.ai_models.facenet.lfw as lfw


class FaceNetModel(tf.keras.Model):
  def __init__(self,
               embedding_size:int,
               image_size:int, # 输入图片大小，默认160。输入图比160大，则裁剪
               backbone:str, # 主体结构：InceptionResNetV1、InceptionResNetV2、InceptionV4、KerasInceptionResNetV2
               dropout_rate:float,
               random_crop:bool, # 随机裁剪
               random_flip:bool, # 随机左右镜像，不建议用
               **kwargs):
    super(FaceNetModel, self).__init__(**kwargs)
    self.embedding_size = embedding_size
    self.image_size = image_size
    self.random_crop = random_crop
    self.random_flip = random_flip
    if backbone == 'InceptionResNetV1':
      self.backbone = InceptionResNetV1(classes=self.embedding_size, 
                                        classifier_activation=None,
                                        dropout_rate=dropout_rate)
    elif backbone == 'InceptionResNetV2':
      self.backbone = InceptionResNetV2(classes=self.embedding_size, 
                                        classifier_activation=None,
                                        dropout_rate=dropout_rate)
    elif backbone == 'InceptionV4':
      self.backbone = InceptionV4(classes=self.embedding_size, 
                                  classifier_activation=None,
                                  dropout_rate=dropout_rate)
    elif backbone == 'KerasInceptionResNetV2':
      self.backbone = tf.keras.applications.InceptionResNetV2(weights=None,
        classes=self.embedding_size,
        classifier_activation=None,
        input_shape=(self.image_size,self.image_size,3),
        include_top=True)
    elif backbone == 'RepVGG':
      self.backbone = get_RepVGG_func_by_name('RepVGG-B2g4')(num_classes=self.embedding_size,deploy=False)

  def call(self, inputs, training=None, mask=None):
    x = inputs
    x = self.backbone(x, training)
    x = tf.math.l2_normalize(x, axis=1, epsilon=1e-10, name='embeddings')
    return x

  @tf.function
  def euclidean_distance(self, embedding1, embedding2, axis=None):
    '''
    计算欧氏距离

    Args:
      embedding1:特征1，(batch_size, embedding_size)
      embedding2:特征2，(batch_size, embedding_size)
      axis:求和维度
    '''
    dist = tf.math.reduce_sum(tf.math.square(tf.math.subtract(embedding1,embedding2)), axis=axis)
    return dist
  
  @tf.function
  def load_image(self, image_path):
    '''
    读取图片

    Args:
      image_path: 图片路径

    Returns:
      images: (height, width, 3)
    '''
    # 读取图片文件
    # tf.print('image_path:', image_path)
    file_contents = tf.io.read_file(image_path)
    image = tf.image.decode_image(file_contents, channels=3)
    
    if self.random_crop:
      # 随机裁剪
      image = tf.image.random_crop(image, [self.image_size, self.image_size, 3])
    else:
      # 裁剪或填充到目标大小
      image = tf.image.resize_with_crop_or_pad(image, self.image_size, self.image_size)
    if self.random_flip:
      # 随机翻转
      image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(tf.cast(image, tf.float32))
    # tf.print('index:', index, tf.shape(image))
    return image

  @tf.function(input_signature=[
    tf.TensorSpec(shape=(None,), dtype=tf.string)])
  def get_embeddings(self, image_paths):
    '''图片列表转特征'''
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    image_dataset = image_dataset.map(self.load_image,
                          num_parallel_calls=tf.data.AUTOTUNE)
    image_dataset = image_dataset.batch(self.batch_size)
    image_dataset = image_dataset.prefetch(tf.data.AUTOTUNE)
    embeddings = tf.zeros((0, self.embedding_size), dtype=tf.float32)
    # 分批，因一次计算所有特征，显存不足
    i = 0
    for images_batch in image_dataset:
      # 对齐batch_size，加快训练速度
      batch_size_now = tf.shape(images_batch)[0]
      paddings = [[0,self.batch_size-batch_size_now], [0,0], [0,0], [0,0]]
      images_batch = tf.pad(images_batch, paddings, "CONSTANT")
      embeddings_batch = self(images_batch, training=False)
      # 对齐batch_size，加快训练速度
      embeddings_batch = embeddings_batch[0:batch_size_now]
      # tf.print('embeddings_batch:', tf.shape(embeddings_batch), type(embeddings_batch))
      embeddings = tf.concat([embeddings, embeddings_batch], axis=0)
    return embeddings

class FaceNetTrainModel(FaceNetModel):
  def __init__(self,
               alpha:float, # 正负距离差，默认0.2
               batch_size:int, # 批次大小
               people_per_batch:int, # 选多少人
               images_per_person:int, # 每人最多选多少张图
               loss_decay:float=0.9, # loss平均移动率，ExponentialMovingAverage
               moving_average:bool=True, # 是否使用MovingAverage
               moving_average_decay:float=0.9999, # 变量平均移动率，ExponentialMovingAverage
               strategy = None,
               **kwargs):
    super(FaceNetTrainModel, self).__init__(**kwargs)
    self.alpha = alpha
    self.batch_size = batch_size
    self.people_per_batch = people_per_batch
    self.images_per_person = images_per_person
    self.loss_decay = loss_decay
    self.moving_average = moving_average
    self.moving_average_decay = moving_average_decay
    self.strategy = strategy

  def build(self, input_shape):
    super(FaceNetTrainModel, self).build(input_shape)
    if self.moving_average:
      self.global_step = self.add_weight(name='global_step',
                                        shape=(), 
                                        dtype=tf.float32, 
                                        trainable=False,
                                        initializer=tf.keras.initializers.Zeros())
      self.shadow_loss = self.add_weight(name='shadow_loss',
                                        shape=(), 
                                        dtype=tf.float32, 
                                        trainable=False,
                                        initializer=tf.keras.initializers.Zeros())
      self.shadow_trainable_variables = []
      tf.print('shadow trainable_variables:', len(self.trainable_variables))
      for var in self.trainable_variables:
        new_weight = self.add_weight(name='shadow_' + var.name,
                                    shape=var.shape, 
                                    dtype=var.dtype, 
                                    trainable=False)
        new_weight.assign(var)
        self.shadow_trainable_variables.append(new_weight)

  @tf.function(input_signature=[
    tf.TensorSpec(shape=(None,None), dtype=tf.float32),
    tf.TensorSpec(shape=(None,None), dtype=tf.float32),
    tf.TensorSpec(shape=(None,None), dtype=tf.float32)])
  def triplet_loss(self, anchor, positive, negative):
    '''计算三元组loss
    
    Args:
      anchor: 参照点特征，(batch_size, embedding_size)
      positive: 同人特征，(batch_size, embedding_size)
      negative: 不同人特征，(batch_size, embedding_size)
    '''
    positive_dist = self.euclidean_distance(anchor, positive, axis=1)
    negative_dist = self.euclidean_distance(anchor, negative, axis=1)

    basic_loss = tf.math.add(tf.math.subtract(positive_dist, negative_dist), self.alpha)
    loss = tf.math.reduce_mean(tf.math.maximum(basic_loss, 0.0), axis=0)
    # loss += tf.math.reduce_mean(tf.math.abs(tf.math.subtract(tf.math.add(positive_dist,negative_dist),1)), axis=0)
    return loss

  @tf.function(input_signature=[
    tf.TensorSpec(shape=(None,None), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int32)])
  def select_triplets(self, embeddings, num_per_class):
    '''
    选择三元组
    
    Args:
      embeddings: 图片特征列表
      image_paths: 图片地址列表
      num_per_class: 同人数量列表
    '''
    triplets = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    embeddings_index = tf.range(tf.shape(embeddings)[0])
    # 遍历所有人
    embeddings_start_index = 0
    triplets_index = 0
    for i in tf.range(self.people_per_batch):
      # 同人有多少张图
      num_per = num_per_class[i]
      # tf.print('embeddings_other:', embeddings_start_index, num_per)
      # 其他人的特征
      embeddings_other = tf.concat([embeddings[0:embeddings_start_index],
                                  embeddings[embeddings_start_index+num_per:]], axis=0)
      embeddings_index_other = tf.concat([embeddings_index[0:embeddings_start_index],
                                  embeddings_index[embeddings_start_index+num_per:]], axis=0)
      # tf.print('embeddings_other:', tf.shape(embeddings_other))
      # tf.print('image_paths_other:', tf.shape(image_paths_other))
      # 遍历这个人的特征，多少图就有多少特征
      for i2 in tf.range(1,num_per):
        # 当前embeddings下标
        a_idx = embeddings_start_index+i2-1
        # 当前人的后面其他特征
        embeddings_one = embeddings[embeddings_start_index+i2:embeddings_start_index+num_per]
        embeddings_index_one = embeddings_index[embeddings_start_index+i2:embeddings_start_index+num_per]
        # 同人距离，(same_people_num,)
        # embeddings[a_idx]: (embedding_size,)
        # embeddings_one: (same_people_num, embedding_size)
        pos_dist_sqr = self.euclidean_distance(embeddings[a_idx], embeddings_one, axis=1)
        # 当前人与其他人的距离，(other_people_num,)
        # embeddings[a_idx]: (embedding_size,)
        # embeddings_other: (other_people_num, embedding_size)
        neg_dists_sqr = self.euclidean_distance(embeddings[a_idx], embeddings_other, axis=1)
        # 插入维度，混合计算所有同人与不同人距离，(同人下标，不同人下标)
        pos_dist_sqr = tf.reshape(pos_dist_sqr, (-1, 1))
        neg_dists_sqr = tf.reshape(neg_dists_sqr, (1, -1))
        # 找到合适的三元组，(同人下标，不同人下标)
        # 按条件判断
        # all_mask = tf.math.logical_and(neg_dists_sqr-pos_dist_sqr<self.alpha, pos_dist_sqr<neg_dists_sqr)
        all_mask = neg_dists_sqr-pos_dist_sqr<self.alpha
        # 遍历所有同人
        for i3 in tf.range(tf.shape(all_mask)[0]):
          # 获取符合条件的其他人
          embeddings_index_other_mask = tf.boolean_mask(embeddings_index_other, all_mask[i3])
          if tf.shape(embeddings_index_other_mask)[0]>0:
            p_idx = i3
            # 随机选择一个作为负样本
            n_idx = tf.random.uniform(shape=[], maxval=tf.shape(embeddings_index_other_mask)[0], dtype=tf.int32)
            # tf.print('triplets:', a_idx, p_idx, n_idx)
            # tf.print('triplets:', image_paths[a_idx], image_paths_one[p_idx], image_paths_other[n_idx])
            triplets = triplets.write(triplets_index,embeddings_index[a_idx])
            triplets_index += 1
            triplets = triplets.write(triplets_index,embeddings_index_one[p_idx])
            triplets_index += 1
            triplets = triplets.write(triplets_index,embeddings_index_other[n_idx])
            triplets_index += 1
      # 起始下标增加
      embeddings_start_index += num_per
    return triplets.stack()

  # @tf.function(input_signature=[
  #   tf.TensorSpec(shape=(None,None,None,3), dtype=tf.float32)])
  def train_step(self, data):
    '''
    训练流程
    '''
    loss = None
    # 对齐batch_size，加快训练速度
    batch_size_now = tf.shape(data)[0]
    paddings = [[0,self.batch_size-batch_size_now], [0,0], [0,0], [0,0]]
    data = tf.pad(data, paddings, "CONSTANT")
    # 记录总训练步数
    self.global_step.assign_add(1)
    with tf.GradientTape() as tape:
      embeddings = self(data, training=True)
      # 对齐batch_size，加快训练速度
      embeddings = embeddings[0:batch_size_now]
      # tf.print('embeddings:', tf.shape(embeddings))
      anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, self.embedding_size]), 3, 1)
      # tf.print('anchor, positive, negative:', tf.shape(anchor), tf.shape(positive), tf.shape(negative))
      loss = self.triplet_loss(anchor, positive, negative)
      # 平均移动loss
      if self.moving_average and self.shadow_loss != 0:
        loss = self.loss_decay * self.shadow_loss + (1 - self.loss_decay) * loss

    # 获取变量集，计算梯度
    trainable_vars = self.trainable_variables
    # tf.print('trainable_variables:', len(trainable_vars))
    gradients = tape.gradient(loss, trainable_vars)
    # 分布式训练时，梯度求平均
    gradients = tf.distribute.get_replica_context().all_reduce('mean', gradients)
    # 按梯度更新所有权重变量
    self.optimizer.apply_gradients(zip(gradients, trainable_vars), experimental_aggregate_gradients=False)
    # self.optimizer.iterations == self.global_step
    # tf.print('optimizer.iterations:', self.optimizer.iterations, self.global_step)
    
    # MovingAverage
    if self.moving_average:
      decay = tf.math.minimum(self.moving_average_decay, (1 + self.global_step) / (10 + self.global_step))
      for i,var in enumerate(trainable_vars):
        ema_trainable_variable = decay * self.shadow_trainable_variables[i] + (1 - decay) * var
        # if i == 0:
        #   tf.print('moving_average:', ema_trainable_variable[0,0,0,0], var[0,0,0,0], self.shadow_trainable_variables[i][0,0,0,0])
        var.assign(ema_trainable_variable)
        self.shadow_trainable_variables[i].assign(ema_trainable_variable)
      # 记录shadow_loss
      self.shadow_loss.assign(loss)
    # 返回一个dict指标名称映射到当前值
    return loss

  @tf.function(input_signature=[
    tf.TensorSpec(shape=(None,None,None,3), dtype=tf.float32)])
  def distributed_train_step(self, dataset_inputs):
    per_replica_losses = self.strategy.run(self.train_step,
                                                      args=(dataset_inputs,))
    return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                           axis=None)

  def test_step(self, data):
    import time
    
    start_time = time.time()
    image_paths, actual_issame, nrof_folds = data
    print('Running forward pass on LFW images: ', end='')
    emb_array = self.get_embeddings(image_paths).numpy()
    
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    
    print('%.3f' % (time.time()-start_time))
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

  def fit(self,
          dataset=None,
          epochs=None,
          steps_per_epoch=None,
          callbacks=None,
          verbose=1,
          lfw_pairs=None,
          lfw_dir=None,
          lfw_nrof_folds=None):
    '''Trains the model for a fixed number of epochs (iterations on a dataset).
    '''
    lfw_image_paths = None
    lfw_actual_issame = None
    if lfw_dir:
      # Read the file containing the pairs used for testing
      pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))

      # Get the paths for the corresponding images
      lfw_image_paths, lfw_actual_issame = lfw.get_paths(os.path.expanduser(lfw_dir), pairs)
      # tf.print('actual_issame:', actual_issame)

    # if self.moving_average:
    #   tf.print('shadow_loss:', self.shadow_loss)
    callbacks = tf.keras.callbacks.CallbackList(
        callbacks,
        add_history=True,
        add_progbar=verbose != 0,
        model=self,
        verbose=verbose,
        epochs=epochs,
        steps=steps_per_epoch)

    train_logs = {}
    self.stop_training = False
    callbacks.on_train_begin()
    # epochs_min = 0
    # loss_sum_min = -1.0
    for epoch in range(epochs):
      callbacks.on_epoch_begin(epoch)
      step = 0
      loss_sum = 0.0
      while step < steps_per_epoch:
        for image_paths, num_per_class in dataset.take(1):
          # 计算图片特征
          # tf.print('\nget_embeddings:', tf.shape(image_paths))
          embeddings = self.get_embeddings(image_paths)
          # 选择三元组
          # tf.print('select_triplets:', tf.shape(embeddings))
          triplets = self.select_triplets(embeddings, num_per_class)
          # tf.print('triplets1:', tf.shape(triplets), triplets.dtype)
          triplets = tf.gather(image_paths, triplets, axis=0)
          # tf.print('triplets2:', tf.shape(triplets), triplets.dtype)
          # tf.print('triplets3:', tf.reshape(triplets, (-1,3)))
          # 拆分三元组进行训练
          # 分批，因一次计算所有特征，显存不足
          triplets_dataset = tf.data.Dataset.from_tensor_slices(triplets)
          triplets_dataset = triplets_dataset.map(self.load_image,
                                                  num_parallel_calls=tf.data.AUTOTUNE)
          triplets_dataset = triplets_dataset.batch(self.batch_size // 3 * 3)
          triplets_dataset = triplets_dataset.prefetch(tf.data.AUTOTUNE)
          # 分批，因一次计算所有特征，显存不足
          for triplets_batch in triplets_dataset:
            callbacks.on_train_batch_begin(step)
            loss = self.distributed_train_step(triplets_batch)
            logs = {'loss': loss}
            step += 1
            # loss_sum += loss
            callbacks.on_train_batch_end(step, logs)
      epoch_logs = {'loss': loss_sum/step}
      train_logs = epoch_logs
      # tf.print('epochs:%d/%d, steps:%d/%d, loss:%.5f' % (epoch+1,epochs,step,steps_per_epoch,loss_sum/step))
      callbacks.on_epoch_end(epoch, epoch_logs)
      if lfw_dir:
        self.test_step((lfw_image_paths, lfw_actual_issame, lfw_nrof_folds))
      # # 动态调整学习速率
      # if loss_sum_min == -1 or loss_sum_min > loss_sum:
      #   loss_sum_min = loss_sum
      #   epochs_min = epoch
      # tf.print('\nloss_sum:%f %f' % (loss_sum, loss_sum_min))
      # if epoch - epochs_min > 10:
      #   self.optimizer.learning_rate.assign(self.optimizer.learning_rate*0.3)
      #   tf.print('\nlearning_rate:%f' % self.optimizer.learning_rate.numpy())
      # # 学习速率低于1E-6则停止训练
      # if self.optimizer.learning_rate < 1E-6:
      #   self.stop_training = True
      if self.stop_training:
        break
    callbacks.on_train_end(logs=train_logs)