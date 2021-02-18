from re import I
import tensorflow as tf
import sys
import os
import numpy as np
import argparse
sys.path.append(os.getcwd())
from ai_api.ai_models.unet.model import UNet
from ai_api.ai_models.losses.focal_loss import FocalLoss
from ai_api.ai_models.losses.focus_loss import FocusLoss
from ai_api.ai_models.unet.dataset_ywb import GetDataSet
import ai_api.ai_models.utils.image_helper as ImageHelper
from ai_api.ai_models.callbacks.save import SaveCallback

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type=str)
parser.add_argument('--val_data_path', type=str)

class UNetTrain(UNet):
  def __init__(self, depth: int=4, filters_base: int=64, output_filters: int=4, *args, **kwargs):
    super(UNetTrain, self).__init__(
      depth=depth,
      filters_base=filters_base,
      output_filters=output_filters,
      *args, **kwargs)

  @tf.function
  def train_step(self, data):
    '''训练'''
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    # print('data:', data)
    x, y_true = data
    # tf.print('train:', tf.shape(x), tf.shape(y_true))
    loss = 0.0
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss += self.loss(y_true, y_pred)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # tf.clip_by_global_norm，将梯度限制在10.0内，防止loss异常导致梯度爆炸
    gradients, gnorm = tf.clip_by_global_norm(gradients,
                                              10.0)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    return {'loss': loss, 'gnorm': gnorm}

  @tf.function
  def test_step(self, data):
    '''评估'''
    x, y_true = data
    # tf.print('train:', tf.shape(x), tf.shape(y_true))
    y_pred = self(x, training=False)
    loss = self.loss(y_true, y_pred)
    def output2Image(img, y_t, y_p):
      img_one = img[0] * 255
      img_one.astype(np.int32)
      # print(y_one.shape)
      ImageHelper.opencvImageToFile('a.jpg',img_one)
      for i in range(y_t.shape[-1]):
        y_one = y_t[0,...,i:i+1] * 255
        y_one.astype(np.int32)
        # print(y_one.shape)
        ImageHelper.opencvImageToFile('b'+str(i)+'.jpg',y_one)
      for i in range(y_p.shape[-1]):
        y_one = y_p[0,...,i:i+1] * 255
        y_one.astype(np.int32)
        # print(y_one.shape)
        ImageHelper.opencvImageToFile('c'+str(i)+'.jpg',y_one)
    tf.numpy_function(output2Image, [x, y_true, y_pred], ())
    return {'loss': loss}


def main():
  model = UNetTrain(output_filters=4, filters_base=16)
  # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=FocalLoss(alpha=0.25, gamma=1.0))
  # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=FocusLoss(threshold=0.5))
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.BinaryCrossentropy())
  # 加载模型
  log_dir = './data/'
  model_path = log_dir + 'unet_model.h5'
  _ = model(tf.ones((1, 512, 512, 3)))
  if os.path.exists(model_path):
      model.load_weights(model_path)
      print('加载模型:{}'.format(model_path))
  model.summary()
  data_set_train, _ = GetDataSet(
      label_path=parser.train_data_path,
      batch_size=1,
      points_num=4,
      input_size=(512, 512),
      output_size=(512, 512))
  data_set_val, _ = GetDataSet(
      label_path=parser.val_data_path,
      batch_size=1,
      points_num=4,
      input_size=(512, 512),
      output_size=(512, 512))
  # 训练回调方法
  early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor='loss', min_delta=0, patience=10, verbose=1)
  print('开始训练')
  steps_per_epoch = 500
  num_epochs = 300
  model.fit(data_set_train,
      steps_per_epoch=steps_per_epoch,
      validation_data=data_set_val,
      validation_steps=1,
      epochs=num_epochs,
      initial_epoch=0,
      callbacks=[early_stopping, SaveCallback(model_path)]
      )
  model.save_weights(model_path)

if __name__ == '__main__':
  main()

