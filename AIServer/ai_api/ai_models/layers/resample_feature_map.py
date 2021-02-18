import tensorflow as tf
import functools
from ai_api.ai_models.layers.attention_conv import AttentionConv2D

class ResampleFeatureMap(tf.keras.layers.Layer):
  def __init__(self,
               target_num_channels: int,
               level_size: int,
               name=None, *args, **kwargs):
    super(ResampleFeatureMap, self).__init__(name=name, *args, **kwargs)
    self.target_num_channels = target_num_channels
    self.level_size = level_size

  def build(self, input_shape):
    height = input_shape[1]
    # width = input_shape[2]
    num_channels = input_shape[3]
    self.conv1 = None
    self.bn1 = None
    self.pool1 = None
    self.upsample1 = None
    if num_channels!=self.target_num_channels:
      # tf.print('ResampleFeatureMap Conv2D')
      self.conv1 = tf.keras.layers.Conv2D(
        self.target_num_channels, (1, 1),
        padding='same',
        name='conv2d')
      # self.conv1 = AttentionConv2D(
      #   self.target_num_channels, (1, 1),
      #   padding='same',
      #   name='conv2d')
      self.bn1 = tf.keras.layers.BatchNormalization(name='bn')
    if height>self.level_size:
      # tf.print('ResampleFeatureMap MaxPooling2D')
      self.pool1 = tf.keras.layers.MaxPooling2D(
        pool_size=[3, 3],
        strides=[2, 2],
        padding='SAME')
    elif height<self.level_size:
      # tf.print('ResampleFeatureMap UpSample2D')
      self.upsample1 = functools.partial(tf.image.resize,size=(self.level_size,self.level_size),method='nearest')
  
  def call(self, inputs, training):
    x = inputs
    if self.conv1:
      x = self.conv1(x)
      x = self.bn1(x, training=training)
    if self.pool1:
      x = self.pool1(x)
    if self.upsample1:
      x = self.upsample1(x)
    return x