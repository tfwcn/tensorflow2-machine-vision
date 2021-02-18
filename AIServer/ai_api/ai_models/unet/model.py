import functools
import tensorflow as tf
from typing import Tuple, Union
from ai_api.ai_models.layers.attention_conv import AttentionConv2D

class UNetConv(tf.keras.layers.Layer):
  def __init__(self, filters: int, kernel_size: Union[int,Tuple[int, int]]=(3, 3), *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.filters = filters
    self.kernel_size = kernel_size

  def build(self, input_shape):
    # print(input_shape)
    self.conv1 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same', kernel_initializer=tf.keras.initializers.he_normal)
    # self.conv1 = AttentionConv2D(self.filters, self.kernel_size, padding='same', kernel_initializer=tf.keras.initializers.he_normal)
    self.bn1 = tf.keras.layers.BatchNormalization()

  def call(self, inputs, training):
    x = inputs
    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)
    return x

class UNetDownSampleLayer(tf.keras.layers.Layer):
  def __init__(self, filters: int, kernel_size: Union[int,Tuple[int, int]]=(3, 3), *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.filters = filters
    self.kernel_size = kernel_size

  def build(self, input_shape):
    # print(input_shape)
    self.conv1 = UNetConv(self.filters, self.kernel_size)
    self.conv2 = UNetConv(self.filters, self.kernel_size)
    self.max_pool1 = tf.keras.layers.MaxPool2D((2, 2))

  def call(self, inputs, training):
    x = inputs
    x = self.conv1(x, training=training)
    x = self.conv2(x, training=training)
    p = x
    x = self.max_pool1(x)
    return p, x

class UNetUpSampleLayer(tf.keras.layers.Layer):
  def __init__(self, filters: int, kernel_size: Union[int,Tuple[int, int]]=(3, 3), *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.filters = filters
    self.kernel_size = kernel_size

  def build(self, input_shape):
    '''
      input_shape：合并时为[p,x]
    '''
    # print('input_shape:', input_shape)
    self.cropping1 = None
    if isinstance(input_shape, list) and len(input_shape)==2:
      # 缩放
      self.cropping1 = functools.partial(tf.image.resize,size=[input_shape[1][1],input_shape[1][2]])
    self.conv1 = UNetConv(self.filters, self.kernel_size)
    self.conv2 = UNetConv(self.filters, self.kernel_size)
    # self.conv_transpose1 = tf.keras.layers.Conv2DTranspose(self.filters, (2, 2), strides=(2, 2), padding='valid')
    self.conv_transpose1 = tf.keras.layers.UpSampling2D(size=(2, 2))
    self.bn1 = tf.keras.layers.BatchNormalization()

  def call(self, inputs, training):
    if self.cropping1:
      # 裁剪与合并
      p, x = inputs
      p = self.cropping1(p)
      x = tf.concat([p, x], axis=-1)
    else:
      x = inputs
    x = self.conv1(x, training=training)
    x = self.conv2(x, training=training)
    p = x
    x = self.conv_transpose1(x)
    x = self.bn1(x, training=training)
    x = tf.nn.sigmoid(x)
    return p, x

class UNet(tf.keras.Model):
  def __init__(self, depth: int=4, filters_base: int=64, output_filters: int=1, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.depth = depth
    self.output_filters = output_filters
    self.filters_base = filters_base

  def build(self, input_shape):
    # print(input_shape)
    # 下采样与上采样层
    self.downsample_layers = []
    self.upsample_layers = []
    for d in range(self.depth):
      self.downsample_layers.append(UNetDownSampleLayer(self.filters_base*2**d))
      self.upsample_layers.append(UNetUpSampleLayer(self.filters_base*2**d))
    self.upsample_layers = self.upsample_layers[::-1]
    self.upsample_layer_last = UNetUpSampleLayer(self.filters_base*2**self.depth)
    self.conv_last = tf.keras.layers.Conv2D(self.output_filters, (1, 1), padding='same')
    # self.conv_last = AttentionConv2D(self.output_filters, (1, 1), padding='same')

  def call(self, inputs, training):
    x = inputs
    # 下采样
    downsample_outputs = []
    for downsample_layer in self.downsample_layers:
      p, x = downsample_layer(x, training=training)
      downsample_outputs.append(p)
    downsample_outputs = downsample_outputs[::-1]
    # 底层
    p, x = self.upsample_layer_last(x, training=training)
    # 上采样
    for i, upsample_layer in enumerate(self.upsample_layers):
      p, x = upsample_layer([downsample_outputs[i], x], training=training)
    x = self.conv_last(p)
    x = tf.nn.sigmoid(x)
    return x


def main():
  m = UNet()
  x = m(tf.zeros((3, 512, 512, 3), dtype=tf.float32), training=False)
  print('x:', x.shape)

if __name__ == '__main__':
  main()