import tensorflow as tf
import tensorflow_addons as tfa
from ai_api.ai_models.backbones.inception_modules import BasicConv2D, Conv2DLinear, ReductionA


class Stem(tf.keras.layers.Layer):
  def __init__(self, weight_decay):
    super(Stem, self).__init__()
    self.conv1 = BasicConv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=2,
                 padding="valid",
                 weight_decay=weight_decay)
    self.conv2 = BasicConv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=1,
                 padding="valid",
                 weight_decay=weight_decay)
    self.conv3 = BasicConv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=1,
                 padding="same",
                 weight_decay=weight_decay)
    self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                         strides=2,
                         padding="valid")
    self.conv4 = BasicConv2D(filters=80,
                 kernel_size=(1, 1),
                 strides=1,
                 padding="same",
                 weight_decay=weight_decay)
    self.conv5 = BasicConv2D(filters=192,
                 kernel_size=(3, 3),
                 strides=1,
                 padding="valid",
                 weight_decay=weight_decay)
    self.conv6 = BasicConv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=2,
                 padding="valid",
                 weight_decay=weight_decay)

  def call(self, inputs, training=None, **kwargs):
    x = self.conv1(inputs, training=training)
    x = self.conv2(x, training=training)
    x = self.conv3(x, training=training)
    x = self.maxpool(x)
    x = self.conv4(x, training=training)
    x = self.conv5(x, training=training)
    x = self.conv6(x, training=training)

    return x


class InceptionResNetA(tf.keras.layers.Layer):
  def __init__(self, weight_decay):
    super(InceptionResNetA, self).__init__()
    self.b1_conv = BasicConv2D(filters=32,
                   kernel_size=(1, 1),
                   strides=1,
                   padding="same",
                   weight_decay=weight_decay)
    self.b2_conv1 = BasicConv2D(filters=32,
                  kernel_size=(1, 1),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.b2_conv2 = BasicConv2D(filters=32,
                  kernel_size=(3, 3),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.b3_conv1 = BasicConv2D(filters=32,
                  kernel_size=(1, 1),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.b3_conv2 = BasicConv2D(filters=32,
                  kernel_size=(3, 3),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.b3_conv3 = BasicConv2D(filters=32,
                  kernel_size=(3, 3),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.conv = Conv2DLinear(filters=256,
                 kernel_size=(1, 1),
                 strides=1,
                 padding="same",
                 weight_decay=weight_decay)

  def call(self, inputs, training=None, **kwargs):
    b1 = self.b1_conv(inputs, training=training)
    b2 = self.b2_conv1(inputs, training=training)
    b2 = self.b2_conv2(b2, training=training)
    b3 = self.b3_conv1(inputs, training=training)
    b3 = self.b3_conv2(b3, training=training)
    b3 = self.b3_conv3(b3, training=training)

    x = tf.concat(values=[b1, b2, b3], axis=-1)
    x = self.conv(x, training=training)

    output = tf.keras.layers.add([x, inputs])
    return tf.nn.relu(output)


class InceptionResNetB(tf.keras.layers.Layer):
  def __init__(self, weight_decay):
    super(InceptionResNetB, self).__init__()
    self.b1_conv = BasicConv2D(filters=128,
                   kernel_size=(1, 1),
                   strides=1,
                   padding="same",
                   weight_decay=weight_decay)
    self.b2_conv1 = BasicConv2D(filters=128,
                  kernel_size=(1, 1),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.b2_conv2 = BasicConv2D(filters=128,
                  kernel_size=(1, 7),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.b2_conv3 = BasicConv2D(filters=128,
                  kernel_size=(7, 1),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.conv = Conv2DLinear(filters=896,
                 kernel_size=(1, 1),
                 strides=1,
                 padding="same",
                 weight_decay=weight_decay)

  def call(self, inputs, training=None, **kwargs):
    b1 = self.b1_conv(inputs, training=training)

    b2 = self.b2_conv1(inputs, training=training)
    b2 = self.b2_conv2(b2, training=training)
    b2 = self.b2_conv3(b2, training=training)

    x = tf.concat(values=[b1, b2], axis=-1)
    x = self.conv(x, training=training)

    output = tf.keras.layers.add([x, inputs])
    return tf.nn.relu(output)


class ReductionB(tf.keras.layers.Layer):
  def __init__(self, weight_decay):
    super(ReductionB, self).__init__()
    self.b1_maxpool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                          strides=2,
                          padding="valid")
    self.b2_conv1 = BasicConv2D(filters=256,
                  kernel_size=(1, 1),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.b2_conv2 = BasicConv2D(filters=384,
                  kernel_size=(3, 3),
                  strides=2,
                  padding="valid",
                  weight_decay=weight_decay)
    self.b3_conv1 = BasicConv2D(filters=256,
                  kernel_size=(1, 1),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.b3_conv2 = BasicConv2D(filters=256,
                  kernel_size=(3, 3),
                  strides=2,
                  padding="valid",
                  weight_decay=weight_decay)
    self.b4_conv1 = BasicConv2D(filters=256,
                  kernel_size=(1, 1),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.b4_conv2 = BasicConv2D(filters=256,
                  kernel_size=(3, 3),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.b4_conv3 = BasicConv2D(filters=256,
                  kernel_size=(3, 3),
                  strides=2,
                  padding="valid",
                  weight_decay=weight_decay)

  def call(self, inputs, training=None, **kwargs):
    b1 = self.b1_maxpool(inputs)

    b2 = self.b2_conv1(inputs, training=training)
    b2 = self.b2_conv2(b2, training=training)

    b3 = self.b3_conv1(inputs, training=training)
    b3 = self.b3_conv2(b3, training=training)

    b4 = self.b4_conv1(inputs, training=training)
    b4 = self.b4_conv2(b4, training=training)
    b4 = self.b4_conv3(b4, training=training)

    return tf.concat(values=[b1, b2, b3, b4], axis=-1)


class InceptionResNetC(tf.keras.layers.Layer):
  def __init__(self, weight_decay):
    super(InceptionResNetC, self).__init__()
    self.b1_conv = BasicConv2D(filters=192,
                   kernel_size=(1, 1),
                   strides=1,
                   padding="same",
                   weight_decay=weight_decay)
    self.b2_conv1 = BasicConv2D(filters=192,
                  kernel_size=(1, 1),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.b2_conv2 = BasicConv2D(filters=192,
                  kernel_size=(1, 3),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.b2_conv3 = BasicConv2D(filters=192,
                  kernel_size=(3, 1),
                  strides=1,
                  padding="same",
                  weight_decay=weight_decay)
    self.conv = Conv2DLinear(filters=1792,
                 kernel_size=(1, 1),
                 strides=1,
                 padding="same",
                 weight_decay=weight_decay)

  def call(self, inputs, training=None, **kwargs):
    b1 = self.b1_conv(inputs, training=training)
    b2 = self.b2_conv1(inputs, training=training)
    b2 = self.b2_conv2(b2, training=training)
    b2 = self.b2_conv3(b2, training=training)

    x = tf.concat(values=[b1, b2], axis=-1)
    x = self.conv(x, training=training)

    output = tf.keras.layers.add([x, inputs])
    return tf.nn.relu(output)


def build_inception_resnet_a(n, weight_decay):
  block = tf.keras.Sequential()
  for _ in range(n):
    block.add(InceptionResNetA(weight_decay))
  return block


def build_inception_resnet_b(n, weight_decay):
  block = tf.keras.Sequential()
  for _ in range(n):
    block.add(InceptionResNetB(weight_decay))
  return block


def build_inception_resnet_c(n, weight_decay):
  block = tf.keras.Sequential()
  for _ in range(n):
    block.add(InceptionResNetC(weight_decay))
  return block


class InceptionResNetV1(tf.keras.Model):
  def __init__(self,
               classes,
               classifier_activation=tf.keras.activations.softmax,
               dropout_rate:float=0.2,
               weight_decay=0.01):
    super(InceptionResNetV1, self).__init__()
    self.stem = Stem(weight_decay)
    self.inception_resnet_a = build_inception_resnet_a(5, weight_decay)
    self.reduction_a = ReductionA(k=192, l=192, m=256, n=384, weight_decay=weight_decay)
    self.inception_resnet_b = build_inception_resnet_b(10, weight_decay)
    self.reduction_b = ReductionB(weight_decay)
    self.inception_resnet_c = build_inception_resnet_c(5, weight_decay)
    self.avgpool = tfa.layers.AdaptiveAveragePooling2D(output_size=(1,1))
    self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
    self.flat = tf.keras.layers.Flatten()
    self.fc = tf.keras.layers.Dense(units=classes,
                    activation=classifier_activation,
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay))

  def call(self, inputs, training=None, mask=None):
    x = self.stem(inputs, training=training)
    x = self.inception_resnet_a(x, training=training)
    x = self.reduction_a(x, training=training)
    x = self.inception_resnet_b(x, training=training)
    x = self.reduction_b(x, training=training)
    x = self.inception_resnet_c(x, training=training)
    x = self.avgpool(x)
    x = self.dropout(x, training=training)
    x = self.flat(x)
    x = self.fc(x)

    return x
