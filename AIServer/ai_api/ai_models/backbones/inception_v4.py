import tensorflow as tf
import tensorflow_addons as tfa
from ai_api.ai_models.backbones.inception_modules import Stem, InceptionBlockA, InceptionBlockB, \
  InceptionBlockC, ReductionA, ReductionB


def build_inception_block_a(n, weight_decay):
  block = tf.keras.Sequential()
  for _ in range(n):
    block.add(InceptionBlockA(weight_decay))
  return block


def build_inception_block_b(n, weight_decay):
  block = tf.keras.Sequential()
  for _ in range(n):
    block.add(InceptionBlockB(weight_decay))
  return block


def build_inception_block_c(n, weight_decay):
  block = tf.keras.Sequential()
  for _ in range(n):
    block.add(InceptionBlockC(weight_decay))
  return block


class InceptionV4(tf.keras.Model):
  def __init__(self,
               classes,
               classifier_activation=tf.keras.activations.softmax,
               dropout_rate:float=0.2,
               weight_decay=0.01):
    super(InceptionV4, self).__init__()
    self.stem = Stem()
    self.inception_a = build_inception_block_a(4,weight_decay)
    self.reduction_a = ReductionA(k=192, l=224, m=256, n=384, weight_decay=weight_decay)
    self.inception_b = build_inception_block_b(7,weight_decay)
    self.reduction_b = ReductionB(weight_decay)
    self.inception_c = build_inception_block_c(3, weight_decay)
    self.avgpool = tfa.layers.AdaptiveAveragePooling2D(output_size=(1,1))
    self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
    self.flat = tf.keras.layers.Flatten()
    self.fc = tf.keras.layers.Dense(units=classes,
                                    activation=classifier_activation,
                                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay))

  def call(self, inputs, training=True, mask=None):
    x = self.stem(inputs, training=training)
    x = self.inception_a(x, training=training)
    x = self.reduction_a(x, training=training)
    x = self.inception_b(x, training=training)
    x = self.reduction_b(x, training=training)
    x = self.inception_c(x, training=training)
    x = self.avgpool(x)
    x = self.dropout(x, training=training)
    x = self.flat(x)
    x = self.fc(x)

    return x
