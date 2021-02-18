import tensorflow as tf
from typing import Tuple, Union

class AttentionConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters:int, 
                 kernel_size:Union[int,Tuple[int,int]], 
                 strides:Union[int,Tuple[int,int]]=(1,1),
                 padding:str='same', 
                 use_bias:bool=False, 
                 kernel_initializer:tf.keras.initializers.Initializer=tf.keras.initializers.he_normal,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.filters=filters
        self.kernel_size=kernel_size
        self.padding=padding
        self.use_bias=use_bias
        self.strides=strides
        self.kernel_initializer=kernel_initializer

    def build(self, input_shape):
        self.W1_1 = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(1, 1), padding='same')
        self.W1_2 = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(1, 1), padding='same')
        self.V1 = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(1, 1), padding='same')
        self.W2_1 = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(1, 1), padding='same')
        self.W2_2 = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(1, 1), padding='same')
        self.V2 = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(1, 1), padding='same')
        self.conv1 = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(1, 1), padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            strides=self.strides,
            padding=self.padding, 
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer)

    def call(self, x):
        # 卷积部分
        o=self.conv1(x)
        o=self.bn1(o)
        o=tf.nn.swish(o)
        # 平面注意力
        o1_1=self.W1_1(x)
        o1_2=self.W1_2(o)
        o1=tf.math.tanh(o1_1+o1_2)
        o1=self.V1(o1)
        o1=tf.math.exp(o1) / tf.math.reduce_sum(tf.math.exp(o1), axis=[1,2], keepdims=True)
        # 通道注意力
        o2_1=self.W2_1(x)
        o2_2=self.W2_2(o)
        o2=tf.math.tanh(o2_1+o2_2)
        o2=self.V2(o2)
        o2=tf.math.exp(o2) / tf.math.reduce_sum(tf.math.exp(o2), axis=-1, keepdims=True)
        # 输入保留率,注意力
        o=tf.concat([o*o1+o*o2,x*(1.0-o1)+x*(1.0-o2)], axis=-1)
        o=self.conv2(o)
        return o

