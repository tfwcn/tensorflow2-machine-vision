import tensorflow as tf


class DorpBlock(tf.keras.layers.Layer):
    '''
    DorpBlock
    '''

    def __init__(self, dist_prob, block_size=5, **kwargs):
        super(DorpBlock, self).__init__(**kwargs)
        self.dist_prob = dist_prob
        self.block_size = block_size

    def build(self, input_shape):
        if self.block_size > (input_shape[1] // 5):
            self.block_size = (input_shape[1] // 5) + 1
        pass

    @tf.function
    def call(self, x, training=False):
        '''x：(batch_size,h,w,c)'''
        # tf.print('training:', training)
        if not training:
            return x
        else:
            if tf.math.equal(tf.rank(x),4):
                x_shape = tf.shape(x)
                x_size = x_shape[1:3]
                x_size_f = tf.cast(x_size, tf.float32)
                # 计算block_size
                x_block_size_f = tf.constant((self.block_size, self.block_size), tf.float32)
                # x_block_size_f = x_size_f * self.block_size
                # x_block_size_f = tf.math.maximum(x_block_size_f, 1)
                x_block_size = tf.cast(x_block_size_f, tf.int32)
                # 根据dist_prob，计算block_num
                x_block_num = (x_size_f[0] * x_size_f[1]) * self.dist_prob / (x_block_size_f[0] * x_block_size_f[1])
                # 计算block在中心区域出现的概率
                x_block_rate = x_block_num / ((x_size_f[0] - x_block_size_f[0] + 1) * (x_size_f[1] - x_block_size_f[1] + 1))
                # tf.print('x_block_rate:', x_block_rate)
                # 根据概率生成block区域
                x_block_center = tf.random.uniform((x_shape[0], x_size[0] - x_block_size[0] + 1, x_size[1] - x_block_size[1] + 1, x_shape[3]), dtype=tf.float32)
                x_block_padding_t = x_block_size[0] // 2
                x_block_padding_b = x_size_f[0] - tf.cast(x_block_padding_t, tf.float32) - (x_size_f[0] - x_block_size_f[0] + 1.0)
                x_block_padding_b = tf.cast(x_block_padding_b, tf.int32)
                x_block_padding_l = x_block_size[1] // 2
                x_block_padding_r = x_size_f[1] - tf.cast(x_block_padding_l, tf.float32) - (x_size_f[1] - x_block_size_f[1] + 1.0)
                x_block_padding_r = tf.cast(x_block_padding_r, tf.int32)
                x_block_padding = tf.pad(x_block_center,[[0, 0],[x_block_padding_t, x_block_padding_b],[x_block_padding_l, x_block_padding_r],[0, 0]])
                x_block = tf.cast(x_block_padding<x_block_rate, tf.float32)
                x_block = tf.nn.max_pool2d(x_block, ksize=[self.block_size, self.block_size], strides=[1, 1], padding='SAME')
                
                x = x * (1-x_block)
                return x
            else:
                return x

    def compute_output_shape(self, input_shape):
        '''计算输出shape'''
        return input_shape
