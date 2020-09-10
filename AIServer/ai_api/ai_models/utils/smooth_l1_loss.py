import tensorflow as tf


class SmoothL1Loss(tf.keras.losses.Loss):
    def __init__(self, beta=0.5, **args):
        super(SmoothL1Loss, self).__init__(**args)
        # 梯度平滑阈值
        self.beta = beta

    def call(self, y_true, y_pred):
        '''获取损失值'''
        a = tf.math.abs(y_pred - y_true)
        loss = tf.where(a < self.beta, 0.5 * a ** 2 / self.beta, a - 0.5 * self.beta)
        return loss
