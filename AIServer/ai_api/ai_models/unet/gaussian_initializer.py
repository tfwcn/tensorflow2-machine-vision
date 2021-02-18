import tensorflow as tf
import math
from ai_api.ai_models.unet.gaussian_kernel_2d import gaussian_kernel_2d


class GaussianInitializer(tf.keras.initializers.Initializer):
  '''初始化高斯核'''

  def __init__(self, sigma=1.0):
    super(GaussianInitializer, self).__init__()
    self.sigma = sigma

  def get_gaussian(self, points):
    '''
    计算每个点的高斯值

    Args:
      points: [...,[y,x]]

    Returns:
      gaussians: [...,]
    '''
    y = points[..., 0] * 2.0 * self.sigma
    x = points[..., 1] * 2.0 * self.sigma
    g = 1.0 / (2.0 * math.pi * self.sigma ** 2.0) * math.e ** -((tf.square(y) + tf.square(x))/(2.0 * self.sigma ** 2.0))
    return g

  def __call__(self, shape, dtype=None):
    # (h, w, input_filters, output_filters) <class 'tuple'>
    # print(shape, type(shape))
    return gaussian_kernel_2d(shape, self.sigma)


def main():
  x = tf.zeros((3, 10, 10, 3), dtype=tf.float32)
  x = tf.keras.layers.Conv2D(
        filters=10,
        kernel_size=(7, 7),
        padding='same',
        kernel_initializer=GaussianInitializer(sigma=1.0),
        use_bias=False,
        trainable=False
      )(x)


if __name__ == '__main__':
  main()
