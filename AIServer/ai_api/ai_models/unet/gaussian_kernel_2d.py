import tensorflow as tf
import math


def get_gaussian(points, sigma=1.0):
  '''
  计算每个点的高斯值

  Args:
    points: [...,[y,x]]

  Returns:
    gaussians: [...,]
  '''
  y = points[..., 0] * 2.0 * sigma
  x = points[..., 1] * 2.0 * sigma
  g = 1.0 / (2.0 * math.pi * sigma ** 2.0) * math.e ** -((tf.square(y) + tf.square(x))/(2.0 * sigma ** 2.0))
  return g

def gaussian_kernel_2d(shape, sigma=1.0):
  '''
  计算每个点的高斯值

  Args:
    shape: (h, w, input_filters, output_filters)

  Returns:
    gaussians: [h, w, input_filters, output_filters,]
  '''
  # (h, w, input_filters, output_filters) <class 'tuple'>
  # print(shape, type(shape))
  h, w, input_filters, output_filters = shape
  h_half = h//2
  w_half = w//2
  y = tf.range(-h_half, h-h_half, dtype=tf.float32) / h_half
  x = tf.range(-w_half, w-w_half, dtype=tf.float32) / w_half
  # tf.print('y:', y)
  # tf.print('x:', x)
  xv, yv = tf.meshgrid(x, y)
  xv = tf.reshape(xv, (h, w, 1, 1, 1))
  yv = tf.reshape(yv, (h, w, 1, 1, 1))
  # tf.print('yv:', yv.shape)
  # tf.print('xv:', xv.shape)
  kernel = tf.concat([yv, xv], axis=-1)
  kernel = tf.tile(kernel, (1, 1, input_filters, output_filters, 1))
  kernel = get_gaussian(kernel)
  return kernel