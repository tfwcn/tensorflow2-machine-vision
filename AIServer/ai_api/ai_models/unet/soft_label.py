import tensorflow as tf
import numpy as np

import sys
import os
sys.path.append(os.getcwd())
import ai_api.ai_models.utils.image_helper as ImageHelper
from ai_api.ai_models.unet.gaussian_kernel_2d import gaussian_kernel_2d

class SoftLabel():
  def __init__(self, image_size, points_num, kernel_size):
    '''
    Args:
      image_size: (h,w)
      points_num: 关键点数量
      kernel_size: (kh,kw)
    '''
    self.image_size = image_size
    self.points_num = points_num
    self.kernel_size = kernel_size

  @tf.function
  def get_target(self, points):
    '''
    Args:
      points: (points_num,2)
    '''
    # tf.print(type(image_size), type(points))
    # points_num = tf.shape(points)[0]
    points_num = self.points_num
    x = tf.zeros((self.image_size[0], self.image_size[1], points_num), dtype=tf.float32)
    indices = tf.TensorArray(dtype=tf.int32,size=0,dynamic_size=True)
    updates = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
    array_index = 0
    for i in range(points_num):
      if points[i,0]>=0 and points[i,1]>=0 and points[i,0]<self.image_size[0] and points[i,1]<self.image_size[1]:
        # 跳过超出范围的点
        indices = indices.write(array_index, (points[i,0],points[i,1],i))
        updates = updates.write(array_index, 1)
        array_index += 1
    x = tf.tensor_scatter_nd_update(x, indices.stack(), updates.stack())
        
    # 高斯核
    kernel_shape = (self.kernel_size[0],self.kernel_size[1],1,1)
    kernel = gaussian_kernel_2d(kernel_shape)
    # 每个关键点，单独用高斯核卷积
    x_list = []
    x = tf.expand_dims(x, axis=0)
    for i in range(points_num):
      x_one = tf.nn.conv2d(
            x[...,i:i+1],
            filters=kernel,
            strides=(1,1),
            padding='SAME'
          )
      x_one = x_one / tf.math.reduce_max(x_one, axis=[1,2])
      x_list.append(x_one)
    x = tf.concat(x_list, axis=-1)
    x = tf.reshape(x, [self.image_size[0], self.image_size[1], points_num])
    return x


def main():
  soft_label = SoftLabel(image_size=(100, 100), points_num=3, kernel_size=(11, 11))
  x = soft_label.get_target(tf.constant([[-10,10],[15,15],[50,50]], dtype=tf.int32))
  print('x:', x.shape, x.dtype)
  for i in range(tf.shape(x)[-1]):
    x_one = x[...,i:i+1].numpy() * 255
    x_one.astype(np.int32)
    print(x_one.shape)
    ImageHelper.opencvImageToFile('a'+str(i)+'.jpg',x_one)

if __name__ == '__main__':
  main()