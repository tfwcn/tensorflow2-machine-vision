import unittest
import tensorflow as tf
import math
import numpy as np

class GridTest(unittest.TestCase):
  """用于测试loss"""
  def test_grid(self):
    '''测试grid代码'''
    def get1():
      grid_shape = [13,13] # height, width
      grid_y = tf.range(0, grid_shape[0], dtype=tf.float32)
      grid_x = tf.range(0, grid_shape[1], dtype=tf.float32)
      grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
      grid_x = tf.reshape(grid_x, (grid_shape[0], grid_shape[1], 1, 1))
      grid_y = tf.reshape(grid_y, (grid_shape[0], grid_shape[1], 1, 1))
      grid_xy = tf.concat([grid_x, grid_y], axis=-1)
      return grid_xy
    def get2():
      grid_shape = [13,13] # height, width
      grid_y = tf.keras.backend.tile(tf.keras.backend.reshape(tf.keras.backend.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
          [1, grid_shape[1], 1, 1])
      grid_x = tf.keras.backend.tile(tf.keras.backend.reshape(tf.keras.backend.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
          [grid_shape[0], 1, 1, 1])
      # (height, width, 1, 2)
      grid = tf.keras.backend.concatenate([grid_x, grid_y])
      grid = tf.keras.backend.cast(grid, tf.float32)
      return grid
    
    out1 = get1().numpy()
    out2 = get2().numpy()
    # print(out1.shape,out2.shape,(out1==out2).all())
    self.assertTrue((out1==out2).all())

unittest.main()