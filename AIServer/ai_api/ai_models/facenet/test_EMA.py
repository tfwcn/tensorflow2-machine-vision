import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, trainable=False)
s_v = tf.Variable(0, dtype=tf.float32, trainable=False)
decay = 0.9999

for i in range(20):
  tmp_decay = tf.math.minimum(decay, (1 + i) / (10 + i))
  tf.print('tmp_decay:', tmp_decay)
  v.assign(10)
  v.assign(tmp_decay * s_v + (1 - tmp_decay) * v)
  s_v.assign(v)
  tf.print(i, v)
