import tensorflow as tf

class SaveCallback(tf.keras.callbacks.Callback):
  def __init__(self, path, only_weights:bool=True):
    '''初始化模型层'''
    super(SaveCallback, self).__init__()
    self.path = path
    self.only_weights = only_weights

  def on_epoch_end(self, batch, logs=None):
    if self.only_weights:
      self.model.save_weights(self.path)
    else:
      self.model.save(self.path)
    # print('\n保存模型:{}'.format(self.path))