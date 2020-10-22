import tensorflow as tf

class DemoModel(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super(DemoModel, self).__init__(*args, **kwargs)
    self.model_layers = []
    for i in range(7):
      self.model_layers.append(tf.keras.layers.Conv2D(20 * (i+1),(3, 3),padding='same'))
      self.model_layers.append(tf.keras.layers.MaxPool2D((2,2),padding='same'))
    self.classes_layers = []
    self.boxes_layers = []
    for i in range(5):
      self.classes_layers.append(tf.keras.layers.Conv2D(9*81,(1, 1),padding='same'))
      self.boxes_layers.append(tf.keras.layers.Conv2D(9*4,(1, 1),padding='same'))
  
  @tf.function
  def call(self, x, training):
    classes_outputs = []
    boxes_outputs = []
    for i in range(7):
      x = self.model_layers[i*2](x)
      x = self.model_layers[i*2+1](x)
      if i>1:
        x_shape = tf.shape(x)
        x_classes = self.classes_layers[i-2](x)
        x_classes = tf.reshape(x_classes,(x_shape[0],x_shape[1],x_shape[2],9,-1))
        classes_outputs.append(x_classes)
        x_boxes = self.boxes_layers[i-2](x)
        x_boxes = tf.reshape(x_boxes,(x_shape[0],x_shape[1],x_shape[2],9,-1))
        boxes_outputs.append(x_boxes)
    return tuple(classes_outputs), tuple(boxes_outputs)

def main():
  m = DemoModel()
  m.build((1, 960, 960, 3))
  m.summary()
  o = m(tf.zeros((2, 960, 960, 3)), training=False)
  print(len(o))
  for i in range(len(o)):
    tf.print(tf.shape(o[i]))

if __name__ == "__main__":
  main()
