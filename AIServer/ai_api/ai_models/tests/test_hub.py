import tensorflow as tf
import tensorflow_hub as tf_hub

m = tf.keras.Sequential([
    tf_hub.KerasLayer("https://tfhub.dev/tensorflow/resnet_50/classification/1")
])
m.build([None, 224, 224, 3])  # Batch input shape.
m.summary()