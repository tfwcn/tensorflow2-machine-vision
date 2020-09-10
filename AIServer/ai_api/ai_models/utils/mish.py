import tensorflow as tf

class Mish(tf.keras.layers.Layer):
    def __init__(self, **args):
        super(Mish, self).__init__(**args)

    def build(self, input_shape):
        pass

    @tf.function
    def call(self, input):
        output = input * tf.math.tanh(tf.keras.activations.softplus(input))
        return output
        
    def compute_output_shape(self, input_shape):
        return input_shape