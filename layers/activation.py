import tensorflow as tf


def leaky_relu(input_tensor, leak=0.2, name="leaky_relu"):
    return tf.maximum(input_tensor, leak * input_tensor)