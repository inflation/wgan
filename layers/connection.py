import tensorflow as tf


def conv2d(input_tensor, output_dim,
           kernel_height=3, kernal_width=3,
           stride_height=2, stride_width=2, stddev=0.02,
           name="conv2d"):

    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_height, kernal_width, input_tensor.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_tensor, w, strides=[1, stride_height, stride_width, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def conv2d_trans(input_tensor, output_shape,
                 kernel_height=3, kernel_width=3,
                 stride_height=2, stride_width=2,
                 stddev=0.02,
                 name="conv2d_transposed", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_height, kernel_width, output_shape[-1],
                            input_tensor.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        conv_trans = tf.nn.conv2d_transpose(input_tensor, w, output_shape=output_shape,
                                            strides=[1, stride_height, stride_width, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        conv_trans = tf.reshape(tf.nn.bias_add(conv_trans, biases), conv_trans.get_shape())

        if with_w:
            return conv_trans, w, biases
        else:
            return conv_trans


def linear(input_tensor, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_tensor.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        w = tf.get_variable("w", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_tensor, w) + bias, w, bias
        else:
            return tf.matmul(input_tensor, w) + bias
