import tensorflow as tf


def WeightVariables(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def BiasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2d(x):
    return tf.nn.max_pool2d(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')


def conv_layer(input_, shape):
    W = WeightVariables(shape)
    b = BiasVariable([shape[3]])
    return tf.nn.relu(conv2d(input_, W) + b)


def full_layer(input_, size):
    in_size = int(input_.get_shape())