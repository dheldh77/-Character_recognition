import tensorflow as tf

@tf.function
def conv(inputs, channel_size=1, layer_size=32, kernel_size=3, pooling=True, conv_strides=[1, 1, 1, 1], maxpool_ksize=[1, 2, 2, 1], maxpool_strides=[1, 2, 2, 1]):
    W_conv = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, channel_size, layer_size], stddev=0.1))
    conv = tf.nn.conv2d(inputs, W_conv, strides=conv_strides, padding='SAME')
    b_conv = tf.Variable(tf.truncated_normal([layer_size], stddev=0.1))
    conv_relu = tf.nn.relu(tf.add(conv, b_conv))
    if pooling:
        conv_maxpool = tf.nn.max_pool2d(conv_relu, ksize=maxpool_ksize, strides=maxpool_strides, padding='SAME')
        return conv_maxpool
    else:
        return conv_relu

@tf.function
def fc(inputs, input_size, layer_size, dropout_rate):
    W_fc = tf.Variable(tf.truncated_normal([input_size, layer_size], stddev=0.1))
    b_fc = tf.Variable(tf.truncated_normal([layer_size], stddev=0.1))
    fc = tf.add(tf.matmul(inputs, W_fc), b_fc)
    fc_relu = tf.nn.relu(fc)
    fc_dropout = tf.nn.dropout(fc_relu, dropout_rate)
    return fc_dropout

@tf.function
def out(input, input_size, class_size=2):
    W_output = tf.Variable(tf.truncated_normal([input_size, class_size], stddev=0.1))
    b_output = tf.Variable(tf.truncated_normal([class_size], stddev=0.1))
    f_output = tf.add(tf.matmul(input, W_output), b_output)
    return f_output