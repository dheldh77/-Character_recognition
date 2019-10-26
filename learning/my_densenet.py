import numpy as np
import tensorflow as tf
from imageResizing import LoadData

def reset_graph(seed = 42):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

height = 28
width = 28
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"
conv2_dropout_rate = 0.25

pool3_fmaps = conv2_fmaps

n_fc1 = 128
fc1_dropout_rate = 0.5

n_outputs = 2

reset_graph()

#(28 X 28 X 1)
with tf.name_scope("inputs"):
    X = tf.compat.v1.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    tf.compat.v1.summary.image('input', X_reshaped, 10)
    y = tf.compat.v1.placeholder(tf.int32, shape=[None], name="y")
    dropout_rate = tf.placeholder(tf.float32)
    training = tf.compat.v1.placeholder_with_default(False, shape=[], name='training')

#(14 X 14 X 32)
with tf.name_scope("conv1"):
    W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
    conv1 = tf.nn.conv2d(X_reshaped, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    b_conv1 = tf.Variable(tf.truncated_normal([32], stddev=0.1))
    conv1_relu = tf.nn.relu(tf.add(conv1, b_conv1))
    conv1_maxpool = tf.nn.max_pool2d(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#(7 X 7 X 64)
with tf.name_scope("conv2"):
    W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
    conv2 = tf.nn.conv2d(conv1_maxpool, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    b_conv2 = tf.Variable(tf.truncated_normal([64], stddev=0.1))
    conv2_relu = tf.nn.relu(tf.add(conv2, b_conv2))
    conv2_maxpool = tf.nn.max_pool2d(conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#(7 X 7 X 64)
with tf.name_scope("conv3"):
    W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    conv3 = tf.nn.conv2d(conv2_maxpool, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
    b_conv3 = tf.Variable(tf.truncated_normal([64], stddev=0.1))
    conv3_relu = tf.nn.relu(tf.add(conv3, b_conv3))

with tf.name_scope("fc1"):

    reshape_data = tf.reshape(conv3_relu, [-1, 7*7*64])
    W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 128], stddev=0.1))
    b_fc1 = tf.Variable(tf.truncated_normal([128], stddev=0.1))
    fc1 = tf.add(tf.matmul(reshape_data, W_fc1), b_fc1)
    fc1_relu = tf.nn.relu(fc1)
    fc1_dropout = tf.nn.dropout(fc1_relu, dropout_rate)

with tf.name_scope("fc2"):
    W_fc2 = tf.Variable(tf.truncated_normal([128, 256], stddev=0.1))
    b_fc2 = tf.Variable(tf.truncated_normal([256], stddev=0.1))
    fc2 = tf.add(tf.matmul(fc1_dropout, W_fc2), b_fc2)
    fc2_relu = tf.nn.relu(fc2)
    fc2_dropout = tf.nn.dropout(fc2_relu, dropout_rate)
    print(tf.shape(fc2_dropout))

with tf.name_scope("output"):
    W_output = tf.Variable(tf.truncated_normal([256, 2], stddev=0.1))
    b_output = tf.Variable(tf.truncated_normal([2], stddev=0.1))
    f_output = tf.add(tf.matmul(fc2_dropout, W_output), b_output)


with tf.name_scope("train"):
    print(f_output.get_shape())
    print(y.get_shape())
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=f_output, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(f_output, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

(X_train, y_train), (X_test, y_test) = LoadData(9)
print('Data loaded')
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28)
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:40], X_train[40:]
y_valid, y_train = y_train[:40], y_train[40:]

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


n_epochs = 150
batch_size = 5

best_loss_val = np.infty
check_interval = 500
checks_since_last_progress = 0
max_checks_without_progress = 20
best_model_params = None

for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
merged = tf.compat.v1.summary.merge_all()
writer = tf.compat.v1.summary.FileWriter('./train')


with tf.Session() as sess:
    init.run()
    tf.compat.v1.global_variables_initializer().run()
    for epoch in range(n_epochs):
        n_batches = len(X_train) // batch_size
        for iteration in range(n_batches):
            X_batch, y_batch = next(shuffle_batch(X_train, y_train, batch_size))
            summary, _ = sess.run([merged, training_op], feed_dict={X: X_batch, y: y_batch, dropout_rate: 0.7})
            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: X_valid, y: y_valid, dropout_rate: 1.0})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
            writer.add_summary(summary, iteration)
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch, dropout_rate: 1.0})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid, dropout_rate: 1.0})
        if checks_since_last_progress > max_checks_without_progress:
            break
    writer.close()

    if best_model_params:
        restore_model_params(best_model_params)
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test, dropout_rate: 1.0})
    save_path = saver.save(sess, "./my_model")
    writer = tf.summary.FileWriter('./mygraph', sess.graph)
