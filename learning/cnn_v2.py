import numpy as np
import tensorflow as tf
from imageResizing import LoadData
import layer_define as ld

def reset_graph(seed = 42):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

height = 28
width = 28
channels = 1
n_inputs = height * width
n_outputs = 2

reset_graph()

with tf.name_scope("inputs"):
    X = tf.compat.v1.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    tf.compat.v1.summary.image('input', X_reshaped, 10)
    y = tf.compat.v1.placeholder(tf.int32, shape=[None], name="y")
    dropout_rate = tf.placeholder(tf.float32)

#(28X28X1)
conv1 = ld.conv(X_reshaped, channel_size=channels, layer_size=32)
#(14X14X32)
conv2 = ld.conv(conv1, channel_size=32, layer_size=64)
#(7X7X64)
conv3 = ld.conv(conv2, channel_size=64, layer_size=64, pooling=False)
#(7X7X64)
flat_size=7*7*64
conv3_flat = tf.reshape(conv3, [-1, flat_size])
#[7X7X64]
fc1 = ld.fc(conv3_flat, flat_size, layer_size=128, dropout_rate=dropout_rate)
fc2 = ld.fc(fc1, 128, 256, dropout_rate=dropout_rate)
output = ld.out(fc2, 256)

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(output, y, 1)
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
X_valid, X_train = X_train[:20], X_train[20:]
y_valid, y_train = y_train[:20], y_train[20:]

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
tf.compat.v2.summary.trace_on(graph=True, profiler=False)

with tf.Session() as sess:
    init.run()
    tf.compat.v1.global_variables_initializer().run()
    for epoch in range(n_epochs):
        n_batches = len(X_train) // batch_size
        for iteration in range(n_batches):
            X_batch, y_batch = next(shuffle_batch(X_train, y_train, batch_size))
            summary, _ = sess.run([merged, training_op], feed_dict={X: X_batch, y: y_batch, training: True})
            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: X_valid, y: y_valid})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
            writer.add_summary(summary, iteration)
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print("에포크 {}, 배치 데이터 정확도: {:.4f}%, 검증 세트 정확도: {:.4f}%, 검증 세트에서 최선의 손실: {:.6f}".format(
                  epoch, acc_batch * 100, acc_val * 100, best_loss_val))
        if checks_since_last_progress > max_checks_without_progress:
            print("조기 종료!")
            break
    writer.close()

    if best_model_params:
        restore_model_params(best_model_params)
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("테스트 세트에서 최종 정확도:", acc_test)
    save_path = saver.save(sess, "./my_model")
    tf.compat.v2.summary.trace_export(name="my_func_trace", profiler_outdir='./train')
