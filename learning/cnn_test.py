import numpy as np
import tensorflow as tf
import os
from imageResizing import LoadData
import datetime

MODEL_PATH = './model'
if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)
MODEL_NAME = '3C3D_model.h5'
MODEL_PATH = os.path.join(MODEL_PATH, MODEL_NAME)

batch_size = 5

dropout_rate = 0.25

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(dropout_rate))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(dropout_rate))
model.add(tf.keras.layers.Dense(18, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

(X_train, y_train), (X_test, y_test) = LoadData()
print(np.shape(X_train))
print(len(X_train))
X_train = X_train.astype(np.float32).reshape(len(X_train), 28, 28, 1)
X_test = X_test.astype(np.float32).reshape(len(X_test), 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
print(np.shape(y_train))
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)
#X_valid, X_train = X_train[:40], X_train[40:]
#y_valid, y_train = y_train[:40], y_train[40:]

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=1)
file_writer = tf.summary.create_file_writer(log_dir)
with file_writer.as_default():
    tf.summary.image("Training data", X_train, step=0, max_outputs=len(X_train))
    tf.summary.image("Test data", X_test, step=0, max_outputs=len(X_test))

history = model.fit(X_train, y_train, epochs=50, callbacks=[ tb])
#validation_data=(X_valid, y_valid)

model.save(MODEL_PATH)
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
with file_writer.as_default():
    tf.summary.scalar("train_loss", train_loss, step=0)
    tf.summary.scalar("train_acc", train_acc, step=0)
    tf.summary.scalar("test_loss", test_loss, step=0)
    tf.summary.scalar("test_acc", test_acc, step=0)

print('Train: %.5f, Test: %.5f, Train loss: %.5f, Test loss: %.5f' % (train_acc, test_acc, train_loss, test_loss))
print('기계학습 모델 생성 완료!')
