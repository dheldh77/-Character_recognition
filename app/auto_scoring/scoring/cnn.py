import numpy as np
import tensorflow as tf
from imageResizing import LoadData
import datetime

def image_learning():
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
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    (X_train, y_train), (X_test, y_test) = LoadData(1)
    print(np.shape(X_train))
    print(len(X_train))
    X_train = X_train.astype(np.float32).reshape(len(X_train), 28, 28, 1)
    X_test = X_test.astype(np.float32).reshape(len(X_test), 28, 28, 1)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:40], X_train[40:]
    y_valid, y_train = y_train[:40], y_train[40:]

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[es, tb])
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
