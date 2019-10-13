import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array

TRAIN_DIR = './data/train/'
TEST_DIR = './data/test/'
p_train_folder_list = array(os.listdir(TRAIN_DIR + 'positive'))
n_train_folder_list = array(os.listdir(TRAIN_DIR + 'negative'))
p_test_folder_list = array(os.listdir(TEST_DIR + 'positive'))
n_test_folder_list = array(os.listdir(TEST_DIR + 'negative'))

train_input = []
train_number = []
train_label = []
test_input = []
test_label = []
"""
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(p_train_folder_list)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print(onehot_encoded)
"""

#이미지 중앙정렬, 사이즈 축소
def ImageTunning(image):
    for y in range(0, len(image)):
        for x in range(0, 290):
            if image[y, x][3] == 0:
                image[y, x] = [255, 255, 255, 1]
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgray = ~imgray
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]

    x, y, w, h = cv2.boundingRect(cnt)
    # imgray = cv2.rectangle(imgray, (x, y), (x + w, y + h), 255, 2)

    M = np.float32([[1, 0, len(image) / 2 - (x + w / 2)], [0, 1, 290 / 2 - (y + h / 2)]])

    imgray = cv2.warpAffine(imgray, M, (len(image), 290))
    image = cv2.resize(imgray, (28, 28), interpolation=cv2.INTER_AREA)
    image = np.array(image)
    image = image.astype(dtype=np.float32)
    image /= 255.0
    return image

def PostProcessing(path, PN):
    path = os.path.join(path, PN) + '/1/'
    img_list = os.listdir(path)
    data_input = []
    data_label = []
    for img in img_list:
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = ImageTunning(img)
        data_input.append(np.array(img))
        if (PN == 'positive'):
            data_label.append(np.array(1))
        else:
            data_label.append(np.array(0))
    return data_input, data_label

def LoadData():
    p_train_input, p_train_label = PostProcessing(TRAIN_DIR, 'positive')
    n_train_input, n_train_label = PostProcessing(TRAIN_DIR, 'negative')
    p_test_input, p_test_label = PostProcessing(TEST_DIR, 'positive')
    n_test_input, n_test_label = PostProcessing(TEST_DIR, 'negative')
    train_data = (np.array(p_train_input + n_train_input), np.array(p_train_label + n_train_label))
    test_data = (np.array(p_test_input + n_test_input), np.array(p_test_label + n_test_label))
    return train_data, test_data

(X_train, y_train), (X_test, y_test) = LoadData()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_train, y_train)
print(X_test, y_test)

# for index in range(len(p_train_folder_list)):
#train input
# p_path = os.path.join(TRAIN_DIR + 'positive', p_train_folder_list[0])
# n_path = os.path.join(TRAIN_DIR + 'negative', n_train_folder_list[0])
# p_path += '/'
# n_path += '/'
# img_list = os.listdir(p_path)
# for img in img_list:
#     img_path = os.path.join(p_path, img)
#     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#     img = ImageTunning(img)
#     #img = np.array(img)
#     img = img.reshape(784)
#     train_input.append(img)
#     # train_number.append([np.array(onehot_encoded[index])])
#     train_label.append(np.array([1]))
# img_list = os.listdir(n_path)
# for img in img_list:
#     img_path = os.path.join(n_path, img)
#     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#     img = ImageTunning(img)
#     #img = np.array(img)
#     img = img.reshape(784)
#     train_input.append(img)
#     # train_number.append([np.array(onehot_encoded[index])])
#     train_label.append(np.array([0]))
#
# train_input = np.array(train_input)
# train_label = np.array(train_label)
#
# #test input
# p_path = os.path.join(TEST_DIR + 'positive', p_test_folder_list[0])
# n_path = os.path.join(TEST_DIR + 'negative', n_test_folder_list[0])
# p_path += '/'
# n_path += '/'
# img_list = os.listdir(p_path)
# for img in img_list:
#     img_path = os.path.join(p_path, img)
#     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#     img = ImageTunning(img)
#     #img = np.array(img)
#     img = img.reshape(784)
#     test_input.append(img)
#     # train_number.append([np.array(onehot_encoded[index])])
#     test_label.append(np.array([1]))
# img_list = os.listdir(n_path)
# for img in img_list:
#     img_path = os.path.join(n_path, img)
#     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#     img = ImageTunning(img)
#     #img = np.array(img)
#     img = img.reshape(784)
#     test_input.append(img)
#     # train_number.append([np.array(onehot_encoded[index])])
#     test_label.append(np.array([0]))
#
# test_input = np.array(test_input)
# test_label = np.array(test_label)


# train = zip(train_input, train_label)
# test = zip(test_input, test_label)
#
# split = dict(train = train, test = test)
#
# options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
#
# for key in split.keys():
#     dataset = split.get(key)
#     writer = tf.python_io.TFRecordWriter(path='./npydata/{}.tfrecords'.format(key), options = options)
#     for data, label in dataset:
#         image = data.tobytes()
#         example = tf.train.Example(features=tf.train.Features(feature = {
#             'label' : tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
#             'image' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image]))
#         }))
#         writer.write(example.SerializeToString())
#     else:
#         writer.close()
#         print('{} was converted to tfrecords'.format(key))


#np.save('./npydata/train/1.npy', np.array([train_input, train_label]))
#np.save('./npydata/test/1.npy', np.array([test_input, test_label]))
#train_input = np.reshape(train_input, (-1, 784))
# train_number = np.reshape(train_number, (-1, 9))
# train_input = np.array(train_input).astype(np.float32)
# train_label = np.array(train_label).astype(np.float32)
#np.save("train_data.npy", train_input)
# np.save("train_number.npy", train_number)
#np.save("train_label.npy", train_label)

#train_input = train_input / 255.0
#test_input = test_input / 255.0

