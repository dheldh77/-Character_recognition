import numpy as np
import cv2
import os
from numpy import array

TRAIN_DIR = './data/train/'
TEST_DIR = './data/test/'
PREDICT_DIR = './data/predict/'
POSITIVE = 1
NEGATIVE = 0

p_train_folder_list = array(os.listdir(TRAIN_DIR + 'positive'))
n_train_folder_list = array(os.listdir(TRAIN_DIR + 'negative'))
p_test_folder_list = array(os.listdir(TEST_DIR + 'positive'))
n_test_folder_list = array(os.listdir(TEST_DIR + 'negative'))
"""
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(p_train_folder_list)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print(onehot_encoded)
"""


#이미지 중앙정렬, 사이즈 축소
def ImageTuning(image):
    for y in range(0, len(image)):
        for x in range(0, 290):
            if image[y, x][3] == 0:
                image[y, x] = [255, 255, 255, 1]
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgray = ~imgray
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

    contours, hierachy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(contours[0])
    X = x + w
    Y = y + h
    for cnt in contours[1:]:
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        X_ = x_ + w_
        Y_ = y_ + h_
        if x_ < x:
            x = x_
        if y_ < y:
            y = y_
        if X_ > X:
            X = X_
        if Y_ > Y:
            Y = Y_

    # imgray = cv2.rectangle(imgray, (x, y), (X, Y), 255, 2)

    M = np.float32([[1, 0, len(image) / 2 - ((X + x) / 2)], [0, 1, 290 / 2 - ((Y + y) / 2)]])

    imgray = cv2.warpAffine(imgray, M, (len(image), 290))
    image = cv2.resize(imgray, (28, 28), interpolation=cv2.INTER_AREA)
    image = np.array(image)
    image = image.astype(dtype=np.float32)
    image /= 255.0
    return image

def PostProcessingWithoutLabel(path):
    """
    해당 path의 이미지를 전처리 합니다.
    :param path: 구체적인 데이터 경로 ex)./data/train/positive/1/1.png
    :return: input(numpy.array type의 28X28크기의 0~1사이 값을 가진 전처리 이미지)
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    print(img)
    img = ImageTuning(img)
    return img


def PostProcessingWithLabel(path, PN, number):
    """
    해당 path의 이미지를 라벨과 함께 전처리 합니다.
    :param path: 구체적인 데이터 경로 ex)./data/train/positive/1/1.png
    :param PN: 1(POSITIVE)-positive, 0(NEGATIVE)-negative
    :return: input, label(28X28크기를 가진 numpy.array type의 0~1사이 값을 가진 전처리 이미지와 1크기를가진 numpy.array type의 0또는 1의 값을 가진 라벨)
    lagel data: P(000000000) N(000000000)
    """
    if (PN == POSITIVE):
        return PostProcessingWithoutLabel(path), number - 1
    elif (PN == NEGATIVE):
        return PostProcessingWithoutLabel(path), number + 9 - 1
    else:
        exit()


def GetDataWithPP(dir):     # Get data with post processing
    p_path = dir + 'positive/'
    n_path = dir + 'negative/'
    input_data = []
    label_data = []

    for idx in range(1, 10):

        path = p_path + str(idx) + '/'
        img_list = os.listdir(path)

        for img in img_list:
            img_path = os.path.join(path, img)
            input, label = PostProcessingWithLabel(img_path, POSITIVE, idx)
            input_data.append(input)
            label_data.append(label)
        print(str(idx) + ' P: done')

        path = n_path + str(idx) + '/'
        img_list = os.listdir(path)

        for img in img_list:
            img_path = os.path.join(path, img)
            input, label = PostProcessingWithLabel(img_path, NEGATIVE, idx)
            input_data.append(input)
            label_data.append(label)
        print(str(idx) + ' N: done')

    return np.array(input_data), np.array(label_data)


# def SaveData():
#     input, label = GetDataWithPP(TRAIN_DIR)
#     path = "./post_data/train"
#     np.savez(path, x=input, y=label)
#     input, label = GetDataWithPP(TEST_DIR)
#     path = "./post_data/test"
#     np.savez(path, x=input, y=label)


def SaveData():     #shuffle version
    input, label = GetDataWithPP(TRAIN_DIR)
    sh = np.arange(input.shape[0])
    np.random.shuffle(sh)
    input = input[sh]
    label = label[sh]
    print(input[0:10])
    path = "./post_data/train"
    np.savez(path, x=input, y=label)

    input, label = GetDataWithPP(TEST_DIR)
    sh = np.arange(input.shape[0])
    np.random.shuffle(sh)
    input = input[sh]
    label = label[sh]
    print(input[0:10])
    path = "./post_data/test"
    np.savez(path, x=input, y=label)


def LoadData():
    path = "./post_data/train.npz"
    if not os.path.isfile(path):
        SaveData()
    train = np.load(path)
    train_input = train['x']
    train_label = train['y']
    path = "./post_data/test.npz"
    if not os.path.isfile(path):
        SaveData()
    test = np.load(path)
    test_input = test['x']
    test_label = test['y']
    return (train_input, train_label), (test_input, test_label)


def LoadTrainDataWithPP(number):
    return GetDataWithPP(TRAIN_DIR)


def LoadTestDataWithPP(number):
    return GetDataWithPP(TEST_DIR)


# def LoadDataWithPP(number):
#     """
#     해당 number의 train, test 데이터를 불러옵니다.
#     :param number: 가져올 데이터의 번호
#     :return: (train_input, train_label), (test_input, test_label)
#     """
#     return (LoadTrainDataWithPP(number)), (LoadTestDataWithPP(number))
