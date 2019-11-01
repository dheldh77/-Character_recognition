import os
import tensorflow as tf
<<<<<<< HEAD
import sys
from imageResizing import PostProcessingWithoutLabel


class Predict:
    def __init__(self):
        self.__MODEL_PATH = './model'
        self.__MODEL_NAME = '3C3D_model.h5'
        self.__MODEL_PATH = os.path.join(self.__MODEL_PATH, self.__MODEL_NAME)
        self.load_model()

    def load_model(self):
        model = tf.keras.models.load_model(self.__MODEL_PATH)

    def predict(self, path):
        predict_data = PostProcessingWithoutLabel(path)
        print(self.model.predict(predict_data))

# MODEL_PATH = './model'
# predict_number = 8
# MODEL_NAME = '3C3D_model_' + str(predict_number) + '.h5'
# MODEL_PATH = os.path.join(MODEL_PATH, MODEL_NAME)
#
# try:
#     model = tf.keras.models.load_model(MODEL_PATH)
# except FileNotFoundError:
#     print(MODEL_PATH + ': File not found')
#     sys.exit()
#
# def predict_model(number, data):
=======
import cv2
from imageResizing import ImageTuning
>>>>>>> c085514faa5f2ad6cf5f73f8340279cbbbee886a

