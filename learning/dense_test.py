import numpy as np
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tensorflow.keras.applications.densenet

from my_densenet import DenseNet

model = DenseNet(reduction=0.5, classes=2)

sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

out = model.predict(im)