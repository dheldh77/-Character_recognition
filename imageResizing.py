import numpy as np
import cv2

img = cv2.imread('./data/train/positive/1/1.png', -1)

for y in range(0, 340):
    for x in range(0, 290):
        if img[y, x][3] == 0:
            img[y, x] = [255, 255, 255, 1]
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgray = ~imgray
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

x, y, w, h = cv2.boundingRect(cnt)
#imgray = cv2.rectangle(imgray, (x, y), (x + w, y + h), 255, 2)

M = np.float32([[1, 0, 340 / 2 - (x + w / 2)], [0, 1, 290 / 2 - (y + h / 2)]])

imgray = cv2.warpAffine(imgray, M, (340, 290))

postImg = cv2.resize(imgray, (28, 28), interpolation = cv2.INTER_AREA)

#290, 340
cv2.imshow('img', postImg)
cv2.waitKey()
cv2.destroyAllWindows()