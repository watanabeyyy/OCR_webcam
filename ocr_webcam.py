import cv2
import numpy as np
import copy

from keras.models import load_model

model = load_model('./model/MNIST_model.hdf5')

capture = cv2.VideoCapture(0)
capture.set(3, 512)  # Width
capture.set(4, 512)  # Heigh
capture.set(5, 30)  # FPS

cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
ret, image = capture.read()

temp = np.zeros((28,28))

while True:

    ret, src = capture.read()

    if ret == False:
        continue

    height, width, channels = src.shape[:3]
    dst = copy.copy(src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # ラベリング処理
    nlabels, labelimg, contours, CoGs = cv2.connectedComponentsWithStats(bin)

    centroid = []

    if nlabels > 0:
        for nlabel in range(1, nlabels):
            x, y, w, h, size = contours[nlabel]
            xg, yg = CoGs[nlabel]

            # 面積フィルタ
            if size >= 100 and size <= 10000:
                centroid.append([x, y, w, h, size])

    label_num = len(centroid)
    img_set = np.zeros((label_num, 28, 28, 1))
    for i, c in enumerate(centroid):
        x = c[0]
        y = c[1]
        w = c[2]
        h = c[3]
        sub_im = bin[y:y + h, x:x + w]
        sub_im = cv2.resize(sub_im, (18, 18))
        temp[5:23,5:23] = sub_im
        temp = temp.astype('float32') / 255.
        img_set[i] = np.reshape(temp, (28, 28, 1))

    if (label_num != 0):
        predict = model.predict(img_set)
        for i, c in enumerate(centroid):
            x = c[0]
            y = c[1]
            w = c[2]
            h = c[3]
            index = np.argmax(predict[i])
            prob = int(predict[i][index]*100)
            if (prob>10):
                cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(src, str(index), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

    cv2.imshow("Capture", src)

    if cv2.waitKey(33) >= 0:
        break

capture.release()
cv2.destroyAllWindows()
