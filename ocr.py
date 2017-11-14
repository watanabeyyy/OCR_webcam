import cv2
import numpy as np
import copy
import sys
from keras.models import load_model
model = load_model('./model/MNIST_model.hdf5')

temp = np.zeros((28,28))

if __name__ == "__main__":
    src = cv2.imread("test.png", cv2.IMREAD_COLOR)
    if src is None:
        print("Failed to load image file.")
        sys.exit(1)


    height, width, channels = src.shape[:3]
    dst = copy.copy(src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # ラベリング処理
    nlabels, labelimg, contours, CoGs = cv2.connectedComponentsWithStats(bin)

    centroid = []

    if nlabels > 0:
        for nlabel in range(1, nlabels):
            x, y, w, h, size = contours[nlabel]
            xg, yg = CoGs[nlabel]

            # 面積フィルタ
            if size >= 30 and size <= 10000:
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
            if (prob>60):
                cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(src, "["+str(index)+"]"+":"+str(prob)+"%", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

    cv2.imshow("img", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()