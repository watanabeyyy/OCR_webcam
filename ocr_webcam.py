import cv2
import numpy as np
import copy
from scipy.misc import imresize

from keras.models import load_model
model = load_model('./model/MNIST_model.hdf5')

# 与えられた引数全てについての論理積を返すメソッドです。
def multi_logical_and(*args):
    result = np.copy(args[0])
    for arg in args:
        result = np.logical_and(result, arg)
    return result

# 2値画像について、周囲1ピクセルをFalseで埋めるメソッドです
def padding(binary_image):
    row, col = np.shape(binary_image)
    result = np.zeros((row+2,col+2))
    result[1:-1, 1:-1] = binary_image[:, :]
    return result

# paddingの逆です
def unpadding(image):
    return image[1:-1, 1:-1]

# そのピクセルの周囲のピクセルの情報を格納したarrayを返します。
def generate_mask(image):
    row, col = np.shape(image)
    p2 = np.zeros((row, col)).astype(bool)
    p3 = np.zeros((row, col)).astype(bool)
    p4 = np.zeros((row, col)).astype(bool)
    p5 = np.zeros((row, col)).astype(bool)
    p6 = np.zeros((row, col)).astype(bool)
    p7 = np.zeros((row, col)).astype(bool)
    p8 = np.zeros((row, col)).astype(bool)
    p9 = np.zeros((row, col)).astype(bool)
    #上
    p2[1:row-1, 1:col-1] = image[0:row-2, 1:col-1]
    #右上
    p3[1:row-1, 1:col-1] = image[0:row-2, 2:col]
    #右
    p4[1:row-1, 1:col-1] = image[1:row-1, 2:col]
    #右下
    p5[1:row-1, 1:col-1] = image[2:row, 2:col]
    #下
    p6[1:row-1, 1:col-1] = image[2:row, 1:col-1]
    #左下
    p7[1:row-1, 1:col-1] = image[2:row, 0:col-2]
    #左
    p8[1:row-1, 1:col-1] = image[1:row-1, 0:col-2]
    #左上
    p9[1:row-1, 1:col-1] = image[0:row-2, 0:col-2]
    return (p2, p3, p4, p5, p6, p7, p8, p9)

# 周囲のピクセルを順番に並べたときに白→黒がちょうど1箇所だけあるかどうかを判定するメソッドです。
def is_once_change(p_tuple):
    number_change = np.zeros_like(p_tuple[0])
    # P2~P9,P2について、隣接する要素の排他的論理和を取った場合のTrueの個数を数えます。
    for i in range(len(p_tuple) - 1):
        number_change = np.add(number_change, np.logical_xor(p_tuple[i], p_tuple[i+1]).astype(int))
    number_change = np.add(number_change, np.logical_xor(p_tuple[7], p_tuple[0]).astype(int))
    array_two = np.ones_like(p_tuple[0]) * 2

    return np.equal(number_change, array_two)

# 周囲の黒ピクセルの数を数え、2以上6以下となっているかを判定するメソッドです。
def is_black_pixels_appropriate(p_tuple):
    number_of_black_pxels = np.zeros_like(p_tuple[0])
    array_two = np.ones_like(p_tuple[0]) * 2
    array_six = np.ones_like(p_tuple[0]) * 6
    for p in p_tuple:
        number_of_black_pxels = np.add(number_of_black_pxels, p.astype(int))
    greater_two = np.greater_equal(number_of_black_pxels, array_two)
    less_six = np.less_equal(number_of_black_pxels, array_six)
    return np.logical_and(greater_two, less_six)

def step1(image, p_tuple):
    #条件1
    condition1 = np.copy(image)

    #条件2
    condition2 = is_once_change(p_tuple)

    #条件3
    condition3 = is_black_pixels_appropriate(p_tuple)

    #条件4
    condition4 = np.logical_not(multi_logical_and(p_tuple[0], p_tuple[2], p_tuple[4]))

    #条件5
    condition5 = np.logical_not(multi_logical_and(p_tuple[2], p_tuple[4], p_tuple[6]))

    return np.logical_xor(multi_logical_and(condition1, condition2, condition3, condition4, condition5), image)

def step2(image, p_tuple):
    #条件1
    condition1 = np.copy(image)

    #条件2
    condition2 = is_once_change(p_tuple)

    #条件3
    condition3 = is_black_pixels_appropriate(p_tuple)

    #条件4
    condition4 = np.logical_not(np.logical_and(p_tuple[0], np.logical_and(p_tuple[2], p_tuple[6])))

    #条件5
    condition5 = np.logical_not(np.logical_and(p_tuple[0], np.logical_and(p_tuple[4], p_tuple[6])))

    return np.logical_xor(multi_logical_and(condition1, condition2, condition3, condition4, condition5), image)

# 黒を1、白を0とする画像を、2値画像に戻すメソッドです
def inv_black_one(inv_bool_image):
    bool_image = ~inv_bool_image.astype(bool)
    return bool_image.astype(int) * 255

# 2値化画像を細線化して返すメソッドです。
def ZhangSuen(image):

    image = padding(image)

    while True:
        old_image = np.copy(image)

        p_tuple = generate_mask(image)
        image = step1(image, p_tuple)
        p_tuple = generate_mask(image)
        image = step2(image, p_tuple)

        if (np.array_equal(old_image, image)):
            break

    return inv_black_one(inv_black_one(unpadding(image)))


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
        sub_im = imresize(sub_im, (18, 18))
        temp[5:23,5:23] = sub_im
        temp = np.where(temp > 10 ,255,0)
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
