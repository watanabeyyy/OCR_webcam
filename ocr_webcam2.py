import cv2
import numpy as np
from  matplotlib import pyplot as plt
import datetime
import train_number

def sort_box_point(data):
    distance = data[:, 0] ** 2 + data[:, 1] ** 2

    sorted = np.copy(data)
    rank = np.argsort(distance)
    sorted[0] = data[rank[3]]
    sorted[2] = data[rank[0]]

    temp = data[rank[3]] - data[rank[1]]
    arg1 = abs(temp[0] / temp[1])
    temp = data[rank[3]] - data[rank[2]]
    arg2 = abs(temp[0] / temp[1])
    if (arg1 > arg2):
        sorted[1] = data[rank[1]]
        sorted[3] = data[rank[2]]
    else:
        sorted[1] = data[rank[2]]
        sorted[3] = data[rank[1]]
    return sorted

def detect_region(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_size = np.array((im.shape[1],im.shape[0]))
    ret, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    img, contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    th_area = img_size[0]*img_size[1] / 20
    contours_large = list(filter(lambda c: cv2.contourArea(c) > th_area, contours))
    #cv2.drawContours(im, contours_large, -1, (255, 0, 0), 3)
    rectness = []
    for (i, cnt) in enumerate(contours_large):
        rect = cv2.minAreaRect(cnt)
        box_area = rect[1][0]*rect[1][1]
        cnt_area = cv2.contourArea(cnt)
        rectness.append(cnt_area/box_area)

    try:
        max_rectness = max(rectness)
    except:
        max_rectness = 0
    if(max_rectness>0.9):
        cnt = contours_large[rectness.index(max_rectness)]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        min_point = img_size / 20
        max_point = img_size*19/20
        if ((box > max_point).any()==False) & ((box<min_point).any() == False):
            #im = cv2.drawContours(im, [box], 0, (0, 0, 255), 2)
            pts1 = np.float32(box)
            pts1 = sort_box_point(pts1)
            pts2 = np.float32([[128 * 4 - 1, 127], [0, 127], [0,0], [128 * 4 - 1, 0]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(im, M, (128*4, 128))
            return True, dst
        else:
            return False, im
    else:
        return False,im

save_im = False
pred_num = True
def read_num(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3),3)

    img_set = np.zeros((4, 128, 128, 1))
    for i in range(4):
        temp = gray[:, 128 * i:128 * (i + 1)]
        if (save_im):
            today = datetime.datetime.today()
            _name = "/" + str(today.year) + str(today.month) + str(today.day)
            _name = _name + "_" + str(today.hour) + str(today.minute) + str(today.second) + "_" + str(today.microsecond)
            if(i == 0):
                cv2.imwrite("./img/6/"+str(i)+str(_name)+".png",temp)
            elif(i == 1):
                cv2.imwrite("./img/7/" +str(i)+ str(_name) + ".png", temp)
            elif (i == 2):
                cv2.imwrite("./img/8/" +str(i)+ str(_name) + ".png", temp)
            else:
                cv2.imwrite("./img/9/" +str(i)+ str(_name) + ".png", temp)
        img_set[i] = np.reshape(temp, (128, 128, 1))

    if (pred_num):
        img_set = img_set[:,10:-10,10:-10,:]
        error,num = train_number.pred(img_set)
        print(error,num)

if (__name__=="__main__"):

    capture = cv2.VideoCapture(2)

    cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
    ret, image = capture.read()

    while True:

        ret, im = capture.read()

        if ret == False:
            continue

        bool , im = detect_region(im)
        if(bool):
            read_num(im)

        cv2.imshow("Capture", im)

        if cv2.waitKey(33) >= 0:
            break

    capture.release()
    cv2.destroyAllWindows()
