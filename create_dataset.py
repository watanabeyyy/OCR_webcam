import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

x = []
y = []
cnt = 0

cwd = os.getcwd()
img_dir = cwd + "/img/"
dir_name = os.listdir(img_dir)
for dname in dir_name:
    dir_path =  img_dir + dname + "/"
    img_name = os.listdir(dir_path)
    one_hot = np.zeros(10)
    one_hot[int(dname)] = 1
    for imname in img_name:
        im_path = dir_path+imname
        img = cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)
        img = img[10:-10,10:-10]
        #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)
        # plt.imshow(img)
        # plt.show()
        x.append(img)
        y.append(one_hot)
        cnt += 1
    print(cnt)
x = np.array(x)
y = np.array(y)
x = x.astype('float32') / 255.
x = np.reshape(x,(cnt, 108, 108, 1))
y = np.reshape(y,(cnt,10))
np.save("./npy/x", x)
np.save("./npy/y", y)
