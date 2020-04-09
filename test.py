#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/8 16:45
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site : 
# @File : test.py
# @Software: PyCharm
#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import cv2
import os
import glob
from sklearn.decomposition import PCA
import pickle
import time

img_path = r'D:\ok_d'
img_list = glob.glob(os.path.join(img_path,'*.jpg'))
model_list = []
if not os.path.exists('okpic'):
    os.mkdir('okpic')
for i in range(6):
    file = open('model\\pca' + str(i) + '.txt', 'rb')
    model = pickle.load(file)
    model_list.append(model)


def test(img):
    small_pics = np.split(img, 6, axis=1)
    img_list = []
    for i, small_pic in enumerate(small_pics):
        data = np.reshape(small_pic, -1)  # (49152,)
        result = model_list[i].transform([data])  # (1, 200)
        back = model_list[i].inverse_transform(result)  # (1, 49152)
        back = np.asarray(back, dtype=np.uint8)
        back_img = np.reshape(back, (img.shape[0], -1))
        img_list.append(back_img)
    final_img = np.hstack((img_list[0],img_list[1],img_list[2],img_list[3],img_list[4],img_list[5]))
    return final_img


if __name__ == '__main__':
    for i,img_name in enumerate(img_list):
        img = cv2.imread(img_name,0)
        start_time = time.time()
        model_img = test(img)
        print('used time is: ',time.time()-start_time)
        diff_img =  cv2.absdiff(img,model_img)
        # diff_img = diff_img
        file_name = 'okpic//'+str(i)+'diff.jpg'
        cv2.imwrite(file_name,diff_img)
        # cv2.imshow('result',diff_img)
        # cv2.waitKey(1000)



