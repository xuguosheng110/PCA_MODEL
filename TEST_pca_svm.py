#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/9 16:45
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site : 数据测试，包括六个pca模型与一个SVM模型
# @File : test.py
# @Software: PyCharm
#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import cv2
import os
import glob
import pickle
import time
dir_name = 'ngresult'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
# img_path = r'G:\IMG\tian\cut_ok'
img_path = r'G:\sky_land_0415\soft_vague\ng'
img_list = glob.glob(os.path.join(img_path,'*.jpg'))
model_list = []
for i in range(8):
    file = open('model\\pca' + str(i) + '.txt', 'rb')
    model = pickle.load(file)
    model_list.append(model)
SVM = pickle.load(open('model\\SVM.txt','rb'))

def test(img):
    small_pics = np.split(img, 8, axis=1)
    img_list = []
    for i, small_pic in enumerate(small_pics):
        data = np.reshape(small_pic, -1)  # (49152,)
        result = model_list[i].transform([data])  # (1, 200)
        back = model_list[i].inverse_transform(result)  # (1, 49152)
        back = np.asarray(back, dtype=np.uint8)
        back_img = np.reshape(back, (img.shape[0], -1))
        img_list.append(back_img)
    final_img = np.hstack((img_list[0],img_list[1],img_list[2],img_list[3],img_list[4],img_list[5],img_list[6],img_list[7]))
    return final_img


if __name__ == '__main__':
    time_list = []
    for i,img_name in enumerate(img_list):
        img = cv2.imread(img_name,0)
        start_time = time.time()
        model_img = test(img)
        diff_img = cv2.absdiff(img,model_img)
        # diff_img = cv2.erode(diff_img, (5, 5), iterations=2)*5
        # diff_img = cv2.Sobel(diff_img,cv2.CV_8U,0,2,ksize=5)
        # diff_img = cv2.GaussianBlur(diff_img, (5, 5), 1.5)*5
        mean = np.mean(diff_img,axis=0) #纵向取平均
        result = SVM.predict([mean])
        print('used time is: ',time.time()-start_time)
        print('result is ok!!') if result[0]==1 else print('result is ng!!')
        diff_img = diff_img*10
        last_img = np.vstack((img,model_img,diff_img))
        if result[0]==1:
            cv2.imwrite(dir_name+'//'+str(i)+'.jpg',last_img)
        # diff_img = diff_img
        # file_name = 'okpic//'+str(i)+'diff.jpg'
        # cv2.imwrite(file_name,diff_img)
        # cv2.imshow('result',last_img)
        # cv2.waitKey(1000)
