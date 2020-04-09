#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/8 18:11
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site : 
# @File : deal_img.py
# @Software: PyCharm
#!/usr/bin/env python
# encoding: utf-8

import os
import glob
import cv2
import numpy as np
if not os.path.exists('data'):
    os.mkdir('data')
img_path = 'okpic' #ok404 #ng890
img_list = glob.glob(os.path.join(img_path,'*.jpg'))
print(len(img_list))
matrix = np.empty((404,1152))
for i,img_name in enumerate(img_list):
    img = cv2.imread(img_name,0) #(128,1152)
    img = cv2.erode(img,(3,3),iterations=5)
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    # cv2.imshow('img',img)
    # cv2.waitKey(1000)
    mean = np.mean(img,axis=0) #纵向取平均
    print(mean)
    matrix[i,:] = mean
np.savetxt('data\\ok.csv', matrix, delimiter=',')
print(matrix.shape)

