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
label = 'ng'
img_path = label + 'pic0415' #ok394 #ng356 #1024 #1152
img_list = glob.glob(os.path.join(img_path,'*.jpg'))
test_img = cv2.imread(img_list[0],0)
x_ = len(img_list)
y_ = test_img.shape[1]
print(x_,y_)
matrix = np.empty((x_,y_))
for i,img_name in enumerate(img_list):
    img = cv2.imread(img_name,0) #(128,1152)
    # img = cv2.erode(img,(3,3),iterations=5)
    # img = cv2.GaussianBlur(img, (5, 5), 1.5)
    # cv2.imshow('img',img)
    # cv2.waitKey(1000)
    mean = np.mean(img,axis=0) #纵向取平均
    print(mean)
    matrix[i,:] = mean
np.savetxt('data\\'+label+'.csv', matrix, delimiter=',')
print(matrix.shape)

