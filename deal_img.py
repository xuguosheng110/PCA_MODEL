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

img_path = r'D:\CODE\pca_backgroung_model\okpic'
img_list = glob.glob(os.path.join(img_path,'*.jpg'))

for img_name in img_list:
    img = cv2.imread(img_name,0)
    img = cv2.erode(img,(3,3),iterations=5)
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    img = img*5
    cv2.imshow('img',img)
    cv2.waitKey(1000)
    mean = np.mean(img,axis=0) #纵向取平均
    print(mean)

