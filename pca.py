#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/7 17:11
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site : 
# @File : pca.py
# @Software: PyCharm
#!/usr/bin/env python
# encoding: utf-8
import cv2
import os
import glob
import numpy as np

img_path = r'D:\ok_d'
img_list = glob.glob(os.path.join(img_path,'*.jpg'))
print(img_list)
# mean = []
#均值为 103.04824216399913
# for img_name in img_list:
#     read_img = cv2.imread(img_name,0)
#     single_mean = np.mean(read_img)
#     print(single_mean)
#     mean.append(single_mean)


def vector_extract():#生成样本矩阵，大小为n×d，n代表样本个数，d代表一个样本的特征数。
    test_img = cv2.imread(img_list[0],0)
    sample_num = len(img_list)
    h,w = test_img.shape
    Container = np.zeros([sample_num,h*w])
    label = np.zeros([sample_num,1])
    for i in range(sample_num):
        img = cv2.imread(img_list[i],0)
        img_vertor = np.reshape(img, -1)
        Container[i,:]=img_vertor[:]
    mean_img = np.mean(Container,0)
    mean_img = np.reshape(mean_img,(h,w))
    mean_img = np.asarray(mean_img,dtype=np.uint8)
    cv2.imshow('new_img',mean_img)
    cv2.waitKey(1000)
    cv2.imwrite('mean_img.jpg',mean_img)
    return Container


def fastPCA(Container,k):#降低到K维
    r,c = Container.shape
    mean_value = np.mean(Container)
    Z = Container - mean_value*np.ones((r,c))
    cov = np.cov(Z) #计算协方差矩阵
    V, D = np.linalg.eig(cov)#求解特征值与特征向量
    V = Z.T*V
    PCA_V = np.matmul(Z,V)#线性变化降维（404，404）



if __name__ == '__main__':
    samples = vector_extract()
    print(samples)
    fastPCA(samples,10)


