#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/8 16:08
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site : 
# @File : train.py
# @Software: PyCharm
#!/usr/bin/env python
# encoding: utf-8
import sklearn
import numpy as np
import cv2
import os
import glob
from sklearn.decomposition import PCA
import pickle
img_path = r'G:\IMG\tian\cut_ok'
img_list = glob.glob(os.path.join(img_path,'*.jpg'))
print(len(img_list))
if not os.path.exists('model'):
    os.mkdir('model')
'''
1.读取图像 2.分三块 1152/3 = 384 3.reshape 
4.PCA降维 
'''
def vector_extract():#生成样本矩阵并分块
    test_img = cv2.imread(img_list[0],0)
    print(test_img.shape)
    sample_num = len(img_list)
    h,w = test_img.shape
    Container = np.zeros([sample_num,h,w])#(404, 128, 1152)
    for i in range(sample_num):
        img_vertor = cv2.imread(img_list[i],0)
        # img_vertor = np.reshape(img, -1)
        Container[i,...]=img_vertor[:]
    datasets = np.split(Container,8,axis=2) #(404, 128, 384)*3
    # test = datasets[0].reshape((404,-1))
    return datasets


def train(data,k,i):
    pca = PCA(k)
    pca.fit_transform(data)  # 降维 404*k
    # pickle.dump(pca, 'model//pca'+str(i)+'.m')
    model_name = 'model//pca'+str(i)+'.txt'
    file = open(model_name, 'wb')
    pickle.dump(pca, file)


if __name__ == '__main__':
    Data_set = vector_extract()
    for i,data in enumerate(Data_set):
        data = data.reshape((data.shape[0],-1))
        train(data,200,i)
