#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/8 13:46
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site : 
# @File : pca0408.py
# @Software: PyCharm
#!/usr/bin/env python
# encoding: utf-8
import cv2
import os
import glob
import numpy as np
from sklearn.decomposition import PCA

img_path = r'D:\ok_d'
img_list = glob.glob(os.path.join(img_path,'*.jpg'))
print(img_list)
if not os.path.exists('pic'):
    os.mkdir('pic')


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
    # mean_img = np.mean(Container,0)
    # mean_img = np.reshape(mean_img,(h,w))
    # mean_img = np.asarray(mean_img,dtype=np.uint8)
    # cv2.imshow('new_img',mean_img)
    # cv2.waitKey(1000)
    # cv2.imwrite('mean_img.jpg',mean_img)
    return Container


def fastPCA(Container,k):#降低到K维
    pca = PCA(k)
    x_data = pca.fit_transform(Container)#降维 404*k
    result = pca.transform([Container[0]])
    print(result.shape)
    '''
    x_inverse = pca.inverse_transform(x_data)#还原 404 *147456
    for i in range(200):
        img = x_inverse[i,:]
        img = np.reshape(img,(128,1152))
        origin_img = Container[i,:]
        origin_img = np.reshape(origin_img, (128, 1152))
        # img = img*5
        diff = img - origin_img
        diff = np.asarray(diff,dtype=np.uint8)*10
        img = np.vstack((origin_img,img))
        img = np.asarray(img,dtype=np.uint8)
        cv2.imshow('img',diff)
        cv2.waitKey(1000)
        cv2.imwrite('pic\\'+str(i)+'.jpg',img)
        print('当前张数： ',i)
        '''
    # figs, objs = plt.subplots(2, 5, figsize=(8, 5), subplot_kw={"xticks": [], "yticks": []})
    # for i, obj in enumerate(objs.flat):
    #     obj.imshow(V[i, :].reshape(128, 1152), cmap="gray")
    #     cv2.waitKey(1000)


if __name__ == '__main__':
    all_imgs = vector_extract()
    fastPCA(all_imgs,200)
