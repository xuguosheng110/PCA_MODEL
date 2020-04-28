#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/28 10:28
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site : 
# @File : Train_.py
# @Software: PyCharm

import numpy as np
import cv2
import os
import glob
from sklearn.decomposition import PCA
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV

'''
目前没做裁剪工作  需增加
1.获取ok图像样本
2.ok图像背景建模 （1.图像分块 2.展平 3.PCA降维 4.训练PCA模型 ）
3.背景建模数据与源图像做差
4.做差之后训练svm分类器
'''
class Train(object):
    def __init__(self,ok_img_path,ng_img_path):
        self.ok_img_list = glob.glob(os.path.join(ok_img_path,'*.jpg'))
        self.ng_img_list = glob.glob(os.path.join(ng_img_path,'*.jpg'))
        self.len_ok_list = len(self.ok_img_list)
        self.len_ng_list = len(self.ng_img_list)
        self.k = 300                             #pca降维保留的主成分维度
        self.spilt_num = 8                       #图像切割数量与模型数量
        self.img_w = 1024                        #图像宽高
        self.img_h = 128                         # W需要是8的倍数!!!!!!!!!!!!!!!!

    def pca_train(self):
        if not os.path.exists('model'):
            os.mkdir('model')
        container_ok = np.zeros([self.len_ok_list,self.img_h,self.img_w])
        for i in range(self.len_ok_list):
            img_vector = cv2.imread(self.ok_img_list[i],0)
            container_ok[i,...] = img_vector[:]
        datasets_ok = np.split(container_ok,self.spilt_num,axis=2)
        for i,data in enumerate(datasets_ok):
            data = data.reshape((data.shape[0], -1))
            pca = PCA(self.k)
            pca.fit_transform(data)
            model_name = 'model//pca_' + str(i) + '.pickle'
            file = open(model_name, 'wb')
            pickle.dump(pca, file)

    def model_img(self,model_list,img):
        small_pics = np.split(img, self.spilt_num, axis=1)
        img_list = []
        for i, small_pic in enumerate(small_pics):
            data = np.reshape(small_pic, -1)  # (49152,)
            result = model_list[i].transform([data])  # (1, 200)
            back = model_list[i].inverse_transform(result)  # (1, 49152)
            back = np.asarray(back, dtype=np.uint8)
            back_img = np.reshape(back, (img.shape[0], -1))
            img_list.append(back_img)
        final_img = np.hstack((img_list[0], img_list[1], img_list[2],
                               img_list[3], img_list[4], img_list[5], img_list[6], img_list[7]))
        return final_img

    def data_parepare(self):
        model_list = []
        if not os.path.exists('data'):
            os.mkdir('data')
        matrix_ok = np.empty((self.len_ok_list, self.img_w))
        matrix_ng = np.empty((self.len_ng_list, self.img_w))

        for i in range(self.spilt_num):
            file = open('model\\pca_' + str(i) + '.pickle', 'rb')
            model = pickle.load(file)
            model_list.append(model)

        for i,img_name in enumerate(self.ok_img_list):
            img = cv2.imread(img_name,0)
            model_img = self.model_img(model_list,img)
            diff_img = cv2.absdiff(img, model_img)
            diff_img = cv2.erode(diff_img, (3, 3), iterations=5)
            blur = cv2.GaussianBlur(diff_img, (5, 5), 1.5)
            mean = np.mean(blur, axis=0)  # 纵向取平均
            matrix_ok[i, :] = mean
        # np.savetxt('data\\'  + '_ok.csv', matrix_ok, delimiter=',')

        for i,img_name in enumerate(self.ng_img_list):
            img = cv2.imread(img_name,0)
            model_img = self.model_img(model_list,img)
            diff_img = cv2.absdiff(img, model_img)
            diff_img = cv2.erode(diff_img, (3, 3), iterations=5)
            blur = cv2.GaussianBlur(diff_img, (5, 5), 1.5)
            mean = np.mean(blur, axis=0)  # 纵向取平均
            matrix_ng[i, :] = mean
        # np.savetxt('data\\'  + '_ng.csv', matrix_ng, delimiter=',')
        label_ok = np.ones((matrix_ok.shape[0], 1))
        label_ng = np.zeros((matrix_ng.shape[0], 1))
        data_ok = np.hstack((matrix_ok, label_ok))
        data_ng = np.hstack((matrix_ng, label_ng))
        data_sets = np.vstack((data_ok, data_ng))  # (1294, 1153)
        np.random.shuffle(data_sets)  # 打乱数据
        X = data_sets[:, 0:-1]
        Y = data_sets[:, -1]
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3)
        return train_x, test_x, train_y, test_y

    def svm_c(self,train_x, test_x, train_y, test_y):
        svc = SVC(kernel='linear', class_weight='balanced')
        c_range = np.logspace(-5, 15, 11, base=2)
        gamma_range = np.logspace(-9, 3, 13, base=2)
        # 网格搜索交叉验证的参数范围，cv=3,3折交叉
        param_grid = [{'kernel': ['linear', 'rbf'], 'C': c_range, 'gamma': gamma_range}]
        grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
        print('start train >>>>>>>>>>>>>>>>>>>>>>>')
        # 训练模型
        grid.fit(train_x, train_y)
        # 计算测试集精度
        score = grid.score(test_x, test_y)
        print('精度为%s' % score)
        model_name = 'model//SVM' + '.pickle'
        file = open(model_name, 'wb')
        pickle.dump(grid, file)


if __name__ == '__main__':
    a = Train('./okpic','./ngpic')
    a.pca_train()
    train_x, test_x, train_y, test_y = a.data_parepare()
    a.svm_c(train_x, test_x, train_y, test_y)
