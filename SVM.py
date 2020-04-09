#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# @Time : 2020/4/9 11:28
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site : 
# @File : SVM.py
# @Software: PyCharm
# ==================================
import os
import numpy as np
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import train_test_split,GridSearchCV


def load_data():
    #数据加载    (890, 1152) (404, 1152)
    matrix_ok = np.loadtxt(open("data\\ok.csv","rb"),delimiter=",",skiprows=0)
    matrix_ng = np.loadtxt(open('data\\ng.csv','rb'),delimiter=',',skiprows=0)
    label_ok = np.ones((matrix_ok.shape[0],1))
    label_ng = np.zeros((matrix_ng.shape[0],1))
    data_ok = np.hstack((matrix_ok,label_ok))
    data_ng = np.hstack((matrix_ng,label_ng))
    data_sets = np.vstack((data_ok,data_ng))#(1294, 1153)
    np.random.shuffle(data_sets)#打乱数据
    X = data_sets[:,0:-1]
    Y = data_sets[:,-1]
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.3)
    return train_x,test_x,train_y,test_y


def svm_c(train_x,test_x,train_y,test_y):
    svc = SVC(kernel='linear',class_weight='balanced')
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'kernel': ['linear'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    print('start train >>>>>>>>>>>>>>>>>>>>>>>')
    # 训练模型
    clf = grid.fit(train_x, train_y)
    # 计算测试集精度
    score = grid.score(test_x, test_y)
    print('精度为%s' % score)
    model_name = 'model//SVM' + '.txt'
    file = open(model_name, 'wb')
    pickle.dump(grid, file)


def test():
    file = open('model\\SVM.txt', 'rb')
    matrix_ng = np.loadtxt(open('data\\ng.csv','rb'),delimiter=',',skiprows=0)
    model = pickle.load(file)
    result = model.predict([matrix_ng[0,:]])
    print(result)


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = load_data()
    svm_c(train_x, test_x, train_y, test_y )
    # test()