#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/28 12:52
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site : 
# @File : Test_.py
# @Software: PyCharm
import pickle
import cv2
import numpy as np

class Model(object):

    def __init__(self):
        self.spilt_num =8
        self.pca_list,self.SVM = self.load_model()

    def load_model(self):
        model_list =[]
        for i in range(self.spilt_num):
            file = open('model\\pca_' + str(i) + '.pickle', 'rb')
            model = pickle.load(file)
            model_list.append(model)
        SVM = pickle.load(open('model\\SVM.pickle', 'rb'))
        return model_list,SVM

    def predict(self,img):
        small_pics = np.split(img, 8, axis=1)
        img_list = []
        for i, small_pic in enumerate(small_pics):
            data = np.reshape(small_pic, -1)  # (49152,)
            result = self.pca_list[i].transform([data])  # (1, 200)
            back = self.pca_list[i].inverse_transform(result)  # (1, 49152)
            back = np.asarray(back, dtype=np.uint8)
            back_img = np.reshape(back, (img.shape[0], -1))
            img_list.append(back_img)
        final_img = np.hstack((img_list[0], img_list[1], img_list[2],
                    img_list[3], img_list[4], img_list[5], img_list[6], img_list[7]))
        diff_img = cv2.absdiff(img,final_img)
        diff_img = cv2.erode(diff_img, (3, 3), iterations=5)
        blur = cv2.GaussianBlur(diff_img, (5, 5), 1.5)
        mean = np.mean(blur, axis=0)  # 纵向取平均
        result = self.SVM.predict([mean])
        print('result is ok!!') if result[0]==1 else print('result is ng!!')

        return 'OK' if result[0]==1 else 'NG'

if __name__ == '__main__':
    model = Model()
    img = cv2.imread('./ngpic/0diff.jpg',0)
    result = model.predict(img)
    print(result)