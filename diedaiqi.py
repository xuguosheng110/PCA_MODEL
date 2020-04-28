#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/14 11:40
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site : 
# @File : diedaiqi.py


import os
import glob
import cv2



class img_pre_deal():
    def __init__(self,img_lists):
        self.img_lists = img_lists
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i<len(self.img_lists):
            self.i += 1
            return self.img_lists[self.i-1]
        else:
            raise StopIteration


class Animal:
    def __init__(self, animal_list):
        self.animals_name = animal_list

    def __getitem__(self, index):
        return self.animals_name[index]

animals = Animal(["dog","cat","fish"])
for animal in animals:
    print(animal)

class iter_test(object):
    def __init__(self,img_list):
        self.img_list = img_list

    def __getitem__(self, item):
        print(item)
        return self.img_list[item]

a = [1,2,3,4,5]
print(a[1::1])
print(a[1:3])
'''
if __name__ == '__main__':
    img_root = r'F:\qizhi\g\IMG\tian\cut_ng'
    img_lists = glob.glob(os.path.join(img_root,'*.jpg'))
    for ll in img_lists:
        print('ll is : ',ll)
    print(img_lists)
    datas = img_pre_deal(img_lists)
    # mi_iter = iter(datas)
    for data in datas:
        print('data is: ',data)
    # lists = iter(img_lists)
    # for img in lists:
    #     print('aaa',img)
    getitem = iter_test(img_lists)
    for lll in getitem:
        print(lll)
'''
