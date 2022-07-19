"""
Created on Tue Apr  3 10:16:07 2018
@author: kuangyongjian
get_data.py
"""
import cv2 as cv
import random
import glob
import os

#加载负样本
def get_neg_samples(foldername,savePath):
    count = 0
    imgs = []
    labels = []
    f = open('neg.txt')
    filenames = glob.iglob(os.path.join(foldername,'*'))
    for filename in filenames:
        # print('filename = ',filename)
        src = cv.imread(filename,1)
        
        if((src.cols >= 64) & (src.rows >= 128)):
            x = random.uniform(0,src.cols - 64)
            y = random.uniform(0,src.rows - 128)
            
            imgRoi = src(cv.Rect(x,y,64,128))
            imgs.append(imgRoi)
            saveName = savePath + 'neg' + str(count) + '.jpg'
            cv.imwrite(saveName,imgRoi)
            
            label = 'neg' + str(count) + '.jpg'
            labels.append(label)
            label = label + '\n'
            f.write(label)
            count += 1
    return imgs,labels
 
 
#读取负样本
def read_neg_samples(foldername):
    imgs = []
    labels = []
    neg_count = 0
    filenames = glob.iglob(os.path.join(foldername,'*'))
    for filename in filenames:
        # print('filename = ',filename)
        src = cv.imread(filename,1)
        imgs.append(src)
        labels.append(-1)
        neg_count += 1

    print ('neg_count = ',neg_count)
    return imgs,labels
        
        
 
#加载正样本
def get_pos_samples(foldername,savePath):
    count = 0
    imgs = []
    labels = []
    f = open('pos.txt')
    filenames = glob.iglob(os.path.join(foldername,'*'))
    for filename in filenames:
        # print('filename = ',filename)
        src = cv.imread(filename)
        imgRoi = src(cv.Rect(16,16,64,128))
        imgs.append(imgRoi)
        saveName = savePath + 'neg' + str(count) + '.jpg'
        cv.imwrite(saveName,imgRoi)
        
        label = 'neg' + str(count) + '.jpg'
        labels.append(label)
        f.write(label)
        count += 1
        
    return imgs,labels
 
 
#读取正样本
def read_pos_samples(foldername):
    imgs = []
    labels = []
    pos_count = 0
    filenames = glob.iglob(os.path.join(foldername,'*'))
    for filename in filenames:
        # print('filename = ',filename)
        src = cv.imread(filename)
        imgs.append(src)
        labels.append(1)
        pos_count += 1

    print ('pos_count = ',pos_count)
    return imgs,labels