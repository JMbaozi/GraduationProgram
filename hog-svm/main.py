# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:47:20 2018
@author: kuangyongjian
test.py
"""
import numpy as np
import cv2 as cv
from nms import py_cpu_nms 
from imutils.object_detection import non_max_suppression


hog = cv.HOGDescriptor()
# hog.load('myHogDector.bin')
hog.load('myHogDector1.bin')

video = cv.VideoCapture('video/test.avi')
# video = cv.VideoCapture('video/PETS2009.avi')

success, frame = video.read()
while success:
    img = frame
    img = cv.resize(img,(960,540))
    cv.imshow('src',img)
    
    rects,scores = hog.detectMultiScale(img,winStride = (8,8),padding = (0,0),scale = 1.03)# https://www.cnblogs.com/klitech/p/5747895.html

    print(rects,scores)

    # sc = []
    # for score in scores:
    #     sc.append(score)
    # # sc = [score for score in scores]
    # sc = np.array(sc)
    # print(sc)
    #转换下输出格式(x,y,w,h) -> (x1,y1,x2,y2)
    for i in range(len(rects)):
        r = rects[i]
        rects[i][2] = r[0] + r[2]
        rects[i][3] = r[1] + r[3]

    
    pick = []
    #非极大值抑制
    print('rects_len',len(rects))
    pick = non_max_suppression(rects, probs = scores, overlapThresh = 0.3)
    # pick = py_cpu_nms(dets=rects,scores=scores,thresh=0.3)
    print('pick_len = ',len(pick))

    #画出矩形框
    for (x,y,xx,yy) in pick:
        cv.rectangle(img, (x, y), (xx, yy), (0, 0, 255), 2)    

    cv.imshow('result', img)  
    cv.waitKey(1)
    success,frame = video.read()

 
