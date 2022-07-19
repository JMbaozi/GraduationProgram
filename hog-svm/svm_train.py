# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 11:47:53 2018
@author: kuangyongjian
svm_train.py
"""
import numpy as np
import cv2 as cv
 
#svm参数配置
def svm_config():
    svm = cv.ml.SVM_create()
    svm.setCoef0(0)
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)
    svm.setC(0.01)
    svm.setType(cv.ml.SVM_EPS_SVR)
 
    return svm
 
#svm训练
def svm_train(svm,features,labels):
    svm.train(np.array(features),cv.ml.ROW_SAMPLE,np.array(labels))
    
#svm参数保存
def svm_save(svm,name):
    svm.save(name)
        
#svm加载参数 
def svm_load(name):
    svm = cv.ml.SVM_load(name)
    
    return svm