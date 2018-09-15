# -*- coding: utf-8 -*-
"""
@author: Wenxuan Xu 
email: rifflexiansen@qq.com
"""

from __future__ import print_function
import numpy as np
from sklearn.svm import SVC

def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    
    (L,D) = traindata.shape
    b = np.zeros(L,dtype=np.float64)
    x = traindata.mean(axis=0)
    for i in range(L):
        a = np.linalg.norm(traindata[i,:]-x)
        b[i] = a
    mx = np.median(b)
    par = 1/np.square(mx)
    
    svcClf = SVC(C=5.0,kernel="rbf",tol=1e-4,gamma=par,probability=True)
#    svcClf = SVC(C=5.0,kernel="rbf",tol=1e-4,gamma='auto',probability=True)
    svcClf.fit(traindata,trainlabel)

    pred_testlabel = svcClf.predict(testdata)
    proba_testlabel = svcClf.predict_proba(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-svm Accuracy:",accuracy)
    return pred_testlabel, proba_testlabel

#============================================================================================
FC_train_feature = np.load('FC_train_feature.npy') 
FC_test_feature = np.load('FC_test_feature.npy')  
y_train_new = np.load('y_train_new.npy')  
y_test_new = np.load('y_test_new.npy') 

#============================================================================================
#SVM
pred_testlabel, proba_testlabel = svc(FC_train_feature,y_train_new,FC_test_feature,y_test_new)
np.save('pred_testlabel.npy',pred_testlabel)
np.save('proba_testlabel.npy',proba_testlabel)
#============================================================================================
#evaluate
pre = pred_testlabel
ins1 = np.where(y_test_new==1)
ins2 = np.where(y_test_new==0)
Sn = float(np.sum(pre[ins1]==1)) / float((np.sum(pre[ins1]==1)+np.sum(pre[ins2]==1)))
Sp = float(np.sum(pre[ins2]==0)) / float((np.sum(pre[ins2]==0)+np.sum(pre[ins1]==0)))
t = float(np.sum(pre[ins2]==0)) / float((np.sum(pre[ins2]==0)+np.sum(pre[ins2]==1)))
SP = float(np.sum(pre[ins1]==1)) / float((np.sum(pre[ins1]==1)+np.sum(pre[ins1]==0)))
ACP = (Sn + Sp + SP + t)/4
print(Sn,Sp,ACP)