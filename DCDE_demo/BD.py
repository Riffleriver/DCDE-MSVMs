# -*- coding: utf-8 -*-
"""
@author: Wenxuan Xu 
email: rifflexiansen@qq.com
"""
from __future__ import print_function
import numpy as np
from sklearn.metrics import roc_auc_score

#============================================================================================
# BD model
PE_pred_testlabel = np.load('PE_pred_testlabel.npy')
PE_proba_testlabel = np.load('PE_proba_testlabel.npy')

PI_pred_testlabel = np.load('PI_pred_testlabel.npy')
PI_proba_testlabel = np.load('PI_proba_testlabel.npy')

PU_pred_testlabel = np.load('PU_pred_testlabel.npy')
PU_proba_testlabel = np.load('PU_proba_testlabel.npy')

y_train_new = np.load('y_train_new1.npy')  
y_test_new = np.load('y_test_new1.npy') 
y_train = np.load('y_train1.npy')  
y_test = np.load('y_test1.npy') 

TT = 0.5 # threshold in manuscript 3.2

#============================================================================================
sum = PE_pred_testlabel + PI_pred_testlabel + PU_pred_testlabel
vote = np.zeros((sum.shape),dtype=int)
vote[np.where(sum>1)] = 1
prob = np.zeros((PU_proba_testlabel.shape),dtype=float)
p = np.column_stack((PE_proba_testlabel[:,1],PI_proba_testlabel[:,1],PU_proba_testlabel[:,1])) 
p_max = p.max(axis = 1)
p_mean = p.mean(axis = 1)
p_min = p.min(axis = 1)

prob[np.where(sum==3),1] = p_max[np.where(sum==3)]
prob[np.where(sum==2),1] = p_mean[np.where(sum==2)]
prob[np.where(sum==1),1] = p_min[np.where(sum==1)]
prob[np.where(sum==0),1] = p_min[np.where(sum==0)]
prob[:,0] = 1 - prob[:,1]

auc = roc_auc_score(y_test,prob)
print(auc)

pre = np.zeros((y_test_new.shape),dtype = int)
pre[np.where(prob[:,1]>TT)] = 1

#============================================================================================
#show performance
ins1 = np.where(y_test_new==1)
ins2 = np.where(y_test_new==0)
Sn = float(np.sum(pre[ins1]==1)) / float((np.sum(pre[ins1]==1)+np.sum(pre[ins2]==1)))
Sp = float(np.sum(pre[ins2]==0)) / float((np.sum(pre[ins2]==0)+np.sum(pre[ins1]==0)))
t = float(np.sum(pre[ins2]==0)) / float((np.sum(pre[ins2]==0)+np.sum(pre[ins2]==1)))
SP = float(np.sum(pre[ins1]==1)) / float((np.sum(pre[ins1]==1)+np.sum(pre[ins1]==0)))
ACP = (Sn + Sp + SP + t)/4
print(Sn,Sp,ACP)

np.save('Pre.npy',pre)
np.save('Prob.npy',prob)
