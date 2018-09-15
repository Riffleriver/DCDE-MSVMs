# -*- coding: utf-8 -*-
"""
@author: Wenxuan Xu 
email: rifflexiansen@qq.com
"""
# informative kmers settlement
from __future__ import print_function
import numpy as np
from kpal.klib import Profile
from Bio import SeqIO
from sklearn.preprocessing import normalize

def Kmer_PWM(handle,kmer,length):
    PWM_k = np.zeros((length,251-kmer,4**kmer),dtype=np.float64)
    for i, seq_record in enumerate(SeqIO.parse(handle, "fasta")):
        print(i)
        seq_array = Profile.from_sequences([str(seq_record.seq)],kmer)
        for j in range(251-kmer):
            t = seq_array.dna_to_binary(str(seq_record.seq[j:j+kmer]))
            PWM_k[i,j,t] = seq_array.counts[t]
    return PWM_k

def Kmer_fren(handle,kmer):
    p = Profile.from_fasta_by_record(handle, kmer)
    t = p.next()
    tt = t.counts.astype(float)/t.total
    for i in p :
        t = i.counts.astype(float)/i.total
        tt = np.row_stack((tt,t))
    return tt

def IFkmer_SD(P, Q, SD, T):
    # SD
    tdd = np.zeros((P.shape),dtype=np.float64)
    # KL ------------------------------------------
    if SD == 'KLD':
        for i in range(len(P)): 
            print(i)
            a = P[i,:]
            div = a * np.log(a / Q)
            div[np.isnan(div)]=0
            div[np.isinf(div)]=0
            div_m = div.mean(axis=0)
            tdd[i,:]=div_m
    # JD ------------------------------------------
    if SD == 'JD':
        for i in range(len(P)): 
            print(i)
            a = P[i,:]
            div1 = a * np.log(a / Q)
            div1[np.isnan(div1)]=0
            div1[np.isinf(div1)]=0 #P->Q
            div2 = Q * np.log(Q / a)
            div2[np.isnan(div2)]=0
            div2[np.isinf(div2)]=0 #Q->P
            div = (div1+div2)/2    
            div_m = div.mean(axis=0)
            tdd[i,:] = div_m   
    #    
    # JSD ------------------------------------------
    if SD == 'JSD':
        for i in range(len(P)): 
            print(i)
            a = P[i,:]
            o = (a+Q)/2
            div1 = a * np.log(a / o)
            div1[np.isnan(div1)]=0
            div1[np.isinf(div1)]=0 #P->Q
            div2 = Q * np.log(Q / 0)
            div2[np.isnan(div2)]=0
            div2[np.isinf(div2)]=0 #Q->P
            div = (div1+div2)/2    
            div_m = div.mean(axis=0)
            tdd[i,:] = div_m 
    #------------------------------------------------
    tdd_m = tdd.mean(axis=0)  
    tdd_mm = -np.sort(-tdd_m)#value
    d=np.argsort(tdd_m)#index
    
    for i in range(len(tdd_mm)): 
        R = np.sum(tdd_mm[0:i])/tdd_mm.sum()
        if  R > T:
            break
    index = d[0:i] 
    return index

def IFkmer_trn_tst(H_trn_p, H_trn_n, H_tst_p, H_tst_n,num_trn_p, num_trn_n,num_tst_p,num_tst_n, kmer, SD, T):
    trn_p = Kmer_fren(open(H_trn_p,'r'), kmer)   
    trn_n = Kmer_fren(open(H_trn_n,'r'), kmer)
    tst_p = Kmer_fren(open(H_tst_p,'r'), kmer)
    tst_n = Kmer_fren(open(H_tst_n,'r'), kmer)
        
    np.save('trn_p.npy',trn_p) 
    np.save('trn_n.npy',trn_n)  
    np.save('tst_p.npy',tst_p)  
    np.save('tst_n.npy',tst_n) 
    
    #==============================================================================
    PWM_trn_p = Kmer_PWM(open(H_trn_p,'r'),kmer,num_trn_p)
    PWM_trn_n = Kmer_PWM(open(H_trn_n,'r'),kmer,num_trn_n)
    PWM_tst_p = Kmer_PWM(open(H_tst_p,'r'),kmer,num_tst_p)
    PWM_tst_n = Kmer_PWM(open(H_tst_n,'r'),kmer,num_tst_n)
    
    np.save('PWM_trn_p.npy',PWM_trn_p) 
    np.save('PWM_trn_n.npy',PWM_trn_n)  
    np.save('PWM_tst_p.npy',PWM_tst_p)  
    np.save('PWM_tst_n.npy',PWM_tst_n)
    
    #==============================================================================
    index = IFkmer_SD(trn_p, trn_n, SD, T)
    np.save('index.npy',index)  
    
    #============================================================================================
    X_trn = np.row_stack((PWM_trn_p,PWM_trn_n)) 
    X_tst = np.row_stack((PWM_tst_p,PWM_tst_n))
    Y_trn = np.zeros(len(X_trn),dtype=np.int64)
    Y_trn[0:num_trn_p]=1
    Y_tst = np.zeros(len(X_tst),dtype=np.int64)
    Y_tst[0:num_tst_p]=1
    
    X_trn = X_trn[:,:,index]
    X_tst = X_tst[:,:,index]
    
#    for i in range(len(X_trn)):
#        X_trn[i,:,:] = normalize(X_trn[i,:,:], norm='l2')     
#    for i in range(len(X_tst)):
#        X_tst[i,:,:] = normalize(X_tst[i,:,:], norm='l2') 
    
    return (X_trn, X_tst, Y_trn, Y_tst)

#==============================================================================
# Main
kmer = 3 # (1,2,3,4,5,6) recommend
num_trn_p = 2000
num_trn_n = 2000
num_tst_p = 100
num_tst_n = 100
SD = 'KLD' # 'KLD' 'JD' 'JSD' 
T = 0.98 # threshold in manuscript 2.1.B
H_trn_p = 'training_2_positive.fasta'
H_trn_n = 'training_2_negative.fasta'
H_tst_p = 'test_2_positive.fasta'
H_tst_n = 'test_2_negative.fasta'

(X_trn, X_tst, Y_trn, Y_tst) = IFkmer_trn_tst(H_trn_p, H_trn_n, H_tst_p, H_tst_n,num_trn_p, num_trn_n,num_tst_p,num_tst_n, kmer, SD, T)

np.save('x_train.npy',X_trn) 
np.save('x_test.npy',X_tst)  
np.save('y_train.npy',Y_trn)  
np.save('y_test.npy',Y_tst) 

