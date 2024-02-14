
import numpy as np
import numpy.linalg as npla
import scipy
import scipy.linalg
#import scipy.io
#import copy as cp
import scipy.sparse
import scipy.sparse.linalg
#import math
#import sys

import openfermion
from openfermion import *
def me(order,H,trial):
    delta = 1e-15
    flag=0
    thresholding = 0
    M = np.zeros((order,order),dtype=float)

    Y = np.zeros(order,dtype=float)
    for i in range(0,order):
        Y[i] = (np.conj(trial).T@Hn(H,(2*order-i-1))@trial).real
        for j in range(i,order):
            if 2*order-i-j-2 != 0:
                M[i,j] = M[j,i] = (np.conj(trial).T@Hn(H,(2*order-i-j-2))@trial).real
            else:
                M[i,j] = M[j,i] = (np.conj(trial).T@trial).real
    #
    print("det,delta,order\n", npla.det(M), delta,order) 
    u,s,v = npla.svd(M)
    if abs(npla.det(M)) < delta:
        print('singlarity encountered!')
        u,s,v = npla.svd(M)
        vp = np.copy(v)
        for i in range(len(s)-1,-1,-1):
            if abs(s[i]) < delta:
                if (thresholding==0):
                    if s[i] < 0:
                        #s[i] += -delta
                        s[i] += 0
                    else:
                        #s[i] += delta
                        s[i] += 0
                #adding new thresholding
                else:
                    vp = np.delete(vp,i,1)
                flag=1
        if flag==1 and thresholding==1:

            print("old M :\n ", M)
            print("old S :\n ", s)
            Mp = np.dot(np.dot(vp.T,M),vp)
            print("new M:\n ", Mp)
            #u,s,v =npla.svd(Mp)
            #print("new S : \n",s)
            Yp = np.dot(Y,vp)
            order_new = Yp.size
            X = npla.solve(Mp,-Yp)
        if flag ==1 and thresholding ==0:
            M = np.dot(u*s,v)
            X = npla.solve(M,-Y)
    if flag==0 or thresholding==0:
        X = npla.solve(M,-Y)
        order_new = Y.size
    coeff = np.zeros(order_new+1,dtype=float)
    coeff[0] = 1.0
    coeff[1:] = X
    roots = np.roots(coeff)
    roots = np.sort(roots)
    print(roots)
    #if flag==1:
    #    exit()
    return roots[0].real#,roots[1],M,X
 
def Hn(Ham,i):
    tmp = np.copy(Ham)
    for j in range(1,i):
        tmp = tmp@Ham
    return tmp
