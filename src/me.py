
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
    delta = 1e-8
    #
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
    if abs(npla.det(M)) < delta:
        # print('singlar encountered!')
        u,s,v = npla.svd(M)
        for i in range(0,len(s)):
            if abs(s[i]) < delta:
                if s[i] < 0:
                    s[i] += -delta
                else:
                    s[i] += delta
        M = np.dot(u*s,v)
    #
    # print(M,-Y)
    X = npla.solve(M,-Y)
    coeff = np.zeros(order+1,dtype=float)
    coeff[0] = 1.0
    coeff[1:] = X
    roots = np.roots(coeff)
    roots = np.sort(roots)
    # print(roots)
    return roots[0]#,roots[1],M,X
 
def Hn(Ham,i):
    tmp = np.copy(Ham)
    for j in range(1,i):
        tmp = tmp@Ham
    return tmp
