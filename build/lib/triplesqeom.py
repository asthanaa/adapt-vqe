#from __future__ import print_function
import numpy as np
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



def createops_ipt(no,nia,nib,nva,nvb,reference_ket):
    ops=[]
    norm=1   
    for i in range(0,nia):
        ia = 2*i
        ib = 2*i+1
        for j in range(0,nib):
            ja = 2*j
            jb = 2*j+1
            for a in range(0,nva):
                aa = nia+nib + 2*a
                ab = nia+nib + 2*a+1
                for k in range(j,nib):
                    ka = 2*k
                    kb = 2*k+1
                    for b in range(a,nva):
                        ba = nia+nib + 2*b
                        bb = nia+nib + 2*b+1 
                        if (a==b) and (j==k):
                            if (k!=i):
                                optmp =  FermionOperator(((ab,1),(ba,1),(ka,0),(jb,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (j!=i):
                                optmp =  FermionOperator(((ab,1),(ba,1),(ka,0),(jb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                        elif (k==j):
                            if (k!=i):
                                optmp =  FermionOperator(((ab,1),(ba,1),(ka,0),(jb,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (j!=i):
                                optmp =  FermionOperator(((ab,1),(ba,1),(ka,0),(jb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (k!=i):
                                optmp =  FermionOperator(((bb,1),(aa,1),(ka,0),(jb,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (j!=i):
                                optmp =  FermionOperator(((bb,1),(aa,1),(ka,0),(jb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)

                        elif (a==b):
                            if (k!=i):
                                optmp =  FermionOperator(((ab,1),(ba,1),(ka,0),(jb,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (j!=i):
                                optmp =  FermionOperator(((ab,1),(ba,1),(ka,0),(jb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (j!=i):
                                optmp =  FermionOperator(((ab,1),(ba,1),(ja,0),(kb,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (k!=i):
                                optmp =  FermionOperator(((ab,1),(ba,1),(ja,0),(kb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                        else:

                            if (k!=i):
                                optmp =  FermionOperator(((ab,1),(ba,1),(ka,0),(jb,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (j!=i):
                                optmp =  FermionOperator(((ab,1),(ba,1),(ka,0),(jb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (i!=j):
                                optmp =  FermionOperator(((ab,1),(ba,1),(ja,0),(kb,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (k!=i):
                                optmp =  FermionOperator(((ab,1),(ba,1),(ja,0),(kb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (k!=i):
                                optmp =  FermionOperator(((bb,1),(aa,1),(ka,0),(jb,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (j!=i):
                                optmp =  FermionOperator(((bb,1),(aa,1),(ka,0),(jb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (i!=j):
                                optmp =  FermionOperator(((bb,1),(aa,1),(ja,0),(kb,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (k!=i):
                                optmp =  FermionOperator(((bb,1),(aa,1),(ja,0),(kb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            #same spin cases  (bb)
                            if (k!=j) and (a!=b):
                                optmp =  FermionOperator(((ab,1),(bb,1),(kb,0),(jb,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (j!=i) and (j!=k) and (k!=i) and (a!=b):
                                optmp =  FermionOperator(((ab,1),(bb,1),(kb,0),(jb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (k!=j) and (a!=b):
                                optmp =  FermionOperator(((ab,1),(bb,1),(jb,0),(kb,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (j!=i) and (j!=k) and (k!=i) and (a!=b):
                                optmp =  FermionOperator(((ab,1),(bb,1),(jb,0),(kb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            #spin aa
                            if (j!=i) and (k!=j) and (k!=i) and (a!=b):
                                optmp =  FermionOperator(((aa,1),(ba,1),(ka,0),(ja,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (k!=j) and (a!=b):
                                optmp =  FermionOperator(((aa,1),(ba,1),(ka,0),(ja,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (j!=i) and (k!=j) and (k!=i) and (a!=b):
                                optmp =  FermionOperator(((aa,1),(ba,1),(ja,0),(ka,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (k!=j) and (a!=b):
                                optmp =  FermionOperator(((aa,1),(ba,1),(ja,0),(ka,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
    spmat_ops = []
    for item in ops:
        spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
    return spmat_ops

def createops_eat(no,nia,nib,nva,nvb,reference_ket):
    ops=[]
    norm=1   
    for c in range(0,nva):
        ca = nia+nib+2*c
        cb = nia+nib+2*c+1
        for i in range(0,nia):
            ia = 2*i
            ib = 2*i+1
            for a in range(0,nva):
                aa = nia+nib + 2*a
                ab = nia+nib + 2*a+1
                for j in range(i,nib):
                    ja = 2*j
                    jb = 2*j+1
                    for b in range(a,nvb):
                        ba = nia+nib + 2*b
                        bb = nia+nib + 2*b+1
                        if (a==b) and (i==j):

                            if (c!=b):
                                optmp =  FermionOperator(((ca,1),(ab,1),(ba,1),(ja,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=a):
                                optmp =  FermionOperator(((cb,1),(ab,1),(ba,1),(ja,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                        elif (a==b):

                            if (c!=b):
                                optmp =  FermionOperator(((ca,1),(ab,1),(ba,1),(ja,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=a):
                                optmp =  FermionOperator(((cb,1),(ab,1),(ba,1),(ja,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=b):
                                optmp =  FermionOperator(((ca,1),(ab,1),(ba,1),(ia,0),(jb,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=a):
                                optmp =  FermionOperator(((cb,1),(ab,1),(ba,1),(ia,0),(jb,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)

                        elif (i==j):

                            if (c!=b):
                                optmp =  FermionOperator(((ca,1),(ab,1),(ba,1),(ja,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=a):
                                optmp =  FermionOperator(((cb,1),(ab,1),(ba,1),(ja,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=a):
                                optmp =  FermionOperator(((ca,1),(bb,1),(aa,1),(ja,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=b):
                                optmp =  FermionOperator(((cb,1),(bb,1),(aa,1),(ja,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                        else:
                            if (c!=b):
                                optmp =  FermionOperator(((ca,1),(ab,1),(ba,1),(ja,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=a):
                                optmp =  FermionOperator(((cb,1),(ab,1),(ba,1),(ja,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=a):
                                optmp =  FermionOperator(((ca,1),(bb,1),(aa,1),(ja,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=b):
                                optmp =  FermionOperator(((cb,1),(bb,1),(aa,1),(ja,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)

                            if (c!=b):
                                optmp =  FermionOperator(((ca,1),(ab,1),(ba,1),(ia,0),(jb,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=a):
                                optmp =  FermionOperator(((cb,1),(ab,1),(ba,1),(ia,0),(jb,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=a):
                                optmp =  FermionOperator(((ca,1),(bb,1),(aa,1),(ia,0),(jb,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=b):
                                optmp =  FermionOperator(((cb,1),(bb,1),(aa,1),(ia,0),(jb,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            #spin aa
                            if (c!=a) and (c!=b) and (a!=b) and (i!=j):
                                optmp =  FermionOperator(((ca,1),(aa,1),(ba,1),(ja,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (b!=a) and (i!=j):
                                optmp =  FermionOperator(((cb,1),(aa,1),(ba,1),(ja,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=a) and (a!=b) and (c!=b) and (i!=j):
                                optmp =  FermionOperator(((ca,1),(ba,1),(aa,1),(ja,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (b!=a) and (i!=j):
                                optmp =  FermionOperator(((cb,1),(ba,1),(aa,1),(ja,0),(ia,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            #spin bb
                            if (c!=a) and (c!=b) and (a!=b) and (i!=j):
                                optmp =  FermionOperator(((cb,1),(ab,1),(bb,1),(jb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (b!=a) and (i!=j):
                                optmp =  FermionOperator(((ca,1),(ab,1),(bb,1),(jb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (c!=a) and (a!=b) and (c!=b) and (i!=j):
                                optmp =  FermionOperator(((cb,1),(bb,1),(ab,1),(jb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
                            if (b!=a) and (i!=j):
                                optmp =  FermionOperator(((ca,1),(bb,1),(ab,1),(jb,0),(ib,0)), norm)
                                optmp = normal_ordered(optmp)
                                ops.append(optmp)
    spmat_ops = []
    for item in ops:
        spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
    return spmat_ops

