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
def barH(parameters,opmats,H):
        """
        Prepare barH:
        exp{-A1}exp{-A2}exp{-A3}...exp{-An}Hexp{A1}exp{A2}exp{A3}...exp{An}
        """
        barH=H
        for k in range(0, len(parameters)):
            barH = scipy.sparse.linalg.expm_multiply((-1*parameters[k]*opmats[k]), barH)
            #barH = scipy.sparse.linalg.expm_multiply(barH,(parameters[k]*opmats[k]))
            barH = barH.dot(scipy.sparse.linalg.expm(parameters[k]*opmats[k]))
        return barH
def expvalue(bra,op,ket):
    """
    give expecttattion value of an operator wrt bra and ket wavefunction
    """
    return bra.dot(op.dot(ket))
def createops(no,nia,nib,nva,nvb,reference_ket):
    #single excitation
    ops=[]
    for i in range(nia):
        ia=2*i
        ib=2*i+1
        for j in range(nva):
            aa=nia+nib+2*j
            ab=nia+nib+2*j+1

            optmp1= FermionOperator(((aa,1),(ia,0)),1.0)
            optmp1 = normal_ordered(optmp1)
            optmp2= FermionOperator(((ab,1),(ib,0)),1.0)
            optmp2 = normal_ordered(optmp2)

            ops.append(optmp1)
            ops.append(optmp2)
            #termA =  FermionOperator(((aa,1),(ia,0)), 1/np.sqrt(2))
            #termA = normal_ordered(termA) 

            #normalize?
            for i in range(0,nia):
                ia = 2*i
                ib = 2*i+1

                for j in range(i,nia):
                    ja = 2*j
                    jb = 2*j+1

                    for a in range(0,nva):
                        aa = nia+nib + 2*a
                        ab = nia+nib + 2*a+1

                        for b in range(a,nva):
                            ba = nia+nib + 2*b
                            bb = nia+nib + 2*b+1

                            optmp =  FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), 1.0)
                            optmp = normal_ordered(optmp)
                            ops.append(optmp)
                            if nia>1:
                                optmp2 =  FermionOperator(((aa,1),(ba,1),(ia,0),(ja,0)), 1.0)
                                optmp2 = normal_ordered(optmp2)
                                ops.append(optmp2)
                            if nia>1:
                                optmp3 =  FermionOperator(((ab,1),(bb,1),(ib,0),(jb,0)), 1.0)
                                optmp3 = normal_ordered(optmp2)
                                ops.append(optmp3)

    spmat_ops = []
    for item in ops:    
        spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
    #print(spmat_ops[0].toarray())
    #print(reference_ket)
    #print(spmat_ops[0].dot(reference_ket))
    #print(spmat_ops)
    #exit()
    return spmat_ops
def comm3(a,b,c):

    mat=0.5*(a.dot(b.dot(c))-a.dot(c.dot(b))-b.dot(c.dot(a))+c.dot(b.dot(a)))
    mat+=0.5*(a.dot(b.dot(c))-b.dot(a.dot(c))-c.dot(a.dot(b))+c.dot(b.dot(a)))
    return mat
