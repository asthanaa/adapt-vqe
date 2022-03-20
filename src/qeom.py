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
    return bra.dot(op.dot(ket)).todense()

def createops(no,nia,nib,nva,nvb,reference_ket):
    #single excitation
    ops=[]
    singlet=1
    triplet=1
    for i in range(nia):
        ia=2*i
        ib=2*i+1
        for j in range(nva):
            aa=nia+nib+2*j
            ab=nia+nib+2*j+1
            if singlet:
                optmp1= FermionOperator(((aa,1),(ia,0)),1.0/np.sqrt(2))
                optmp1+= FermionOperator(((ab,1),(ib,0)),1.0/np.sqrt(2))
                optmp1 = normal_ordered(optmp1)
                ops.append(optmp1)
                optmp3= FermionOperator(((aa,1),(ib,0)),1.0)
                optmp3 = normal_ordered(optmp3)
                optmp4= FermionOperator(((ab,1),(ia,0)),1.0)
                optmp4 = normal_ordered(optmp4)
                ops.append(optmp3)
                ops.append(optmp4)
            if triplet:
                optmp2= FermionOperator(((aa,1),(ia,0)),1.0/np.sqrt(2))
                optmp2-= FermionOperator(((ab,1),(ib,0)),1.0/np.sqrt(2))
                optmp2 = normal_ordered(optmp2)
                ops.append(optmp2)

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

                    if (i==j):
                                if (a==b):
                                    if singlet:
                                        optmp =  FermionOperator(((aa,1),(bb,1),(jb,0),(ia,0)), 1.0)
                                        optmp = normal_ordered(optmp)
                                        ops.append(optmp)
                                else:
                                    #correct tthe problem with spin adaptation here
                                    if singlet:

                                        optmp =  FermionOperator(((aa,1),(bb,1),(jb,0),(ia,0)), 1.0/np.sqrt(2))
                                        optmp +=  FermionOperator(((ba,1),(ab,1),(jb,0),(ia,0)), 1.0/np.sqrt(2))
                                        optmp = normal_ordered(optmp)
                                        ops.append(optmp)
                                        optmp =  FermionOperator(((aa,1),(ba,1),(jb,0),(ia,0)), 1.0)
                                        optmp = normal_ordered(optmp)
                                        ops.append(optmp)
                                        optmp =  FermionOperator(((ab,1),(bb,1),(jb,0),(ia,0)), 1.0)
                                        optmp = normal_ordered(optmp)
                                        ops.append(optmp)
                                    if triplet:
                                        optmp =  FermionOperator(((aa,1),(bb,1),(jb,0),(ia,0)), 1.0/np.sqrt(2))
                                        optmp -=  FermionOperator(((ba,1),(ab,1),(jb,0),(ia,0)), 1.0/np.sqrt(2))
                                        optmp = normal_ordered(optmp)
                                        ops.append(optmp)




                                    '''
                                    #optmp =  FermionOperator(((aa,1),(bb,1),(ja,0),(ib,0)), 1.0)
                                    #optmp = normal_ordered(optmp)
                            #ops.append(optmp)
                             if nia>1:
                                optmp2 =  FermionOperator(((aa,1),(ba,1),(ia,0),(ja,0)), 1.0)
                                optmp2 = normal_ordered(optmp2)
                                ops.append(optmp2)
                            if nia>1:
                                optmp3 =  FermionOperator(((ab,1),(bb,1),(ib,0),(jb,0)), 1.0)
                                optmp3 = normal_ordered(optmp2)
                                ops.append(optmp3)

                                    '''

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
    mat1=0.5*(a.dot(b.dot(c))-a.dot(c.dot(b))-b.dot(c.dot(a))+c.dot(b.dot(a)))
    mat2=0.5*(a.dot(b.dot(c))-b.dot(a.dot(c))-c.dot(a.dot(b))+c.dot(b.dot(a)))
    return mat1+mat2
def comm2(a,b):
    mat=a.dot(b)-b.dot(a)
    return mat







def createops_eefull(no,nia,nib,nva,nvb,reference_ket):
    #single excitation
    ops=[]
    singlet=1
    triplet=1
    norm=1
    for i in range(nia):
        ia=2*i
        ib=2*i+1
        for j in range(nva):
            aa=nia+nib+2*j
            ab=nia+nib+2*j+1
            optmp= FermionOperator(((aa,1),(ia,0)),norm)
            optmp = normal_ordered(optmp)
            ops.append(optmp)
            optmp= FermionOperator(((ab,1),(ib,0)),norm)
            optmp = normal_ordered(optmp)
            ops.append(optmp)
            #spin forbidden excitations not needed
            #optmp= FermionOperator(((aa,1),(ib,0)),norm)
            #optmp = normal_ordered(optmp)
            #ops.append(optmp)
            #optmp= FermionOperator(((ab,1),(ia,0)),norm)
            #optmp = normal_ordered(optmp)
            #ops.append(optmp)

            #spin adapted ones?
            #optmp1= FermionOperator(((aa,1),(ia,0)),1.0/np.sqrt(2))
            #optmp1+= FermionOperator(((ab,1),(ib,0)),1.0/np.sqrt(2))
            #optmp1 = normal_ordered(optmp1)
            #ops.append(optmp1)
            #optmp1= FermionOperator(((aa,1),(ia,0)),1.0/np.sqrt(2))
            #optmp1-= FermionOperator(((ab,1),(ib,0)),1.0/np.sqrt(2))
            #optmp1 = normal_ordered(optmp1)
            #ops.append(optmp1)
    for i in range(0,nia):
        ia = 2*i
        ib = 2*i+1

        for j in range(i,nib):
            ja = 2*j
            jb = 2*j+1

            for a in range(0,nva):
                aa = nia+nib + 2*a
                ab = nia+nib + 2*a+1

                for b in range(a,nvb):
                    ba = nia+nib + 2*b
                    bb = nia+nib + 2*b+1

                    if (i==j) and (a==b):
                        optmp =  FermionOperator(((aa,1),(bb,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)
                    elif (i==j) or (a==b):
                        optmp =  FermionOperator(((aa,1),(bb,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)
                        optmp =  FermionOperator(((ab,1),(ba,1),(ja,0),(ib,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)
                    else:

                        optmp =  FermionOperator(((aa,1),(ba,1),(ja,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)
                        optmp =  FermionOperator(((ab,1),(bb,1),(jb,0),(ib,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)

                        optmp =  FermionOperator(((aa,1),(bb,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)
                        optmp =  FermionOperator(((ab,1),(ba,1),(ja,0),(ib,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)

                        optmp =  FermionOperator(((ba,1),(ab,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)
                        optmp =  FermionOperator(((bb,1),(aa,1),(ja,0),(ib,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)

    spmat_ops = []
    for item in ops:    
        spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
    return spmat_ops

def createops_basic(no,nia,nib,nva,nvb,reference_ket):
    #single excitation
    ops=[]
    singlet=1
    triplet=1
    norm=1
    for i in range(nia):
        ia=2*i
        ib=2*i+1
        for j in range(nva):
            aa=nia+nib+2*j
            ab=nia+nib+2*j+1
            if singlet:

                #optmp1= FermionOperator(((aa,1),(ia,0)),1.0/np.sqrt(2))
                #optmp1+= FermionOperator(((ab,1),(ib,0)),1.0/np.sqrt(2))
                #optmp1 = normal_ordered(optmp1)
                #ops.append(optmp1)
                optmp3= FermionOperator(((aa,1),(ia,0)),norm)
                optmp3 = normal_ordered(optmp3)
                optmp4= FermionOperator(((ab,1),(ib,0)),norm)
                optmp4 = normal_ordered(optmp4)
                ops.append(optmp3)
                ops.append(optmp4)
            #if triplet:

            #    optmp2= FermionOperator(((aa,1),(ia,0)),1.0/np.sqrt(2))
            #    optmp2-= FermionOperator(((ab,1),(ib,0)),1.0/np.sqrt(2))
            #    optmp2 = normal_ordered(optmp2)
            #    ops.append(optmp2)

            #optmp2= FermionOperator(((ab,1),(ia,0)),1.0)
            #optmp2 = normal_ordered(optmp2)
            #ops.append(optmp2)
            #optmp2= FermionOperator(((aa,1),(ib,0)),1.0)
            #optmp2 = normal_ordered(optmp2)
            #ops.append(optmp2)


    for i in range(0,nia):
        ia = 2*i
        ib = 2*i+1

        for j in range(i,nia):
            ja = 2*j
            jb = 2*j+1

            for a in range(0,nva):
                aa = nia+nib + 2*a
                ab = nia+nib + 2*a+1

                for b in range(0,nva):
                    ba = nia+nib + 2*b
                    bb = nia+nib + 2*b+1

                    if (i==j):
                                if singlet:

                                    #optmp =  FermionOperator(((ab,1),(ba,1),(ja,0),(ib,0)), 1.0)
                                    #optmp = normal_ordered(optmp)
                                    #ops.append(optmp)
                                    optmp =  FermionOperator(((aa,1),(bb,1),(jb,0),(ia,0)), norm)
                                    optmp = normal_ordered(optmp)
                                    ops.append(optmp)

    spmat_ops = []
    for item in ops:    
        spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
    return spmat_ops
def createops_ip(no,nia,nib,nva,nvb,reference_ket):
    #single excitation
    ops=[]
    singlet=1
    triplet=1
    norm=1
    for i in range(nia):
        ia=2*i
        ib=2*i+1
        if singlet:
            optmp3= FermionOperator((ia,0),norm)
            optmp3 = normal_ordered(optmp3)
            optmp4= FermionOperator((ib,0),norm)
            optmp4 = normal_ordered(optmp4)
            ops.append(optmp3)
            ops.append(optmp4)
        #correct only for h2 sto3g

        for j in range(nia):
            ja=2*j
            jb=2*j+1
            optmp5= FermionOperator(((ia,0),(jb,0)),norm)
            optmp5 = normal_ordered(optmp5)
            ops.append(optmp5)
    for i in range(0,nia):
        ia = 2*i
        ib = 2*i+1

        for j in range(i,nia):
            ja = 2*j
            jb = 2*j+1

            for a in range(0,nva):
                aa = nia+nib + 2*a
                ab = nia+nib + 2*a+1

                if (i==j):
                    if singlet:
                        optmp =  FermionOperator(((ab,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)
                        optmp =  FermionOperator(((aa,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)
    spmat_ops = []
    for item in ops:    
        spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
    return spmat_ops
def createops_ipfull(no,nia,nib,nva,nvb,reference_ket):
    #single excitation
    ops=[]
    singlet=1
    triplet=1
    norm=1
    for i in range(nia):
        ia=2*i
        ib=2*i+1
        optmp= FermionOperator(((ia,0)),norm)
        optmp = normal_ordered(optmp)
        ops.append(optmp)
        optmp= FermionOperator(((ib,0)),norm)
        optmp = normal_ordered(optmp)
        ops.append(optmp)
        #for j in range(nia):
        #    ja=2*j
        #    jb=2*j+1
        #    optmp= FermionOperator(((ia,0),(jb,0)),norm)
        #    optmp = normal_ordered(optmp)
        #    ops.append(optmp)
    for i in range(0,nia):
        ia = 2*i
        ib = 2*i+1

        for j in range(i,nib):
            ja = 2*j
            jb = 2*j+1

            for a in range(0,nva):
                aa = nia+nib + 2*a
                ab = nia+nib + 2*a+1

                if (i==j):

                        optmp =  FermionOperator(((ab,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)
                        optmp =  FermionOperator(((aa,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)

                else:
                        optmp =  FermionOperator(((aa,1),(ja,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)
                        optmp =  FermionOperator(((ab,1),(jb,0),(ib,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)

                        optmp =  FermionOperator(((ab,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)
                        optmp =  FermionOperator(((ab,1),(ja,0),(ib,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)

                        optmp =  FermionOperator(((aa,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)
                        optmp =  FermionOperator(((aa,1),(ja,0),(ib,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)

    spmat_ops = []
    for item in ops:    
        spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
    return spmat_ops

def createops_eafull(no,nia,nib,nva,nvb,reference_ket):
    #single excitation
    ops=[]
    singlet=1
    triplet=1
    norm=1
    for j in range(nva):

        aa=nia+nib+2*j
        ab=nia+nib+2*j+1
        optmp3= FermionOperator((aa,1),norm)
        optmp3 = normal_ordered(optmp3)
        ops.append(optmp3)
        optmp4= FermionOperator((ab,1),norm)
        optmp4 = normal_ordered(optmp4)
        ops.append(optmp4)
        #2e ea (for H2 sto3g, add more for others)
        #for k in range(nva):
        #    ba=nia+nib+2*k
        #bb=nia+nib+2*k+1
        #optmp5 = FermionOperator(((aa,1),(bb,1)),norm)
        #optmp5 = normal_ordered(optmp5)
        #ops.append(optmp5)
            

    for i in range(0,nia):
        ia = 2*i
        ib = 2*i+1


        for a in range(0,nva):
            aa = nia+nib + 2*a
            ab = nia+nib + 2*a+1

            for b in range(a,nva):
                ba = nia+nib + 2*b
                bb = nia+nib + 2*b+1

                if (a==b):
                    optmp =  FermionOperator(((aa,1),(bb,1),(ib,0)), norm)
                    optmp = normal_ordered(optmp)
                    ops.append(optmp)
                    optmp =  FermionOperator(((aa,1),(bb,1),(ia,0)), norm)
                    optmp = normal_ordered(optmp)
                    ops.append(optmp)
                else:
                        optmp =  FermionOperator(((aa,1),(ba,1),(ia,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)
                        optmp =  FermionOperator(((ab,1),(bb,1),(ib,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)

                        optmp =  FermionOperator(((ab,1),(ba,1),(ia,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)
                        optmp =  FermionOperator(((ab,1),(ba,1),(ib,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)

                        optmp =  FermionOperator(((aa,1),(bb,1),(ia,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)
                        optmp =  FermionOperator(((aa,1),(bb,1),(ib,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)
                    

    spmat_ops = []
    for item in ops:    
        spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
    return spmat_ops

def createops_ea(no,nia,nib,nva,nvb,reference_ket):
    #single excitation
    ops=[]
    singlet=1
    triplet=1
    norm=1
    for j in range(nva):

        aa=nia+nib+2*j
        ab=nia+nib+2*j+1
        if singlet:

            optmp3= FermionOperator((aa,1),norm)
            optmp3 = normal_ordered(optmp3)
            optmp4= FermionOperator((ab,1),norm)
            optmp4 = normal_ordered(optmp4)
            ops.append(optmp3)
            ops.append(optmp4)
        #2e ea (for H2 sto3g, add more for others)
        for k in range(nva):
            ba=nia+nib+2*k
            bb=nia+nib+2*k+1
            optmp5 = FermionOperator(((aa,1),(bb,1)),norm)
            optmp5 = normal_ordered(optmp5)
            ops.append(optmp5)
            

    for i in range(0,nia):
        ia = 2*i
        ib = 2*i+1


        for a in range(0,nva):
            aa = nia+nib + 2*a
            ab = nia+nib + 2*a+1

            for b in range(0,nva):
                ba = nia+nib + 2*b
                bb = nia+nib + 2*b+1

                if (i==j):
                                if singlet:


                                    optmp =  FermionOperator(((aa,1),(bb,1),(ib,0)), norm)
                                    optmp = normal_ordered(optmp)
                                    ops.append(optmp)
                                    optmp =  FermionOperator(((aa,1),(bb,1),(ia,0)), norm)
                                    optmp = normal_ordered(optmp)
                                    ops.append(optmp)

    spmat_ops = []
    for item in ops:    
        spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
    return spmat_ops
def createops_eefullwithI(no,nia,nib,nva,nvb,reference_ket):
    #single excitation
    ops=[]
    singlet=1
    triplet=1
    norm=1
    optmp= FermionOperator('')
    optmp = normal_ordered(optmp)
    ops.append(optmp)
    for i in range(nia):
        ia=2*i
        ib=2*i+1
        for j in range(nva):
            aa=nia+nib+2*j
            ab=nia+nib+2*j+1

            optmp= FermionOperator(((aa,1),(ia,0)),norm)
            optmp = normal_ordered(optmp)
            ops.append(optmp)
            optmp= FermionOperator(((ab,1),(ib,0)),norm)
            optmp = normal_ordered(optmp)
            ops.append(optmp)
            #spin forbidden excitations not needed
            #optmp= FermionOperator(((aa,1),(ib,0)),norm)
            #optmp = normal_ordered(optmp)
            #ops.append(optmp)
            #optmp= FermionOperator(((ab,1),(ia,0)),norm)
            #optmp = normal_ordered(optmp)
            #ops.append(optmp)

            #spin adapted ones?
            #optmp1= FermionOperator(((aa,1),(ia,0)),1.0/np.sqrt(2))
            #optmp1+= FermionOperator(((ab,1),(ib,0)),1.0/np.sqrt(2))
            #optmp1 = normal_ordered(optmp1)
            #ops.append(optmp1)
            #optmp1= FermionOperator(((aa,1),(ia,0)),1.0/np.sqrt(2))
            #optmp1-= FermionOperator(((ab,1),(ib,0)),1.0/np.sqrt(2))
            #optmp1 = normal_ordered(optmp1)
            #ops.append(optmp1)
    for i in range(0,nia):
        ia = 2*i
        ib = 2*i+1

        for j in range(i,nib):
            ja = 2*j
            jb = 2*j+1

            for a in range(0,nva):
                aa = nia+nib + 2*a
                ab = nia+nib + 2*a+1

                for b in range(a,nvb):
                    ba = nia+nib + 2*b
                    bb = nia+nib + 2*b+1

                    if (i==j) and (a==b):
                        optmp =  FermionOperator(((aa,1),(bb,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)
                    elif (i==j) or (a==b):
                        optmp =  FermionOperator(((aa,1),(bb,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)
                        optmp =  FermionOperator(((ab,1),(ba,1),(ja,0),(ib,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)
                    else:

                        optmp =  FermionOperator(((aa,1),(ba,1),(ja,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)
                        optmp =  FermionOperator(((ab,1),(bb,1),(jb,0),(ib,0)), norm)
                        optmp = normal_ordered(optmp)
                        ops.append(optmp)

                        optmp =  FermionOperator(((aa,1),(bb,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)
                        optmp =  FermionOperator(((ab,1),(ba,1),(ja,0),(ib,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)

                        optmp =  FermionOperator(((ba,1),(ab,1),(jb,0),(ia,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)
                        optmp =  FermionOperator(((bb,1),(aa,1),(ja,0),(ib,0)), norm)
                        optmp = normal_ordered(optmp)                           
                        ops.append(optmp)

    spmat_ops = []
    for item in ops:    
        spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
    return spmat_ops

