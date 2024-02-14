import scipy
import openfermion
import openfermionpsi4
import os
import numpy as np
import copy
import random
import sys

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo,  cc
from pyscf.cc import ccsd

import operator_pools
import vqe_methods
from tVQE import *
from orbital_opt import *
from openfermion import *

import pyscf_helper

def prepare_u(parameters,singles):
    u=scipy.sparse.linalg.expm(parameters[-1]*singles[-1])
    for k in reversed(range(0, len(parameters)-1)):
        u = scipy.sparse.linalg.expm_multiply((parameters[k]*singles[k]),u)
    return u
def trial_e(k,singles,curr_state,h):
    u=prepare_u(k,singles)
    #print("after prepare u", u)
    #form cost function - energy wrt effective Hamiltonian with some Ps
    Hnew= u.conj().transpose().dot(h.dot(u)) #flag
    Enew=curr_state.transpose().conj().dot(Hnew.dot(curr_state))
    Enew=Enew.todense()
    return Enew.real[0,0]
def ops_singles(no,nia,nib,nva,nvb):
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
            optmp -= hermitian_conjugated(optmp)
            ops.append(optmp)
            optmp= FermionOperator(((ab,1),(ib,0)),norm)
            #optmp = normal_ordered(optmp)
            optmp -= hermitian_conjugated(optmp)
            ops.append(optmp)

    spmat_ops = []
    for item in ops:    
        spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
    return spmat_ops

def ops_singlesGSD_unitary(no,nia,nib,nva,nvb):
        print(" Form singlet SD operators")
        ops = []
        n_orb = nia+nva
        for p in range(0,n_orb):
            pa = 2*p
            pb = 2*p+1
            for q in range(p,n_orb):
                qa = 2*q
                qb = 2*q+1
                termA =  FermionOperator(((pa,1),(qa,0)))
                termA += FermionOperator(((pb,1),(qb,0)))
                termA -= hermitian_conjugated(termA)
                termA = normal_ordered(termA)
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    ops.append(termA)
        spmat_ops = []
        for item in ops:    
            spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
        return spmat_ops

def ops_singles_unitary(no,nia,nib,nva,nvb):
        print(" Form singlet SD operators")
        ops = []
        n_occ = nia
        n_vir = nva
        for i in range(0,n_occ):
            ia = 2*i
            ib = 2*i+1
            for a in range(0,n_vir):
                aa = 2*n_occ + 2*a
                ab = 2*n_occ + 2*a+1
                termA =  FermionOperator(((aa,1),(ia,0)), 1/np.sqrt(2))
                termA += FermionOperator(((ab,1),(ib,0)), 1/np.sqrt(2))
                termA -= hermitian_conjugated(termA)
                termA = normal_ordered(termA)
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    ops.append(termA)
        spmat_ops = []
        for item in ops:    
            spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
        return spmat_ops
def orbopt(no,nia,nib,nva,nvb,curr_state,h,min_options,singles,k):#one cycle singles
    #min_options = {'gtol': theta_thresh, 'disp':False}-change in case different thresh
    #form single operator pool
    #singles=ops_singles(no,nia,nib,nva,nvb)
    #parameters
    #k=[]
    #np.random.seed(42)
    #for i in range(len(singles)):
    #    k.insert(0, np.random.random()/10)  #how to initialize the parameters?
    #k=np.random.uniform(low=0.0, high=0.1, size=(len(singles)))
    #k=np.zeros(len(singles))
    #print("singles", singles)
    #print("parameters", k)
    # test
    #optimize Ps to find minima in energy- this will form your Hnew and first P vector.
    print("orbital optimization running")
    opt_result=scipy.optimize.minimize(trial_e, k, args=(singles, curr_state,h),options = min_options, method = 'BFGS', callback=None) 
    print(opt_result)
    k = list(opt_result['x'])
    print("optimized e",trial_e(k,singles,curr_state,h))
    #return the Hnew
    u=prepare_u(k,singles)
    Hnew= u.conj().transpose().dot(h.dot(u))
    return Hnew,k 
#def orboptDIIS(state,h):#DIIS orb opt sent to adapt

