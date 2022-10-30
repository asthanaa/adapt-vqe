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


def trial_e(k,singles,curr_state,h):
    #form effective Hamiltonain using some initian (0s?) Ps
    for i in len(singles):
        expf=k[i]*singles[i]
    U=numpy.expm(expf)
    #form cost function - energy wrt effective Hamiltonian with some Ps
    Hnew= U.conj().transpose().dot(h.dot(U))
    return curr_state.transpose().conj().dot(Hnew.dot(curr_state))
    

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
    print('singles',ops)
    spmat_ops = []
    for item in ops:    
        spmat_ops.append(transforms.get_sparse_operator(item, n_qubits = nia+nib+nva+nvb))
    return spmat_ops
def orbopt(no,nia,nib,nva,nvb,curr_state,h):#one cycle singles
    k=numpy.zeros(100)
    #form single operator pool
    singles=ops_singles(no,nia,nib,nva,nvb)

    #Enew=trial_e(k,singles,curr_state,Hnew)

    #compute gradients of the singles
    #for oi in range(singles):
    #    gi = singles.compute_gradient_i(oi, curr_state, sig)
    #optimize Ps to find minima in energy- this will form your Hnew and first P vector.
    opt_result = scipy.optimize.minimize(trial_e, k, args=(singles, curr_state,h)
                    options = min_options, method = 'BFGS', callback=trial.callback) 
    #return the Hnew


def orboptDIIS(state,h):#DIIS orb opt sent to adapt

