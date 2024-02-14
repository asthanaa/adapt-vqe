import sys
import scipy
#import vqe_methods 
#import operator_pools
#import pyscf_helper 

import numpy as np
import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo,  cc,ci
from pyscf.cc import ccsd,rccsd

#import openfermion 
#from openfermion import *
#from tVQE import *

#import qeom
from scipy.sparse.linalg import eigs
def test():
    for h in range(11):
        st=1.5+0.1*h
        geometry = [('H', (0,  0, 0)),
            ('H', (0, 0, 1.5)),
            ('H', (0, st, 0)),
            ('H', (0, st,  1.5))]
        charge = 0
        spin = 0
        basis = 'sto3g'
        mol = gto.M(
        atom = geometry,  # in Angstrom
        basis = 'sto3g',
        )
        #mf = scf.HF(mol).run()
        mf = scf.RHF(mol).run()
        myci =fci.FCI(mf)
        e_ee, c_ee = myci.kernel(nroots=10)
        print('FCI total energy = ', e_ee)
        original_stdout = sys.stdout 
        with open('states', 'a') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print(st,',',repr(np.sort(e_ee)))
            sys.stdout = original_stdout
        with open('ee', 'a') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print(st,repr(np.sort(e_ee)-e_ee[0]))
            sys.stdout = original_stdout
if __name__== "__main__":
    test()
