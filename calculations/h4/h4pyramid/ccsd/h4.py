import sys
import scipy
#import vqe_methods 
#import operator_pools
#import pyscf_helper 

import numpy as np
import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo,  cc
from pyscf.cc import ccsd,rccsd

#import openfermion 
#from openfermion import *
#from tVQE import *

#import qeom
from scipy.sparse.linalg import eigs
def test():
    for h in range(20):
        height=1.0*h*0.1+0.8
        geometry = [('H', (0.458280907664828,  0.317506, -0.0000000000)),
            ('H', (-0.229140453832414, 0.317506, -0.3968829)),
            ('H', (-0.229140453832414, 0.317506, 0.3968829)),
            #('H', (0.0000000000, -0.952519,  0.0000000000))]
            ('H', (0.0000000000, height,  0.0000000000))]
        charge = 0
        spin = 0
        basis = 'sto3g'
        mol = gto.M(
            atom = geometry,  # in Angstrom
                basis = 'sto3g',
                    )
        #mf = scf.HF(mol).run()
        mf = scf.RHF(mol).x2c().run()
        mycc = cc.rccsd.RCCSD(mf).run() 
        #mycc1 = cc.rccsd(mycc) 
        #mycc.kernel()
        e_ee, c_ee = mycc.eeccsd(nroots=15)
        print('rCCSD total energy = ', mycc.e_tot)
        original_stdout = sys.stdout 
        with open('dataccsd', 'a') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print(height,',',mycc.e_tot,',', repr(np.sort(e_ee)+mycc.e_tot))
            sys.stdout = original_stdout
if __name__== "__main__":
    test()
