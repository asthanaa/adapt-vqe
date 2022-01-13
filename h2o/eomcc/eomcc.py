#!/usr/bin/env python

import numpy as np
from pyscf import gto, scf, cc
mol = gto.Mole()
mol.verbose = 5
mol.unit = 'B'
mol.atom = 'O; H 1 1.80847; H 1 1.80847 2 104.5'
mol.basis = 'sto-3g'
mol.build()
print(gto.Mole.energy_nuc)
mf = scf.RHF(mol)
mf.verbose = 7
mf.scf()

mycc = cc.RCCSD(mf)
mycc.verbose = 7
mycc.frozen = 1
mycc.ccsd()

eS = 27.2114 * np.array(mycc.eomee_ccsd_singlet(nroots=5)[0])
print(eS)
eT = 27.2114 * np.array(mycc.eomee_ccsd_triplet(nroots=5)[0])
print('S-S: ', eS)
print('S-T: ', eT)
