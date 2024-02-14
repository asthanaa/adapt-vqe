#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run FCI
'''

import pyscf

mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 0.7',  # in Angstrom
    #atom = 'O 0 0 0; H 0.95700111 0.00000  0.00000000; H -0.23961394 0.00000  0.92651836',
    basis = 'sto3g',
    symmetry = True,
)
myhf = mol.RHF().run()

#
# create an FCI solver based on the SCF object
#


cisolver = pyscf.fci.FCI(myhf)
print('E(FCI) = %.12f' % cisolver.kernel()[0])
cisolver = pyscf.fci.FCI(myhf)

mycc = pyscf.cc.CCSD(myhf).run()
print('CCSD total energy', mycc.e_tot)
et = mycc.ccsd_t()

print('CCSD(T) total energy', mycc.e_tot + et)
e_ee, c_ee = mycc.eeccsd(nroots=15)
print('EOM-CCSD(T) total energy', e_ee+mycc.e_tot + et)



norb = myhf.mo_coeff.shape[1]
nelec = mol.nelec


fs = pyscf.fci.addons.fix_spin_(pyscf.fci.FCI(mol, myhf.mo_coeff), 0.5)
fs.nroots = 4
e, c = fs.kernel(verbose=5)
for i, x in enumerate(c):
    print('state %d, E = %.12f  2S+1 = %.7f' %
          (i, e[i], pyscf.fci.spin_op.spin_square0(x, norb, nelec)[1]))
fs = pyscf.fci.addons.fix_spin_(pyscf.fci.FCI(mol, myhf.mo_coeff), ss=2)
fs.nroots = 15
nelec=(2,0)
e, c = fs.kernel(nelec=nelec)
print('nelec',nelec)

for i, x in enumerate(c):
    print('state %d, E = %.12f  2S+1 = %.7f' %
          (i, e[i], pyscf.fci.spin_op.spin_square0(x, norb, nelec)[1]))
