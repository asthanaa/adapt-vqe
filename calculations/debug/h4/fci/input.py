import scipy
import vqe_methods
import operator_pools
import pyscf_helper
import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo,  cc
from pyscf.cc import ccsd
import openfermion
from openfermion import *
from tVQE import *
r = 2.0
geometry = [('H',(0, 0, 0)),('H', (0, 0, 1.5)),('H', (0, r, 0)),('H', (0, r, 1.5))]
charge = 0
spin = 0
basis = 'sto-3g'
ref = 'rhf'
[n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis,n_frzn_occ=0,n_act=4)
sq_ham = pyscf_helper.SQ_Hamiltonian()
sq_ham.init(h, g, C, S)
print(" HF Energy: %21.15f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))
fermi_ham  = sq_ham.export_FermionOperator()
print(eigenspectrum(fermi_ham)[0]+E_nuc)
hamiltonian = openfermion.transforms.get_sparse_operator(fermi_ham)
s2 = vqe_methods.Make_S2(n_orb)
#print(s2)
#build reference configuration
occupied_list = []
for i in range(n_a):
    occupied_list.append(i*2)
for i in range(n_b):
    occupied_list.append(i*2+1)
print(" Build reference state with %4i alpha and %4i beta electrons" %(n_a,n_b), occupied_list)
reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(occupied_list, 2*n_orb)).transpose()
[e,v] = scipy.sparse.linalg.eigsh(hamiltonian.real,which='SA',v0=reference_ket.todense())
for ei in range(len(e)):
    S2 = v[:,ei].conj().T.dot(s2.dot(v[:,ei]))
    #print(" State %4i: %20.15f au  <S2>: %12.8f" %(ei,e[ei]+E_nuc,S2))
    print(" EE %4i: %20.15f au  <S2>: %12.8f" %(ei,e[ei]-e[0],S2))
    # for i in range(len(v[:,ei])):
    #   if(np.abs(v[i,ei])>1e-10):
    #       print("%d, %.5e"%(i,v[i,ei]))
