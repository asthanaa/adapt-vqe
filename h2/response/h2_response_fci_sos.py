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

import qeom
from scipy.sparse.linalg import eigs
from scipy import linalg

def test():
    r =0.7
    geometry = [('H', (0,0,0)), ('H', (0,0,1*r))]


    charge = 0
    spin = 0
    #basis = '6-31g'
    basis = 'sto-3g'

    [n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis)

    print(" n_orb: %4i" %n_orb)
    print(" n_a  : %4i" %n_a)
    print(" n_b  : %4i" %n_b)

    sq_ham = pyscf_helper.SQ_Hamiltonian()
    sq_ham.init(h, g, C, S)
    print(" HF Energy: %12.8f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))
    #ehf = E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))
    #assert(abs(ehf - -1.82913741) < 1e-7)
    fermi_ham  = sq_ham.export_FermionOperator()
    hamiltonian = openfermion.transforms.get_sparse_operator(fermi_ham)
    hamiltonian_arr = hamiltonian.real.toarray()
    print(hamiltonian_arr.shape)
    #exit()

    s2 = vqe_methods.Make_S2(n_orb)

    #build reference configuration
    occupied_list = []
    for i in range(n_a):
        occupied_list.append(i*2)
    for i in range(n_b):
        occupied_list.append(i*2+1)

    print(" Build reference state with %4i alpha and %4i beta electrons" %(n_a,n_b), occupied_list)

    reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(occupied_list, 2*n_orb)).transpose()

    shift = 0
    number_op = openfermion.FermionOperator()
    number_vec = np.ones(n_orb)
    for p in range(n_orb):
        pa = 2*p + shift
        pb = 2*p+1 +  shift
        number_op += openfermion.FermionOperator(((pa,1),(pa,0)), number_vec[p])
        number_op += openfermion.FermionOperator(((pb,1),(pb,0)), number_vec[p])
    number_op = openfermion.transforms.get_sparse_operator(number_op)
   
    index_states = [] 
    [e,v] = scipy.sparse.linalg.eigsh(hamiltonian_arr,16,which='SA',v0=reference_ket.todense())
    for ei in range(len(e)):
        S2 = v[:,ei].conj().T.dot(s2.dot(v[:,ei]))
        number = v[:,ei].conj().T.dot(number_op.dot(v[:,ei]))
        if np.round(number.real, 6) == 2 and ei !=0:
            index_states.append(ei) 
        print(" State %4i: %12.8f au  <S2>: %12.8f" %(ei,e[ei]+E_nuc,S2))

    print('index_states: ', index_states)
    ex_states = np.zeros((16))
    for state in index_states:
        ex_states[state] = e[state]-e[0]
    print('ex_states', ex_states)

     
    # reading dipole integrals
    dipole_ao = mol.intor('int1e_r_sph')
    print('dipole (ao): ', dipole_ao)
    dipole_mo = []
    # convert from AO to MO basis
    for i in range(3):
        dipole_mo.append(np.einsum("pu,uv,vq->pq", C.T, dipole_ao[i], C))
    print('dipole (mo): ', dipole_mo)

    fermi_dipole_mo_op = []
    shift = 0
    for i in range(3):
        fermi_op = openfermion.FermionOperator()
        is_all_zero = np.all((dipole_mo[i] == 0)) 
        if not is_all_zero:
            #MU
            for p in range(dipole_mo[0].shape[0]):
                pa = 2*p + shift
                pb = 2*p+1 +  shift
                for q in range(dipole_mo[0].shape[1]):
                    qa = 2*q +shift
                    qb = 2*q+1 +shift
                    fermi_op += openfermion.FermionOperator(((pa,1),(qa,0)), dipole_mo[i][p,q])
                    fermi_op += openfermion.FermionOperator(((pb,1),(qb,0)), dipole_mo[i][p,q])
            fermi_dipole_mo_op.append(openfermion.transforms.get_sparse_operator(fermi_op))
        else:
            fermi_dipole_mo_op.append([])
    print('fermi_dipole_mo_op: ', fermi_dipole_mo_op)

    omega = 0.077357 
    # resp_state[i] = sum_i 2 * omega * < 0 | X | i > < 0 | Y | i > / (omega^2 - omega_i^2)
    cart = ['X', 'Y', 'Z']
    polar = {}
    for x in range(3):
        for y in range(3):
            key = cart[x] + cart[y]
            polar[key] = []
            if isinstance(fermi_dipole_mo_op[x], list) or  isinstance(fermi_dipole_mo_op[y], list):
                polar[key] = [0.0]
            else:
                for state in index_states:
                    term1  =  v[:,0].transpose().conj().dot(fermi_dipole_mo_op[x].dot(v[:,state]))
                    term1 *= v[:,state].transpose().conj().dot(fermi_dipole_mo_op[y].dot(v[:,0]))
                    term1 *= (1/(omega - ex_states[state]))

                    term2  =  v[:,0].transpose().conj().dot(fermi_dipole_mo_op[y].dot(v[:,state]))
                    term2 *= v[:,state].transpose().conj().dot(fermi_dipole_mo_op[x].dot(v[:,0]))
                    term2 *= (1/(omega + ex_states[state]))

                    term = term1 - term2
                    print(term) 
                    polar[key].append(term)

            polar[key] = sum(polar[key]).real


    print('polarizability: ', polar)

if __name__== "__main__":
    test()
