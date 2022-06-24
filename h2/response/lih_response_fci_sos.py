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
import pickle

def test(prop_list):
    dist = np.arange(1.5,3.70,0.1)
    dist = np.arange(2.2,4.2,0.025)
    dist = np.arange(2.652,2.677,0.004)
    dist = np.append(dist, np.arange(2.67620,2.67690,.0002))
    #dist = np.arange(2.6771,2.6800,0.0002)
    #dist = np.append(dist, np.arange(2.6980,2.6999,.0002))
    #dist = np.arange(3.409,3.424,0.002)
    #dist = np.append(dist, 3.40750)
    results = []
    for r in dist:
        #geometry = [('H', (0,0,0)), ('H', (0,0,1*r))]
        geometry = [('Li', (0,0,0)), ('H', (0,0,1*r))]
 

        #geometry = '''
        #           H
        #           H 1 {0}
        #           H 1 {1} 2 {2}
        #           H 3 {0} 1 {2} 2 {3}
        #           '''.format(r, 2.00, 90.0, 60.0)
        #           #'''.format(r, 2*r, 90.0, 60.0)
        #           #'''.format(0.75, 1.5, 90.0, 60.0)

        #geometry = '''
        #           H            0.000000000000    -0.750000000000    -0.324759526419
        #           H           -0.375000000000    -0.750000000000     0.324759526419 
        #           H            0.000000000000     0.750000000000    -0.324759526419 
        #           H            0.375000000000     0.750000000000     0.324759526419 
        #           '''

        #geometry = '''
        #           O     -0.028962160801    -0.694396279686    -0.049338350190
        #           O      0.028962160801     0.694396279686    -0.049338350190
        #           H      0.350498145881    -0.910645626300     0.783035421467
        #           H     -0.350498145881     0.910645626300     0.783035421467
        #           '''


        # distorted (H2)2 --> to check symmetry of linear response function for exact wavefunction!
        #geometry = '''
        #           H            0.000000000000    -0.750000000000    -0.324759526419
        #           H           -0.375000000000    -0.760000000000     0.324759526419
        #           H            0.000000000000     0.750000000000    -0.324759526419
        #           H            0.375000000000     0.850000000000     0.45
        #           '''

        #geometry = '''
        # #O            0.059724787792     0.052257861912     0.000000000000   
        # #H            0.059724787792    -1.147742138088     0.000000000000   
        # #H           -1.007600511112     0.318371947072     0.000000000000   
        # # distorted geometry of h2o
        #  O            0.043340048199     0.042342768297     0.000000000000   
        #  H            0.043340048199    -0.914657231703     0.000000000000  
        #  H           -0.731178064104     0.242646771541     0.000000000000 
        # '''

        charge = 0
        spin = 0
        #basis = '6-31g'
        basis = 'sto-3g'
        #basis = 'ano-rcc-mb'

        [n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis, n_frzn_occ=1, n_act=5)
        #[n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis)

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
  
        num_states = 4**n_orb 
        index_states = [] 
        S2_states = [] 
        [e,v] = scipy.sparse.linalg.eigsh(hamiltonian_arr,num_states,which='SA',v0=reference_ket.todense())
        for ei in range(len(e)):
            S2 = v[:,ei].conj().T.dot(s2.dot(v[:,ei]))
            S2_states.append(S2) 
            number = v[:,ei].conj().T.dot(number_op.dot(v[:,ei]))
            #if np.round(number.real, 3) == n_a + n_b and ei !=0 and np.round(S2.real, 3) == 0:
            if np.round(number.real, 3) == n_a + n_b and ei !=0 :
                index_states.append(ei) 
            print(" State %4i: %12.8f au  <S2>: %12.8f" %(ei,e[ei]+E_nuc,S2))

        print('index_states: ', index_states)
        ex_states = np.zeros((num_states))
        for state in index_states:
            if state != 0:
                ex_states[state] = e[state]-e[0]
        #print('ex_states', ex_states)

         
        # reading electric dipole integrals
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
        if 'optrot' in prop_list:
            # make angular momentum integrals!
            # reading electric dipole integrals
            angmom_ao = mol.intor('int1e_cg_irxp')
            print('angmom (ao): ', angmom_ao)
            angmom_mo = []
            # convert from AO to MO basis
            for i in range(3):
                angmom_mo.append(-0.5*np.einsum("pu,uv,vq->pq", C.T, angmom_ao[i], C))
                print('angmom (mo): ', angmom_mo[i])

            fermi_angmom_mo_op = []
            shift = 0
            for i in range(3):
                fermi_op = openfermion.FermionOperator()
                is_all_zero = np.all((angmom_mo[i] == 0)) 
                if not is_all_zero:
                    #MU
                    for p in range(angmom_mo[0].shape[0]):
                        pa = 2*p + shift
                        pb = 2*p+1 +  shift
                        for q in range(angmom_mo[0].shape[1]):
                            qa = 2*q +shift
                            qb = 2*q+1 +shift
                            fermi_op += openfermion.FermionOperator(((pa,1),(qa,0)), angmom_mo[i][p,q])
                            fermi_op += openfermion.FermionOperator(((pb,1),(qb,0)), angmom_mo[i][p,q])
                    fermi_angmom_mo_op.append(openfermion.transforms.get_sparse_operator(fermi_op))
                else:
                    fermi_angmom_mo_op.append([])
            print('fermi_angmom_mo_op: ', fermi_angmom_mo_op)

        omega = 0.077357 
        # resp_state[i] = sum_i  < 0 | X | i > < i | Y | 0 > / (omega - omega_i)
        #                        + < 0 | Y | i > < i | X | 0 > / (omega + omega_i) 
        cart = ['X', 'Y', 'Z']
        polar = {}
        OS = {}
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
                        polar[key].append(term)

                polar[key] = -1.0 * sum(polar[key]).real
        OS = []
        num_OS_states = len(index_states)
        print('v0: ', len(v[:,0]))
        print('v1: ', len(v[:,1]))
        print('dipole_mo_op: ', fermi_dipole_mo_op[2].shape)
        for state in index_states[:num_OS_states]:
            termx = 0
            termy = 0
            termz = 0
            if not isinstance(fermi_dipole_mo_op[0], list):
                termx  =  v[:,0].transpose().conj().dot(fermi_dipole_mo_op[0].dot(v[:,state]))
            if not isinstance(fermi_dipole_mo_op[1], list):
                termy  =  v[:,0].transpose().conj().dot(fermi_dipole_mo_op[1].dot(v[:,state]))
            if not isinstance(fermi_dipole_mo_op[2], list):
                termz  =  v[:,0].transpose().conj().dot(fermi_dipole_mo_op[2].dot(v[:,state]))
            term =  2.0/3.0 * ex_states[state] * (termx**2 + termy**2 + termz**2)
            #OS.append((term, np.abs(np.round(S2_states[state],3)), ex_states[state]))
            OS.append((term, ex_states[state]))
                
                 
        print('polarizability: ', polar)
        print('OS: ', OS)

        if 'optrot' in prop_list:
            optrot = {}
            RS = []
            for x in range(3):
                for y in range(3):
                    key = cart[x] + cart[y]
                    optrot[key] = []
                    if isinstance(fermi_dipole_mo_op[x], list) or  isinstance(fermi_angmom_mo_op[y], list):
                        optrot[key] = [0.0]
                    else:
                        for state in index_states:
                            term1 = 0
                            term2 = 0
                            if not isinstance(fermi_dipole_mo_op[x], list) and not isinstance(fermi_angmom_mo_op[y], list):
                                term1  =  v[:,0].transpose().conj().dot(fermi_dipole_mo_op[x].dot(v[:,state]))
                                print('x: y: ', x, y)
                                term1 *= v[:,state].transpose().conj().dot(fermi_angmom_mo_op[y].dot(v[:,0]))
                                term1 *= (1/(omega - ex_states[state]))

                            if not isinstance(fermi_dipole_mo_op[y], list) and not isinstance(fermi_angmom_mo_op[x], list):
                                term2  =  v[:,0].transpose().conj().dot(fermi_angmom_mo_op[y].dot(v[:,state]))
                                term2 *= v[:,state].transpose().conj().dot(fermi_dipole_mo_op[x].dot(v[:,0]))
                                term2 *= (1/(omega + ex_states[state]))
                            term = term1 - term2
                            optrot[key].append(term)

                    optrot[key] = sum(optrot[key]).real
            print('optical rotation: ', optrot)
            RS = []
            num_RS_states = len(index_states)
            for state in index_states[:num_RS_states]:
                term_mux = 0
                term_muy = 0
                term_muz = 0
                term_Lx = 0
                term_Ly = 0
                term_Lz = 0
                if not isinstance(fermi_dipole_mo_op[0], list) and not isinstance(fermi_angmom_mo_op[0], list):
                    term_mux  =  v[:,0].transpose().conj().dot(fermi_dipole_mo_op[0].dot(v[:,state]))
                    term_Lx   =  v[:,state].transpose().conj().dot(fermi_angmom_mo_op[0].dot(v[:,0])) 
                #term_Lx_1   =  v[:,0].transpose().conj().dot(fermi_angmom_mo_op[0].dot(v[:,state]))
                # term_Lx and term_Lx_1 are negatives of each other!

                if not isinstance(fermi_dipole_mo_op[1], list) and not isinstance(fermi_angmom_mo_op[1], list):
                    term_muy  =  v[:,0].transpose().conj().dot(fermi_dipole_mo_op[1].dot(v[:,state]))
                    term_Ly   =  v[:,state].transpose().conj().dot(fermi_angmom_mo_op[1].dot(v[:,0])) 

                if not isinstance(fermi_dipole_mo_op[2], list) and not isinstance(fermi_angmom_mo_op[2], list):
                    term_muz  =  v[:,0].transpose().conj().dot(fermi_dipole_mo_op[2].dot(v[:,state]))
                    term_Lz   =  v[:,state].transpose().conj().dot(fermi_angmom_mo_op[2].dot(v[:,0])) 

                term =  term_mux*term_Lx + term_muy*term_Ly + term_muz*term_Lz
                #RS.append((term, np.abs(np.round(S2_states[state],3)), ex_states[state]))
                RS.append((term, ex_states[state]))

                #print('State: ', state)
                #print('term_mux: ', term_mux)
                #print('term_muy: ', term_muy)
                #print('term_muz: ', term_muz)

                #print('term_Lx: ', term_Lx)
                #print('term_Ly: ', term_Ly)
                #print('term_Lz: ', term_Lz)
            print('RS: ', RS)
            #print('ex_energies: ', ex_states[ex_states != 0][:15])
            print('ex_energies: ', ex_states[ex_states != 0])
        temp = {}
        temp['polarizability'] = polar
        temp['isotropic_polarizability'] = 1/3.0 * (polar['XX'] + polar['YY'] + polar['ZZ'])
        temp['OS'] = OS
        if 'optrot' in prop_list:
            temp['rotation(589nm)'] = optrot
            temp['trace_rotation(589nm)'] = optrot['XX'] + optrot['YY'] + optrot['ZZ']
            temp['RS'] = RS
        else:
            temp['rotation(589nm)'] = 0
            temp['trace_rotation(589nm)'] = 0 
            temp['RS'] = 0
        results.append(temp)
    return results

if __name__== "__main__":
    #results = test(['polar', 'optrot'])
    results = test(['polar'])
    #print('results: ', results)
    #np.savetxt('h2_fci_sos.txt', results)
    output = open('lih_fci_sos_1.dat', 'wb')
    pickle.dump(results, output) # converts array to binary and writes to output
    input_ = open('lih_fci_sos_1.dat', 'rb')
    results = pickle.load(input_) # Reads 
    print('results after reading: ', results)
