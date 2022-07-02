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
    #dist = np.arange(0.2,2.70,0.1)
    dist = [0.7]
    results = []
    for r in dist:
        #geometry = [('H', (0,0,0)), ('H', (0,0,1*r))]
        #geometry = '''
        #           H
        #           H 1 {0}
        #           H 1 {1} 2 {2}
        #           H 3 {0} 1 {2} 2 {3}
        #           '''.format(0.75, 1.5, 90.0, 60.0)
   
        geometry = '''
                   H            0.000000000000    -0.750000000000    -0.324759526419
                   H           -0.375000000000    -0.750000000000     0.324759526419
                   H            0.000000000000     0.750000000000    -0.324759526419
                   H            0.375000000000     0.750000000000     0.324759526419
                   '''

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
        #print(hamiltonian.toarray())
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

        [e,v] = scipy.sparse.linalg.eigsh(hamiltonian.real,1,which='SA',v0=reference_ket.todense())
        for ei in range(len(e)):
            S2 = v[:,ei].conj().T.dot(s2.dot(v[:,ei]))
            print(" State %4i: %12.8f au  <S2>: %12.8f" %(ei,e[ei]+E_nuc,S2))
        
        fermi_ham += FermionOperator((),E_nuc)
        #pyscf.molden.from_mo(mol, "full.molden", sq_ham.C)

        #   Francesco, change this to singlet_GSD() if you want generalized singles and doubles
        pool = operator_pools.singlet_SD()
        pool.init(n_orb, n_occ_a=n_a, n_occ_b=n_b, n_vir_a=n_orb-n_a, n_vir_b=n_orb-n_b)
        for i in range(pool.n_ops):
            print(pool.get_string_for_term(pool.fermi_ops[i]))
        [e,v,params,ansatz_mat] = vqe_methods.adapt_vqe(fermi_ham, pool, reference_ket, theta_thresh=1e-9)

        print(" Final ADAPT-VQE energy: %12.8f" %e)
        print(" <S^2> of final state  : %12.8f" %(v.conj().T.dot(s2.dot(v))[0,0].real))

        #store FCI values for checking
        #a,b=eigs(hamiltonian)
        #fci_levels=a+E_nuc

        #create operators single and double for each excitation
        op=qeom.createops_eefull(n_orb,n_a,n_b,n_orb-n_a,n_orb-n_b,reference_ket)
        #print('op[0] is',op[0])
        #exit()

        #transform H with e^{sigma}
        barH=qeom.barH(params, ansatz_mat, hamiltonian)
        #print("barH, H", barH,'\n\n',hamiltonian)
        #a,b=eigs(barH)
        #print('ex energy',a)
        #print('reference state',reference_ket)
        #print('energy 1',qeom.expvalue(v.transpose().conj(),hamiltonian,v))
        print('barH based energy diff=0?',qeom.expvalue(reference_ket.transpose().conj(),barH,reference_ket)[0,0].real-e+E_nuc)

        #create ex operator

        Hmat=np.zeros((len(op),len(op)))
        V=np.zeros((len(op),len(op)))
        for i in range(len(op)):
            for j in range(len(op)):
                #mat=op[i].transpose().conj().dot(barH.dot(op[j]))
                mat=qeom.comm3(op[i].transpose().conj(),barH,op[j])
                #print(mat.toarray())
                Hmat[i,j]=qeom.expvalue(reference_ket.transpose().conj(),mat,reference_ket)[0,0].real
                #mat3=qeom.comm2(op[i].transpose().conj(),op[j])
                #V[i,j]=qeom.expvalue(reference_ket.transpose().conj(),mat3,reference_ket)[0,0]
        #Diagonalize ex operator-> eigenvalues are excitation energies
        print('M: ', Hmat)
        eig,aval=scipy.linalg.eig(Hmat)
        #ex_energies =  27.2114 * np.sort(eig.real)
        ex_energies =  np.sort(eig.real)
        ex_energies = np.array([i for i in ex_energies if i >0])
        print('final excitation energies (adapt in qiskit ordering) in eV: ', ex_energies)
        final_total_energies = np.sort(eig.real)+e
        print('final total energies (adapt in qiskit ordering)', final_total_energies)


        # Response part now!

        omega = 0.077357
        identity = np.eye(len(op))
        # reading dipole integrals
        dipole_ao = mol.intor('int1e_r_sph')
        #print('dipole (ao): ', dipole_ao) 
        dipole_mo = []
        # convert from AO to MO basis
        for i in range(3):
            dipole_mo.append(np.einsum("pu,uv,vq->pq", C.T, dipole_ao[i], C))
        #print('dipole (mo): ', dipole_mo) 

        fermi_dipole_mo_op = []
        shift = 0
        for i in range(3):
            fermi_op = openfermion.FermionOperator()
            #H
            for p in range(dipole_mo[0].shape[0]):
                pa = 2*p + shift
                pb = 2*p+1 +  shift
                for q in range(dipole_mo[0].shape[1]):
                    qa = 2*q +shift
                    qb = 2*q+1 +shift
                    fermi_op += openfermion.FermionOperator(((pa,1),(qa,0)), dipole_mo[i][p,q])
                    fermi_op += openfermion.FermionOperator(((pb,1),(qb,0)), dipole_mo[i][p,q])
            fermi_dipole_mo_op.append(openfermion.transforms.get_sparse_operator(fermi_op))  

        #print('fermi_dipole_mo_op: ', fermi_dipole_mo_op)
        bar_dipole_mo = []
        #print('params: ', params)
        #print('ansatz_mat: ', ansatz_mat)
        for i in range(3):
            is_all_zero = np.all((dipole_mo[i] == 0))
            if is_all_zero:
                print('Array contains only 0')
                bar_dipole_mo.append([])
            else:
                bar_dipole_mo.append(qeom.barH(params, ansatz_mat, fermi_dipole_mo_op[i]))

        #print('bar_dipole_mo: ', bar_dipole_mo[2].toarray())
        if 'optrot' in prop_list:
            # make angular momentum integrals!
            # reading electric dipole integrals
            angmom_ao = mol.intor('int1e_cg_irxp')
            #print('angmom (ao): ', angmom_ao)
            angmom_mo = []
            # convert from AO to MO basis
            for i in range(3):
                angmom_mo.append(-0.5*np.einsum("pu,uv,vq->pq", C.T, angmom_ao[i], C))
                #print('angmom (mo): ', angmom_mo[i])

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
            #print('fermi_angmom_mo_op: ', fermi_angmom_mo_op)
            bar_angmom_mo = []
            for i in range(3):
                is_all_zero = np.all((angmom_mo[i] == 0))
                if is_all_zero:
                    #print('Array contains only 0')
                    bar_angmom_mo.append([])
                else:
                    bar_angmom_mo.append(qeom.barH(params, ansatz_mat, fermi_angmom_mo_op[i]))



        # < UCC | [Y, Qi] | UCC > --> < HF | [Ybar, Qi] | HF >
        rhs_vec_dip_xyz = []
        x_plus_x_dag_dip_xyz = []

        # one equation rather than two
        # (Hmat^2 - omega^2*I)(x - x^dag) = rhs
        Hmat_sq = Hmat.dot(Hmat)
        omega_sq = omega * omega
        Hmat_sq -= omega_sq * identity
        
        for x in range(3):
            final_rhs = np.zeros((len(op)))
            if not isinstance(bar_dipole_mo[x], list):
                for i in range(len(op)):
                    mat = qeom.comm2(bar_dipole_mo[x], op[i])
                    final_rhs[i]= qeom.expvalue(reference_ket.transpose().conj(),mat,reference_ket)[0,0].real
            rhs_vec_dip_xyz.append(final_rhs)
            print('final_rhs: ', final_rhs)
            final_rhs_one = 2*omega*final_rhs
            x_minus_x_dag = linalg.solve(Hmat_sq, final_rhs_one) 
            # x + xdag = Hmat * (x - xdag)/omega
            x_plus_x_dag = Hmat.dot(x_minus_x_dag)/omega
            x_plus_x_dag_dip_xyz.append(x_plus_x_dag)


        cart = ['X', 'Y', 'Z']
        polar = {}
        for x in range(3):
            for y in range(3):
                key = cart[x] + cart[y]
                polar[key] = rhs_vec_dip_xyz[x].dot(x_plus_x_dag_dip_xyz[y])

        print('polarizability: ', polar)

        #polar = final_rhs.dot(x_plus_x_dag)
        #print('polar: ', polar)

        # Oscillator strength!
        eig_val,eig_vec=scipy.linalg.eig(Hmat)
        ex_energies =  eig_val.real
        ex_data = []
        for i in range(len(ex_energies)):
            ex_data.append([eig_vec[:,i]])


        OS = {}
        num_OS_states = len(ex_energies)
        OS_tmp = {}

        for x in range(3):
            shape0 = fermi_dipole_mo_op[x].shape[0]
            OS[x] = []
            OS_tmp[x] = []
            is_all_zero = True
            if shape0 > 1:
                is_all_zero = False
            if not is_all_zero:
                for state in range(num_OS_states):
                    term = 0
                    for i in range(len(op)):
                        coeff_i = ex_data[state][0][i]
                        #mat1 = fermi_dipole_mo_op[x].dot(op[i])*coeff_i
                        #term += qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                        mat1 = bar_dipole_mo[x].dot(op[i])*coeff_i
                        term += qeom.expvalue(reference_ket.transpose().conj(),mat1,reference_ket)[0,0]
                    OS[x].append(term)
                    OS_tmp[x].append((term, ex_energies[state]))
            else:
                for state in range(num_OS_states):
                    OS[x].append(0.0)
                    #OS[x].append(np.zeros(num_OS_states))

        '''
        print('OS[0]: \n\n')
        for items in OS_tmp[0]:
            print(items)
        print('OS[1]: \n\n')
        for items in OS_tmp[1]:
            print(items)
        print('OS[2]: \n\n')
        for items in OS_tmp[2]:
            print(items)
        '''

        OS_final = []
        for state in range(num_OS_states):
            termx = OS[0][state]
            termy = OS[1][state]
            termz = OS[2][state]
            term =  2.0/3.0 * ex_energies[state] * (termx**2 + termy**2 + termz**2)
            OS_final.append((term, ex_energies[state]))

        OS_final = sorted(OS_final, key=lambda x: x[1])
        print('OS_final: ', OS_final)


        if 'optrot' in prop_list:
            rhs_vec_angmom_xyz = []
            x_minus_x_dag_angmom_xyz = []

            # one equation rather than two
            # (Hmat^2 - omega^2*I)(x - x^dag) = rhs
            
            for x in range(3):
                final_rhs = np.zeros((len(op)))
                if not isinstance(bar_angmom_mo[x], list):
                    for i in range(len(op)):
                        mat = qeom.comm2(bar_angmom_mo[x], op[i])
                        final_rhs[i]= qeom.expvalue(reference_ket.transpose().conj(),mat,reference_ket)[0,0].real
                rhs_vec_angmom_xyz.append(final_rhs)
                final_rhs_one = 2*omega*final_rhs
                x_minus_x_dag = linalg.solve(Hmat_sq, final_rhs_one) 
                x_minus_x_dag_angmom_xyz.append(x_minus_x_dag)
            optrot = {}
            for x in range(3):
                for y in range(3):
                    key = cart[x] + cart[y]
                    optrot[key] = rhs_vec_dip_xyz[x].dot(x_minus_x_dag_angmom_xyz[y])
            print('optrot: ', optrot)

            #print('rhs_vec_dip_xyz :', rhs_vec_dip_xyz[0])
            #print('x_plus_x_dag_dip_xyz :', x_plus_x_dag_dip_xyz[0])
            #print('rhs_vec_angmom_xyz :', rhs_vec_angmom_xyz[0])
            #print('x_minus_x_dag_angmom_xyz :', x_minus_x_dag_angmom_xyz[0])

            RS = {}
            num_RS_states =  len(ex_energies)

            for x in range(3):
                shape0_mu = fermi_dipole_mo_op[x].shape[0]
                shape0_L = 1
                if not isinstance(fermi_angmom_mo_op[x], list):
                    shape0_L =  fermi_angmom_mo_op[x].shape[0]
                RS[x] = []
                is_all_zero = True
                if shape0_mu > 1 and shape0_L > 1:
                    is_all_zero = False
                if not is_all_zero:
                    for state in range(num_RS_states):
                        term_mu = 0
                        term_L = 0
                        for i in range(len(op)):
                            coeff_i = ex_data[state][0][i]
                            mat1 = fermi_dipole_mo_op[x].dot(op[i])*coeff_i
                            term_mu += qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                            mat1 = fermi_angmom_mo_op[x].dot(op[i])*coeff_i
                            term_L += qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                        RS[x].append((term_mu, term_L, ex_energies[state]))
                else:
                    for state in range(num_RS_states):
                        RS[x].append((0,0,ex_energies[state]))

            
            for x in range(3):
                RS[x] = sorted(RS[x], key=lambda x: x[2]) 
           
            '''
            print('RS[0]\n')
            for items in RS[0]:
                print(items)

            print('RS[1]\n')
            for items in RS[1]:
                print(items)

            print('RS[1]\n')
            for items in RS[2]:
                print(items)
            '''

            RS_final = []
            for state in range(num_RS_states):
                term_mux = RS[0][state][0]
                term_Lx = RS[0][state][1]

                term_muy = RS[1][state][0]
                term_Ly = RS[1][state][1]

                term_muz = RS[2][state][0]
                term_Lz = RS[2][state][1]

                term =  -1.0*(term_mux*term_Lx + term_muy*term_Ly + term_muz*term_Lz)
                RS_final.append((term, ex_energies[state]))

            RS_final = sorted(RS_final, key=lambda x: x[1])
            print('RS_final: ', RS_final)
        temp = {}
        temp['polarizability'] = polar
        temp['isotropic_polarizability'] = 1/3.0 * (polar['XX'] + polar['YY'] + polar['ZZ'])
        temp['rotation(589nm)'] = optrot
        temp['trace_rotation(589nm)'] = optrot['XX'] + optrot['YY'] + optrot['ZZ']
        temp['OS'] = OS_final
        temp['RS'] = RS_final
        results.append(temp)
    return results

if __name__== "__main__":
    results = test(['polar', 'optrot'])
    print('results: ', results)
    #output = open('h2_scqeom.dat', 'wb')
    #pickle.dump(results, output) # converts array to binary and writes to output
    #input_ = open('h2_scqeom.dat', 'rb')
    #results = pickle.load(input_) # Reads 
    #print('results after reading: ', results)
