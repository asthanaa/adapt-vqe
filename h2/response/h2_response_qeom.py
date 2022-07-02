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

        geometry = [('H', (0,0,0)), ('H', (0,0,1*r))]
         
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
        pool = operator_pools.singlet_GSD()
        pool.init(n_orb, n_occ_a=n_a, n_occ_b=n_b, n_vir_a=n_orb-n_a, n_vir_b=n_orb-n_b)
        for i in range(pool.n_ops):
            print(pool.get_string_for_term(pool.fermi_ops[i]))
        [e,v,params,ansatz_mat] = vqe_methods.adapt_vqe(fermi_ham, pool, reference_ket, theta_thresh=1e-9)
        print('len(pool): ', pool.n_ops)

        print(" Final ADAPT-VQE energy: %12.8f" %e)
        print(" <S^2> of final state  : %12.8f" %(v.conj().T.dot(s2.dot(v))[0,0].real))


        #create operators single and double for each excitation
        op=qeom.createops_eefull(n_orb,n_a,n_b,n_orb-n_a,n_orb-n_b,reference_ket)

        # Response part now!
        # Christiansen's paper!
        # lets first do this using the q-eom approach first 
        # and then with the Ayush's approach later!
        
        M=np.zeros((len(op),len(op)))
        Q=np.zeros((len(op),len(op)))
        V=np.zeros((len(op),len(op)))
        W=np.zeros((len(op),len(op)))
        Hmat=np.zeros((len(op)*2,len(op)*2))
        S=np.zeros((len(op)*2,len(op)*2))
        for i in range(len(op)):
            for j in range(len(op)):
                mat1=qeom.comm3(op[i],hamiltonian,op[j].transpose().conj())
                M[i,j]=qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                mat2=qeom.comm3(op[i],hamiltonian,op[j])
                Q[i,j]=-qeom.expvalue(v.transpose().conj(),mat2,v)[0,0]
                mat3=qeom.comm2(op[i],op[j].transpose().conj())
                V[i,j]=qeom.expvalue(v.transpose().conj(),mat3,v)[0,0]
                mat4=qeom.comm2(op[i],op[j])
                W[i,j]=-qeom.expvalue(v.transpose().conj(),mat4,v)[0,0]

        #print('M: ', M)
        #print('Q: ', Q)
        #print('V: ', V)
        #print('W: ', W)
        Hmat=np.bmat([[M,Q],[Q.conj(),M.conj()]])
        S=np.bmat([[V,W],[-W.conj(),-V.conj()]])

        eig,aval=scipy.linalg.eig(Hmat,S)
        ex_energies =  np.sort(eig.real)
        ex_energies = np.array([i for i in ex_energies if i >0])
        print('final excitation energies (adapt in qiskit ordering) in eV: ', ex_energies)
        final_total_energies = np.sort(eig.real)+e
        print('final total energies (adapt in qiskit ordering)', final_total_energies)


        omega = 0.077357

        # LHS Matrix 

        Hmat_res = Hmat - omega * S

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
   
        response_amp_dip_xyz = [] 
        rhs_vec_dip_xyz = []
        for x in range(3):
            dip_up = np.zeros((len(op)))
            dip_down = np.zeros((len(op)))
            rhs_vec = np.zeros((2 * len(op)))
            shape0 = fermi_dipole_mo_op[x].shape[0] 
            is_all_zero = True
            if shape0 > 1:
                is_all_zero = False
            for i in range(len(op)):
                if not is_all_zero:
                    mat1=qeom.comm2(fermi_dipole_mo_op[x],op[i])
                    dip_up[i]=qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                    mat2=qeom.comm2(fermi_dipole_mo_op[x],op[i].transpose().conj())
                    dip_down[i]=-1.0 * qeom.expvalue(v.transpose().conj(),mat2,v)[0,0]
                    #dip_down[i]=qeom.expvalue(v.transpose().conj(),mat2,v)[0,0]
            for i in range(2 * len(op)):
                if i < len(op):
                    rhs_vec[i] = dip_up[i]
                else:
                    rhs_vec[i] = dip_down[i-len(op)]

            rhs_vec_dip_xyz.append(rhs_vec)
            response_amp_dip_xyz.append(linalg.solve(Hmat_res, rhs_vec))


        cart = ['X', 'Y', 'Z']   
        polar = {}
        for x in range(3):
            for y in range(3):
                key = cart[x] + cart[y]
                polar[key] = rhs_vec_dip_xyz[x].dot(response_amp_dip_xyz[y])

        print('polarizability: ', polar)
        print('response_amp_dip_xyz: ', response_amp_dip_xyz)
        print('rhs_vec_dip_xyz: ', rhs_vec_dip_xyz)

        # Oscillator strength!
        eig_val,eig_vec=scipy.linalg.eig(Hmat,S)
        ex_energies =  eig_val.real
        print('ex_energies: ', ex_energies)
        old_order = np.argsort(ex_energies)
        ex_energies = sorted(ex_energies)#, key=lambda x: x[-1])
        new_eig_vec = np.zeros((2*len(op), 2*len(op)))
        for i in range(2*len(op)):
            new_eig_vec[i] = eig_vec[:,old_order[i]]
        
         

        print('number of eigenstates: ', len(ex_energies))
        #print('eig_val: ', eig_val)
        #print('eig_vec: ', eig_vec[0])
        #print('new_eig_vec: ', new_eig_vec)
        #print('ex_energies: ', ex_energies)
        ex_data = []
        ex_data_neg = []
        size_ex_energies = len(ex_energies)
        for i in range(int(size_ex_energies/2)):
            if ex_energies[i] < 0:
                ex_data_neg.append((ex_energies[i], [new_eig_vec[i]]))
                ex_data.append((ex_energies[size_ex_energies-i-1], [new_eig_vec[size_ex_energies-i-1]]))
            #print(ex_energies[i])

        #print('ex_data_neg: ', ex_data_neg)
        #print('ex_data: ', ex_data)
        num_OS_states = len(ex_data)

        print('plus minus') 
        for state in range(num_OS_states):
            print(ex_data[state][0], ex_data_neg[state][0])

        OS = {}

        for x in range(3):
            shape0 = fermi_dipole_mo_op[x].shape[0] 
            OS[x] = []
            is_all_zero = True
            if shape0 > 1:
                is_all_zero = False
            if not is_all_zero:
                print('x:: ', x)
                for state in range(num_OS_states):
                    ex_energy = ex_data[state][0]
                    #if ex_energy > 0:
                    print('ex_energy: ', ex_energy)
                    term = 0
                    for i in range(2*len(op)):
                        coeff_i = ex_data[state][1][0][i]
                        if i < len(op):
                            #mat1 = fermi_dipole_mo_op[x].dot(op[i])*coeff_i
                            #mat1 = fermi_dipole_mo_op[x].dot(op[i].transpose().conj())*coeff_i
                            mat1 = op[i].dot(fermi_dipole_mo_op[x])*coeff_i
                            temp = qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                            term += temp	
                            #print(' smaller state op term: ', state, i, temp)
                            #print('coeff_i: ', coeff_i)
                        else:
                            #mat1 = fermi_dipole_mo_op[x].dot(op[i- len(op)].transpose().conj())*coeff_i
                            #mat2 = fermi_dipole_mo_op[x].dot(op[i- len(op)])*coeff_i
                            #mat1 = op[i-len(op)].dot(fermi_dipole_mo_op[x])*coeff_i
                            mat1 = op[i-len(op)].transpose().conj().dot(fermi_dipole_mo_op[x])*coeff_i
                            temp = qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                            #temp = qeom.expvalue(v.transpose().conj(),mat2,v)[0,0]
                            #term += qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                            #mat1 = fermi_dipole_mo_op[x].dot(op[i- len(op)])*coeff_i
                            #temp = qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                            term += temp	
                            #print('greater state op term: ', state, i, temp)
                            #print('coeff_i: ', coeff_i)
                            #print('term_expt: ', temp)
                    #OS[x].append((term, np.round(ex_data[state][0], 4)))     
                    OS[x].append(term)     
            else:
                for state in range(num_OS_states):
                    #ex_energy = ex_data[state][0]
                    #print('ex_energy: ', ex_energy)
                    #if ex_energy > 0:
                    OS[x].append(0.0)

        ############################################################################################
        OS_neg = {}
        print('OS_neg: ')
        for x in range(3):
            shape0 = fermi_dipole_mo_op[x].shape[0] 
            OS_neg[x] = []
            is_all_zero = True
            if shape0 > 1:
                is_all_zero = False
            if not is_all_zero:
                for state in range(num_OS_states):
                    ex_energy = ex_data_neg[state][0]
                    print('ex_energy: ', ex_energy)
                    term = 0
                    for i in range(2*len(op)):
                        coeff_i = ex_data_neg[state][1][0][i]
                        if i < len(op):
                            mat1 = fermi_dipole_mo_op[x].dot(op[i])*coeff_i
                            temp = qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                            term += temp	
                            #print(' smaller state op term: ', state, i, temp)
                            #print('coeff_i: ', coeff_i)
                        else:
                            mat1 = fermi_dipole_mo_op[x].dot(op[i- len(op)].transpose().conj())*coeff_i
                            temp = qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                            term += temp	
                            #print('greater state op term: ', state, i, temp)
                            #print('coeff_i: ', coeff_i)
                    OS_neg[x].append(term)     
            else:
                for state in range(num_OS_states):
                    #ex_energy = ex_data[state][0]
                    #print('ex_energy: ', ex_energy)
                    #if ex_energy > 0:
                    OS_neg[x].append(0.0)

        #OS_final = OS
        #OS_final = OS
        #for i in range(len(op)):
        # mu_z
        # fix state and mu_z 
        '''
        op_exc = []
        op_de_exc = []
        state = 2
        print(2/3 * ex_data[state][0])
        for i in range(len(op)):
            op_exc.append(op[i])
        for i in range(len(op)):
            op_exc.append(op[i].transpose().conj())
        #for state in range(num_OS_states):
        #    print('ex_energy: ', ex_data[state][0]) 
        #    coeff_i = ex_data[state][1][0][i]
        A = 0
        B = 0
        C = 0
        D = 0
        for i in range(6):
            #mat1 = fermi_dipole_mo_op[2].dot(op_exc[i])*ex_data[state][1][0][i]
            #print(qeom.expvalue(v.transpose().conj(),mat1,v)[0,0])
            mat1 = op_exc[i].dot(fermi_dipole_mo_op[2])*ex_data[state][1][0][i]
            print(qeom.expvalue(v.transpose().conj(),mat1,v)[0,0])
            #mat1 = fermi_dipole_mo_op[2].dot(op_de_exc[i])*ex_data[state][1][0][3+i]
            #A += qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
            #B += qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
            #mat1 = op_de_exc[i].dot(fermi_dipole_mo_op[2])*ex_data[state][1][0][i]
            #C += qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
            #mat1 = op_exc[i].dot(fermi_dipole_mo_op[2])*ex_data[state][1][0][3+i]
            #D += qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]

        #print('A: ', A)
        #print('B: ', B)
        #print('C: ', C)
        #print('D: ', D)
        print('\n')
        print(ex_data[state][1][0][0])
        print(ex_data[state][1][0][1])
        print(ex_data[state][1][0][2])
        print(ex_data[state][1][0][3])
        print(ex_data[state][1][0][4])
        print(ex_data[state][1][0][5])
        '''


        OS_plus = []
        for state in range(num_OS_states):
            termx_plus = OS[0][state]
            termy_plus = OS[1][state]
            termz_plus = OS[2][state]
            termx_minus = OS_neg[0][state]
            termy_minus = OS_neg[1][state]
            termz_minus = OS_neg[2][state]
            termx = 0.5 * (termx_plus + termx_minus)
            termy = 0.5 * (termy_plus + termy_minus)
            termz = 0.5 * (termz_plus + termz_minus)
            print(' + termx, termy, termz: ', termx, termy, termz)
            term =  2.0/3.0 * ex_data[state][0] * (termx**2 + termy**2 + termz**2)
            OS_plus.append((term, np.round(ex_data[state][0], 4)))

        OS_final = sorted(OS_plus, key=lambda x: x[1])

        print('OS_final: ', OS_final)

        if 'optrot' in prop_list:
            optrot = {}
            response_amp_angmom_xyz = [] 
            rhs_vec_angmom_xyz = []
            for x in range(3):
                dip_up = np.zeros((len(op)))
                dip_down = np.zeros((len(op)))
                rhs_vec = np.zeros((2 * len(op)))
                if isinstance(fermi_angmom_mo_op[x], list):
                    shape0 = 1 
                else:
                    shape0 = fermi_angmom_mo_op[x].shape[0]
                is_all_zero = True
                if shape0 > 1:
                    is_all_zero = False
                for i in range(len(op)):
                    if not is_all_zero:
                        mat1=qeom.comm2(fermi_angmom_mo_op[x],op[i])
                        dip_up[i]=qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                        mat2=qeom.comm2(fermi_angmom_mo_op[x],op[i].transpose().conj())
                        dip_down[i]=-1.0 * qeom.expvalue(v.transpose().conj(),mat2,v)[0,0]
                        #dip_down[i]=qeom.expvalue(v.transpose().conj(),mat2,v)[0,0]
                for i in range(2 * len(op)):
                    if i < len(op):
                        rhs_vec[i] = dip_up[i]
                    else:
                        rhs_vec[i] = dip_down[i-len(op)]

                rhs_vec_angmom_xyz.append(rhs_vec)
                response_amp_angmom_xyz.append(linalg.solve(Hmat_res, rhs_vec))
            for x in range(3):
                for y in range(3):
                    key = cart[x] + cart[y]
                    optrot[key] = -1.0 * rhs_vec_dip_xyz[x].dot(response_amp_angmom_xyz[y])
            print('optrot: ', optrot) 

            '''
            print('rhs_vec_dip_xyz: ', rhs_vec_dip_xyz[0])
            print('response_amp_dip_xyz: ',  response_amp_dip_xyz[0])
            print('rhs_vec_angmom_xyz: ', rhs_vec_angmom_xyz[0])
            print('response_amp_angmom_xyz: ',  response_amp_angmom_xyz[0])
            '''
   
            RS = {}
            num_RS_states = len(ex_energies)

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
                        for i in range(2*len(op)):
                            coeff_i = ex_data[state][1][0][i]
                            if i < len(op):
                                mat1 = fermi_dipole_mo_op[x].dot(op[i])*coeff_i
                                term_mu -= qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                                mat1 = fermi_angmom_mo_op[x].dot(op[i])*coeff_i
                                term_L -= qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                            else:
                                mat1 = fermi_dipole_mo_op[x].dot(op[i-len(op)].transpose().conj())*coeff_i
                                term_mu -= qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                                mat1 = fermi_angmom_mo_op[x].dot(op[i-len(op)].transpose().conj())*coeff_i
                                term_L -= qeom.expvalue(v.transpose().conj(),mat1,v)[0,0]
                        RS[x].append((term_mu, term_L, ex_data[state][0]))
                else:
                    for state in range(num_RS_states):
                        RS[x].append((0,0,ex_data[state][0]))

            for x in range(3):
                RS[x] = sorted(RS[x], key=lambda x: x[2])


            RS_plus_minus = []
            for state in range(num_RS_states):

                ex_energy = np.round(RS[0][state][2], 4)
                #print('ex_energy: ', ex_energy)
                term_mux = RS[0][state][0]
                term_Lx = RS[0][state][1]

                term_muy = RS[1][state][0]
                term_Ly = RS[1][state][1]

                term_muz = RS[2][state][0]
                term_Lz = RS[2][state][1]

                #print('term_mux: ', term_mux)
                #print('term_muy: ', term_muy)
                #print('term_muz: ', term_muz)
                #print('term_Lx: ', term_Lx)
                #print('term_Ly: ', term_Ly)
                #print('term_Lz: ', term_Lz)

                term =  -1.0*(term_mux*term_Lx + term_muy*term_Ly + term_muz*term_Lz)
                RS_plus_minus.append((term, np.round(RS[0][state][2], 4)))

            RS_plus_minus = sorted(RS_plus_minus, key=lambda x: x[1])

            RS_final = []
            for i in range(int(num_RS_states/2)):
                RS = RS_plus_minus[i][0] - RS_plus_minus[num_OS_states-i-1][0]
                RS_final.append((RS, abs(RS_plus_minus[i][1])))
            RS_final = sorted(RS_final, key=lambda x: x[-1])
            print('RS_final: ', RS_final)
        temp = {}
        temp['polarizability'] = polar
        temp['isotropic_polarizability'] = 1/3.0 * (polar['XX'] + polar['YY'] + polar['ZZ'])
        temp['OS'] = OS_final
        if 'optrot' in prop_list:
            temp['rotation(589nm)'] = optrot
            temp['trace_rotation(589nm)'] = optrot['XX'] + optrot['YY'] + optrot['ZZ']
            temp['RS'] = RS_final
        else:
            temp['rotation(589nm)'] = 0
            temp['trace_rotation(589nm)'] = 0
            temp['RS'] = 0
        results.append(temp)
    return results

if __name__== "__main__":
    #results = test(['polar', 'optrot'])
    results = test(['polar'])
    print('results: ', results)
    #output = open('h2_qeom.dat', 'wb')
    #pickle.dump(results, output) # converts array to binary and writes to output
    #input_ = open('h2_qeom.dat', 'rb')
    #results = pickle.load(input_) # Reads 
    #print('results after reading: ', results)
