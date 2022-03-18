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

def test(prop_list):
    #r =0.7
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
    pool = operator_pools.singlet_GSD()
    pool.init(n_orb, n_occ_a=n_a, n_occ_b=n_b, n_vir_a=n_orb-n_a, n_vir_b=n_orb-n_b)
    for i in range(pool.n_ops):
        print(pool.get_string_for_term(pool.fermi_ops[i]))
    [e,v,params,ansatz_mat] = vqe_methods.adapt_vqe(fermi_ham, pool, reference_ket, theta_thresh=1e-9)

    print(" Final ADAPT-VQE energy: %12.8f" %e)
    print(" <S^2> of final state  : %12.8f" %(v.conj().T.dot(s2.dot(v))[0,0].real))


    #create operators single and double for each excitation
    op=qeom.createops_eefull(n_orb,n_a,n_b,n_orb-n_a,n_orb-n_b,reference_ket)

    #transform H with e^{sigma}
    barH=qeom.barH(params, ansatz_mat, hamiltonian)
    print('barH based energy diff=0?',qeom.expvalue(reference_ket.transpose().conj(),barH,reference_ket)[0,0].real-e+E_nuc)

    #create ex operator

    Hmat=np.zeros((len(op),len(op)))
    V=np.zeros((len(op),len(op)))
    for i in range(len(op)):
        for j in range(len(op)):
            mat=qeom.comm3(op[i].transpose().conj(),barH,op[j])
            #print(mat.toarray())
            Hmat[i,j]=qeom.expvalue(reference_ket.transpose().conj(),mat,reference_ket)[0,0].real
    #Diagonalize ex operator-> eigenvalues are excitation energies
    eig,aval=scipy.linalg.eig(Hmat)
    ex_energies =  27.2114 * np.sort(eig.real)
    ex_energies = np.array([i for i in ex_energies if i >0])
    print('final excitation energies (adapt in qiskit ordering) in eV: ', ex_energies)
    final_total_energies = np.sort(eig.real)+e
    print('final total energies (adapt in qiskit ordering)', final_total_energies)


    
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

    Hmat=np.bmat([[M,Q],[Q.conj(),M.conj()]])
    S=np.bmat([[V,W],[-W.conj(),-V.conj()]])


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

    #print('rhs_vec_dip_xyz[z]: ', rhs_vec_dip_xyz[2])
    #print('response_amp_dip_xyz[z]: ', response_amp_dip_xyz[2])

    if 'optrot' in prop_list:
        optrot = {}
        response_amp_angmom_xyz = [] 
        rhs_vec_angmom_xyz = []
        for x in range(3):
            dip_up = np.zeros((len(op)))
            dip_down = np.zeros((len(op)))
            rhs_vec = np.zeros((2 * len(op)))
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
        print('rhs_vec_dip_xyz: ', rhs_vec_dip_xyz[0])
        print('response_amp_dip_xyz: ',  response_amp_dip_xyz[0])
        print('rhs_vec_angmom_xyz: ', rhs_vec_angmom_xyz[0])
        print('response_amp_angmom_xyz: ',  response_amp_angmom_xyz[0])
        



if __name__== "__main__":
    test(['polar', 'optrot'])
