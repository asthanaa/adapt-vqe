import scipy
import vqe_methods
import vqeorbopt 
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

def test():
    r0 = 3.0 
    geometry = '''
    H 0 0 0
    H 0 0 {0}
    H 0 0 {1}
    H 0 0 {2}
    '''.format(r0, 2*r0, 3*r0)
    charge = 0
    spin = 0
    basis = 'sto3g'
    [n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis,n_frzn_occ=0,n_act=4)
    print(" n_orb: %4i" %n_orb)
    print(" n_a  : %4i" %n_a)
    print(" n_b  : %4i" %n_b)
    sq_ham = pyscf_helper.SQ_Hamiltonian()
    sq_ham.init(h, g, C, S)
    print(" HF Energy: %12.8f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))
    fermi_ham  = sq_ham.export_FermionOperator()
    hamiltonian = openfermion.transforms.get_sparse_operator(fermi_ham)
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
    #change operators to only doubles later on
    pool = operator_pools.singlet_GSD()
    pool.init(n_orb, n_occ_a=n_a, n_occ_b=n_b, n_vir_a=n_orb-n_a, n_vir_b=n_orb-n_b)
    for i in range(pool.n_ops):
        print(pool.get_string_for_term(pool.fermi_ops[i]))
    #[e,v,params,ansatz_mat] = vqe_methods.adapt_vqe(fermi_ham, pool, reference_ket, theta_thresh=1e-9)
    [e,v,params,ansatz_mat] = vqeorbopt.ooadapt(n_orb,n_a,n_b,n_orb-n_a,n_orb-n_b,fermi_ham, pool, reference_ket, theta_thresh=1e-9, adapt_thresh=1e-9)


    print(" Final ADAPT-VQE energy: %12.8f" %e)
    print(" <S^2> of final state  : %12.8f" %(v.conj().T.dot(s2.dot(v))[0,0].real))
    
    
    #finished ground state calculation
    exit()


    #store FCI values for checking
    #a,b=eigs(hamiltonian)
    #fci_levels=a+E_nuc

    #create operators single and double for each excitation
    op=qeom.createops_basic(n_orb,n_a,n_b,n_orb-n_a,n_orb-n_b,reference_ket)
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
        factor=1

        if i in (0,2,4,9,12,13): factor=-1
        for j in range(len(op)):
            factorj=factor
            if j in (0,2,4,9,12,13): factorj=factorj*-1
            print("state",i,j,op[i].dot(reference_ket),op[j].dot(reference_ket),factorj)
            #mat=op[i].transpose().conj().dot(barH.dot(op[j]))
	    #phase corrected
            #mat=qeom.comm3(op[i].transpose().conj(),barH,op[j])
            #phase un corrected
            mat=factorj*qeom.comm3(op[i].transpose().conj(),barH,op[j])
            #print(mat.toarray())
            Hmat[i,j]=qeom.expvalue(reference_ket.transpose().conj(),mat,reference_ket)[0,0].real
            mat3=qeom.comm2(op[i].transpose().conj(),op[j])
            V[i,j]=qeom.expvalue(reference_ket.transpose().conj(),mat3,reference_ket)[0,0]
    #Diagonalize ex operator-> eigenvalues are excitation energies
    eig,aval=scipy.linalg.eig(Hmat)
    print('M',Hmat)
    print('final excited states',np.sort(eig.real)+e)
    print('final excitation energies',np.sort(eig.real))
if __name__== "__main__":
    test()
