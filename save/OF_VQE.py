import scipy
import openfermion
import openfermionpsi4
import os
import numpy as np
import copy
<<<<<<< HEAD:OF_VQE.py
import random
import logging
import argparse
import math
from Classical_Amps import Harvest_CCSD_Amps

#Parse Initial Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--log', type=str, default='app.log', help='Output File for logging.')
parser.add_argument('-m', '--mode', type=str, default='DEBUG', help='Mode for logging.')
parser.add_argument('-rw', '--readwrite', type=str, default='w', help='Obliterate existing log file?')
parser.add_argument('-s', '--seed', type=int, default=111596, help='Random seed.')
parser.add_argument('-p', '--protocol', type=str, default='random', help='Protocol for ordering terms in Trotter approximation.')
parser.add_argument('-sys', '--system', type=str, default='H2O', help='Which chemical system listed do you want to solve?')
parser.add_argument('-d', '--dis', type=float, default=0.0, help='Dissociation parameter in angstroms')
parser.add_argument('-cc', '--ccfit', type=bool, default=False, help='Use CCSD amplitudes as initial thetas?')
parser.add_argument('-b', '--basis', type=str, default='sto-3g', help='Basis set')
parser.add_argument('-f', '--filter', type=bool, default=False, help='Use Hamiltonian as a criteria for removing irrelevant terms')
parser.add_argument('-c', '--config', type=str, default='ijab', help='What types of transitions do we allow?')
args = parser.parse_args()

#Seed RNG
random.seed(args.seed)

#Logging Preferences
logging.basicConfig(filename=args.log, filemode=args.readwrite, format='%(message)s')
logging.getLogger().setLevel(eval('logging.%s' %args.mode))
logging.debug(args)

#Manually initialize stateSS
basis = args.basis
r = args.dis
if args.system == 'H2O':
    multiplicity = 1
    ry = .763239
    rz = .477047
    angle = math.atan(rz/ry)
    y = abs(r*math.sin(angle))
    z = abs(r*math.cos(angle))
    geometry = [('H', (0, ry+y, -rz-z)), ('H', (0, -ry-y, -rz-z)), ('O', (0, 0, 0.119262))]
elif args.system == 'BeH2':
    multiplicity = 1
    rz = 1.4276
    geometry = [('H', (0, 0, -rz-r)), ('H', (0, 0, rz+r)), ('Be', (0, 0, 0))]
elif args.system == 'LiH':
    multiplicity = 1
    rz = 1.64
    geometry = [('H', (0, 0, rz+r)), ('Li', (0, 0, 0))]
elif args.system == 'H8':
    multiplicity = 1
    geometry = [('H', (0,0,0)), ('H', (.74+r,0,0)), ('H', (4.48,0,0)), ('H', (5.22+r,0,0)), ('H', (8.96,0,0)), ('H', (9.7+r,0,0)), ('H', (13.44, 0, 0)), ('H', (14.18+r, 0,0))]
elif args.system == 'H4':
    multiplicity = 1
    geometry = [('H', (0,0,0)), ('H', (.74+r,0,0)), ('H', (4.48,0,0)), ('H', (5.22+r,0,0))]
elif args.system == 'H2':
    multiplicity = 1
    geometry = [('H', (0,0,0)), ('H', (.74+r,0,0))]
else:
   logging.critical('Unsupported system.')


molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
molecule.filename = args.system
molecule = openfermionpsi4.run_psi4(molecule, run_scf = 1, run_mp2 = 1, run_ccsd = 1, run_fci=1, delete_input=False, delete_output=False)
molecule = Harvest_CCSD_Amps(molecule)
logging.debug('Molecule: '+str(geometry))
logging.debug('Qubits: '+str(molecule.n_qubits))
logging.debug('Spin-Orbitals: '+str(molecule.n_orbitals*2))
=======
import random 

#Manually initialize state
basis = 'sto-3g'
multiplicity = 1
geometry = [('H', (0,0,1.5)),('H', (0, 0, 3)), ('H', (0,0,4.5)), ('H', (0, 0, 6))]
geometry = [('H', (0,0,1.5)),('H', (0, 0, 3)), ('H', (0,0,4.5)), ('H', (0, 0, 6)), ('H', (0, 0, 7.5)), ('H', (0, 0, 9))]
r1 = 1.5
geometry = [('H', (0,0,1*r1)), ('H', (0,0,2*r1)), ('H', (0,0,3*r1)), ('H', (0,0,4*r1)), ('H', (0,0,5*r1)), ('H', (0,0,6*r1)), ('H', (0,0,7*r1)), ('H', (0,0,8*r1))]
molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
molecule = openfermionpsi4.run_psi4(molecule, run_scf = 1, run_mp2=1, run_cisd=1, run_ccsd = 1, run_fci=1, delete_input=0)
>>>>>>> packaging:save/OF_VQE.py
n_spinorbitals = int(molecule.n_orbitals*2)
print('HF energy      %20.16f au' %(molecule.hf_energy))
print('MP2 energy     %20.16f au' %(molecule.mp2_energy))
print('CISD energy    %20.16f au' %(molecule.cisd_energy))
print('CCSD energy    %20.16f au' %(molecule.ccsd_energy))
print('FCI energy     %20.16f au' %(molecule.fci_energy))

global global_der 
global global_energy  
global global_iter  
global_der = np.array([])
global_energy = 0.0 
global_iter = 0 

#Build p-h reference and map it to JW transform
reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
reference_bra = reference_ket.transpose().conj()

#JW transform Hamiltonian computed classically with OFPsi4
tensor_hamiltonian = molecule.get_molecular_hamiltonian()
singles_hamiltonian = tensor_hamiltonian.one_body_tensor
doubles_hamiltonian = tensor_hamiltonian.two_body_tensor
hamiltonian = openfermion.transforms.get_sparse_operator(tensor_hamiltonian)

#Thetas
parameters = []

#Second_quantized operations (not Jordan-Wignered)
SQ_CC_ops = []

def ops_pqrs():

    #Doubles
    pairs = []
    for i in range(0, n_spinorbitals):
        for j in range(i+1, n_spinorbitals):
            pairs.append([i,j])
    for pair1 in range(0, len(pairs)):
        for pair2 in range(pair1, len(pairs)):
            a, b = pairs[pair2]
            i, j = pairs[pair1]
            two_elec = openfermion.FermionOperator(((b,1),(a,1),(j,0),(i,0)))-openfermion.FermionOperator(((i,1),(j,1),(a,0),(b,0)))
            if args.filter==False or abs(doubles_hamiltonian[j][i][a][b])>1e-8 or abs(doubles_hamiltonian[b][a][i][j])>1e-8:
                if args.ccfit == True:
                    parameters.append(molecule.ccsd_double_amps[b][a][j][i])
                else:
                    parameters.append(0)
                SQ_CC_ops.append(two_elec)
    #Singles
    for i in range(0, n_spinorbitals):
        for a in range(i, n_spinorbitals):
            one_elec = openfermion.FermionOperator(((a,1),(i,0)))-openfermion.FermionOperator(((i,1),(a,0)))
            if args.filter==False or abs(singles_hamiltonian[a][i])>1e-8 or abs(singles_hamiltonian[i][a])>1e-8:
                if args.ccfit == True:
                    parameters.append(molecule.ccsd_single_amps[a][i])
                else:
                    parameters.append(0)
                SQ_CC_ops.append(one_elec)
    return(SQ_CC_ops, parameters)

def ops_ijab():
    #Form alpha_occs, beta_occs, alpha_noccs, beta_noccs
    orbitals = range(0, n_spinorbitals)
    a_occs = [k for k in orbitals if k%2==0 and k<molecule.n_electrons]
    b_occs = [k for k in orbitals if k%2==1 and k<molecule.n_electrons]
    a_noccs = [k for k in orbitals if k%2==0 and k>=molecule.n_electrons]
    b_noccs = [k for k in orbitals if k%2==1 and k>=molecule.n_electrons]

    #aa singles
    for i in a_occs:
        for a in a_noccs:
            one_elec = openfermion.FermionOperator(((a,1),(i,0)))-openfermion.FermionOperator(((i,1),(a,0)))
            if args.filter==False or abs(singles_hamiltonian[i][a])>1e-8 or abs(singles_hamiltonian[a][i])>1e-8:
                if args.ccfit == True:
                    parameters.append(molecule.ccsd_single_amps[a][i])
                else:
                    parameters.append(0)
                SQ_CC_ops.append(one_elec)

    #bb singles
    for i in b_occs:
        for a in b_noccs:
            one_elec = openfermion.FermionOperator(((a,1),(i,0)))-openfermion.FermionOperator(((i,1),(a,0)))
            if args.filter==False or abs(singles_hamiltonian[i][a])>1e-8 or abs(singles_hamiltonian[a][i])>1e-8:
                if args.ccfit == True:
                    parameters.append(molecule.ccsd_single_amps[a][i])
                else:
                    parameters.append(0)
                SQ_CC_ops.append(one_elec)

    ##aa doubles
    for i in a_occs:
        for j in [k for k in a_occs if k>i]:
            for a in a_noccs:
                for b in [k for k in a_noccs if k>a]:
                    two_elec = openfermion.FermionOperator(((b,1),(a,1),(j,0),(i,0)))-openfermion.FermionOperator(((i,1),(j,1),(a,0),(b,0)))
                    if args.filter==False or abs(doubles_hamiltonian[j][i][a][b])>1e-8 or abs(doubles_hamiltonian[b][a][i][j])>1e-8:
                        if args.ccfit == True:
                            parameters.append(molecule.ccsd_double_amps[b][a][j][i])
                        else:
                            parameters.append(0)
                        SQ_CC_ops.append(two_elec)

    ##bb doubles
    for i in b_occs:
        for j in [k for k in b_occs if k>i]:
            for a in b_noccs:
                for b in [k for k in b_noccs if k>a]:
                    two_elec = openfermion.FermionOperator(((b,1),(a,1),(j,0),(i,0)))-openfermion.FermionOperator(((i,1),(j,1),(a,0),(b,0)))
                    if args.filter==False or abs(doubles_hamiltonian[j][i][a][b])>1e-8 or abs(doubles_hamiltonian[b][a][i][j])>1e-8:
                        if args.ccfit == True:
                            parameters.append(molecule.ccsd_double_amps[b][a][j][i])
                        else:
                            parameters.append(0)
                        SQ_CC_ops.append(two_elec)

    #ab doubles
    for i in a_occs:
        for j in b_occs:
            for a in a_noccs:
                for b in b_noccs:
                    two_elec = openfermion.FermionOperator(((b,1),(a,1),(j,0),(i,0)))-openfermion.FermionOperator(((i,1),(j,1),(a,0),(b,0)))
                    if args.filter==False or abs(doubles_hamiltonian[j][i][a][b])>1e-8 or abs(doubles_hamiltonian[b][a][i][j])>1e-8:
                        if args.ccfit == True:
                            parameters.append(molecule.ccsd_double_amps[b][a][j][i])
                        else:
                            parameters.append(0)
                        SQ_CC_ops.append(two_elec)

    return (SQ_CC_ops, parameters)

if args.config == 'ijab':
    SQ_CC_ops, parameters = ops_ijab()
if args.config == 'pqrs':
    SQ_CC_ops, parameters = ops_pqrs()

print(" Number of parameters: ", len(parameters))
#Jordan_Wigners into the Pauli Matrices, then computes their products as sparse matrices.
JW_CC_ops = []
for classical_op in SQ_CC_ops:
    JW_CC_ops.append(openfermion.transforms.get_sparse_operator(classical_op, n_qubits = molecule.n_qubits))

#Commutator Evaluation Functions:
ham_ket = hamiltonian.dot(reference_ket)

#[e^A,H]
def Commutator(op_no):
    op = JW_CC_ops[op_no]
    bra = ham_ket.transpose()
    ket = scipy.sparse.linalg.expm_multiply(op, reference_ket)
    comm = bra.dot(ket)
    bra = (scipy.sparse.linalg.expm_multiply(-op, reference_ket)).transpose().conj()
    ket = ham_ket
    comm = comm-bra.dot(ket)
    return abs(comm.toarray()[0][0])

#[A,H]
def Commutator2(op_no):
    op = JW_CC_ops[op_no]
    bra = ham_ket.transpose()
    ket = op.dot(reference_ket)
    comm = bra.dot(ket)
    bra = reference_bra.dot(op)
    ket = ham_ket
    comm = comm-bra.dot(ket)
    return abs(comm.toarray()[0][0])


#Special Orders of Interest:
if args.protocol == 'random':
    op_indices = []
    for i in range(0, len(JW_CC_ops)):
        op_indices.append(i)
    random.shuffle(op_indices)
    new_ops = []
    new_parameters = []
    for i in op_indices:
        new_ops.append(JW_CC_ops[i])
        new_parameters.append(parameters[i])
    logging.debug(op_indices)
    JW_CC_ops=new_ops
    parameters=new_parameters

if args.protocol == 'increasing_comms':
    op_indices = []
    for i in range(0, len(JW_CC_ops)):
        op_indices.append(i)
    op_indices = sorted(op_indices, key=Commutator)
    new_ops = []
    new_parameters = []
    for i in op_indices:
        new_ops.append(JW_CC_ops[i])
        new_parameters.append(parameters[i])
    logging.info('Increasing Commutators:')
    logging.debug(op_indices)
    JW_CC_ops=new_ops
    parameters = new_parameters

elif args.protocol == 'decreasing_comms':
    op_indices = []
    for i in range(0, len(JW_CC_ops)):
        op_indices.append(i)
    op_indices = sorted(op_indices, key=Commutator, reverse=True)
    new_ops = []
    new_parameters = []
    for i in op_indices:
        new_ops.append(JW_CC_ops[i])
        new_parameters.append(parameters[i])
    logging.info('Decreasing Commutators:')
    logging.debug(op_indices)
    JW_CC_ops=new_ops
    parameters = new_parameters

elif args.protocol == 'increasing_unexp_comms':
    logging.info('Correlation: '+str(molecule.fci_energy-molecule.hf_energy))
    op_indices = []
    new_parameters = []
    for i in range(0, len(JW_CC_ops)):
        op_indices.append(i)
    op_indices = sorted(op_indices, key=Commutator2)
    new_ops = []
    for i in op_indices:
        new_ops.append(JW_CC_ops[i])
        new_parameters.append(parameters[i])
    logging.info('Increasing Unexponentiated Commutators:')
    logging.debug(op_indices)
    JW_CC_ops=new_ops
    parameters = new_parameters

elif args.protocol == 'decreasing_unexp_comms':
    op_indices = []
    for i in range(0, len(JW_CC_ops)):
        op_indices.append(i)
    op_indices = sorted(op_indices, key=Commutator2, reverse=True)
    new_ops = []
    new_parameters = []
    for i in op_indices:
        new_ops.append(JW_CC_ops[i])
        new_parameters.append(parameters[i])
    logging.info('Decreasing Unexponentiated Commutators:')
    logging.debug(op_indices)
    parameters = new_parameters
    JW_CC_ops=new_ops

logging.debug(str(len(parameters))+' parameters.')
'''
SPE based on a traditional, untrotterized ansatz
v'=exp(a+b+...+n)v
'''
def SPE(parameters):
    generator = scipy.sparse.csc_matrix((2**(molecule.n_qubits), 2**(molecule.n_qubits)), dtype = complex)
    for mat_op in range(0,len(JW_CC_ops)):
        generator = generator+parameters[mat_op]*JW_CC_ops[mat_op]
    new_state = scipy.sparse.linalg.expm_multiply(generator, reference_ket)
    new_bra = new_state.transpose().conj()
    assert(new_bra.dot(new_state).toarray()[0][0]-1<0.0000001)
    energy = new_bra.dot(hamiltonian.dot(new_state))
    return energy.toarray()[0][0].real

'''
SPE based on full, 1st-order Trotter decomposition
v'=exp(n)exp(n-1)...exp(a)v
'''
def Trotter_SPE(parameters):
    global global_energy  
    new_state = reference_ket
    for k in reversed(range(0, len(parameters))):
        new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*JW_CC_ops[k]), new_state)
    new_bra = new_state.transpose().conj()
    assert(new_bra.dot(new_state).toarray()[0][0]-1<0.0000001)
    energy = new_bra.dot(hamiltonian.dot(new_state))
    global_energy = energy.toarray()[0][0].real
    assert(global_energy.imag <  1e-14)
    global_energy = global_energy.real
    return global_energy 

<<<<<<< HEAD:OF_VQE.py
=======

                 
                
#Numerical trotterized gradient
def Numerical_Trot_Grad(parameters):
    global global_der
    step_size = 1e-6
    grad = []
    for k in reversed(range(0, len(parameters))):
        para = copy.copy(parameters)
        para[k]+=step_size
        diff = Trotter_SPE(para)
        para[k]-=2*step_size
        diff -= Trotter_SPE(para)
        grad.append(diff/(step_size*2))
    global_der = np.asarray(grad)
    return np.asarray(grad)

def Five_Point_Grad(parameters):
    grad = []
    for k in reversed(range(0, len(parameters))):
        forw = copy.copy(parameters)
        forw2 = copy.copy(parameters)
        reve = copy.copy(parameters)
        reve2 = copy.copy(parameters)
        forw[k]+=1e-7
        forw2[k]+=2e-7
        reve[k]-=1e-7
        reve2[k]-=2e-7
        f2 = Trotter_SPE(forw2)
        f1 = Trotter_SPE(forw)
        r1 = Trotter_SPE(reve)
        r2 = Trotter_SPE(reve2)
        diff = (-f2+8*f1-8*r1+r2)/(1.2e-6)
        grad.append(diff)
    return np.asarray(grad)

>>>>>>> packaging:save/OF_VQE.py
#Analytical trotter gradient
def Trotter_Gradient(parameters):
    global global_der 
    grad = []
    new_state = copy.copy(reference_ket)
    for k in reversed(range(0, len(parameters))):
        new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*JW_CC_ops[k]), new_state)
    new_bra = new_state.transpose().conj()
    hbra = new_bra.dot(hamiltonian)
    term = 0
    ket = copy.copy(new_state)
    grad = Recurse(parameters, grad, hbra, ket, term)
    global_der = grad
    return np.asarray(grad)

#Recursive component of analytical trotter gradient
def Recurse(parameters, grad, hbra, ket, term):
    if term == 0:
        hbra = hbra
        ket = ket
    else:
        hbra = (scipy.sparse.linalg.expm_multiply(-JW_CC_ops[term-1]*parameters[term-1], hbra.transpose().conj())).transpose().conj()
        ket = scipy.sparse.linalg.expm_multiply(-JW_CC_ops[term-1]*parameters[term-1], ket)
    grad.append((2*hbra.dot(JW_CC_ops[term]).dot(ket).toarray()[0][0].real))
    if term<len(parameters)-1:
        term += 1
        Recurse(parameters, grad, hbra, ket, term)
    return np.asarray(grad)

#Callback Function
def callback(parameters):
<<<<<<< HEAD:OF_VQE.py
    global iterations
    logging.debug(Trotter_SPE(parameters))
    iterations+= 1

logging.debug('HF = '+str(molecule.hf_energy))
logging.debug('CCSD = '+str(molecule.ccsd_energy))
logging.debug('FCI = '+str(molecule.fci_energy))
logging.debug('Optimizing:')
=======
    global global_der 
    global global_energy  
    global global_iter 
    err = np.sqrt(np.vdot(global_der, global_der))
    print(" Iter:%4i Current Energy = %20.16f Gradient Norm %10.1e Gradient Max %10.1e" %(global_iter,
        global_energy, err, np.max(np.abs(global_der))))
    global_iter += 1


#for p in range(len(parameters)):
#    parameters[p] = (random.random()-.5)*.001

#der_num = Numerical_Trot_Grad(parameters)
#der_ana = Trotter_Gradient(parameters)
#print(" Numerical: ")
#print(der_num)
#print("\n Analytical: ")
#print(der_ana)
#print("\n Error: ")
#print(np.linalg.norm(der_num-der_ana))
>>>>>>> packaging:save/OF_VQE.py

global iterations
iterations = 1
optimization = scipy.optimize.minimize(Trotter_SPE, parameters, jac=Trotter_Gradient, options = {'gtol': 1e-5, 'disp': True}, method = 'BFGS', callback=callback)
logging.debug('Converged in '+str(iterations)+ ' iterations.')
logging.info(Trotter_SPE(optimization.x)-molecule.fci_energy)
