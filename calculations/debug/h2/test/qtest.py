from qiskit_nature.drivers import PySCFDriver, UnitsType, Molecule
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit_nature.circuit.library import HartreeFock
import sys
#sys.path.append('/home/akumar1/trans_H_calcs')
#from quccsd import QUCCSD
from qiskit import BasicAer
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import QasmSimulator
from qiskit.algorithms import VQE # qiskit_nature has VQEUCCFactory --> need to check this!
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import VQEUCCFactory
from qiskit_nature.algorithms import GroundStateEigensolver, QEOM
from qiskit_nature.transformers import FreezeCoreTransformer
from qiskit_nature.circuit.library.ansatzes import UCCSD

import numpy as np
import scipy

#from qiskit.algorithms import NumPyMinimumEigensolver, NumPyEigensolver

import logging
logging.basicConfig(filename='7_debug.log', level=logging.DEBUG)

molecule = Molecule(geometry = [['H', [0.000000000000,   0.000000000000,   0.000000000000]],
                               ['H',  [0.000000000000,   0.000000000000,   0.7]]],
                               charge=0, multiplicity=1)

driver = PySCFDriver(molecule = molecule, unit=UnitsType.ANGSTROM, basis='631-g')
remove_list = []
fcore = False #frozen core or not
freezeCoreTransformer = FreezeCoreTransformer(freeze_core=fcore, remove_orbitals=remove_list)
es_problem = ElectronicStructureProblem(driver, [freezeCoreTransformer])
qi = QuantumInstance(Aer.get_backend('statevector_simulator'))
include_custom = False

num_spin_orbitals = 4
num_particles = [1,1]

qubit_converter = QubitConverter(JordanWignerMapper())

optimizer = L_BFGS_B()
init_state = HartreeFock(num_spin_orbitals, num_particles, qubit_converter)
ansatz = UCCSD(qubit_converter, num_particles, num_spin_orbitals, initial_state=init_state)

#store the intermediate result
counts = []
values = []
def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)

#solver = VQE(ansatz, optimizer, callback=store_intermediate_result,include_custom=include_custom,quantum_instance=qi)

solver = VQEUCCFactory(qi)

gsc = GroundStateEigensolver(qubit_converter, solver)
#res = gsc.solve(es_problem)

##########################################
'''
# Trying to make some measurements
# VQE energy!

from qiskit.opflow import StateFn
second_q_op = es_problem.second_q_ops()
qubit_op = qubit_converter.convert(second_q_op[0])
measurement = StateFn(qubit_op).adjoint()
print(measurement.eval(solver.get_optimal_circuit()))
# the above lines work, so now need to play with my Hamiltonian
# to evaluate single and double commutators
excitations = ansatz.operators
print('excitations: ', excitations)
new_qubit_op = (qubit_op @ excitations[0]) - (excitations[0] @ qubit_op)
measurement = StateFn(new_qubit_op).adjoint()
print(measurement.eval(solver.get_optimal_circuit()))

# I need to get q-eom excitation operators instead!
'''
#####################################################
from typing import List, Union, Optional, Tuple, Dict, cast

qeom_es = QEOM(gsc, 'sd')
#qeom_results = qeom_es.solve(es_problem)
#print(qeom_results)

gs_result = qeom_es._gsc.solve(es_problem)
qeom_es._untapered_qubit_op_main = qeom_es._gsc._qubit_converter.map(es_problem.second_q_ops()[0])

data = es_problem.hopping_qeom_ops(qubit_converter, 'sd')

hopping_operators, type_of_commutativities, excitation_indices = data
#print(hopping_operators)
size = int(len(list(excitation_indices.keys())) // 2)

eom_matrix_operators = qeom_es._build_all_commutators(
            hopping_operators, type_of_commutativities, size)

print('eom_matrix_operators: ', eom_matrix_operators)

measurement_results = qeom_es._gsc.evaluate_operators(
            gs_result.eigenstates[0], eom_matrix_operators)

measurement_results = cast(Dict[str, List[float]], measurement_results)

m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std = qeom_es._build_eom_matrices(measurement_results, size)

energy_gaps, expansion_coefs = qeom_es._compute_excitation_energies(m_mat, v_mat, q_mat, w_mat)

#print(energy_gaps)
#print(m_mat)

print("m matrix\n",m_mat)

Hmat=np.bmat([[m_mat,q_mat],[q_mat.conj(),m_mat.conj()]])
S=np.bmat([[v_mat,w_mat],[-w_mat.conj(),-v_mat.conj()]])
#Diagonalize ex operator-> eigenvalues are excitation energies
eig,aval=scipy.linalg.eig(Hmat,S)
#ex_energies =  27.2114 * np.sort(eig.real)
#ex_energies = np.array([i for i in ex_energies if i >0])
#print('final excitation energies in eV: ', ex_energies)
ex_energies_hartree =  1.0 * np.sort(eig.real)
ex_energies_hartree = np.array([i for i in ex_energies_hartree if i >0])
print('final excitation energies in Hartree: ', ex_energies_hartree)
