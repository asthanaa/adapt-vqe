import numpy as np
from qiskit.aqua.operators import WeightedPauliOperator as op
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli
import openfermion
import openfermionpsi4
import operator_pools
from openfermion import *
import random
import re
from qiskit import transpile
from qiskit.providers.aer import Aer

r = 0.7
geometry = [('H', (0,0,0)), ('H', (0,0,r))]

basis = "sto-3g"
multiplicity = 1

molecule = openfermion.chem.MolecularData(geometry, basis, multiplicity)
print("here")
molecule = openfermionpsi4.run_psi4(molecule,
            run_scf=1, 
            run_mp2=1, 
            run_cisd=0, 
            run_ccsd=0, 
            run_fci=1
            #delete_input=1
            )
print("psi4 complete")
print(molecule.get_molecular_hamiltonian())
pool = operator_pools.singlet_SD_JW()
pool.init(molecule)

qc = QuantumCircuit(pool.n_spin_orb)

q = QuantumRegister(pool.n_spin_orb, name='q')

# Import ADAPT ansatz 

f = open("h2.out", "r")
searchlines = f.readlines()
f.close()
ops = []
for i, line in enumerate(searchlines):  
    if "Add operator" in line:
        ind = int(line.split()[2])
        A = pool.fermi_ops[ind]
        #A = openfermion.transforms.jordan_wigner(A[0])
        sum = []
        for term in A.terms:
            zz = 0
            coeff = np.imag(A.terms[term])
            if coeff > 0:
                sign = '+'
            if coeff < 0:
                sign = '-'
            term = str(term)
            Bin = np.zeros((2 * pool.n_spin_orb,), dtype=int)
            X_pat = re.compile("(\d{,2}), 'X'")
            X = X_pat.findall(term)
            if X:
                for i in X:
                    i = int(i)
                    Bin[pool.n_spin_orb + i] = 1
            Y_pat = re.compile("(\d{,2}), 'Y'")
            Y = Y_pat.findall(term)
            if Y:
                for i in Y:
                    i = int(i)
                    Bin[pool.n_spin_orb + i] = 1
                    Bin[i] = 1
            Z_pat = re.compile("(\d{,2}), 'Z'")
            Z = Z_pat.findall(term)
            if Z:
                for i in Z:
                    i= int(i)
                    Bin[i] = 1
            string = ''
            for k in range(pool.n_spin_orb):
                if Bin[k] == 0:
                    if Bin[k + pool.n_spin_orb] == 1:
                        string += 'X'
                    if Bin[k + pool.n_spin_orb] == 0:
                        string += 'I'
                if Bin[k] == 1:
                    if Bin[k + pool.n_spin_orb] == 1:
                        string += 'Y'
                    if Bin[k + pool.n_spin_orb] == 0:
                        string += 'Z'
            p = Pauli.from_label(string)
            sum.append([-1*random.random(), p])
        ops.append(op(paulis=sum))

circuit_list = [
    A.evolve(state_in=None,
              evo_time=1,
              num_time_slices=1,
              quantum_registers=q,
              expansion_mode='trotter',
              expansion_order=1)
    for A in ops
]

for circ in circuit_list:
    final_circ = transpile(circ, 
                           basis_gates=['u3', 'cx', 'id'], 
                           optimization_level=0, 
                           backend=Aer.get_backend('statevector_simulator')
                           )
    print(final_circ.count_ops())
    print(final_circ)

