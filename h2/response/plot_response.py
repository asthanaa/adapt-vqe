import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

#mol_name = 'h2'
#mol_name = 'lih'
mol_name = str(sys.argv[1])

dist = np.arange(0.2,2.50,0.1)
dist = np.arange(1.0,2.80,0.1)
dist = np.arange(1.5,3.70,0.1)
dist = np.arange(1.5,3.70,0.1)
dist = np.arange(2.2,4.2,0.025)
#input_ = open('h2_fci_sos.dat', 'rb')
input_ = open(mol_name + '_fci_sos.dat', 'rb')
results_fci = pickle.load(input_) 
#input_ = open('h2_qeom.dat', 'rb')
input_ = open(mol_name + '_qeom.dat', 'rb')
results_qeom = pickle.load(input_) 
#input_ = open('h2_scqeom.dat', 'rb')
#input_ = open(mol_name + '_scqeom.dat', 'rb')
#results_scqeom = pickle.load(input_) 
#input_ = open('h2_qse.dat', 'rb')
#input_ = open(mol_name + '_qse.dat', 'rb')
#results_qse = pickle.load(input_) 

iso_polar_fci = []
iso_polar_qeom = []
iso_polar_scqeom = []
iso_polar_qse = []

# isotropic polarizabilities 
for i in range(len(dist)):
    iso_polar_fci.append(results_fci[i]['isotropic_polarizability'])
    iso_polar_qeom.append(results_qeom[i]['isotropic_polarizability'])
    #iso_polar_scqeom.append(results_scqeom[i]['isotropic_polarizability'])
    #iso_polar_qse.append(results_qse[i]['isotropic_polarizability'])

print(list(zip(dist,iso_polar_fci)))
print(list(zip(dist,iso_polar_qeom)))

'''
fig, ax = plt.subplots()
#color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12
ax.plot(dist, iso_polar_fci, label="FCI", color='grey', linestyle='solid', linewidth=1.5)
#ax.plot(dist, iso_polar_fci, label="FCI", color='grey', marker = 'o', markersize = 0.5, linestyle='None')
ax.plot(dist, iso_polar_qeom, label="qEOM", marker = 'x', color = 'red',  linestyle='None', markersize=3.5)
#ax.plot(dist, iso_polar_scqeom, label="VQE-EOM", marker = '.', color = 'blue',  linestyle='None')
#ax.plot(dist, iso_polar_qse, label="QSE", marker = '.', color = 'orange',  linestyle='None')
plt.xlabel('Li-H bond length ($\mathbf{\AA}$)', fontsize=17, fontweight='bold')
plt.ylabel("Isotropic polarizability (au)", fontsize=15, fontweight='bold')
#ax.set_ylim([-5000, 8000])
ax.tick_params(axis='y', which='major', labelsize=16)
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)
ax.legend(loc='upper left', fontsize=9)
name_fig = mol_name + '_polarizability_sto_3g_fci_qeom.pdf'
#fig.savefig('h2_polarizability_sto_3g.pdf', dpi=600, bbox_inches="tight")
fig.savefig(name_fig, dpi=600, bbox_inches="tight")
'''

'''
dist = np.arange(0.2,2.70,0.1)
input_ = open('h2_2_fci_sos.dat', 'rb')
results_fci = pickle.load(input_) 
trace_optrot_fci = []
for i in range(len(dist)):
    #trace_optrot_fci.append(results_fci[i]['trace_rotation(589nm)'])
    trace_optrot_fci.append(results_fci[i]['isotropic_polarizability'])
fig, ax = plt.subplots()
#ax.plot(dist[6:], trace_optrot_fci[6:], label="FCI", color='grey', linestyle='solid', linewidth=1.5)
ax.plot(dist, trace_optrot_fci, label="FCI", color='grey', linestyle='solid', linewidth=1.5)
plt.xlabel('Inter-hydrogen distance ($\mathbf{\AA}$)', fontsize=17, fontweight='bold')
plt.ylabel("Trace optical rotation(au)", fontsize=15, fontweight='bold')
ax.tick_params(axis='y', which='major', labelsize=16)
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)
ax.legend(loc='upper left', fontsize=13)
fig.savefig('h2_2_optrot_sto_3g.pdf', dpi=600, bbox_inches="tight")
print(trace_optrot_fci)
'''
