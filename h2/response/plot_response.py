import pickle
import matplotlib.pyplot as plt
import numpy as np

dist = np.arange(0.2,2.50,0.1)
input_ = open('h2_fci_sos.dat', 'rb')
results_fci = pickle.load(input_) 
input_ = open('h2_qeom.dat', 'rb')
results_qeom = pickle.load(input_) 
input_ = open('h2_scqeom.dat', 'rb')
results_scqeom = pickle.load(input_) 

iso_polar_fci = []
iso_polar_qeom = []
iso_polar_scqeom = []

# isotropic polarizabilities 
for i in range(len(dist)):
    iso_polar_fci.append(results_fci[i]['isotropic_polarizability'])
    iso_polar_qeom.append(results_qeom[i]['isotropic_polarizability'])
    iso_polar_scqeom.append(results_scqeom[i]['isotropic_polarizability'])

fig, ax = plt.subplots()
#color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12
ax.plot(dist, iso_polar_fci, label="FCI", color='grey', linestyle='solid', linewidth=1.5)
ax.plot(dist, iso_polar_qeom, label="qEOM", marker = 'x', color = 'red',  linestyle='None')
ax.plot(dist, iso_polar_scqeom, label="SC-qEOM", marker = '.', color = 'blue',  linestyle='None')
plt.xlabel('Inter-hydrogen distance ($\mathbf{\AA}$)', fontsize=17, fontweight='bold')
plt.ylabel("Isotropic polarizability (au)", fontsize=15, fontweight='bold')
ax.tick_params(axis='y', which='major', labelsize=16)
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)
ax.legend(loc='upper left', fontsize=13)
fig.savefig('h2_polarizability_sto_3g.pdf', dpi=600, bbox_inches="tight")

dist = np.arange(0.2,2.70,0.1)
input_ = open('h2_2_fci_sos.dat', 'rb')
results_fci = pickle.load(input_) 
trace_optrot_fci = []
for i in range(len(dist)):
    trace_optrot_fci.append(results_fci[i]['trace_rotation(589nm)'])
fig, ax = plt.subplots()
ax.plot(dist[6:], trace_optrot_fci[6:], label="FCI", color='grey', linestyle='solid', linewidth=1.5)
plt.xlabel('Inter-hydrogen distance ($\mathbf{\AA}$)', fontsize=17, fontweight='bold')
plt.ylabel("Trace optical rotation(au)", fontsize=15, fontweight='bold')
ax.tick_params(axis='y', which='major', labelsize=16)
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)
ax.legend(loc='upper left', fontsize=13)
fig.savefig('h2_2_optrot_sto_3g.pdf', dpi=600, bbox_inches="tight")
print(trace_optrot_fci)
