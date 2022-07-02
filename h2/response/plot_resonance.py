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
    #iso_polar_qeom.append(results_qeom[i]['isotropic_polarizability'])
    #iso_polar_scqeom.append(results_scqeom[i]['isotropic_polarizability'])
    #iso_polar_qse.append(results_qse[i]['isotropic_polarizability'])

#print(list(zip(dist,iso_polar_fci)))
#print(list(zip(dist,iso_polar_qeom)))


# first part!
input_ = open(mol_name + '_fci_sos_1.dat', 'rb')
results_fci_1 = pickle.load(input_) 
print(len(results_fci_1))
fci_1 = []
dist_1 = np.arange(2.2,2.651,0.025) 
size_1 = len(dist_1)
for i in range(len(dist_1)):
    fci_1.append(results_fci[i]['isotropic_polarizability'])
dist_1 = np.append(dist_1,np.arange(2.652, 2.677, 0.004))
dist_1 = np.append(dist_1,np.arange(2.67620, 2.67690, 0.0002))
print(len(dist_1)-size_1)
for i in range(len(dist_1)-size_1):
    fci_1.append(results_fci_1[i]['isotropic_polarizability'])
dist_1 = np.append(dist_1,2.67695)
fci_1.append(949882.1738246339)
#print(dist_1)
#print(fci_1)
print(list(zip(dist_1,fci_1)))

# second part!
input_ = open(mol_name + '_fci_sos_2.dat', 'rb')
results_fci_2 = pickle.load(input_) 

fci_2 = []
dist_2 = np.array(2.6770)
fci_2.append(-52481697.01020853)
dist_2 = np.append(dist_2, np.arange(2.6771,2.6800,0.0002))
for i in range(len(dist_2)-1):
    fci_2.append(results_fci_2[i]['isotropic_polarizability'])
size_2 = len(np.arange(2.7,3.41,0.025))
dist_2 = np.append(dist_2, np.arange(2.7,3.41,0.025))
#size_2 = len(dist_2)-1
for i in range(size_2):
    fci_2.append(results_fci[20 + i]['isotropic_polarizability'])

input_ = open(mol_name + '_fci_sos_22.dat', 'rb')
results_fci_2 = pickle.load(input_) 
temp = len(np.arange(3.40100,3.40701,0.001))
dist_2 = np.append(dist_2, np.arange(3.40100,3.40701,0.001))
dist_2 = np.append(dist_2, 3.40750)
for i in range(temp+1):
    fci_2.append(results_fci_2[i]['isotropic_polarizability'])
dist_2 = np.append(dist_2, 3.40805)
fci_2.append(-5719978.786997339)
#print(dist_2)
#print(fci_2)

# third part
input_ = open(mol_name + '_fci_sos_3.dat', 'rb')
results_fci_3 = pickle.load(input_) 
fci_3 = []
dist_3 = np.array(3.40808)
fci_3.append(13377009.898852158)
dist_3 = np.array(3.4085)
fci_3.append(280363.4743775823)

dist_3 = np.append(dist_3, np.arange(3.409,3.424,0.002))
for i in range(len(dist_3)-2):
    fci_3.append(results_fci_3[i]['isotropic_polarizability'])

#temp = len(np.arange(3.425,4.19,0.025))
temp = len(np.arange(3.425,4.01,0.025))
#dist_3 = np.append(dist_3, np.arange(3.425,4.19,0.025))
dist_3 = np.append(dist_3, np.arange(3.425,4.01,0.025))
#size_3 = len(dist_3) - 1
#print(size_1)
#print(size_2)
#print(size_3)
#print(size_1 + size_2 + size_3)
#print(len(results_fci))
for i in range(temp):
    fci_3.append(results_fci[49 + i]['isotropic_polarizability'])
#print(dist_3)
#print(fci_3)

qeom_1 = []
dist_1_qeom = np.arange(2.2,2.651,0.025)
#dist_1_qeom = np.append(dist_1_qeom,np.arange(2.652, 2.677, 0.008))
#dist_1_qeom = np.append(dist_1_qeom,np.arange(2.67620, 2.67690, 0.0002))
#dist_1_qeom = np.append(dist_1_qeom,2.67695)
input_ = open(mol_name + '_qeom_1.dat', 'rb')
qeom_results_1 = pickle.load(input_) 
for i in range(len(dist_1_qeom)):
    qeom_1.append(qeom_results_1[i]['isotropic_polarizability'])

print(list(zip(dist_1_qeom,qeom_1)))


scqeom_1 = []
input_ = open(mol_name + '_scqeom_1.dat', 'rb')
scqeom_results_1 = pickle.load(input_) 
for i in range(len(dist_1_qeom)):
    scqeom_1.append(scqeom_results_1[i]['isotropic_polarizability'])

print(list(zip(dist_1_qeom,scqeom_1)))
    
qse_1 = []
input_ = open(mol_name + '_qse_1.dat', 'rb')
qse_results_1 = pickle.load(input_) 
for i in range(len(dist_1_qeom)):
    qse_1.append(qse_results_1[i]['isotropic_polarizability'])

print(list(zip(dist_1_qeom,qse_1)))

qeom_2 = []
dist_2_qeom = np.array(2.70)
qeom_2.append(-2091.7656239700877)
dist_2_qeom = np.append(dist_2_qeom, np.arange(2.725,3.31,0.025))
#dist_1_qeom = np.append(dist_1_qeom,np.arange(2.652, 2.677, 0.008))
#dist_1_qeom = np.append(dist_1_qeom,np.arange(2.67620, 2.67690, 0.0002))
#dist_1_qeom = np.append(dist_1_qeom,2.67695)
input_ = open(mol_name + '_qeom_2.dat', 'rb')
qeom_results_2 = pickle.load(input_) 
for i in range(len(dist_2_qeom)-1):
    qeom_2.append(qeom_results_2[i]['isotropic_polarizability'])
dist_2_qeom = np.append(dist_2_qeom, np.arange(3.325,3.399,0.025))
qeom_2.append(-1310.7559284695355)
qeom_2.append(-1926.4437379743035)
qeom_2.append(-3484.2452517870524)
#qeom_2.append(-14739.172009680407)

print(list(zip(dist_2_qeom,qeom_2)))


scqeom_2 = []
input_ = open(mol_name + '_scqeom_2.dat', 'rb')
scqeom_results_2 = pickle.load(input_) 
for i in range(len(dist_2_qeom)):
    scqeom_2.append(scqeom_results_2[i]['isotropic_polarizability'])

qse_2 = []
input_ = open(mol_name + '_qse_2.dat', 'rb')
qse_results_2 = pickle.load(input_) 
for i in range(len(dist_2_qeom)):
    qse_2.append(qse_results_2[i]['isotropic_polarizability'])

#dist_3_qeom  = np.arange(3.425,4.19,0.025)
dist_3_qeom  = np.arange(3.425,4.01,0.025)
qeom_3 = []
input_ = open(mol_name + '_qeom_3.dat', 'rb')
qeom_results_3 = pickle.load(input_) 
for i in range(len(dist_3_qeom)):
    qeom_3.append(qeom_results_3[i]['isotropic_polarizability'])

scqeom_3 = []
input_ = open(mol_name + '_scqeom_3.dat', 'rb')
scqeom_results_3 = pickle.load(input_) 
for i in range(len(dist_3_qeom)):
    scqeom_3.append(scqeom_results_3[i]['isotropic_polarizability'])

qse_3 = []
input_ = open(mol_name + '_qse_3.dat', 'rb')
qse_results_3 = pickle.load(input_) 
for i in range(len(dist_3_qeom)):
    qse_3.append(qse_results_3[i]['isotropic_polarizability'])


fig, ax = plt.subplots()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12
ax.plot(dist_1, fci_1, label="FCI", color='grey', linestyle='solid', linewidth=2)#, marker = 'o', markersize=3)
ax.plot(dist_2, fci_2, color='grey', linestyle='solid', linewidth=2)#, marker = 'o', markersize=3)
ax.plot(dist_3, fci_3, color='grey', linestyle='solid', linewidth=2)#, marker = 'o', markersize=3)
ax.plot(dist_1_qeom, qeom_1, label="qEOM", marker = 'x', color = 'red',  linestyle='None', markersize=2.5)
ax.plot(dist_2_qeom, qeom_2, marker = 'x', color = 'red',  linestyle='None', markersize=2.5)
ax.plot(dist_3_qeom, qeom_3, marker = 'x', color = 'red',  linestyle='None', markersize=2.5)
ax.plot(dist_1_qeom, scqeom_1, label="EOM-VQE", marker = 'x', color = 'blue',  linestyle='None', markersize=2.5)
ax.plot(dist_2_qeom, scqeom_2, marker = 'x', color = 'blue',  linestyle='None', markersize=2.5)
ax.plot(dist_3_qeom, scqeom_3, marker = 'x', color = 'blue',  linestyle='None', markersize=2.5)
ax.plot(dist_1_qeom, qse_1, label="QSE", marker = 'x', color = 'orange',  linestyle='None', markersize=2.5)
ax.plot(dist_2_qeom, qse_2, marker = 'x', color = 'orange',  linestyle='None', markersize=2.5)
ax.plot(dist_3_qeom, qse_3, marker = 'x', color = 'orange',  linestyle='None', markersize=2.5)
plt.xlabel('Li-H bond length ($\mathbf{\AA}$)', fontsize=17, fontweight='bold')
plt.ylabel("Isotropic polarizability (au)", fontsize=15, fontweight='bold')
ax.set_ylim([-200000, 200000])
ax.tick_params(axis='y', which='major', labelsize=16)
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)
#ax.legend(loc='upper left', fontsize=9)
ax.legend(loc='upper center', fontsize=11, frameon=False)
name_fig = mol_name + 'part1_markers.pdf'
#fig.savefig('h2_polarizability_sto_3g.pdf', dpi=600, bbox_inches="tight")
fig.savefig(name_fig, dpi=600, bbox_inches="tight")

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
