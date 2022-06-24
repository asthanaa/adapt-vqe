import psi4

mol = {}

mol["n2"] = psi4.geometry("""
N 0 0 1.05723396
N 0 0 -1.05723396
units bohr
symmetry c1
""") 

mol["h2o2"] = psi4.geometry("""
O     -0.028962160801    -0.694396279686    -0.049338350190                                                                  
O      0.028962160801     0.694396279686    -0.049338350190                                                                  
H      0.350498145881    -0.910645626300     0.783035421467                                                                  
H     -0.350498145881     0.910645626300     0.783035421467                                                                  
#noreorient
symmetry c1        
""")

# OR45 (S)-methyloxirane Molecule 1.
mol["(S)-mox"] = psi4.geometry("""
C -1.0434290  0.6153280 -0.0615190
C  0.1515990 -0.0360920  0.4888630
H  0.1531120 -0.2526560  1.5570100
O -0.8257760 -0.7892730 -0.2415100
H -0.9522010  1.2186250 -0.9624750
H -1.8701630  0.8793260  0.5941530
C  1.5090730  0.0997910 -0.1483000
H  2.0832840  0.9000390  0.3286960
H  1.4131530  0.3248980 -1.2127570
H  2.0755690 -0.8302050 -0.0468100
symmetry c1
""")

# OR45 Dimethylallene Molecule 17.
mol["dma"] = psi4.geometry("""
C -0.0000020 -0.4140180 -0.0000820
C  1.2460740 -0.4124290 -0.3898150
C -1.2459090 -0.4119140  0.3901890
H -1.5373900 -1.0897580  1.1922810
C  2.3353570  0.4600290  0.1851310
H  1.5379560 -1.0912810 -1.1909070
C -2.3355030  0.4597100 -0.1854250
H  1.9512980  1.1010180  0.9798820
H  2.7742720  1.0968600 -0.5903120
H  3.1462750 -0.1507210  0.5961800
H -3.1466880 -0.1516020 -0.5951030
H -1.9518880  1.0995170 -0.9813430
H -2.7739380  1.0977020  0.5893370
symmetry C1
""")

mol["h2_2"] = psi4.geometry("""
H
H 1 R
H 1 D 2 P
H 3 R 1 P 2 T

R = 0.75
D = 1.5
P = 90.0
T = 60.0
symmetry c1
""")

mol["h2_3"] = psi4.geometry("""
H
H 1 R
H 1 D 2 P
H 3 R 1 P 2 T
H 3 D 4 P 1 X
H 5 R 3 P 4 T

R = 0.75
D = 1.5
P = 90.0
T = 60.0
X = 180.0
symmetry c1
""")

mol["h2_4"] = psi4.geometry("""
H
H 1 R
H 1 D 2 P
H 3 R 1 P 2 T
H 3 D 4 P 1 X
H 5 R 3 P 4 T
H 5 D 6 P 3 X
H 7 R 5 P 6 T

R = 0.75
D = 1.5
P = 90.0
T = 60.0
X = 180.0
symmetry c1
""")

mol["h2_5"] = psi4.geometry("""
H
H 1 R
H 1 D 2 P
H 3 R 1 P 2 T
H 3 D 4 P 1 X
H 5 R 3 P 4 T
H 5 D 6 P 3 X
H 7 R 5 P 6 T
H 7 D 8 P 5 X
H 9 R 7 P 8 T

R = 0.75
D = 1.5
P = 90.0
T = 60.0
X = 180.0
symmetry c1
""")

mol["h2_6"] = psi4.geometry("""
H
H 1 R
H 1 D 2 P
H 3 R 1 P 2 T
H 3 D 4 P 1 X
H 5 R 3 P 4 T
H 5 D 6 P 3 X
H 7 R 5 P 6 T
H 7 D 8 P 5 X
H 9 R 7 P 8 T
H 9 D 10 P 7 X
H 11 R 9 P 10 T

R = 0.75
D = 1.5
P = 90.0
T = 60.0
X = 180.0
symmetry c1
""")

mol["h2_7"] = psi4.geometry("""
H
H 1 R
H 1 D 2 P
H 3 R 1 P 2 T
H 3 D 4 P 1 X
H 5 R 3 P 4 T
H 5 D 6 P 3 X
H 7 R 5 P 6 T
H 7 D 8 P 5 X
H 9 R 7 P 8 T
H 9 D 10 P 7 X
H 11 R 9 P 10 T
H 11 D 12 P 9 X
H 13 R 11 P 12 T

R = 0.75
D = 1.5
P = 90.0
T = 60.0
X = 180.0
symmetry c1
""")

# Geometry from Harley's cerebro area
mol['1-fluoro-propane'] = psi4.geometry("""
C
H 1 B1
C 1 B2 2 A1
C 3 B3 1 A2 2 D1
H 4 B4 3 A3 1 D2
H 1 B5 2 A4 3 D3
H 1 B6 2 A5 3 D4
H 3 B7 1 A6 2 D5
H 3 B8 1 A7 2 D6
H 4 B9 3 A8 1 D7
F 4 B10 3 A9 1 D8

A5  =  108.16228279
D6  = -59.81524907
A6  =  109.98124081
D7  =  58.54006537
A7  =  110.28985606
D8  = -60.5529762
A8  =  111.37724991
A9  =  109.846557
B1  =  1.08503577
B2  =  1.52724872
B3  =  1.51413535
B4  =  1.08401589
B5  =  1.08671956
B6  =  1.08371249
B7  =  1.08658269
B8  =  1.08806091
B9  =  1.08491428
B10 =  1.37421632
D1  =  179.30667262
D2  = -179.79478803
A1  =  110.93125751
D3  = -121.88723206
A2  =  112.95018677
D4  =  121.60266638
A3  =  111.40071383
A4  =  107.81289087
D5  =  57.83253083
symmetry c1
""")

# Geometry from Harley's cerebro area
mol['1-fluoro-pentane'] = psi4.geometry("""
C
C 1 B1
H 2 B2 1 A1
C 1 B3 2 A2 3 D1
C 4 B4 1 A3 2 D2
H 5 B5 4 A4 1 D3
H 1 B6 2 A5 3 D4
H 1 B7 2 A6 3 D5
H 4 B8 1 A7 2 D6
H 4 B9 1 A8 2 D7
H 5 B10 4 A9 1 D8 
F 5 B11 4 A10 1 D9 
H 2 B12 1 A11 4 D10 
C 2 B13 1 A12 4 D11 
H 14 B14 2 A13 1 D12 
H 14 B15 2 A14 1 D13 
H 14 B16 2 A15 1 D14 

A10 =  109.92940827
A11 =  109.26718995
A12 =  112.93301229
B1  =  1.52897205
A13 =  111.16686238
B2  =  1.08868021
A14 =  111.11274314
B3  =  1.52890237
A15 =  111.22521817
B4  =  1.51399239
B5  =  1.08402213
B6  =  1.08960136
B7  =  1.08621416
B8  =  1.08735962
B10 =  1.08480538
B9  =  1.08889959
B11 =  1.37446813
B12 =  1.0883279
B13 =  1.52768243
B14 =  1.08657451
B15 =  1.08638026
B16 =  1.08565649
D10 = -58.07801641
D11 =  179.98379612
D12 = -60.09788277
D13 =  59.79415776
D14 =  179.85020768
D1  =  57.89320707
D2  =  179.12340816
D3  =  179.68095582
A1  =  109.43196583
D4  = -64.01522274
A2  =  113.04749973
D5  =  179.65244818
A3  =  113.25868353
A4  =  111.35605561
D6  =  57.52173029
D7  = -59.92501902
A5  =  109.29363077
D8  =  58.05460656
A6  =  109.62964216
D9  = -61.11775468
A7  =  109.92499717
A8  =  110.14209083
A9  =  111.43453718
symmetry c1
""")

# Geometry from Harley's cerebro area
mol['1-fluoro-heptane'] = psi4.geometry("""
C
C 1 B1
H 2 B2 1 A1
C 1 B3 2 A2 3 D1
C 4 B4 1 A3 2 D2
H 5 B5 4 A4 1 D3
H 1 B6 2 A5 3 D4
H 1 B7 2 A6 3 D5
H 4 B8 1 A7 2 D6
H 4 B9 1 A8 2 D7
H 5 B10 4 A9 1 D8
F 5 B11 4 A10 1 D9
H 2 B12 1 A11 4 D10
C 2 B13 1 A12 4 D11
H 14 B14 2 A13 1 D12
H 14 B15 2 A14 1 D13
C 14 B16 2 A15 1 D14
H 17 B17 14 A16 2 D15
H 17 B18 14 A17 2 D16
C 17 B19 14 A18 2 D17
H 20 B20 17 A19 14 D18
H 20 B21 17 A20 14 D19
H 20 B22 17 A21 14 D20

A10 =  109.93574393
A11 =  109.2394584
A12 =  113.18591919
B1  =  1.52900622
A13 =  109.31883819
B2  =  1.08947607
A14 =  109.2674421
B3  =  1.52902146
A15 =  113.35033189
B4  =  1.5139787
A16 =  109.31036469
B5  =  1.08403001
A17 =  109.29297649
B6  =  1.08948119
A18 =  113.0688008
B7  =  1.08608892
A19 =  111.16099951
B8  =  1.08734571
B10 =  1.08481165
B9  =  1.08888769
B11 =  1.37449836
B12 =  1.08912144
B13 =  1.52920525
B14 =  1.08933099
B15 =  1.08911288
B16 =  1.52937245
B17 =  1.08851071
B18 =  1.08842669
B19 =  1.52784126
A20 =  111.26048612
A21 =  111.1520733
D10 = -58.04792647
D11 =  179.97375508
D12 = -58.09370282
D13 =  57.73073861
D14 =  179.82052453
D15 =  57.93020027
D16 = -57.95807663
D17 =  179.9933299
D18 = -59.95129035
B20 =  1.08661835
D19 =  179.99081223
B21 =  1.08574912
B22 =  1.0865609
D20 =  59.93436291
D1  =  57.84408754
D2  =  179.15705807
D3  =  179.68981581
A1  =  109.40875951
D4  = -64.02206555
A2  =  113.00755349
D5  =  179.5618211
A3  =  113.25543554
A4  =  111.35532249
D6  =  57.54991328
D7  = -59.89864749
A5  =  109.35618022
D8  =  58.06607565
A6  =  109.69270442
D9  = -61.11078631
A7  =  109.93095425
A8  =  110.14645497
A9  =  111.43526365
symmetry c1   
""")

# Geometry from Harley's cerebro area
mol['(S)-1-phenylethanol'] = psi4.geometry("""
C
C 1 B1
C 2 B2 1 A1
C 3 B3 2 A2 1 D1
C 4 B4 3 A3 2 D2
C 5 B5 4 A4 3 D3
C 2 B6 3 A5 4 D4
H 7 B7 2 A6 3 D5
H 6 B8 5 A7 4 D6
H 5 B9 4 A8 3 D7
H 4 B10 3 A9 2 D8
H 3 B11 2 A10 7 D9
O 1 B12 2 A11 3 D10
H 13 B13 1 A12 2 D11
C 1 B14 2 A13 3 D12
H 15 B15 1 A14 2 D13
H 15 B16 1 A15 2 D14
H 15 B17 1 A16 2 D15
H 1 B18 2 A17 3 D16

A10 =  118.95682987
A11 =  112.13128267
A12 =  107.36335412
B1  =  1.52226159
A13 =  112.28401776
B2  =  1.40100004
A14 =  110.12821695
B3  =  1.39527559
A15 =  110.68039197
B4  =  1.3965682
A16 =  109.91004656
B5  =  1.39614969
A17 =  108.18849291
B6  =  1.39978966
B7  =  1.08849054
B8  =  1.08700862
B10 =  1.08711802
B9  =  1.08680915
B11 =  1.08632945
B12 =  1.4290682
B13 =  0.9705727
B14 =  1.52922876
B15 =  1.0949227
B16 =  1.09464134
B17 =  1.09493433
B18 =  1.10303747
D10 = -34.91585746
D11 = -53.61535731
D12 =  84.3590047
D13 = -59.69966893
D14 =  60.32965811
D15 = -179.28376684
D16 = -156.13039484
D1  = -178.57963128
D2  =  0.08186297
D3  = -0.1919996
A1  =  120.51019098
D4  =  0.26185585
A2  =  120.54469912
D5  =  179.30774015
A3  =  120.21143279
A4  =  119.63243767
D6  = -179.51794634
D7  = -179.89016222
A5  =  118.83262254
D8  = -179.86097711
A6  =  119.49233932
D9  = -179.22055484
A7  =  120.11709223
A8  =  120.20249318
A9  =  119.76730779
symmetry c1
""")

# OR45 (1R,4R)-Norbornenone Molecule 20
mol["norbornenone"] = psi4.geometry("""
C  1.1757860 -0.0069620 -0.0484540
C -0.8100310 -1.2831360 -0.5280540
C  0.1167890 -0.9119490  0.6298460
C  0.4248100  1.2892770 -0.4231390
C -1.0266460  0.9550630  0.0290500
C -1.4930270 -0.1871510 -0.8725130
H -0.8056540 -2.2442270 -1.0246230
H  0.5436000 -1.7256560  1.2107680
H  0.5260180  1.5173680 -1.4847770
H  0.8578210  2.1195540  0.1439830
H -1.6922410  1.8132420  0.1087690
H -2.1795940 -0.0787570 -1.7023480
O  2.3380230 -0.2538320 -0.2305820
C -0.7412690  0.1667950  1.3346370
H -1.6420920 -0.2528140  1.7856890
H -0.1905180  0.7503100  2.0789600
symmetry c1
""")


# OR45 (1R,5R)-beta-pinene Molecule 32
mol["beta-pinene"] = psi4.geometry("""
C -1.0765220 -0.2835340  0.1959240
C -0.7359440  1.2327740 -0.0281470
C  0.5025910  1.6488230  0.7809710
C  1.7474220  0.8034850  0.3838420
C  1.3930670 -0.5310990 -0.2632700
C  0.0294050 -0.5549880 -0.9031190
C -0.2985700  0.8600630 -1.4727610
C  2.2198510 -1.5762850 -0.2703180
C -2.4821180 -0.6405730 -0.3078410
C -0.8869840 -0.8948300  1.5837660
H  0.7137880  2.7080540  0.6039680
H  0.5221230  1.4266450 -1.9176140
H -1.1274850  0.8410600 -2.1783090
H -1.5554650  1.9464780  0.0983040
H  2.4065690  0.6475090  1.2413240
H  2.3346560  1.3737350 -0.3441290
H  0.1077140 -0.7265960  1.9961020
H -1.6203250 -0.4848090  2.2867480
H  0.2937110  1.5515200  1.8501550
H -0.1081920 -1.4286340 -1.5449590
H -2.6064710 -1.7272160 -0.3569990
H -2.6993320 -0.2374910 -1.2985370
H -3.2404840 -0.2519480  0.3800210
H -1.0392650 -1.9781860  1.5427650
H  1.9588160 -2.5001820 -0.7756160
H  3.1864510 -1.5429480  0.2224840
symmetry c1
""")

# B3LYP/aug-cc-pVTZ optimized structure 
mol["fluorooxirane"] = psi4.geometry("""
units Bohr
C          0.2403332017           -0.0885103560            0.9274082395
C         -1.9467968844            1.2009064480           -0.1231423794
O         -1.5011809111           -1.4989261089           -0.4180476302
F          2.5328284210            0.2211492546           -0.2147489576
H          0.4922277650           -0.5359990755            2.9107014665
H         -3.5435469826            1.6940643818            1.0604258094
H         -1.6929812215            2.2889552787           -1.8383495383

symmetry c1
""")

import sys
import numpy as np
sys.path.append('../')
from fvno_plus_plus_lg import guess_calculate 
from fvno_plus_plus_lg import optrot_calculate 
from numpy import linalg as LA

np.set_printoptions(precision=15, linewidth=200, suppress=True)
#psi4.core.set_memory(int(2e9), False)
psi4.set_memory(int(5e9), False)

filt_singles = False
if sys.argv[3] == "true":
    filt_singles = True

if filt_singles:
    output_file = sys.argv[1] + '_' + sys.argv[2] + '_singles_frozen' + '.dat'
else:
    output_file = sys.argv[1] + '_' + sys.argv[2] + '.dat'
if sys.argv[4] == 'testing':
    output_file = 'testing.dat' 
  
  
psi4.core.set_output_file(output_file, False)

psi4.set_options({'basis': 'aug-cc-pvdz',
                  'guess': 'sad',
                  'scf_type': 'pk',
                  'e_convergence': 1e-9,
                  'd_convergence': 1e-9})
#psi4.set_num_threads(24)
psi4.set_options({'guess': 'sad'})

print('Computing RHF reference.')
psi4.core.set_active_molecule(mol[sys.argv[1]])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)  
print('RHF Final Energy                          % 16.10f\n' % rhf_e)


Emat_ab = guess_calculate(mol[sys.argv[1]], rhf_e, rhf_wfn, 'CCSD', 1, 0, 0, 10, False, False, sys.argv[2])
#Evec_ab = guess_calculate(mol[sys.argv[1]], rhf_e, rhf_wfn, 'CCSD', maxiter_cc, maxiter_lambda, maxiter_pert, memory, 
#transform into local basis or not -> edv, Filt singles->False while creating guess, sys.argv[2]->Density_type)
C = psi4.core.Matrix.to_array(rhf_wfn.Ca())
F = psi4.core.Matrix.to_array(rhf_wfn.Fa())

nmo = rhf_wfn.nmo()
occ = rhf_wfn.doccpi()[0]
vir = nmo - occ
C_occ = C[:, :occ]
C_vir = C[:, occ:]
F_mo  = np.einsum('ui,vj,uv', C, C, F)
F_mo_occ = F_mo[:occ,:occ]
F_mo_vir = F_mo[occ:, occ:]

frz_vir = [0, int(.05 * vir), int(.1 * vir), int(.15 * vir), int(.20 * vir), int(.25 * vir), int(.30 * vir)]
Emat_ab1 = np.zeros_like(Emat_ab)

for k in frz_vir:

    print('\nTruncation : %d \n' % k)

    Emat_ab1 = Emat_ab.copy()
    Emat_view = Emat_ab1[:,vir-k:]
    Emat_view.fill(0)

    #C_occ_no = np.einsum('pi,ij->pj', C_occ, Emat_ij)
    C_vir_no = np.einsum('pa,ab->pb', C_vir, Emat_ab1)

    #F_no_occ  = np.einsum('ki,lj,kl', Emat_ij, Emat_ij, F_mo_occ)
    F_no_vir  = np.einsum('ca,db,cd', Emat_ab1, Emat_ab1, F_mo_vir)

    #tmp_occ_ev, tmp_occ_mat = LA.eig(F_no_occ)
    tmp_vir_ev, tmp_vir_mat = LA.eigh(F_no_vir)

    #C_occ_sc = np.einsum('pi,ij->pj', C_occ_no, tmp_occ_mat)
    C_vir_sc = np.einsum('pa,ab->pb', C_vir_no, tmp_vir_mat)

    #F_occ_sc  = np.einsum('ui,vj,uv', C_occ_sc, C_occ_sc, F)
    F_vir_sc  = np.einsum('ua,vb,uv', C_vir_sc, C_vir_sc, F)

    #C_np_sc = np.concatenate((C_occ_sc, C_vir_sc), axis=1)
    C_np_sc = np.concatenate((C_occ, C_vir_sc), axis=1)

    C_psi4_sc = psi4.core.Matrix.from_array(C_np_sc)

    rhf_wfn.Ca().copy(C_psi4_sc)
    optrot_calculate(mol[sys.argv[1]], rhf_e, rhf_wfn, 'CCSD', 100, 100, 100, 10, False, filt_singles)
#optrot_calculate(mol[sys.argv[1]], rhf_e, rhf_wfn, 'CCSD', 100, 100, 100, 10, True, filt_singles, Evec_ab)
