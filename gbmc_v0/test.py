import util_funcs as uf
import pad_dump_file as pdf
import vv_props as vvp
import lammps_dump_writer as ldw
import lammps_script_writer as lsw
import ovito.data as ovd
from ovito.pipeline import StaticSource, Pipeline
import ovito.modifiers as ovm
from shutil import copyfile
import numpy as np
import os
# directory = '/home/leila/Leila_sndhard/codes/GBMC-master/data/sym_100/Al_S13_1_N1_0_-1_5_N2_0_-1_-5/accepted_steps/'
directory = '/home/leila/Leila_sndhard/codes/gbmc_python/gbmc_v0/gbmc_v0/gbmc_v0/lammps_dump/test/accepted/'
a = os.listdir(directory)
lat_par = 4.074
non_p = 1
weight_1= 0.5
CohEng= -3.443962  #  calculated from in.cohesive

rCut = 2*lat_par
Tm = 1000
weight_1 = .5
tol_fix_reg = 5 * lat_par  # the width of rigid traslation region
SC_tol = 5 * lat_par
energy = []
iteration = []
j = 0
# a = ['dump.0']
for i in a:
    filename_0 = directory + i
    data = uf.compute_ovito_data(filename_0)
    eng = uf.cal_GB_E(data, weight_1, non_p, lat_par, CohEng)
    energy = energy + [eng]
    print(eng)
    iteration = iteration + [j]
    j += 1

# import matplotlib.pyplot as plt
# fig, axes = plt.subplots()
# axes.plot(iteration, energy, 'o-', markeredgewidth=0)
# plt.show()

# np.savetxt('energy_mycod.txt', np.column_stack([iteration,energy]))