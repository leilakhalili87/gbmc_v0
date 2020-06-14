import pickle as pkl
import numpy as np
import os
import sys

import util_funcs as uf
import pad_dump_file as pdf
import vv_props as vvp
import numpy.linalg as nla
import lammps_dump_writer as ldw
import lammps_script_writer as lsw


lat_par = 4.074
rCut = 2*lat_par  
tol_fix_reg = 5 * lat_par # the width of rigid traslation region
lammps_exe_path = '/home/leila/Downloads/mylammps/src/lmp_mpi'
pot_path = './lammps_dump/' # the path for the potential

filename0 = 'tests/data/dump_1' # the name of the dump file that
# lammps creates from the pkl file
box_bound, dump_lamp, box_type = ldw.lammps_box(lat_par, './tests/data/gb_attr.pkl')
ldw.write_lammps_dump(filename0, box_bound, dump_lamp, box_type) # writing the dump file


fil_name = 'in.min' # the initila minimization lammps script
# write the in.min script and run it and create dump_minimized
run_lammps_min(filename0, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path, './lammps_dump/')

filename1 = './lammps_dump/dump_minimized' # the output of previous step
fil_name1 = 'in.anneal' # lammps script for heating, equillibrium, cooling and the minimizing
# write the in.anneal script and run it and dump dump_minimized
run_lammps_anneal(filename1, fil_name1, pot_path, lat_par, tol_fix_reg, lammps_exe_path, './lammps_dump/')

filename = 'dump_minimized' # the output of previous step
ovito_data = uf.compute_ovito_data(filename)
non_p = uf.identify_pbc(ovito_data)
pts_w_imgs, gb1_inds, inds_arr = pdf.pad_dump_file(ovito_data, lat_par, rCut, non_p)
tri_vertices, gb_tri_inds = vvp.triang_inds(pts_w_imgs, gb1_inds, inds_arr)
cc_coors, cc_rad = vvp.vv_props(pts_w_imgs, tri_vertices, gb_tri_inds, lat_par)
cc_coors1 = vvp.wrap_cc(ovito_data.cell, cc_coors)