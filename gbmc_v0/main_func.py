import util_funcs as uf
import pad_dump_file as pdf
import vv_props as vvp
import lammps_dump_writer as ldw
import lammps_script_writer as lsw
import ovito.data as ovd
from ovito.pipeline import StaticSource, Pipeline
import ovito.modifiers as ovm

lat_par = 4.05
rCut = 2*lat_par
CohEng= -3.8
Tm = 933
tol_fix_reg = 5 * lat_par  # the width of rigid traslation region
lammps_exe_path = '/home/leila/Downloads/mylammps/src/lmp_mpi'
pot_path = './lammps_dump/'  # the path for the potential

# filename0 = 'tests/data/dump_1'  # the name of the dump file that
# box_bound, dump_lamp, box_type = ldw.lammps_box(lat_par, './tests/data/gb_attr.pkl') # lammps creates from the pkl file
# ldw.write_lammps_dump(filename0, box_bound, dump_lamp, box_type)  # writing the dump file

# fil_name = 'in.min'  # the initila minimization lammps script write the in.min script and run it and create dump_minimized
# lsw.run_lammps_min(filename0, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path, './lammps_dump/')

# filename1 = './lammps_dump/dump_minimized'  # the output of previous step
# # # lammps script for heating, equillibrium, cooling and the minimizing write the in.anneal script and run it and dump dump_minimized
# # fil_name1 = 'in.anneal'
# # lsw.run_lammps_anneal(filename1, fil_name1, pot_path, lat_par, tol_fix_reg, lammps_exe_path, './lammps_dump/')

filename = './lammps_dump/dump_minimized'  # the output of previous step
#  read the data
data = uf.compute_ovito_data(filename)
non_p = uf.identify_pbc(data)
#  find the gb atoms
GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pdf.GB_finder(data, lat_par, non_p)

choice = uf.choos_rem_ins()
choice = "removal"
p_rm = uf.RemProb(data, CohEng, GbIndex)
ID2change = uf.RemIns_decision(p_rm)
uf.atom_removal(filename, './lammps_dump/test/', GbIndex[ID2change])
fil_name = 'in.min'  # the initila minimization lammps script write the in.min script and run it and create dump_minimized
# lsw.run_lammps_min(filename, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path, './lammps_dump/test/')
filename = './lammps_dump/test/dump_minimized'
data1 = uf.compute_ovito_data(filename)
weight_1 = .5
E1 = uf.cal_GB_E(data1, weight_1, non_p, lat_par, CohEng)
E0 = uf.cal_GB_E(data, weight_1, non_p, lat_par, CohEng)
area = uf.cal_area(data1, non_p)
p_boltz = uf.p_boltz_func(E0, E1, area, Tm)
decision = uf.decide(p_boltz)
if decision == "accept":
    filename = "new"
else:
    filename = "old"


# if choice == "removal":
    # E_rm = uf.RemProb(data, CohEng, GbIndex)
    # ID2change = uf.RemIns_decision(Prob)
    # uf.atom_removal(filename0, path2dump, ID2change)
    # fil_name = 'in.min'  # the initila minimization lammps script write the in.min script and run it and create dump_minimized
    # lsw.run_lammps_min(filename, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path, './lammps_dump/')
    # data = compute_ovito_data(filename0)
    # E_GB = cal_GB_E(data, weight_1, non_p, lat_par, E_coh)
    # p_boltz = p_boltz_func(E0, E1, area, Tm)
    # decision = uf.decide(p_boltz)
    # if decision == "accept":
    #     filename = "new"
    # else:
    #     filename = "old"
# else:
#     pts_w_imgs, gb1_inds, inds_arr = pdf.pad_dump_file(data, lat_par, rCut, non_p)
#     tri_vertices, gb_tri_inds = vvp.triang_inds(pts_w_imgs, gb1_inds, inds_arr)
#     cc_coors, cc_rad = vvp.vv_props(pts_w_imgs, tri_vertices, gb_tri_inds, lat_par)
#     cc_coors1 = vvp.wrap_cc(ovito_data.cell, cc_coors)
#     rad_norm = radi_normaliz(cc_rad)
#     ID2change = RemIns_decision(Prob)
#     atom_insertion(filename0, path2dump, cc_coors1)




