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
weight_1 = .5
tol_fix_reg = 5 * lat_par  # the width of rigid traslation region
lammps_exe_path = '/home/leila/Downloads/mylammps/src/lmp_mpi'
pot_path = './lammps_dump/'  # the path for the potential
dump_path = './lammps_dump/test/'
pkl_file = './tests/data/gb_attr.pkl'
initial_dump = 'tests/data/dump_1'  # the name of the dump file that

box_bound, dump_lamp, box_type = ldw.lammps_box(lat_par, pkl_file) # lammps creates from the pkl file
ldw.write_lammps_dump(initial_dump, box_bound, dump_lamp, box_type)  # writing the dump file

filename_0 = dump_path + 'dump.0' # the output of previous step
fil_name = 'in.min'  # the initila minimization lammps script write the in.min script and run it and create dump_minimized
lsw.run_lammps_min(initial_dump, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path, filename_0, step=1)

data_0 = uf.compute_ovito_data(filename_0)
for i in range(1, 10, 1):
    #  read the data
    data_0 = uf.compute_ovito_data(filename_0)
    non_p = uf.identify_pbc(data_0)
    #  find the gb atoms
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pdf.GB_finder(data_0, lat_par, non_p)

    #  decide between remove and insertion
    choice = uf.choos_rem_ins()
    choice = "insertion"
    #  if the choice is removal
    if choice == "removal":
        p_rm = uf.RemProb(data_0, CohEng, GbIndex)
        ID2change = uf.RemIns_decision(p_rm)
        uf.atom_removal(filename_0, dump_path , GbIndex[ID2change])
        fil_name = 'in.min'  # the initila minimization lammps script write the in.min script and run it and create dump_minimized
        filename_rem = dump_path + 'rem_dump'
        lsw.run_lammps_min(filename_rem, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path, dump_path + 'dump.' + str(i))
        filename_1 = dump_path + 'dump.' + str(i)
        data_1 = uf.compute_ovito_data(filename_1)

        E_1 = uf.cal_GB_E(data_1, weight_1, non_p, lat_par, CohEng)  #  after removal
        E_0 = uf.cal_GB_E(data_0, weight_1, non_p, lat_par, CohEng)
        dE = E_1 - E_0
        if dE < 0:
            decision = "accept"
            print("finally accepted in removal")
        else:
            area = uf.cal_area(data_1, non_p)
            p_boltz = uf.p_boltz_func(dE, area, Tm)
            decision = uf.decide(p_boltz)

        if decision == "accept":
            filename_0 = filename_1
        else:
            pass

    else:
        pts_w_imgs, gb1_inds, inds_arr = pdf.pad_dump_file(data_0, lat_par, rCut, non_p)
        tri_vertices, gb_tri_inds = vvp.triang_inds(pts_w_imgs, gb1_inds, inds_arr)
        cc_coors, cc_rad = vvp.vv_props(pts_w_imgs, tri_vertices, gb_tri_inds, lat_par)
        cc_coors1 = vvp.wrap_cc(data_0.cell, cc_coors)
        Prob = uf.radi_normaliz(cc_rad)
        ID2change = uf.RemIns_decision(Prob)
        pos_add_atom = cc_coors[ID2change]
        uf.atom_insertion(filename_0, dump_path, pos_add_atom)
        fil_name = 'in.min'  # the initila minimization lammps script write the in.min script and run it and create dump_minimized
        filename_ins = dump_path + 'ins_dump'
        lsw.run_lammps_min(filename_ins, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path, dump_path + 'dump.' + str(i))
        filename_1 = dump_path + 'dump.' + str(i)
        data_1 = uf.compute_ovito_data(filename_1)

        E_1 = uf.cal_GB_E(data_1, weight_1, non_p, lat_par, CohEng)  #  after removal
        E_0 = uf.cal_GB_E(data_0, weight_1, non_p, lat_par, CohEng)
        dE = E_1 - E_0
        print(dE)
        if dE < 0:
            decision = "accept"
            print("finally accepted in insertion")
        else:
            area = uf.cal_area(data_1, non_p)
            p_boltz = uf.p_boltz_func(dE, area, Tm)
            decision = uf.decide(p_boltz)

        if decision == "accept":
            filename_0 = filename_1
        else:
            pass






