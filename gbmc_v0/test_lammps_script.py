
import numpy as np
import util_funcs as uf;
import ovito.io as oio

def file_gen(fil_name):

    # file_name = fil_name[0] + '/in.'+ fil_name[1]
    fiw = open(fil_name, 'w')
    return fiw, fil_name


def lammps_script_var(fiw, lat_par):
    overlap_cte = np.sqrt(2) * lat_par / 4
    line = []
    line.append('# Minimization Parameters -------------------------\n')
    line.append('\n')
    line.append('variable Etol equal 1e-25\n')
    line.append('variable Ftol equal 1e-25\n')
    line.append('variable MaxIter equal 5000\n')
    line.append('variable MaxEval equal 10000\n')
    line.append('\n')   
    line.append('# Structural variables------------------------------\n')
    line.append('\n')
    line.append('variable LatParam equal ' + str(lat_par) + '\n')
    line.append('# ------------------------------------------------\n')
    line.append('\n')
    line.append('variable cnt equal 1\n')
    line.append('# ------------------------------------------------\n')
    line.append('\n')
    line.append('variable OverLap equal ' + str(overlap_cte) + '\n')
    line.append('\n')
    for i in line:
        fiw.write(i)

    return True


def script_init_sim(fiw, non_p):
    if non_p ==0:
        bound = 'f p p'
    elif non_p == 1:
        bound = 'p f p'
    else:
        bound = 'p p f'

    line = []
    line.append('# -----------Initializing the Simulation-----------\n')
    line.append('\n')
    line.append('clear\n')
    line.append('units metal\n')
    line.append('dimension 3\n')
    line.append('boundary ' +  str(bound) +  '\n')
    line.append('atom_style atomic\n')
    line.append('\n')
    for i in line:
        fiw.write(i)

    return True


def define_box(fiw, untilted, tilt, box_type):
    """
    """
    if box_type == 'block':
        whole_box = 'region whole block ' + str(untilted[0][0]) + ' ' + str(untilted[0][1]) + ' ' +\
                    str(untilted[1][0]) +' ' + str(untilted[1][1]) + ' ' + str(untilted[2][0]) + ' '\
                    + str(untilted[2][1]) +' ', ' units box'
    elif box_type == 'prism':
        whole_box = 'region whole prism '+ str(untilted[0][0]) +  ' ' + str(untilted[0][1]) + ' ' +\
                    str(untilted[1][0]) + ' ' + str(untilted[1][1]) + ' ' + str(untilted[2][0]) + ' '\
                    + str(untilted[2][1]) + ' ' + str(tilt[0])+ ' ' + str(tilt[1]) + ' ' + str(tilt[2]) + ' units box'
    create_box = 'create_box 2 whole\n'

    line =[]
    line.append('# ---------Creating the Atomistic Structure--------\n')
    line.append('\n')
    line.append('lattice fcc ${LatParam}\n')
    line.append(str(whole_box) + '\n')
    line.append(str(create_box) + '\n')

    for i in line:
        fiw.write(i)

    return True



def define_fix_rigid(fiw, box_bound, tol_fix_reg, non_p):
    """
    """
    untilted, tilt, box_type = uf.define_bounds(box_bound)
    untilted[non_p, :] = untilted[non_p, :] + np.array([tol_fix_reg, -tol_fix_reg])

    if box_type == 'block':
        rigid_reg = 'region reg_fix block ' + str(untilted[0][0]) + ' ' + str(untilted[0][1]) + ' ' +\
                    str(untilted[1][0]) + ' ' + str(untilted[1][1]) + ' ' + str(untilted[2][0]) + ' ' +\
                    str(untilted[2][1]) + ' units box'
    elif box_type == 'prism':
        rigid_reg = 'region reg_fix prism ' + str(untilted[0][0]) + ' ' + str(untilted[0][1]) + ' ' +\
                    str(untilted[1][0]) + ' ' + str(untilted[1][1]) + ' ' + str(untilted[2][0]) + ' '\
                    + str(untilted[2][1]) + ' ' + str(tilt[0]) + ' ' + str(tilt[1]) + ' ' + str(tilt[2]) + ' units box'
    line =[]
    line.append(str(rigid_reg) + '\n')
    line.append('\n')
    line.append("#---------Defining the fix rigid group--------------\n")
    line.append('group non_fix region reg_fix\n')
    line.append('group fix_reg subtract all non_fix\n')
    line.append('\n')
  

    for i in line:
        fiw.write(i)

    return True


def script_pot(fiw, pot_path):
    line =[]
    line.append('# -------Defining the potential functions----------\n')
    line.append('\n')
    line.append('pair_style eam/alloy\n')
    line.append('pair_coeff * * ' + str(pot_path) + 'Al99.eam.alloy Al Al\n')

    line.append('neighbor 2 bin\n')
    line.append('neigh_modify delay 10 check yes\n')

    for i in line:
        fiw.write(i)

    return True

def script_read_dump(fiw, dump_name):
    line = []
    line.append('read_dump ' + str(dump_name) + ' 0 x y z box yes add yes \n')
    for i in line:
        fiw.write(i)

    return True


def script_overlap(fiw, tol_fix_reg):
    delta_low = -tol_fix_reg
    delta_high = tol_fix_reg
    line =[]
    line.append('group lower type 2 \n')
    line.append('group upper type 1\n')
    line.append('delete_atoms overlap ${OverLap}  upper lower\n')
    line.append('change_box all z delta ' + str(delta_low) + ' ' + str(delta_high) + '\n')

    for i in line:
        fiw.write(i)

    return True


def script_compute(fiw):
    line =[]
    line.append('\n')
    line.append('# ---------Computing Simulation Parameters---------\n')
    line.append('\n')
    line.append('compute csym all centro/atom fcc\n')
    line.append('compute eng all pe/atom\n')
    line.append('compute eatoms all reduce sum c_eng\n')
    line.append('compute MinAtomEnergy all reduce min c_eng\n')
    line.append('\n')

    for i in line:
        fiw.write(i)

    return True


def script_min_sec(fiw):
    line =[]
    line.append('#----------------------minimization--------------------\n')
    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('thermo 100\n')
    line.append('thermo_style custom step pe lx ly lz xy xz yz xlo xhi ylo yhi zlo zhi press pxx pyy pzz '
                'c_eatoms c_MinAtomEnergy\n')
    line.append('thermo_modify lost ignore\n')
    line.append('dump 1 all custom 10 ./lammps_dump/dump_minimized id type x y z c_csym c_eng\n')
    line.append('fix 1 all box/relax x 0 y 0 xy 0\n')
    line.append('fix 2 fix_reg rigid single reinit yes\n')
    line.append('fix 2 fix_reg rigid single reinit yes\n')
    line.append('min_style cg\n')
    line.append('minimize ${Etol} ${Ftol} ${MaxIter} ${MaxEval}\n')
    line.append('undump 1\n')
    line.append('unfix 1\n')
    line.append('unfix 2\n')

    for i in line:
        fiw.write(i)

    return True

def script_main_min(fil_name, lat_par, tol_fix_reg, dump_name, pot_path):
    fiw, file_name = file_gen(fil_name)
    lammps_script_var(fiw, lat_par)
    script_init_sim(fiw, non_p)
    box_bound = uf.box_size_reader(dump_name)
    untilted, tilt, box_type = uf.define_bounds(box_bound)
    define_box(fiw, untilted, tilt, box_type)
    script_read_dump(fiw, dump_name)
    define_fix_rigid(fiw, box_bound, tol_fix_reg, non_p)
    script_pot(fiw, pot_path)
    script_overlap(fiw, tol_fix_reg)
    script_compute(fiw)
    script_min_sec(fiw)
import os
filename0 = './tests/data/dump_1'
fil_name = 'in.min'
lat_par = 4.05
dump_name = './tests/data/dump_1'
pot_path = './lammps_dump/'
tol_fix_reg = 5*lat_par
data = uf.compute_ovito_data(filename0)
non_p = uf.identify_pbc(data)
script_main_min(fil_name, lat_par, tol_fix_reg, dump_name, pot_path)
lammps_exe_path = '/home/leila/Downloads/mylammps/src/lmp_mpi'
os.system(str(lammps_exe_path) + '< ./' + 'in.min')
# def script_heating(fiw):
#     MaxEval0 = 1000
#     line =[]
#     line.append('# -----------------heating step-----------------\n')
#     line.append('\n')
#     line.append('reset_timestep 0\n')
#     line.append('timestep 0.001\n')
#     line.append('fix 1 all npt temp .1 466.75 .1 couple xy  x 0.0 0.0 1.0  y 0.0 0.0 1.0\n')
#     line.append('thermo 100\n')
#     line.append('thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n')
#     line.append('dump 1 all custom 10 ' + str(path) + 'dump_step_heat id type x y z c_csym c_eng\n')
#     line.append('run ' + str(MaxEval0) + '\n')
#     line.append('unfix 1\n')
#     line.append('undump 1\n')
#     line.append('\n')
#     for i in line:
#         fiw.write(i)

#     return True


# def script_cooling(fiw):
#     line =[]
#     line.append('# -----------------equilibrium step-----------------\n')
#     line.append('\n')
#     line.append('reset_timestep 0\n')
#     line.append('timestep 0.001\n')
#     line.append('fix 1 all npt temp 466.75 466.75 0.1 couple xy  x 0.0 0.0 1.0  y 0.0 0.0 1.0\n')
#     line.append('thermo 100\n')
#     line.append('thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n')
#     line.append('dump 1 all custom 10 ' + str(path) + 'dump_step_cool id type x y z c_csym c_eng\n')
#     line.append('run ${MaxEval}\n')
#     line.append('unfix 1\n')
#     line.append('undump 1\n')
#     line.append('\n')
#     for i in line:
#         fiw.write(i)

#     return True

# def script_equil(fiw):
#     MaxEval1 = 12000
#     line =[]
#     line.append('# -----------------cooling step-----------------\n')
#     line.append('\n')
#     line.append('reset_timestep 0\n')
#     line.append('timestep 0.001\n')
#     line.append('fix 1 all npt temp 466.75 .1 .1 couple xy  x 0.0 0.0 1.0  y 0.0 0.0 1.0\n')
#     line.append('thermo 10\n')
#     line.append('thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n')
#     line.append('dump 1 all custom 10 ' + str(path) + 'dump_step_equil id type x y z c_csym c_eng\n')
#     line.append('run ' + str(MaxEval1) + '\n')
#     line.append('unfix 1\n')
#     line.append('undump 1\n')
#     line.append('\n')
#     for i in line:
#         fiw.write(i)

#     return True



