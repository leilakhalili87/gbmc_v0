
import numpy as np
import util_funcs as uf
import os


def file_gen(fil_name):
    fiw = open(fil_name, 'w')
    return fiw, fil_name


def lammps_script_var(fiw, lat_par, Etol, Ftol, MaxIter, MaxEval):
    overlap_cte = np.sqrt(2) * lat_par / 4 
    line = []
    line.append('# Minimization Parameters -------------------------\n')
    line.append('\n')
    line.append('variable Etol equal ' + str(Etol) + '\n')
    line.append('variable Ftol equal ' + str(Ftol) + '\n')
    line.append('variable MaxIter equal ' + str(MaxIter) + '\n')
    line.append('variable MaxEval equal ' + str(MaxEval) + '\n')
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


def lammps_script_var_anneal(fiw, lat_par, Etol, Ftol, MaxIter, MaxEval, Iter_heat, Iter_equil, Iter_cool):
    overlap_cte = np.sqrt(2) * lat_par / 4 
    line = []
    line.append('# Minimization Parameters -------------------------\n')
    line.append('\n')
    line.append('variable Etol equal ' + str(Etol) + '\n')
    line.append('variable Ftol equal ' + str(Ftol) + '\n')
    line.append('variable MaxIter equal ' + str(MaxIter) + '\n')
    line.append('variable MaxEval equal ' + str(MaxEval) + '\n')
    line.append('variable Iter_heat equal ' + str(Iter_heat) + '\n')
    line.append('variable Iter_equil equal ' + str(Iter_equil) + '\n')
    line.append('variable Iter_cool equal ' + str(Iter_cool) + '\n')
    line.append('\n')
    line.append('# Structural variables------------------------------\n')
    line.append('\n')
    line.append('variable LatParam equal ' + str(lat_par) + '\n')
    line.append('# ------------------------------------------------\n')
    line.append('\n')
    line.append('variable cnt equal 1\n')
    line.append('# ------------------------------------------------\n')
    line.append('\n')

    for i in line:
        fiw.write(i)

    return True



def script_init_sim(fiw, non_p):
    if non_p == 0:
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
    line.append('boundary ' + str(bound) + '\n')
    line.append('atom_style atomic\n')
    line.append('atom_modify map array\n')
    line.append('\n')
    for i in line:
        fiw.write(i)

    return True


def define_box(fiw, untilted, tilt, box_type):
    """
    """
    if box_type == 'block':
        whole_box = 'region whole block ' + str(untilted[0][0]) + ' ' + str(untilted[0][1]) + ' ' +\
                     str(untilted[1][0]) + ' ' + str(untilted[1][1]) + ' ' + str(untilted[2][0]) + ' '\
                     + str(untilted[2][1]) + ' units box'
    elif box_type == 'prism':
        whole_box = 'region whole prism ' + str(untilted[0][0]) + ' ' + str(untilted[0][1]) + ' ' +\
                     str(untilted[1][0]) + ' ' + str(untilted[1][1]) + ' ' + str(untilted[2][0]) + ' '\
                     + str(untilted[2][1]) + ' ' + str(tilt[0]) + ' ' + str(tilt[1]) + ' ' + str(tilt[2])\
                     + ' units box'
    create_box = 'create_box 2 whole\n'
    line = []
    line.append('# ---------Creating the Atomistic Structure--------\n')
    line.append('\n')
    line.append('lattice fcc ${LatParam}\n')
    line.append(str(whole_box) + '\n')
    line.append(str(create_box) + '\n')

    for i in line:
        fiw.write(i)

    return True


def define_fix_rigid(fiw, untilted, tilt, box_type, tol_fix_reg, non_p, step):
    """
    """
    if step == 1:
        untilted[non_p, :] = untilted[non_p, :] + 2 * np.array([tol_fix_reg, - tol_fix_reg])
    else:
        untilted[non_p, :] = untilted[non_p, :] + 3 * np.array([tol_fix_reg, - tol_fix_reg])

    if box_type == 'block':
        rigid_reg = 'region reg_fix block ' + str(untilted[0][0]) + ' ' + str(untilted[0][1]) + ' ' +\
                    str(untilted[1][0]) + ' ' + str(untilted[1][1]) + ' ' + str(untilted[2][0]) + ' ' +\
                    str(untilted[2][1]) + ' units box'
    elif box_type == 'prism':
        rigid_reg = 'region reg_fix prism ' + str(untilted[0][0]) + ' ' + str(untilted[0][1]) + ' ' +\
                     str(untilted[1][0]) + ' ' + str(untilted[1][1]) + ' ' + str(untilted[2][0]) + ' '\
                     + str(untilted[2][1]) + ' ' + str(tilt[0]) + ' ' + str(tilt[1]) + ' ' + str(tilt[2])\
                     + ' units box'
    line = []
    line.append(str(rigid_reg) + '\n')
    line.append('\n')
    line.append("#---------Defining the fix rigid group--------------\n")
    line.append('group non_fix region reg_fix\n')
    line.append('group fix_reg subtract all non_fix\n')
    line.append('fix 2 fix_reg rigid single reinit yes\n')
    line.append('\n')

    for i in line:
        fiw.write(i)

    return True


def script_pot(fiw, pot_path):
    line = []
    line.append('# -------Defining the potential functions----------\n')
    line.append('\n')
    line.append('pair_style eam/alloy\n')
    line.append('pair_coeff * * ' + str(pot_path) + 'Al99.eam.alloy Al Al\n')

    line.append('neighbor 2 bin\n')
    line.append('neigh_modify delay 5 check yes\n')

    for i in line:
        fiw.write(i)

    return True


def script_read_dump(fiw, dump_name):
    line = []
    line.append('read_dump ' + str(dump_name) + ' 0 x y z box yes add yes \n')
    for i in line:
        fiw.write(i)

    return True


def script_overlap(fiw, untilted, tol_fix_reg, non_p, step):
    untilted[non_p, :] = untilted[non_p, :] + np.array([-tol_fix_reg, tol_fix_reg])
    if non_p == 0:
        var = 'x'
    elif non_p == 1:
        var = 'y'
    else:
        var = 'z'
    line = []
    line.append('group lower type 2 \n')
    line.append('group upper type 1\n')
    if step == 1:
        line.append('delete_atoms overlap ${OverLap}  upper lower\n')
        # line.append('change_box all ' + str(var) + ' final ' + str(untilted[non_p, 0]) + ' ' + str(untilted[non_p, 1])
        #             + ' units box\n')

    for i in line:
        fiw.write(i)

    return True


def script_compute(fiw):
    line = []
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


def script_min_sec(fiw, output, non_p, box_type):
    line = []
    line.append('\n')
    line.append('#----------------------Run minimization 1--------------------\n')
    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('thermo 100\n')
    line.append('thermo_style custom step pe lx ly lz xy xz yz xlo xhi ylo yhi zlo zhi press pxx pyy pzz '
                'c_eatoms c_MinAtomEnergy\n')
    line.append('thermo_modify lost ignore\n')
    line.append('min_style cg\n')
    line.append('minimize 1e-5 1e-5 5000 10000\n')

    line.append('\n')
    line.append('#----------------------Run minimization 2--------------------\n')
    line.append('\n')
    line.append('thermo 100\n')
    line.append('thermo_style custom step pe lx ly lz xy xz yz xlo xhi ylo yhi zlo zhi press pxx pyy pzz '
                'c_eatoms c_MinAtomEnergy\n')
    line.append('thermo_modify lost ignore\n')
    if non_p == 0:
        if box_type == "block":
            line.append('fix 1 all box/relax y 0 z 0 vmax .001\n')
        else:
            line.append('fix 1 all box/relax y 0 z 0 yz 0 vmax .001\n')
    elif non_p == 1:
        if box_type == "block":
            line.append('fix 1 all box/relax x 0 z 0 vmax .001\n')
        else:
            line.append('fix 1 all box/relax x 0 z 0 xz 0 vmax .001\n')
    else:
        if box_type == "block":
            line.append('fix 1 all box/relax x 0 y 0 vmax .001\n')
        else:
            line.append('fix 1 all box/relax x 0 y 0 xy 0 vmax .001\n')
    line.append('min_style cg\n')
    line.append('minimize ${Etol} ${Ftol} ${MaxIter} ${MaxEval}\n')
    line.append('\n')
    line.append('#----------------------Run 0 to dump--------------------\n')
    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('reset_ids\n')
    line.append('thermo 100\n')
    line.append('thermo_style custom step pe lx ly lz xy xz yz xlo xhi ylo yhi zlo zhi press pxx pyy pzz '
                'c_eatoms c_MinAtomEnergy\n')
    line.append('thermo_modify lost ignore\n')
    line.append('dump 1 all custom ${MaxIter} ' + str(output) + ' id type x y z c_csym c_eng\n')
    line.append('dump_modify 1 every ${MaxIter} sort id first yes\n')
    line.append('run 0\n')
    line.append('unfix 1\n')
    line.append('undump 1\n')

    for i in line:
        fiw.write(i)

    return True


def script_heating(fiw, output, Tm, non_p):
    T = Tm / 2
    line = []

    line.append('velocity all create ' + str(T) + ' 235911\n')
    line.append('fix 1 all nve \n')
#thermalize at temperature
    line.append('thermo 100\n')
    line.append('thermo_modify flush yes\n')
    line.append('run 10000\n')
    line.append('unfix 1\n')

    line.append('# -----------------heating step-----------------\n')
    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('timestep 0.001\n')
    if non_p == 0:
        line.append('fix 1 all npt temp .01 ' + str(T) + ' $(100.0*dt) couple yz  y 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    elif non_p == 1:
        line.append('fix 1 all npt temp .01 ' + str(T) + ' $(100.0*dt) couple xz  x 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    else:
        line.append('fix 1 all npt temp .01 ' + str(T) + ' $(100.0*dt) couple xy  x 0.0 0.0 $(1000.0*dt)  y 0.0 0.0 $(1000.0*dt)\n')

    # line.append('fix 1 all npt temp .1 ' + str(T) + ' .1 couple xy  x 0.0 0.0 1.0  y 0.0 0.0 1.0\n')
    line.append('thermo 100\n')
    line.append('thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n')
    line.append('thermo_modify lost ignore\n')
    line.append('dump 1 all custom 100 ./lammps_dump/heat_0 id type x y z c_csym c_eng\n')
    line.append('run ${Iter_heat}\n')
    # line.append('dump 1 all custom ${Iter_heat} ' + str(output) + ' id type x y z c_csym c_eng\n')
    # line.append('dump_modify 1 every ${Iter_heat} sort id first yes\n')
    line.append('unfix 1\n')
    line.append('undump 1\n')
    line.append('\n')
    for i in line:
        fiw.write(i)

    return True


def script_equil(fiw, output, Tm, non_p):
    T = Tm / 2
    line = []
    line.append('# -----------------equilibrium step-----------------\n')
    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('timestep 0.0001\n')
    if non_p == 0:
        line.append('fix 1 all npt temp ' + str(T) + ' ' + str(T) + ' $(100.0*dt) couple yz  y 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    elif non_p == 1:
        line.append('fix 1 all npt temp ' + str(T) + ' ' + str(T) + ' $(100.0*dt) couple xz  x 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    else:
        line.append('fix 1 all npt temp ' + str(T) + ' ' + str(T) + ' $(100.0*dt) couple xy  x 0.0 0.0 $(1000.0*dt)  y 0.0 0.0 $(1000.0*dt)\n')

    # line.append('fix 1 all npt temp  ' + str(T) + ' ' + str(T) + ' 0.1 couple xy  x 0.0 0.0 1.0  y 0.0 0.0 1.0\n')
    line.append('thermo 100\n')
    line.append('thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n')
    line.append('thermo_modify lost ignore\n')
    line.append('dump 1 all custom 10 ./lammps_dump/equil id type x y z c_csym c_eng\n')
    line.append('run ${Iter_equil}\n')
    # line.append('dump 1 all custom ${Iter_equil} ' + str(output) + ' id type x y z c_csym c_eng\n')
    # line.append('dump_modify 1 every ${Iter_equil} sort id first yes\n')
    line.append('unfix 1\n')
    line.append('undump 1\n')
    line.append('\n')
    for i in line:
        fiw.write(i)

    return True


def script_cooling(fiw, output, Tm, non_p):
    T = Tm / 2
    line = []
    line.append('# -----------------cooling step-----------------\n')
    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('timestep 0.001\n')
    if non_p == 0:
        line.append('fix 1 all npt temp ' + str(T) + ' .01 $(100.0*dt) couple yz  y 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    elif non_p == 1:
        line.append('fix 1 all npt temp ' + str(T) + ' .01 $(100.0*dt) couple xz  x 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    else:
        line.append('fix 1 all npt temp ' + str(T) + ' .01 $(100.0*dt) couple xy  x 0.0 0.0 $(1000.0*dt)  y 0.0 0.0 $(1000.0*dt)\n')
    # line.append('fix 1 all npt temp ' + str(T) + ' .1 .1 couple xy  x 0.0 0.0 1.0  y 0.0 0.0 1.0\n')
    line.append('thermo 10\n')
    line.append('thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n')
    line.append('thermo_modify lost ignore\n')
    # line.append('dump 1 all custom 100 ' + str(output) + ' id type x y z c_csym c_eng\n')
    line.append('run ${Iter_cool}\n')
    line.append('dump 1 all custom ${Iter_cool} ' + str(output) + ' id type x y z c_csym c_eng\n')
    line.append('dump_modify 1 every ${Iter_cool} sort id first yes\n')
    line.append('run 0\n')
    line.append('unfix 1\n')
    line.append('undump 1\n')
    line.append('\n')
    for i in line:
        fiw.write(i)

    return True


def script_nve(fiw, output, Tm, non_p, Iter_heat, Iter_equil, Iter_cool):
    T = Tm / 2
    line = []

    line.append('# -----------------heating step-----------------\n')

    line.append('fix 1 all nvt temp .01 ' + str(T) + ' 1\n')
    line.append('thermo 100\n')
    line.append('run ${Iter_heat}\n')
    line.append('unfix 1\n')
    line.append('# -----------------equilibrium step-----------------\n')

    line.append('fix 1 all nvt temp ' + str(T) + ' ' + str(T) + ' 1\n')
    line.append('thermo 100\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run ${Iter_equil}\n')
    line.append('unfix 1\n')

    line.append('# -----------------cooling step-----------------\n')
    line.append('fix 1 all nvt temp  ' + str(T) + ' .01 1\n')
    line.append('thermo 100\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run ${Iter_cool}\n')
    line.append('unfix 1\n')

    line.append('fix 1 all nvt temp .01 .01 1\n')
    line.append('thermo 100\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run ${Iter_equil}\n')
    line.append('unfix 1\n')

    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('timestep 0.001\n')
    if non_p == 0:
        line.append('fix 1 all npt temp .01 .01 .1 couple yz  y 0.0 0.0 1  z 0.0 0.0 1\n')
    elif non_p == 1:
        line.append('fix 1 all npt temp .01 .01 .1 couple xz  x 0.0 0.0 1  z 0.0 0.0 1\n')
    else:
        line.append('fix 1 all npt temp .01 .01 .1 couple xy  x 0.0 0.0 1  y 0.0 0.0 1\n')
    # line.append('fix 1 all npt temp ' + str(T) + ' .1 .1 couple xy  x 0.0 0.0 1.0  y 0.0 0.0 1.0\n')
    line.append('thermo 100\n')
    line.append('thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n')
    line.append('thermo_modify lost ignore\n')
    # line.append('dump 1 all custom 1000 ' + str(output) + ' id type x y z c_csym c_eng\n')
    line.append('run ${Iter_equil}\n')
    line.append('dump 1 all custom ${Iter_equil} ' + str(output) + ' id type x y z c_csym c_eng\n')
    line.append('dump_modify 1 every ${Iter_equil} sort id first yes\n')
    line.append('run 0\n')
    line.append('unfix 1\n')
    line.append('undump 1\n')
    line.append('\n')

    for i in line:
        fiw.write(i)

    return True

def script_just_nvt(fiw, output, Tm, non_p, Iter_heat, Iter_equil, Iter_cool):

    T = Tm / 2
    line = []
    line.append('velocity all create ' + str(T) + ' 235911\n')
    line.append('fix 1 all nve \n')
#thermalize at temperature
    line.append('thermo 100\n')
    line.append('thermo_modify flush yes\n')
    line.append('run 1000\n')
    line.append('unfix 1\n')

    line.append('# -----------------heating step-----------------\n')
    line.append('\n')

    line.append('fix 1 all nvt temp .01 ' + str(T) + ' $(100.0*dt)\n')
    line.append('thermo 100\n')
    line.append('run ${Iter_heat}\n')
    line.append('unfix 1\n')
    line.append('# -----------------equilibrium step-----------------\n')
    line.append('\n')

    line.append('fix 1 all nvt temp ' + str(T) + ' ' + str(T) + ' $(100.0*dt)\n')
    line.append('dump 1 all custom 100 equi id type x y z c_csym c_eng\n')
    line.append('thermo 100\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run ${Iter_equil}\n')
    line.append('unfix 1\n')
    line.append('undump 1\n')

    line.append('# -----------------cooling step-----------------\n')
    line.append('\n')

    line.append('fix 1 all nvt temp  ' + str(T) + ' .01 $(100.0*dt)\n')
    line.append('thermo 100\n')
    line.append('thermo_modify lost ignore\n')
    line.append('dump 1 all custom 100 cool id type x y z c_csym c_eng\n')
    line.append('run ${Iter_cool}\n')
    line.append('unfix 1\n')
    line.append('undump 1\n')
    line.append('fix 1 all nvt temp .01 .01 1\n')
    line.append('thermo 100\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run ${Iter_equil}\n')
    line.append('unfix 1\n')

    line.append('# -----------------heating step-----------------\n')
    line.append('\n')

    if non_p == 0:
        line.append('fix 1 all npt temp .01 ' + str(T) + ' $(100.0*dt) couple yz  y 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    elif non_p == 1:
        line.append('fix 1 all npt temp .01 ' + str(T) + ' $(100.0*dt) couple xz  x 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    else:
        line.append('fix 1 all npt temp .01 ' + str(T) + ' $(100.0*dt) couple xy  x 0.0 0.0 $(1000.0*dt)  y 0.0 0.0 $(1000.0*dt)\n')
    line.append('thermo 100\n')
    line.append('thermo_modify lost ignore\n')


    line.append('dump 1 all custom ${Iter_equil} ' + str(output) + ' id type x y z c_csym c_eng\n')
    line.append('dump_modify 1 every ${Iter_equil} sort id first yes\n')
    line.append('run 0\n')
    line.append('undump 1\n')
    line.append('\n')

    for i in line:
        fiw.write(i)

    return True

def script_nve_nvt(fiw, output, Tm, non_p, Iter_heat, Iter_equil, Iter_cool):

    T = Tm / 2
    line = []
    line.append('velocity all create ' + str(T) + ' 235911\n')
    line.append('fix 1 all nve \n')
#thermalize at temperature
    line.append('thermo 100\n')
    line.append('thermo_modify flush yes\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run 1000\n')
    line.append('unfix 1\n')

    line.append('# -----------------equilibrium step-----------------\n')
    line.append('\n')

    line.append('fix 1 all nvt temp ' + str(T) + ' ' + str(T) + ' $(100.0*dt)\n')
    line.append('dump 1 all custom 100 equi id type x y z c_csym c_eng\n')
    line.append('thermo 100\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run ${Iter_equil}\n')
    line.append('unfix 1\n')
    line.append('undump 1\n')

    line.append('# -----------------cooling step-----------------\n')
    line.append('\n')

    line.append('fix 1 all nvt temp  ' + str(T) + ' .01 .01\n')
    line.append('thermo 100\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run ${Iter_cool}\n')
    line.append('dump 1 all custom ${Iter_cool} ' + str(output) + ' id type x y z c_csym c_eng\n')
    line.append('dump_modify 1 every ${Iter_cool} sort id first yes\n')
    line.append('run 0\n')
    line.append('undump 1\n')
    line.append('\n')

    for i in line:
        fiw.write(i)

    return True


def script_main_min(fil_name, lat_par, tol_fix_reg, dump_name, pot_path, non_p, output,\
                    step, Etol, Ftol, MaxIter, MaxEval):
    fiw, file_name = file_gen(fil_name)
    lammps_script_var(fiw, lat_par, Etol, Ftol, MaxIter, MaxEval)
    script_init_sim(fiw, non_p)
    box_bound = uf.box_size_reader(dump_name)
    untilted, tilt, box_type = uf.define_bounds(box_bound)
    define_box(fiw, untilted, tilt, box_type)
    script_read_dump(fiw, dump_name)
    script_pot(fiw, pot_path)
    script_overlap(fiw, untilted, tol_fix_reg, non_p, step)
    # define_fix_rigid(fiw, untilted, tilt, box_type, tol_fix_reg, non_p, step)
    script_compute(fiw)
    script_min_sec(fiw, output, non_p, box_type)


def run_lammps_min(filename0, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path,\
                   output, step=2, Etol=1e-25, Ftol=1e-25, MaxIter=5000, MaxEval=10000):
    data = uf.compute_ovito_data(filename0)
    non_p = uf.identify_pbc(data)
    script_main_min(fil_name, lat_par, tol_fix_reg, filename0, pot_path, non_p, output,step, Etol, Ftol, MaxIter, MaxEval)
    os.system(str(lammps_exe_path) + '< ./' + fil_name)


def script_main_anneal(fil_name, lat_par, tol_fix_reg, dump_name, pot_path, non_p, output,\
                       Tm, step, Etol, Ftol, MaxIter, MaxEval, Iter_heat, Iter_equil, Iter_cool):
    fiw, file_name = file_gen(fil_name)
    # lammps_script_var(fiw, lat_par, Etol, Ftol, MaxIter, MaxEval)
    lammps_script_var_anneal(fiw, lat_par, Etol, Ftol, MaxIter, MaxEval, Iter_heat, Iter_equil, Iter_cool)
    script_init_sim(fiw, non_p)
    box_bound = uf.box_size_reader(dump_name)
    untilted, tilt, box_type = uf.define_bounds(box_bound)
    define_box(fiw, untilted, tilt, box_type)
    
    script_read_dump(fiw, dump_name)
    script_pot(fiw, pot_path)
    script_compute(fiw)
    # define_fix_rigid(fiw, untilted, tilt, box_type, tol_fix_reg, non_p, step)
    # script_heating(fiw, output, Tm, non_p)
    # script_equil(fiw, output, Tm, non_p)
    # script_cooling(fiw, output, Tm, non_p)
    
    # script_min_sec(fiw, output, non_p, box_type)
    script_nve_nvt(fiw, output, Tm, non_p, Iter_heat, Iter_equil, Iter_cool)



def run_lammps_anneal(filename0, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path,\
                      output, Tm,  step=2, Etol=1e-25, Ftol=1e-25, MaxIter=5000, MaxEval=10000, Iter_heat=1000, Iter_equil=10000, Iter_cool=12000):
    data = uf.compute_ovito_data(filename0)
    non_p = uf.identify_pbc(data)
    script_main_anneal(fil_name, lat_par, tol_fix_reg, filename0, pot_path, non_p, output,\
                       Tm, step, Etol, Ftol, MaxIter, MaxEval, Iter_heat, Iter_equil, Iter_cool)
    os.system(str(lammps_exe_path) + '< ./' + fil_name)


# lammps_exe_path = '/home/leila/Downloads/mylammps/src/lmp_mpi'
# lat_par = 4.05
# tol_fix_reg = lat_par * 5
# filename0 = './tests/data/dump_1'
# fil_name = 'in.min_1'
# pot_path = './lammps_dump/'
# Tm = 1000
# out_min_1 = './lammps_dump/dump.0'
# run_lammps_min(filename0, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path, out_min_1,\
#                step=1, Etol=1e-25, Ftol=1e-25, MaxIter=10000, MaxEval=10000)

# fil_name1 = 'in.anneal'
# out_heat = './lammps_dump/heat'
# run_lammps_anneal(out_min_1, fil_name1, pot_path, lat_par, tol_fix_reg, lammps_exe_path, out_heat,\
#                    Tm,  step=2, Etol=1e-25, Ftol=1e-25, MaxIter=10000, MaxEval=10000, Iter_heat=1000, Iter_equil=20000, Iter_cool=10000)

# lines = open(out_heat, 'r').readlines()
# lines[1] = '0\n'
# out = open(out_heat, 'w')
# out.writelines(lines)
# out.close()


# fil_name2 = 'in.min_2'
# out_min_2 = './lammps_dump/final'
# run_lammps_min(out_heat, fil_name2, pot_path, lat_par, tol_fix_reg, lammps_exe_path, out_min_2,\
#                step=2, Etol=1e-25, Ftol=1e-25, MaxIter=10000, MaxEval=10000)