import numpy as np
import pickle as pkl
from numpy import linalg as LA
import os


def lammps_box(pkl_name):
    """
    Function calculates the box bound and the atom coordinates of the GB simulation.

    Parameters
    ------------
    pkl_name :
        The name of the pkl file which contains the simulation cell ( a 3*4 numpy array
        where the first 3 columns are the cell vectors and the last column is the box origin),
        the cordinates of the upper and lower grain.

    Returns
    ----------
    box_bound :
        The box bound needed to write lammps dump file which is 9 parameters: xlo, xhi, ylo,
        yhi, zlo, zhi, xy, xz, yz
    dump_lamp :
        A numpy nd.array having atom ID, atom type( 1 for upper grain and 2 for lower grain), x, y, z
    """
    
    jar = open(pkl_name, 'rb'); gb_attr = pkl.load(jar); jar.close();
    

    u_pts = gb_attr['upts']
    len_u = np.shape(u_pts)[0]
    u_type = np.zeros((len_u, 1)) + 1
    upper = np.concatenate((u_type, u_pts), axis=1)

    l_pts = gb_attr['lpts']
    len_l = np.shape(l_pts)[0]
    l_type = np.zeros((len_l, 1)) + 2
    lower = np.concatenate((l_type, l_pts), axis=1)

    all_atoms = np.concatenate((lower, upper))
    num_atoms = len_u + len_l
    ID = np.arange(num_atoms).reshape(num_atoms, 1) + 1
    dump_lamp = np.concatenate((ID, all_atoms), axis=1)

    sim_cell = gb_attr['cell'];
    # origin_o = np.array([np.min(sim_cell[0, 0:3]), np.min(sim_cell[1, 0:3]), np.min(sim_cell[2, 0:3])])
    origin_o = sim_cell[:,3];

    cell = sim_cell[:, 0:3] + sim_cell[:, 3].reshape(3, 1)

    xlo1, xhi1 = np.min(cell[0]) - origin_o[0], np.max(cell[0])
    ylo1, yhi1 = np.min(cell[1]) - origin_o[1], np.max(cell[1])
    zlo1, zhi1 = np.min(cell[2]) - origin_o[2], np.max(cell[2])
    

    vec_a = sim_cell[:, 0]
    vec_b = sim_cell[:, 1]
    vec_c = sim_cell[:, 2]

    len_b = LA.norm(vec_b)
    len_c = LA.norm(vec_c)

    cos_gamma = np.dot(vec_a, vec_b) / LA.norm(vec_b) / LA.norm(vec_a)
    cos_beta = np.dot(vec_a, vec_c) / LA.norm(vec_a) / LA.norm(vec_c)
    cos_alpha = np.dot(vec_b, vec_c) / LA.norm(vec_b) / LA.norm(vec_c)

    xz1 = cos_beta * len_c
    xy1 = cos_gamma * len_b
    yz1 = (len_b * len_c * cos_alpha - xy1 * xz1) / (len_b * len_b - xy1 * xy1)

    ####
    # “origin” at (xlo,ylo,zlo)
    xlo = origin_o[0]; ylo = origin_o[1]; zlo = origin_o[2];
    # a = (xhi-xlo,0,0);
    xhi = sim_cell[0,0] + xlo;
    # b = (xy,yhi-ylo,0);
    xy = sim_cell[0,1]; yhi = sim_cell[1,1]+ylo;
    # c = (xz,yz,zhi-zlo)
    xz = sim_cell[0,2]; yz = sim_cell[1,2]; zhi = sim_cell[2,2]+zlo;

    print("------------------------")
    print("xlo diff: "+str(xlo-xlo1))
    print("ylo diff: "+str(ylo-ylo1))
    print("zlo diff: "+str(zlo-zlo1))
    print("xhi diff: "+str(xhi-xhi1))
    print("yhi diff: "+str(yhi-yhi1))
    print("zhi diff: "+str(zhi-zhi1))
    print("xy diff: "+str(xy-xy1))
    print("xz diff: "+str(xz-xz1))
    print("yz diff: "+str(yz-yz1))
    print("------------------------")

    xlo_bound = xlo + np.min(np.array([0, xy, xz, xy + xz]))
    xhi_bound = xhi + np.max(np.array([0, xy, xz, xy + xz]))
    ylo_bound = ylo + np.min(np.array([0, yz]))
    yhi_bound = yhi + np.max(np.array([0, yz]))
    zlo_bound = zlo
    zhi_bound = zhi

    box_bound = np.array([[xlo_bound, xhi_bound, xy], [ylo_bound, yhi_bound,  xz], [zlo_bound, zhi_bound, yz]])
    return box_bound, dump_lamp


def write_lammps_dump(filename0, box_bound, dump_lamp):
    """
    Function writes the lammps dump file.

    Parameters
    ------------
    filename0 :
        Name of the lammps dump file
    box_bound :
        The box bound needed to write lammps dump file which is 9 parameters: xlo, xhi, ylo, yhi,
        zlo, zhi, xy, xz, yz
    dump_lamp :
        A numpy nd.array having atom ID, atom type( 1 for upper grain and 2 for lower grain), x, y, z

    Returns
    ----------
    """
    num_atoms = np.shape(dump_lamp)[0]
    file = open(filename0, "w")
    file.write("ITEM: TIMESTEP\n")
    file.write("0\n")
    file.write("ITEM: NUMBER OF ATOMS\n")
    file.write(str(num_atoms) + "\n")
    # file.write("ITEM: BOX BOUNDS xy xz yz pp ff pp\n")
    file.write("ITEM: BOX BOUNDS xy xz yz pp pp ff\n")
    file.write(' '.join(map(str, box_bound[0])) + "\n")
    file.write(' '.join(map(str, box_bound[1])) + "\n")
    file.write(' '.join(map(str, box_bound[2])) + "\n")
    file.write("ITEM: ATOMS id type x y z\n")
    file.close()
    mat = np.matrix(dump_lamp)
    with open(filename0, 'a') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%d %d %.10f %.10f %.10f')


def write_lammps_script(dump_name, path, script_name,  box_bound):
    """
    Function writes the lammps script to minimize the simulation box.

    Parameters
    ------------
    dump_name :
        Name of the lammps dump file.
    path :
        The path that the lammps dump files will be saved.
    script_name :
        The name of the lammps script created for minimization.
    box_bound :
        The box bound needed to write lammps dump file which is 9 parameters: xlo, xhi, ylo, yhi,
        zlo, zhi, xy, xz, yz


    Returns
    ----------
    """
    fiw = open(str(path) + str(script_name), 'w')
    line = []
    line.append('# Minimization Parameters -------------------------\n')
    line.append('\n')
    line.append('variable Etol equal 1e-25\n')
    line.append('variable Ftol equal 1e-25\n')
    line.append('variable MaxIter equal 5000\n')
    line.append('variable MaxEval equal 10000\n')
    line.append('variable Infn equal -10000000000\n')
    line.append('variable Inf equal 10000000000\n')
    line.append('# ------------------------------------------------\n')
    line.append('\n')
    line.append('variable LatParam equal 4.05\n')
    line.append('# ------------------------------------------------\n')
    line.append('\n')
    line.append('variable GBname index in.minimize0\n')
    line.append('variable cnt equal 1\n')
    line.append('# ------------------------------------------------\n')
    line.append('\n')
    line.append('variable OverLap equal 1.43\n')
    line.append('\n')
    line.append('# -----------Initializing the Simulation-----------\n')
    line.append('\n')
    line.append('clear\n')
    line.append('units metal\n')
    line.append('dimension 3\n')
    # line.append('boundary '+bcon[0]+' '+bcon[1]+' '+bcon[2]+'\n')
    line.append('boundary p p f \n')
    line.append('atom_style atomic\n')
    line.append('\n')
    line.append('# ---------Creating the Atomistic Structure--------\n')
    line.append('\n')
    line.append('lattice fcc ${LatParam}\n')

    line.append('region whole prism ' + str(box_bound[0][0]) + ' ' +
                str(box_bound[0][1]) + ' ' + str(box_bound[1][0]) + ' ' + str(box_bound[1][1]) + ' ' +
                str(box_bound[2][0]) + ' ' + str(box_bound[2][1]) + ' ' + str(box_bound[0][2])
                + ' ' + str(box_bound[1][2]) + ' ' + str(box_bound[2][2]) + '\n')

    line.append('create_box 2 whole\n')
    line.append('read_dump ' + str(dump_name) + ' 0 x y z box yes add yes \n')
    line.append('\n')

    # line.append('region lower prism ' + str(box_bound[0][0]) + ' ' + str(box_bound[0][1]) + ' ' +
    #             str(box_bound[1][0]) + ' ' + str(box_bound[1][1]) + ' 0 ' + str(box_bound[2][1]) + ' ' +
    #             str(box_bound[0][2]) + ' ' + str(box_bound[1][2]) + ' ' + str(box_bound[2][2]) + '\n')

    # line.append('region upper prism ' + str(box_bound[0][0]) + ' ' + str(box_bound[0][1]) + ' ' +
    #             str(box_bound[1][0]) + ' ' + str(box_bound[1][1]) + ' ' + str(box_bound[2][0]) + ' 0 ' +
    #             str(box_bound[0][2]) + ' ' + str(box_bound[1][2]) + ' ' + str(box_bound[2][2]) + '\n')

    line.append('group lower type 2\n')
    line.append('group upper type 1\n')
    line.append('\n')
    line.append('# -------Defining the potential functions----------\n')
    line.append('\n')
    line.append('pair_style eam/alloy\n')
    line.append('pair_coeff * * ' + str(path) + 'Al99.eam.alloy Al Al\n')
    line.append('delete_atoms overlap ${OverLap}  upper lower\n')
    line.append('neighbor 2 bin\n')
    line.append('neigh_modify delay 10 check yes\n')
    line.append('\n')
    line.append('# ---------Computing Simulation Parameters---------\n')
    line.append('\n')
    line.append('compute csym all centro/atom fcc\n')
    line.append('compute eng all pe/atom\n')
    line.append('compute eatoms all reduce sum c_eng\n')
    line.append('compute MinAtomEnergy all reduce min c_eng\n')
    line.append('\n')
    line.append('# ------1st Minimization:Relaxing the bi-crystal------\n')
    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('thermo 10\n')
    line.append('thermo_style custom step pe lx ly lz xy xz yz xlo xhi ylo yhi zlo zhi ' +
                'press pxx pyy pzz c_eatoms c_MinAtomEnergy\n')
    line.append('thermo_modify lost ignore\n')
    line.append('dump 1 all custom 10 ' + str(path) + 'dump_befor.${cnt} id type x y z c_csym c_eng\n')
    line.append('fix 1 all box/relax x 0 y 0 xy 0\n')
    line.append('min_style cg\n')
    line.append('minimize ${Etol} ${Ftol} ${MaxIter} ${MaxEval}\n')
    line.append('undump 1\n')
    line.append('\n')
    line.append('# -----------------Dumping Outputs-----------------\n')
    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('timestep 0.001\n')
    line.append('thermo 10\n')
    line.append('thermo_style custom step pe lx ly lz press pxx pyy pzz c_eatoms\n')
    line.append('thermo_modify lost ignore\n')
    line.append('dump 1 all custom 10 ' + str(path) + 'dump_after.${cnt} id type x y z c_csym c_eng\n')
    line.append('run 0\n')
    line.append('undump 1\n')
    for i in line:
        fiw.write(i)
    fiw.close()
    return True


box_bound, dump_lamp = lammps_box('./tests/data/gb_attr.pkl')
write_lammps_dump("./tests/data/dump_1", box_bound, dump_lamp)
write_lammps_script('./tests/data/dump_1', './lammps_dump/', 'in.minimize0', box_bound)
lammps_exe_path = '/home/leila/Downloads/lammps-stable/lammps-7Aug19/src/lmp_mpi'
os.system(str(lammps_exe_path) + '< ./lammps_dump/' + 'in.minimize0')

# box_bound, dump_lamp = lammps_box('./tests/data/gb_attr.pkl')
# write_lammps("./tests/data/dump_1", box_bound, dump_lamp)
