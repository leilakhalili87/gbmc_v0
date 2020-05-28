import numpy as np
import pickle as pkl
from numpy import linalg as LA


def lammps_box(pkl_name):
    """
    Function calculates the box bound and the atom coordinates of the GB.

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
    jar = open(pkl_name, 'rb')
    gb_attr = pkl.load(jar)
    jar.close()
    sim_cell = gb_attr['cell']
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

    origin_o = np.array([np.min(sim_cell[0, 0:3]), np.min(sim_cell[1, 0:3]), np.min(sim_cell[2, 0:3])])

    cell = sim_cell[:, 0:3] + sim_cell[:, 3].reshape(3, 1)

    xlo, xhi = np.min(cell[0]) - origin_o[0], np.max(cell[0])
    ylo, yhi = np.min(cell[1]) - origin_o[1], np.max(cell[1])
    zlo, zhi = np.min(cell[2]) - origin_o[2], np.max(cell[2])

    vec_a = sim_cell[:, 0]
    vec_b = sim_cell[:, 1]
    vec_c = sim_cell[:, 2]

    len_b = LA.norm(vec_b)
    len_c = LA.norm(vec_c)

    cos_gamma = np.dot(vec_a, vec_b) / LA.norm(vec_b) / LA.norm(vec_a)
    cos_beta = np.dot(vec_a, vec_c) / LA.norm(vec_a) / LA.norm(vec_c)
    cos_alpha = np.dot(vec_b, vec_c) / LA.norm(vec_b) / LA.norm(vec_c)

    xz = cos_beta * len_c
    xy = cos_gamma * len_b
    yz = (len_b * len_c * cos_alpha - xy * xz) / (len_b * len_b - xy * xy)

    xlo_bound = xlo + np.min(np.array([0, xy, xz, xy + xz]))
    xhi_bound = xhi + np.max(np.array([0, xy, xz, xy + xz]))
    ylo_bound = ylo + np.min(np.array([0, yz]))
    yhi_bound = yhi + np.max(np.array([0, yz]))
    zlo_bound = zlo
    zhi_bound = zhi

    box_bound = np.array([[xlo_bound, xhi_bound, xy], [ylo_bound, yhi_bound,  xz], [zlo_bound, zhi_bound, yz]])
    return box_bound, dump_lamp


def write_lammps(filename0, box_bound, dump_lamp):
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


box_bound, dump_lamp = lammps_box('gb_attr.pkl')
write_lammps("my_dump.txt", box_bound, dump_lamp)
