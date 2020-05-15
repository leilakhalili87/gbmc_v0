import numpy as np
# import ovito.modifiers as ovm
# from ovito.io import import_file, export_file


def pad_dump_file(data, lat_par, rCut):
    """
    Function to take as input the dump data (from OVITO),find the GB atoms and
    add padding to the GB atoms (including images) within rCut.
    These atoms (with padding) will be triangulated to compute Voronoi vertices and their radii.

    Parameters
    ------------
    data :
        Data object computed using OVITO I/O
    lat_par :
        Lattice parameter for the crystal being simulated
    rCut :
        Cut-off radius for computing Delaunay triangulations

    Returns
    ----------
    pts_w_imgs :
        Points of interest (GB atoms and neighbors) on which Delaunay triangulation is called.
    gb1_inds :
        Indices of the GB atoms
    """

    GbRegion, GbIndex, GbWidth, w_left_SC, w_right_SC = GB_finder(data, lat_par)

    sim_cell = data.cell[...]
    sim_avec = np.array(sim_cell[:, 0])
    sim_bvec = np.array(sim_cell[:, 1])
    sim_cvec = np.array(sim_cell[:, 2])
    sim_orig = np.array(sim_cell[:, 3])

    x1_vec = np.array([sim_avec[0], sim_avec[2]])
    z1_vec = np.array([sim_cvec[0], sim_cvec[2]])
    [nx, nz] = num_rep_2d(x1_vec, z1_vec, rCut)

    pts1, gb1_inds = pad_gb_perp(data, GbRegion, GbIndex, rCut)
    pts_w_imgs = create_imgs(pts1, nx, nz, sim_avec, sim_cvec)
    pts_w_imgs, gb1_inds = (slice_along_planes(sim_orig,
                                               sim_avec, sim_bvec, sim_cvec, rCut,
                                               pts_w_imgs, gb1_inds))
    return pts_w_imgs, gb1_inds


def GB_finder(data, lat_par):
    """
    The function finds the GB region usning Polyhedral Template Matching.

    Parameters
    --------------
    filename0 :
        The lammps dump file
    lat_par:
        The lattice parameter

    Returns
    -----------
    GbRegion:
        The maximum and Minimum value of postion of atoms in Y direction  in the GB region.
    GbWidth :
        GbRegion[1] - GbRegion[0]
    GbIndex :
        The index of atoms in GB
    w_left_SC :'
        The width of the region on the left side of GB which have single crystal structure
    w_right_SC :
        The width of the region on the right side of GB which have single crystal structure
    """

    # num_particles = data.particles.count
    ptm_struct = data.particles['Structure Type'][...]
    position_Y = data.particles['Position'][...][:, 1]

    # length_box = np.max(position_Y) - np.min(position_Y)

    NoSurfArea = []
    # Find the smallest single crystal range
    a = 1
    pos_min = np.min(position_Y)

    while a != 0:
        pos_max = pos_min + lat_par
        a = len(np.where((ptm_struct == 0) & (position_Y < pos_max) & (position_Y > pos_min))[0])
        pos_min += lat_par

    NoSurfArea = NoSurfArea + [pos_min]

    # Find the largest single crystal range
    a = 1
    pos_max = np.max(position_Y)
    while a != 0:
        pos_min = pos_max - lat_par
        a = len(np.where((ptm_struct == 0) & (position_Y < pos_max) & (position_Y > pos_min))[0])
        pos_max -= lat_par

    NoSurfArea = NoSurfArea + [pos_min + lat_par]

    gb_index = np.where((ptm_struct == 0) & (position_Y < NoSurfArea[1]) & (position_Y > NoSurfArea[0]))[0]
    gb_mean = np.mean(position_Y[gb_index])
    gb_std = np.std(position_Y[gb_index])

    # Delete the outliers of gb
    var = position_Y[gb_index] - gb_mean
    GbIndex = gb_index[np.where((var < 3*gb_std) & (var > -3*gb_std))[0]]

    GbY = position_Y[GbIndex]
    GbRegion = [np.min(GbY), np.max(GbY)]
    GbWidth = GbRegion[1] - GbRegion[0]
    w_left_SC = GbRegion[0] - NoSurfArea[0]
    w_right_SC = NoSurfArea[1] - GbRegion[1]

    return GbRegion, GbIndex, GbWidth, w_left_SC, w_right_SC


def num_rep_2d(xvec, yvec, rCut):
    """
    Function finds the number of replications necessary such that thecircle of radius rCut at the
    center of the primitive-cell lies completely inside the super-cell.

    Parameters
    ------------
    xvec :
        The basis vector in x direction in x-z plane
    yvec :
        The basis vector in z direction in x-z plane
    rCut
        Cut-off radius for computing Delaunay triangulations

    Returns
    ------------
    [int(m_x), int(m_y)] :
        int(m_x) is the number of replications in x direction, int(m_y)
        is the number of replication in z direction.

    """
    c_vec_norm = np.linalg.norm(np.cross(xvec, yvec))
    d_y = c_vec_norm/(np.linalg.norm(yvec))
    d_x = c_vec_norm/(np.linalg.norm(xvec))
    m_x = np.ceil(rCut/d_y)
    m_y = np.ceil(rCut/d_x)

    return [int(m_x), int(m_y)]


def pad_gb_perp(data, GbRegion, GbIndex, rCut):
    """
    Function to take as input the dump data (from OVITO), find the GB atoms
    and add padding to the GB atoms  within rCut in Y direction.

    Parameters
    -------------
    data :
        Data object computed using OVITO I/O
    GbRegion :
        Indices of atoms in GB area
    GbIndex :
        Indices of atoms in GB area
    rCut
        Cut-off radius for computing Delaunay triangulations

    Returns
    ---------
    pts1 :
        Indices of the atoms which Y value is in range [GBRegion[0] - rCut, GBRegion[1] + rCut].
    gb1_inds :
        Indices of the GB atoms
    """
    position_X = data.particles['Position'][...][:, 0]
    position_Y = data.particles['Position'][...][:, 1]
    position_Z = data.particles['Position'][...][:, 2]

    Ymin, Ymax = GbRegion[0] - rCut, GbRegion[1] + rCut

    pad1_inds = np.where((position_Y <= Ymax) & (position_Y >= Ymin))[0]

    int1, a1, a2 = np.intersect1d(pad1_inds, GbIndex, return_indices=True)
    gb1_inds = a1

    # Replicate the GB structure along X and Z direction (nx and nz times)
    num1 = np.size(pad1_inds)
    pts1 = np.zeros((num1, 3))
    pts1[:, 0] = np.array(position_X[pad1_inds])
    pts1[:, 1] = np.array(position_Y[pad1_inds])
    pts1[:, 2] = np.array(position_Z[pad1_inds])

    return pts1, gb1_inds


def create_imgs(pts1, nx, nz, sim_avec, sim_cvec):
    """
    Creates the replicates of the main cell in X and Z direction.

    Parameters
    -------------
    pts1 :
        Indices of the atoms which Y value is in range [GBRegion[0] - rCut, GBRegion[1] + rCut].
    nx :
        Number of replications in x direction
    nz :
        Number of replications in z direction
    sim_avec :
        The simulation cell basis vector in a direction
    sim_cvec :
        The simulation cell basis vector in c direction

    Returns
    ----------
    pts_w_imgs :
        The position of atoms after replicating the box n_x and n_z times in X and Z direction.
    """
    num1 = np.shape(pts1)[0]
    pts_w_imgs = np.zeros((num1*(2*nx+1)*(2*nz+1), 3))

    # The first set of atoms correspond to the main
    # cell.
    ct1 = 0
    ind_st = num1*ct1
    ind_stop = num1*(ct1+1)-1
    pts_w_imgs[ind_st:ind_stop+1, :] = pts1
    ct1 = ct1 + 1

    # Array for translating the main cell
    nx_val = np.linspace(-nx, nx, 2*nx+1)
    nz_val = np.linspace(-nz, nz, 2*nz+1)
    mval = np.meshgrid(nx_val, nz_val)
    mx = np.ndarray.flatten(mval[0])
    mz = np.ndarray.flatten(mval[1])
    i1 = np.where((mx == 0) & (mz == 0))[0][0]
    mx = np.delete(mx, i1)
    mz = np.delete(mz, i1)
    x_trans = np.tile(sim_avec, (num1, 1))
    z_trans = np.tile(sim_cvec, (num1, 1))

    # Creating the images
    for ct2 in range(np.size(mx)):
        mx1 = mx[ct2]
        mz1 = mz[ct2]
        pts_trans = pts1 + mx1*x_trans + mz1*z_trans
        # xs = pts_trans[:, 0]
        # ys = pts_trans[:, 1]
        # zs = pts_trans[:, 2]
        ind_st = num1*ct1
        ind_stop = num1*(ct1+1)-1
        pts_w_imgs[ind_st:ind_stop+1, :] = pts_trans
        ct1 = ct1 + 1

    return pts_w_imgs


def slice_along_planes(orig, sim_avec, sim_bvec, sim_cvec, rCut, pts_w_imgs, gb1_inds):
    """

    1. Descriptions
    2. Input Parameters
        1. `orig` - The origin of the main cell.
        2. `sim_avec` -  The simulation cell basis vector in a direction
        3. `sim_bvec` - The simulation cell basis vector in b direction
        4. `sim_cvec` - The simulation cell basis vector in c direction
        5. `rCut` - Cut-off radius for computing Delaunay triangulations
        6. `pts_w_imgs` - The position of atoms after replicating the box n_x and n_z times in X and Z direction.
        7. `gb1_inds` - Indices of the GB atoms
    3. Return Parameters
        1. `pts_w_imgs` - The position of atoms after replicating the box, n_x and n_z times in x and z direction.
        2. `gb1_inds` - Indices of the GB atoms
    """

    au_vec = sim_avec/np.linalg.norm(sim_avec)
    bu_vec = sim_bvec/np.linalg.norm(sim_bvec)
    cu_vec = sim_cvec/np.linalg.norm(sim_cvec)

    xcut_nvec = np.cross(au_vec, bu_vec)
    xcut_nvec = xcut_nvec/np.linalg.norm(xcut_nvec)
    zcut_nvec = np.cross(cu_vec, bu_vec)
    zcut_nvec = zcut_nvec/np.linalg.norm(zcut_nvec)

    pl_nvecs = np.vstack((xcut_nvec, xcut_nvec, zcut_nvec, zcut_nvec))
    lvals = ([[0, 0, 0, -1], [0, 0, 1, 1], [0, -1, 0, 0], [1, 1, 0, 0]])
    # pl_pts = np.zeros((4, 3))
    ct1 = 0
    for l1 in lvals:
        pt1 = orig + sim_avec*l1[0] + au_vec*rCut*l1[1] + sim_cvec*l1[2] + cu_vec*rCut*l1[3]
        pl_nvec = pl_nvecs[ct1]
        inds_keep1 = inds_to_keep(pl_nvec, pt1, orig, pts_w_imgs)
        pts_w_imgs, gb1_inds = del_inds(inds_keep1, pts_w_imgs, gb1_inds)
        ct1 = ct1 + 1

    return pts_w_imgs, gb1_inds


def del_inds(ind1, pts1, gb1_inds):
    """
    Function deletes the indices of atoms outside of the main box plus rCut margin around it

    Parameters
    ------------
    ind1 :
        The indices of atoms we want to keep
    pts1 :
        The position of atoms after replicating the box n_x and n_z times in x and z direction.
    gb1_ind
        Indices of the GB atoms

    Returns
    ---------
    pts1 :
        The position of atoms we want to keep
    gb1_inds :
        Indices of the GB atoms
    """

    int1, a1, a2 = np.intersect1d(ind1, gb1_inds, return_indices=True)
    gb1_inds = a1
    pts1 = pts1[ind1, :]
    return pts1, gb1_inds


def inds_to_keep(norm_vec, pl_pt, orig, pts):
    """
    Function identifies the indices of atoms which are in inside the main box  plus a rCut margin around it

    Parameters
    -------------
    norm_vec :
        Plane normal within rCut distance from the considered  box face
    pl_pt :
        A point on a plane within rCut distance from the considered box face
    orig :
        The origin of the main cell.
    pts :
        The position of atoms after replicating the box, n_x and n_z times in x and z direction.

    Returns
    ----------
    inds_keep :
        The indices of atoms within the replicates which are within rCut distance of the main cell
    """
    # Sign-values for pts_w_imgs
    # npts = np.shape(pts)[0]
    # pts1 = pts - np.tile(pl_pt, (npts,1))
    pts1 = pts - pl_pt  # numpy does broadcasting
    sign_vals = np.sign(pts1[:, 0]*norm_vec[0] + pts1[:, 1]*norm_vec[1] + pts1[:, 2]*norm_vec[2])
    orig_sign_val = np.sign(np.dot((orig - pl_pt), norm_vec))

    if (orig_sign_val > 0):
        inds_keep = np.where(sign_vals > 0)[0]
    else:
        inds_keep = np.where(sign_vals < 0)[0]

    return inds_keep
