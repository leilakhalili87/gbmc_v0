import numpy as np
# import ovito.modifiers as ovm
# from ovito.io import import_file, export_file

def p_arr(non_p):
    arr = np.array([0, 1, 2])
    arr0 = np.delete(arr, non_p)
    return arr0

def pad_dump_file(data, lat_par, rCut, non_p):
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

    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = GB_finder(data, lat_par, non_p)
    arr = p_arr(non_p)

    sim_cell = data.cell[...]
    sim_nonp_vec = np.array(sim_cell[:, non_p])
    sim_1vec = np.array(sim_cell[:, arr[0]])
    sim_2vec = np.array(sim_cell[:, arr[1]])
    sim_orig = np.array(sim_cell[:, 3])

    p1_vec = np.array([sim_1vec[arr[0]], sim_1vec[arr[1]]])
    p2_vec = np.array([sim_2vec[arr[0]], sim_2vec[arr[1]]])
    [n1, n2] = num_rep_2d(p1_vec, p2_vec, rCut)

    pts1, gb1_inds = pad_gb_perp(data, GbRegion, GbIndex, rCut, non_p)
    pts_w_imgs = create_imgs(pts1, n1, n2, sim_1vec, sim_2vec, non_p)
    pts_w_imgs, gb1_inds = (slice_along_planes(sim_orig,
                                               sim_1vec, sim_2vec, sim_nonp_vec, rCut,
                                               pts_w_imgs, gb1_inds, non_p))
    return pts_w_imgs, gb1_inds


def GB_finder(data, lat_par, non_pbc):
    """
    The function finds the GB region usning Polyhedral Template Matching.

    Parameters
    --------------
    filename0 :
        The lammps dump file
    lat_par:
        The lattice parameter
    non_pbc : int
        The non-periodic direction. 0 , 1 or 2 which corresponds to
        x, y and z direction, respectively. 

    Returns
    -----------
    GbRegion:
        The maximum and Minimum value of postion of atoms in Z direction  in the GB region.
    GbWidth :
        GbRegion[1] - GbRegion[0]
    GbIndex :
        The index of atoms in GB
    w_bottom_SC :'
        The width of the region on the bottom side of GB which have single crystal structure
    w_top_SC :
        The width of the region on the top side of GB which have single crystal structure
    """

    # num_particles = data.particles.count
    ptm_struct = data.particles['Structure Type'][...]
    position_np = data.particles['Position'][...][:, non_pbc]

    NoSurfArea = []
    # Find the smallest single crystal range
    a = 1
    pos_min = np.min(position_np)

    while a != 0:
        pos_max = pos_min + lat_par
        a = len(np.where((ptm_struct == 0) & (position_np < pos_max) & (position_np > pos_min))[0])
        pos_min += lat_par

    NoSurfArea = NoSurfArea + [pos_min]

    # Find the largest single crystal range
    a = 1
    pos_max = np.max(position_np)
    while a != 0:
        pos_min = pos_max - lat_par
        a = len(np.where((ptm_struct == 0) & (position_np < pos_max) & (position_np > pos_min))[0])
        pos_max -= lat_par

    NoSurfArea = NoSurfArea + [pos_min + lat_par]

    gb_index = np.where((ptm_struct == 0) & (position_np < NoSurfArea[1]) & (position_np > NoSurfArea[0]))[0]
    gb_mean = np.mean(position_np[gb_index])
    gb_std = np.std(position_np[gb_index])

    # Delete the outliers of gb
    var = position_np[gb_index] - gb_mean
    GbIndex = gb_index[np.where((var < 3*gb_std) & (var > -3*gb_std))[0]]

    GbZ = position_np[GbIndex]
    GbRegion = [np.min(GbZ), np.max(GbZ)]
    GbWidth = GbRegion[1] - GbRegion[0]
    w_bottom_SC = GbRegion[0] - NoSurfArea[0]
    w_top_SC = NoSurfArea[1] - GbRegion[1]

    return GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC


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


def pad_gb_perp(data, GbRegion, GbIndex, rCut, non_p):
    """
    Function to take as input the dump data (from OVITO), find the GB atoms
    and add padding to the GB atoms  within rCut in Z direction.

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
        Indices of the atoms which Z value is in range [GBRegion[0] - rCut, GBRegion[1] + rCut].
    gb1_inds :
        Indices of the GB atoms
    """
    arr0 = p_arr(non_p)

    position_nonp = data.particles['Position'][...][:, non_p]
    position_p1 = data.particles['Position'][...][:, arr0[0]]
    position_p2 = data.particles['Position'][...][:, arr0[1]]

    Zmin, Zmax = GbRegion[0] - rCut, GbRegion[1] + rCut

    pad1_inds = np.where((position_nonp <= Zmax) & (position_nonp >= Zmin))[0]

    int1, a1, a2 = np.intersect1d(pad1_inds, GbIndex, return_indices=True)
    gb1_inds = a1

    # Replicate the GB structure along periodic directions (nx and ny times)
    num1 = np.size(pad1_inds)
    pts1 = np.zeros((num1, 3))
    pts1[:, arr0[0]] = np.array(position_p1[pad1_inds])
    pts1[:, arr0[1]] = np.array(position_p2[pad1_inds])
    pts1[:, non_p] = np.array(position_nonp[pad1_inds])

    return pts1, gb1_inds


def create_imgs(pts1, n1, n2, sim_1vec, sim_2vec, non_p):
    """
    Creates the replicates of the main cell in X and Z direction.

    Parameters
    -------------
    pts1 :
        Indices of the atoms which Y value is in range [GBRegion[0] - rCut, GBRegion[1] + rCut].
    n1 :
        Number of replications in 1st periodic direction
    n2 :
        Number of replications in 2nd periodic direction
    sim_1vec :
        The simulation cell basis vector in 1st periodic direction
    sim_2vec :
        The simulation cell basis vector in 2nd periodic direction
    non_pbc : int
        The non-periodic direction. 0 , 1 or 2 which corresponds to
        x, y and z direction, respectively. 

    Returns
    ----------
    pts_w_imgs :
        The position of atoms after replicating the box n_x and n_z times in X and Z direction.
    """
    arr = p_arr(non_p)

    num1 = np.shape(pts1)[0]
    pts_w_imgs = np.zeros((num1*(2*n1+1)*(2*n2+1), 3))

    # The first set of atoms correspond to the main
    # cell.
    ct1 = 0
    ind_st = num1*ct1
    ind_stop = num1*(ct1+1)-1
    pts_w_imgs[ind_st:ind_stop+1, :] = pts1
    ct1 = ct1 + 1

    # Array for translating the main cell
    n1_val = np.linspace(-n1, n1, 2*n1+1)
    n2_val = np.linspace(-n2, n2, 2*n2+1)
    mval = np.meshgrid(n1_val, n2_val)
    m1 = np.ndarray.flatten(mval[0])
    m2 = np.ndarray.flatten(mval[1])
    i1 = np.where((m1 == 0) & (m2 == 0))[0][0]
    m1 = np.delete(m1, i1)
    m2 = np.delete(m2, i1)
    p1_trans = np.tile(sim_1vec, (num1, 1))
    p2_trans = np.tile(sim_2vec, (num1, 1))

    # Creating the images
    for ct2 in range(np.size(m1)):
        mp1 = m1[ct2]
        mp2 = m2[ct2]
        pts_trans = pts1 + mp1*p1_trans + mp2*p2_trans

        pos_nonp = pts_trans[:, non_p]
        pos_p1 = pts_trans[:, arr[0]]
        pos_p2 = pts_trans[:, arr[1]]
        ind_st = num1*ct1
        ind_stop = num1*(ct1+1)-1
        pts_w_imgs[ind_st:ind_stop+1, :] = pts_trans
        ct1 = ct1 + 1

    return pts_w_imgs


def slice_along_planes(orig, sim_1vec, sim_2vec, sim_nonp_vec, rCut, pts_w_imgs, gb1_inds, non_p):
    """

    Function 

    Parameters
    ------------
    orig
        The origin of the main cell.
    sim_1vec
        The simulation cell basis vector in a direction
    sim_2vec
        The simulation cell basis vector in b direction
    sim_nonp_vec
        The simulation cell basis vector in c direction
    rCut
        Cut-off radius for computing Delaunay triangulations
    pts_w_imgs
        The position of atoms after replicating the box n_x and n_z times in X and Z direction.
    gb1_inds
        Indices of the GB atoms
    non_pbc : int
        The non-periodic direction. 0 , 1 or 2 which corresponds to
        x, y and z direction, respectively. 

    Returns
    ------------
    pts_w_imgs
            The position of atoms after replicating the box, n_x and n_z times in x and z direction.
    gb1_inds
            Indices of the GB atoms
    """
    p1u_vec = sim_1vec/np.linalg.norm(sim_1vec)
    p2u_vec = sim_2vec/np.linalg.norm(sim_2vec)
    nonp_u_vec = sim_nonp_vec/np.linalg.norm(sim_nonp_vec)

    p1cut_nvec = np.cross(p1u_vec, nonp_u_vec)
    p1cut_nvec = p1cut_nvec/np.linalg.norm(p1cut_nvec)
    p2cut_nvec = np.cross(p2u_vec, nonp_u_vec)
    p2cut_nvec = p2cut_nvec/np.linalg.norm(p2cut_nvec)

    pl_nvecs = np.vstack((p1cut_nvec, p1cut_nvec, p2cut_nvec, p2cut_nvec))
    lvals = ([[0, 0, 0, -1], [0, 0, 1, 1], [0, -1, 0, 0], [1, 1, 0, 0]])
    # pl_pts = np.zeros((4, 3))
    ct1 = 0
    for l1 in lvals:
        pt1 = orig + sim_1vec*l1[0] + p1u_vec*rCut*l1[1] + sim_2vec*l1[2] + p2u_vec*rCut*l1[3]
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
