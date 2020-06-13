import numpy as np
import ovito.io as oio
import ovito.modifiers as ovm
from itertools import islice
import pad_dump_file as pdf


def compute_ovito_data(filename0):
    """
    Computes the attributes of ovito

    Parameters
    ------------
    filename0 : string
        The name of the input file.

    Returns
    --------
    data : class
        all the attributes of data
    """
    pipeline = oio.import_file(filename0, sort_particles=True)
    dmod = ovm.PolyhedralTemplateMatchingModifier(rmsd_cutoff=.1)
    pipeline.modifiers.append(dmod)
    data = pipeline.compute()
    return data


def identify_pbc(data):
    """
    Function finds the non-periodic direction

    Parameters
    ------------
    data : class
        all the attributes of data

    Returns
    --------
    non_pbc : int
        The non-periodic direction. 0 , 1 or 2 which corresponds to
        x, y and z direction, respectively.
    """
    pbc = data.cell.pbc
    pbc = np.asarray(pbc) + 0
    non_pbc = np.where(pbc == 0)[0][0]
    return non_pbc


def box_size_reader(dump_name):
    with open(dump_name) as lines:
        box_bound = np.genfromtxt(islice(lines, 5, 8))
    return box_bound


def define_bounds(box_bound):
    # boundaries of fix rigid
    siz_box = np.shape(box_bound)[1]
    if siz_box == 2:
        box_type = "block"
        xy = 0
        xz = 0
        yz = 0
        tilt = []
    else:
        box_type = "prism"
        xy = box_bound[0, 2]
        xz = box_bound[1, 2]
        yz = box_bound[2, 2]
        tilt = np.array([xy, xz, yz])
    xlo = box_bound[0, 0] - np.min(np.array([0, xy, xz, xy + xz]))
    xhi = box_bound[0, 1] - np.max(np.array([0, xy, xz, xy + xz]))
    ylo = box_bound[1, 0] - np.min(np.array([0, yz]))
    yhi = box_bound[1, 1] - np.max(np.array([0, yz]))
    zlo = box_bound[2, 0]
    zhi = box_bound[2, 1]
    untilted = np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]])

    return untilted, tilt, box_type


def RemProb(data, CohEng, GbIndex):
    """
    The function finds The atomic removal probabilty.

    Parameters
    --------------
    filename0 : string
        The lammps dump file
    CohEng	: float
        The cohesive energy

    Return
    ----------------

    AtomicRemProb : float
        The probabilty of removing an atom
    """

    GbAtomicEng = data.particle_properties['c_eng'][GbIndex]
    Excess_Eng = (GbAtomicEng - CohEng)
    Excess_Eng[Excess_Eng < 0] = 0
    Excess_Eng_Tot = np.sum(Excess_Eng)

    return Excess_Eng/Excess_Eng_Tot


def check_SC_reg(data, lat_par, rCut, non_p, tol_fix_reg, SC_tol):
    """
    Function to identify whether single crystal region on eaither side of the GB is
    bigger than a tolerance (SC_tol)

    Parameters
    ------------
    data :
        Data object computed using OVITO I/O
    lat_par :
        Lattice parameter for the crystal being simulated
    rCut :
        Cut-off radius for computing Delaunay triangulations
    non_pbc : int
        The non-periodic direction. 0 , 1 or 2 which corresponds to
        x, y and z direction, respectively.
    tol_fix_reg : float
        The user defined tolerance for the size of rigid translation region in lammps simulation.
    SC_tol : float
        The user defined tolerance for the minimum size of single crystal region.
    Returns
    ----------
    SC_boolean :
        A boolean list for low/top or left/right single crytal region. True means the width > SC_tol.
    """
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pdf.GB_finder(data, lat_par, non_p)

    SC_boolean = [True, True]
    w_bottom_SC = w_bottom_SC - tol_fix_reg
    w_top_SC = w_top_SC - tol_fix_reg
    if w_bottom_SC < SC_tol:
        SC_boolean[0] = False
    if w_top_SC < SC_tol:
        SC_boolean[1] = False
    return SC_boolean
