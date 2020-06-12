import numpy as np
import ovito.io as oio
import ovito.modifiers as ovm
from itertools import islice 

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
