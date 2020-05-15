import numpy as np
import ovito.io as oio
import ovito.modifiers as ovm


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
    num_particles = data.particles.count
    return num_particles


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
