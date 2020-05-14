import pickle as pkl
import numpy as np
import os
import sys
import ovito.io as oio;
import ovito.modifiers as ovm;
from ovito.io import import_file, export_file
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist

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
    pipeline =  oio.import_file(filename0,sort_particles=True);
    dmod = ovm.PolyhedralTemplateMatchingModifier(rmsd_cutoff=.1);
    pipeline.modifiers.append(dmod);
    return pipeline.compute();


def RemProb(data, CohEng, GbIndex ):
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
	Excess_Eng = (GbAtomicEng - CohEng);
	Excess_Eng[Excess_Eng < 0] = 0;
	Excess_Eng_Tot = np.sum(Excess_Eng);

	return Excess_Eng/Excess_Eng_Tot;

