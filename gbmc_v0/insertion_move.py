import pickle as pkl
import numpy as np
import os
import sys
# import ovito.io as oio;
# import ovito.modifiers as ovm;
# from ovito.io import import_file, export_file
# from scipy.spatial import Delaunay
# from scipy.spatial.distance import pdist

import util_funcs as uf;
import pad_dump_file as pdf;
import vv_props as vvp;
import numpy.linalg as nla;

###############
lat_par = 4.074; rCut = 2*lat_par;
filename0 = '../data/dump_1';

ovito_data = uf.compute_ovito_data(filename0);
############################################################################
pts_w_imgs, gb1_inds, inds_arr = pdf.pad_dump_file(ovito_data, lat_par, rCut);
tri_vertices, gb_tri_inds = vvp.triang_inds(pts_w_imgs, gb1_inds, inds_arr);
cc_coors, cc_rad = vvp.vv_props(pts_w_imgs, tri_vertices, gb_tri_inds, lat_par);
cc_coors1 = vvp.wrap_cc(ovito_data.cell, cc_coors);
####################################################################################



