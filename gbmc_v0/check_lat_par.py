import numpy as np
import util_funcs as uf
from ovito.data import NearestNeighborFinder

lat_par = 4.05
cut_off = 10  # in Ang.
num_neighbors = 12
non_p_dir = 2   # which direction is non-periodic. In our case is 2, in previous simulations it was 1 (y direction)

data = uf.compute_ovito_data('./lammps_dump/dump_befor.1')

ptm_struct = data.particles['Structure Type'][...]
position = data.particles['Position'][...]
need_atom = np.where((position[:, non_p_dir] > 0) & (ptm_struct == 1))[0]
pos_sc = position[need_atom, :]
min_Z, max_Z = np.min(pos_sc[:, non_p_dir]) + cut_off, np.max(pos_sc[:, non_p_dir]) - cut_off

area = np.where((position[:, non_p_dir] < max_Z) & (position[:, non_p_dir] > min_Z))[0]
num_particl = np.shape(area)[0]
finder = NearestNeighborFinder(num_neighbors, data)

distances = np.zeros(num_particl)
i = 0
for index in area:
    # Iterate over the neighbors of the current particle, starting with the closest:
    for neigh in finder.find(index):
        distances[i] = neigh.distance + distances[i]
    i += 1

cal_lat_par = np.sqrt(2) * np.mean(distances) / num_neighbors
print("The calculate larrice parameter is = ", cal_lat_par)
