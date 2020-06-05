import numpy as np
import pytest
import gbmc_v0.pad_dump_file as pad
@pytest.mark.parametrize('filename0, lat_par',
                         [("data/dump_1", 4.05)])
def pad_gb_perp(filename0, lat_par):
    data = uf.compute_ovito_data(filename0)
    rCut = 2 * lat_par
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pad.GB_finder(data, lat_par)
    pts1, gb1_inds = pad_gb_perp(data, GbRegion, GbIndex, rCut)
    p_pos = pts1[gb1_inds]
    d_pos = data.particles['Position'][...][GbIndex, :]

    assert p_pos == d_pos
