
import numpy as np
import pytest
import gbmc_v0.pad_dump_file as pdf
import gbmc_v0.util_funcs as uf


@pytest.mark.parametrize('filename0, rCut, lat_par',
                         [("data/dump_1", 8.1, 4.05),
                          ("data/dump_1", 30, 4.05)])
def test_create_imgs(filename0, rCut, lat_par):
    data = uf.compute_ovito_data(filename0)
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pdf.GB_finder(data, lat_par)
    sim_cell = data.cell[...]
    sim_avec = np.array(sim_cell[:, 0])
    sim_bvec = np.array(sim_cell[:, 1])
    sim_cvec = np.array(sim_cell[:, 2])

    x1_vec = np.array([sim_avec[0], sim_avec[1]])
    y1_vec = np.array([sim_bvec[0], sim_bvec[1]])
    [nx, ny] = pdf.num_rep_2d(x1_vec, y1_vec, rCut)
    pts1, gb1_inds = pdf.pad_gb_perp(data, GbRegion, GbIndex, rCut)
    pts_w_imgs = pdf.create_imgs(pts1, nx, ny, sim_avec, sim_cvec)

    num0 = pts_w_imgs.shape[0]/pts1.shape[0]
    num1 = np.power(nx+ny+1, 2)
    assert np.allclose(num0, num1)
