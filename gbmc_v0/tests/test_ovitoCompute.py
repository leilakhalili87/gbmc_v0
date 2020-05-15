import numpy as np
import ovito.io as oio
import ovito.modifiers as ovm

from gbmc_v0 import *

filename0 = "./data/dump_1"
def test_compute_ovito_data(filename0):
    actual = 3588
    expected = compute_ovito_data(filename0)
    assert actual == expected