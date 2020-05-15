from ..util_funcs import compute_ovito_data


def test_compute_ovito_data():
    filename0 = "gbmc_v0/tests/data/dump_1"
    actual = 3588
    expected = compute_ovito_data(filename0)
    assert actual == expected
