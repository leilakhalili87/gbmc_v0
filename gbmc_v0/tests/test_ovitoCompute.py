from ..util_funcs import compute_ovito_data


def test_compute_ovito_data():
    filename0 = "/data/dump_1"
    actual = 3588
    expected = compute_ovito_data(filename0)
    assert actual == expected
