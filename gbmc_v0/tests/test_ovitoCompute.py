from ..util_funcs import compute_ovito_data


def test_compute_ovito_data():
    filename0 = "gbmc_v0/tests/data/dump_1"
    actual = 5447
    expected = data.particles.count
    assert actual == expected
