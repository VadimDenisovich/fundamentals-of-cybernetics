from codingtheory.metrics import two_proportion_ztest, wald_ci


def test_wald_ci_contains_point():
    lo, hi = wald_ci(0.1, 1000)
    assert lo < 0.1 < hi
    assert abs((hi - lo) / 2 - 1.96 * (0.1 * 0.9 / 1000) ** 0.5) < 1e-12


def test_wald_ci_clipped():
    lo, hi = wald_ci(0.0, 100)
    assert lo == 0.0


def test_ztest_identical_proportions():
    z, p = two_proportion_ztest(50, 1000, 50, 1000)
    assert z == 0.0 and p == 1.0


def test_ztest_detects_difference():
    z, p = two_proportion_ztest(100, 1000, 50, 1000)
    assert p < 0.001


def test_ztest_small_difference_not_significant():
    z, p = two_proportion_ztest(52, 1000, 50, 1000)
    assert p > 0.05
