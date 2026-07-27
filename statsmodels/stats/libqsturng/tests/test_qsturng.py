"""The 'handful' tests are intended to aid refactoring. The tests with the
@pytest.mark..slow are empirical (test within error limits) and intended to more
extensively ensure the stability and accuracy of the functions"""

from statsmodels.compat.python import lmap, lzip

from pathlib import Path

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
import pytest

from statsmodels.stats.libqsturng import psturng, qsturng


def read_ch(fname):
    with Path(fname).open(encoding="utf-8") as f:
        lines = f.readlines()
    ps, rs, vs, qs = lzip(*[L.split(",") for L in lines])
    return (lmap(float, ps), lmap(float, rs), lmap(float, vs), lmap(float, qs))


class TestQsturng:

    def test_scalar(self):
        assert_almost_equal(4.43645545899562, qsturng(0.9, 5, 6), 5)

    def test_vector(self):
        assert_array_almost_equal(
            np.array([3.98832389, 4.56835318, 6.26400894]),
            qsturng([0.8932, 0.9345, 0.9827], [4, 4, 4], [6, 6, 6]),
            5,
        )

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            qsturng(-0.1, 5, 6)
        with pytest.raises(ValueError):
            qsturng(0.9991, 5, 6)
        with pytest.raises(ValueError):
            qsturng(0.89, 5, 1)
        with pytest.raises(ValueError):
            qsturng(0.9, 5, 0)
        with pytest.raises(ValueError):
            qsturng(0.9, 1, 2)

    def test_handful_to_tbl(self):
        cases = [
            (0.75, 30.0, 12.0, 5.01973488482),
            (0.975, 15.0, 18.0, 6.00428263999),
            (0.1, 8.0, 11.0, 1.76248712658),
            (0.995, 6.0, 17.0, 6.13684839819),
            (0.85, 15.0, 18.0, 4.65007986215),
            (0.75, 17.0, 18.0, 4.33179650607),
            (0.75, 60.0, 16.0, 5.50520795792),
            (0.99, 100.0, 2.0, 50.3860723433),
            (0.9, 2.0, 40.0, 2.38132493732),
            (0.8, 12.0, 20.0, 4.15361239056),
            (0.675, 8.0, 14.0, 3.35011529943),
            (0.75, 30.0, 24.0, 4.77976803574),
            (0.75, 2.0, 18.0, 1.68109190167),
            (0.99, 7.0, 120.0, 5.00525918406),
            (0.8, 19.0, 15.0, 4.70694373713),
            (0.8, 15.0, 8.0, 4.80392205906),
            (0.5, 12.0, 11.0, 3.31672775449),
            (0.85, 30.0, 2.0, 10.2308503607),
            (0.675, 20.0, 18.0, 4.23706426096),
            (0.1, 60.0, 60.0, 3.69215469278),
        ]
        for p, r, v, q in cases:
            assert_almost_equal(q, qsturng(p, r, v), 5)

    @pytest.mark.skip
    def test_all_to_tbl(self):
        from statsmodels.stats.libqsturng.make_tbls import R, T

        ps, rs, vs, qs = ([], [], [], [])
        for p in T:
            for v in T[p]:
                for r in R.keys():
                    ps.append(p)
                    vs.append(v)
                    rs.append(r)
                    qs.append(T[p][v][R[r]])
        qs = np.array(qs)
        errors = np.abs(qs - qsturng(ps, rs, vs)) / qs
        assert_equal(np.array([]), np.where(errors > 0.03)[0])

    def test_handful_to_ch(self):
        cases = [
            (0.8699908, 10.0, 465.4956, 3.997799075635331),
            (0.8559087, 43.0, 211.7474, 5.1348419692951675),
            (0.6019187, 11.0, 386.5556, 3.338310148769882),
            (0.658888, 51.0, 74.652, 4.810888048315373),
            (0.6183604, 77.0, 479.8493, 4.986405932173287),
            (0.9238978, 77.0, 787.5278, 5.787105300302294),
            (0.8408322, 7.0, 227.3483, 3.555579831141358),
            (0.5930279, 60.0, 325.3461, 4.76580231238824),
            (0.6236158, 61.0, 657.5285, 4.820781275598787),
            (0.9344575, 72.0, 846.4138, 5.801434132925911),
            (0.8761198, 56.0, 677.8171, 5.362460718311719),
            (0.7901517, 41.0, 131.525, 4.922283134195054),
            (0.6396423, 44.0, 624.3828, 4.601512725008315),
            (0.8085966, 14.0, 251.4224, 4.079305842471975),
            (0.716179, 45.0, 136.7055, 4.805549808934009),
            (0.8204, 6.0, 290.9876, 3.3158771384085597),
            (0.8705345, 83.0, 759.6216, 5.596933456448538),
            (0.8249085, 18.0, 661.9321, 4.3283725986180395),
            (0.9503, 2.0, 4.434, 3.7871158594867262),
            (0.7276132, 95.0, 91.43983, 5.410038486849989),
        ]
        for p, r, v, q in cases:
            assert_almost_equal(q, qsturng(p, r, v), 5)

    @pytest.mark.slow
    def test_10000_to_ch(self):
        import os

        curdir = Path(__file__).resolve().parent
        ps, rs, vs, qs = read_ch(
            os.path.split(os.path.split(curdir)[0])[0] + "/tests/results/bootleg.csv"
        )
        qs = np.array(qs)
        errors = np.abs(qs - qsturng(ps, rs, vs)) / qs
        assert_equal(np.array([]), np.where(errors > 0.03)[0])


class TestPsturng:

    def test_scalar(self):
        """scalar input -> scalar output"""
        assert_almost_equal(0.1, psturng(4.43645545899562, 5, 6), 5)

    def test_vector(self):
        """vector input -> vector output"""
        assert_array_almost_equal(
            np.array([0.10679889, 0.06550009, 0.01730145]),
            psturng([3.98832389, 4.56835318, 6.26400894], [4, 4, 4], [6, 6, 6]),
            5,
        )

    def test_v_equal_one(self):
        assert_almost_equal(0.1, psturng(0.2, 5, 1), 5)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            psturng(-0.1, 5, 6)
        with pytest.raises(ValueError):
            psturng(0.9, 1, 2)

    def test_handful_to_known_values(self):
        cases = [
            (0.7149957872611143, 67, 956.7074248839239, 5.051765844307069),
            (0.4297423485506767, 16, 723.5026173650232, 3.3303582093701354),
            (0.9493642935954842, 2, 916.1867328010926, 2.7677975546417244),
            (0.8535738177072504, 66, 65.67055060832368, 5.564743810827011),
            (0.8737210802190093, 74, 626.4236947499363, 5.535554057070111),
            (0.5389196056471373, 49, 862.6379943848578, 4.510864592337715),
            (0.9881865955566457, 18, 36.269686711464274, 6.090664375088616),
            (0.5303199489603763, 50, 265.29558652727917, 4.5179640079726795),
            (0.7318857887397332, 59, 701.414975522512, 4.9980139875409915),
            (0.653320193689827, 61, 591.0118366419591, 4.870658176670689),
            (0.5540322165724856, 77, 907.3415672540519, 4.878613591798463),
            (0.30783916857266, 83, 82.44692348798088, 4.439640124285829),
            (0.2932172024241566, 16, 709.6438257555301, 3.030427754070273),
            (0.27146478168880306, 31, 590.0059468357417, 3.5870031664477215),
            (0.6734879695843378, 81, 608.0270611112766, 5.109619997443294),
            (0.3277439394596894, 18, 17.70622439925084, 3.211903816376543),
            (0.7081637474795982, 72, 443.10678914889695, 5.099003088941065),
            (0.3335493927675786, 47, 544.0772192199048, 4.061335296419328),
            (0.6041214394736305, 36, 895.8352693327155, 4.381717596850172),
            (0.8873905230066598, 77, 426.0366551155826, 5.633392948034131),
        ]
        for p, r, v, q in cases:
            assert_almost_equal(1.0 - p, psturng(q, r, v), 5)

    @pytest.mark.slow
    def test_100_random_values(self):
        n = 100
        random_state = np.random.RandomState(12345)
        ps = random_state.random_sample(n) * (0.999 - 0.1) + 0.1
        rs = random_state.randint(2, 101, n)
        vs = random_state.random_sample(n) * 998.0 + 2.0
        qs = qsturng(ps, rs, vs)
        estimates = psturng(qs, rs, vs)
        actuals = 1.0 - ps
        errors = estimates - actuals
        assert_equal(np.array([]), np.where(errors > 1e-05)[0])
