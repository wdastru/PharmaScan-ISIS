# test_reader_ng.py
import numpy as np
import pytest
from pathlib import Path
from scipy.integrate import simpson

from reader_ng import (
    find_maximum,
    compute_regions_integrals,
    METABOLITE_REGIONS,
    N_POINTS_FIT,
    parameter_extract,
    _init_stats_lists,
    _accumulate_averages,
    _accumulate_squared_diffs,
    _finalize_std_dev,
    _compute_integrals_stats,
    spline_fit
)

# ----------------------------------------------------------------------
# Tests for find_maximum
# ----------------------------------------------------------------------
class TestFindMaximum:
    def test_full_array(self):
        arr = np.array([1, 3, 2, 5, 4])
        val, idx = find_maximum(arr)
        assert val == 5
        assert idx == 3

    def test_with_limits(self):
        arr = np.array([1, 3, 5, 2, 4])
        val, idx = find_maximum(arr, start=1, end=4)  # indices 1,2,3 → values 3,5,2
        assert val == 5
        assert idx == 2

# ----------------------------------------------------------------------
# Tests for compute_regions_integrals
# ----------------------------------------------------------------------
class TestComputeRegionsIntegrals:
    def test_constant_y(self):
        """With y=1, integral = region width."""
        x_fit = np.linspace(-10, 10, N_POINTS_FIT)
        y_fit = np.ones_like(x_fit)
        integrals = compute_regions_integrals(x_fit, y_fit)
        for region, (start, end) in METABOLITE_REGIONS.items():
            expected = end - start
            assert np.isclose(integrals[region], expected, rtol=1e-6, atol=1e-9), \
                f"Region {region}: expected {expected}, got {integrals[region]}"

    @pytest.mark.parametrize("region", list(METABOLITE_REGIONS.keys()))
    def test_triangle(self, region):
        """Triangle peaking at 1 → integral = width/2."""
        start, end = METABOLITE_REGIONS[region]
        x_fit = np.linspace(start, end, N_POINTS_FIT)
        centre = (start + end) / 2
        width = end - start
        y_fit = 1.0 - np.abs(x_fit - centre) / (width / 2)

        integrals = compute_regions_integrals(x_fit, y_fit)
        expected = width / 2
        assert np.isclose(integrals[region], expected, rtol=1e-4, atol=1e-7), \
            f"Region {region}: expected {expected}, got {integrals[region]}"

        # Other regions must be zero (x_fit covers only this region)
        for other in METABOLITE_REGIONS:
            if other != region:
                assert integrals[other] == 0.0, \
                    f"Region {other} should be 0, got {integrals[other]}"

    def test_oscillatory(self):
        """Compare with high‑resolution Simpson reference."""
        def oscillatory(x):
            return 0.5 + 0.3 * np.sin(2 * np.pi * x / 5) + 0.2 * np.cos(2 * np.pi * x / 2.5)

        # Dense reference grid
        x_full = np.linspace(-10, 10, 2000)
        y_full = oscillatory(x_full)
        reference = {}
        for region, (start, end) in METABOLITE_REGIONS.items():
            mask = (x_full >= start) & (x_full <= end)
            if np.any(mask):
                x_reg = x_full[mask]
                y_reg = y_full[mask]
                ref = simpson(y_reg, x_reg) if len(x_reg) > 3 else np.trapezoid(y_reg, x_reg)
            else:
                ref = 0.0
            reference[region] = ref

        # Coarse grid used by the actual function
        x_fit = np.linspace(-10, 10, N_POINTS_FIT)
        y_fit = oscillatory(x_fit)
        computed = compute_regions_integrals(x_fit, y_fit)

        for region in METABOLITE_REGIONS:
            assert np.isclose(computed[region], reference[region], rtol=1e-2, atol=1e-5), \
                f"Region {region}: reference {reference[region]:.6f}, got {computed[region]:.6f}"

# ----------------------------------------------------------------------
# Tests for parameter_extract
# ----------------------------------------------------------------------
# Minimal method‑file snippets
SAT_TRANS_FL_BLOCK = (
    "##$PVM_SatTransFL= ( 19 )\n"
    "-1094.20948068296 1094.20948068296 -972.630649495964 972.630649495964\n"
    "-851.051818308968 851.051818308968 -729.472987121973 729.472987121973\n"
    "-607.894155934977 607.894155934977 -486.315324747982 486.315324747982\n"
    "-364.736493560986 364.736493560986 -243.157662373991 243.157662373991\n"
    "-121.578831186995 121.578831186995 0\n"
    "##$PVM_SatTransRepetitions=19\n"
    "##$PVM_SatTransRefScan=Off\n"
)

WORK_OFFSET_BLOCK = (
    "##$PVM_FrqWorkOffset= ( 8 )\n"
    "-252.155018851338 0 0 0 0 0 0 0\n"
    "##$PVM_FrqWork= ( 8 )\n"
    "121.578831186995 0 0 0 0 0 0 0\n"
)

EXPECTED_SAT_TRANS_FL = [
    -1094.20948068296, 1094.20948068296, -972.630649495964, 972.630649495964,
    -851.051818308968, 851.051818308968, -729.472987121973, 729.472987121973,
    -607.894155934977, 607.894155934977, -486.315324747982, 486.315324747982,
    -364.736493560986, 364.736493560986, -243.157662373991, 243.157662373991,
    -121.578831186995, 121.578831186995, 0.0
]

EXPECTED_WORK_OFFSET = [-252.155018851338, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

class TestParameterExtract:
    def test_sat_trans_fl(self, tmp_path):
        file = tmp_path / "method"
        file.write_text(SAT_TRANS_FL_BLOCK)
        result = parameter_extract(file, "PVM_SatTransFL")
        assert result == EXPECTED_SAT_TRANS_FL

    def test_work_offset(self, tmp_path):
        file = tmp_path / "method"
        file.write_text(WORK_OFFSET_BLOCK)
        result = parameter_extract(file, "PVM_FrqWorkOffset")
        assert result == EXPECTED_WORK_OFFSET

    def test_missing_header(self, tmp_path):
        file = tmp_path / "method"
        file.write_text("##$SomeOther= ( 2 )\n1.0 2.0\n")
        with pytest.raises(ValueError, match="non trovato"):
            parameter_extract(file, "PVM_SatTransFL")

    def test_too_many_numbers(self, tmp_path, capsys):
        block = (
            "##$PVM_SatTransFL= ( 2 )\n"
            "10.0 20.0 30.0\n"
            "##$NextParam= ( 1 )\n"
            "100.0\n"
        )
        file = tmp_path / "method"
        file.write_text(block)
        result = parameter_extract(file, "PVM_SatTransFL")
        assert result == [10.0, 20.0]
        captured = capsys.readouterr().out
        assert "Attenzione" in captured

# ----------------------------------------------------------------------
# Tests for statistical accumulation helpers
# ----------------------------------------------------------------------
class TestStatistics:
    def test_init_stats_lists(self):
        length = 5
        stats = _init_stats_lists(length)
        assert len(stats["max_vals"]) == length
        assert all(v == 0.0 for v in stats["max_vals"])
        assert len(stats["sd_max_vals"]) == length

    def test_accumulate_averages(self):
        acc = _init_stats_lists(3)
        total_folders = 2
        val1 = ([10, 20, 30], [1.0, 2.0, 3.0], [100.0, 200.0, 300.0])
        _accumulate_averages(acc, val1, total_folders)
        val2 = ([30, 40, 50], [3.0, 4.0, 5.0], [300.0, 400.0, 500.0])
        _accumulate_averages(acc, val2, total_folders)

        expected_idx = [20, 30, 40]
        expected_val = [2, 3, 4]
        expected_freq = [200, 300, 400]
        for i in range(3):
            assert np.isclose(acc["max_indexes"][i], expected_idx[i])
            assert np.isclose(acc["max_vals"][i], expected_val[i])
            assert np.isclose(acc["sat_trans_hz"][i], expected_freq[i])

    def test_std_dev(self):
        acc = _init_stats_lists(2)
        val1 = ([0, 0], [1.0, 2.0], [10.0, 20.0])
        val2 = ([0, 0], [3.0, 6.0], [30.0, 60.0])

        _accumulate_averages(acc, val1, 2)
        _accumulate_averages(acc, val2, 2)

        sd_acc = _init_stats_lists(2)
        avg = {
            "max_indexes": acc["max_indexes"],
            "max_vals": acc["max_vals"],
            "sat_trans_hz": acc["sat_trans_hz"],
        }
        _accumulate_squared_diffs(sd_acc, val1, avg)
        _accumulate_squared_diffs(sd_acc, val2, avg)
        _finalize_std_dev(sd_acc, 2)

        # For two elements, sample std = |a-b|/sqrt(2)
        assert np.isclose(sd_acc["sd_max_vals"][0], np.sqrt(((1-2)**2 + (3-2)**2) / 1))
        assert np.isclose(sd_acc["sd_max_vals"][1], np.sqrt(((2-4)**2 + (6-4)**2) / 1))
        assert np.isclose(sd_acc["sd_sat_trans_hz"][0], np.sqrt(((10-20)**2 + (30-20)**2) / 1))

    def test_compute_integrals_stats(self):
        analysis_results = {
            "s1": {"integrals": {"regionA": 1.0, "regionB": 2.0}},
            "s2": {"integrals": {"regionA": 2.0, "regionB": 4.0}},
            "s3": {"integrals": {"regionA": 3.0, "regionB": 6.0}},
        }
        keys = ["s1", "s2", "s3"]
        stats = _compute_integrals_stats(keys, analysis_results)
        mean, std = stats["mean"], stats["std"]

        assert np.isclose(mean["regionA"], 2.0)
        assert np.isclose(mean["regionB"], 4.0)
        assert np.isclose(std["regionA"], 1.0)   # sample std of 1,2,3
        assert np.isclose(std["regionB"], 2.0)   # sample std of 2,4,6

        # Single sample → std = 0
        single = _compute_integrals_stats(["s1"], analysis_results)
        assert single["mean"]["regionA"] == 1.0
        assert single["std"]["regionA"] == 0.0

# ----------------------------------------------------------------------
# Tests for Spline fit
# ----------------------------------------------------------------------
class TestSplineFit:
    def test_spline_passes_through_original_points(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 1, 3])
        # Build an x_fit that includes the original x values
        x_fit = np.linspace(1, 4, 50)
        res = spline_fit(x, y, x_fit=x_fit)
        assert res["fit_successful"]
        # Find indices of original x in x_fit
        idx = [np.argmin(np.abs(x_fit - xi)) for xi in x]
        np.testing.assert_allclose(res["y_fit"][idx], y, rtol=1e-3)

    def test_x_fit_custom(self):
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 0])
        custom_x = np.linspace(0, 2, 10)
        res = spline_fit(x, y, x_fit=custom_x)
        assert np.array_equal(res["x_fit"], custom_x)

    def test_spline_fit_returns_valid_output(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 1.0, 3.0])
        res = spline_fit(x, y, n_points=50)

        assert res["fit_successful"] is True
        assert isinstance(res["x"], np.ndarray)
        assert isinstance(res["y"], np.ndarray)
        assert isinstance(res["x_fit"], np.ndarray)
        assert isinstance(res["y_fit"], np.ndarray)
        assert res["x_fit"].shape == (50,)
        assert res["y_fit"].shape == (50,)
        np.testing.assert_array_equal(res["x"], x)
        np.testing.assert_array_equal(res["y"], y)                