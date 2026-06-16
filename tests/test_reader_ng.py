# test_reader_ng.py
import numpy as np
import pytest
from pathlib import Path
from scipy.integrate import simpson
from scipy.interpolate import PchipInterpolator

from reader_ng import (
    find_maximum,
    compute_regions_integrals,
    DEFAULT_METABOLITE_REGIONS,
    N_POINTS_FIT,
    parameter_extract,
    _compute_group_stats,
    spline_fit,
    build_common_ppm_grid,
    collect_replicate_differences,
    constrained_lorentzian
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
        for region, (start, end) in DEFAULT_METABOLITE_REGIONS.items():
            expected = end - start
            assert np.isclose(integrals[region], expected, rtol=1e-6, atol=1e-9), \
                f"Region {region}: expected {expected}, got {integrals[region]}"

    @pytest.mark.parametrize("region", list(DEFAULT_METABOLITE_REGIONS.keys()))
    def test_triangle(self, region):
        """Triangle peaking at 1 → integral = width/2."""
        start, end = DEFAULT_METABOLITE_REGIONS[region]
        x_fit = np.linspace(start, end, N_POINTS_FIT)
        centre = (start + end) / 2
        width = end - start
        y_fit = 1.0 - np.abs(x_fit - centre) / (width / 2)

        integrals = compute_regions_integrals(x_fit, y_fit)
        expected = width / 2
        assert np.isclose(integrals[region], expected, rtol=1e-4, atol=1e-7), \
            f"Region {region}: expected {expected}, got {integrals[region]}"

    def test_oscillatory(self):
        """Compare with high‑resolution Simpson reference."""
        def oscillatory(x):
            return 0.5 + 0.3 * np.sin(2 * np.pi * x / 5) + 0.2 * np.cos(2 * np.pi * x / 2.5)

        # Dense reference grid
        x_full = np.linspace(-10, 10, 2000)
        y_full = oscillatory(x_full)
        reference = {}
        for region, (start, end) in DEFAULT_METABOLITE_REGIONS.items():
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

        for region in DEFAULT_METABOLITE_REGIONS:
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
    def test_compute_integrals_stats(self):
        analysis_results = {
            "s1": {"integrals": {"regionA": 1.0, "regionB": 2.0}},
            "s2": {"integrals": {"regionA": 2.0, "regionB": 4.0}},
            "s3": {"integrals": {"regionA": 3.0, "regionB": 6.0}},
        }
        keys = ["s1", "s2", "s3"]
        stats = _compute_group_stats(keys, analysis_results) 
        mean, std = stats["mean"], stats["std"]
        
        assert np.isclose(mean["regionA"], 2.0)
        assert np.isclose(mean["regionB"], 4.0)
        assert np.isclose(std["regionA"], 1.0)   # sample std of 1,2,3
        assert np.isclose(std["regionB"], 2.0)   # sample std of 2,4,6

        # Single sample → std = 0
        single = _compute_group_stats(["s1"], analysis_results)
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

class TestBuildCommonPpmGrid:
    def test_basic(self):
        """Two replicates with overlapping ppm ranges."""
        analysis_results = {
            "rep1": {"zero_corrected_ppm": [-2.0, 1.0, 5.0]},
            "rep2": {"zero_corrected_ppm": [ 0.0, 3.0, 6.0]}
        }
        x = build_common_ppm_grid(["rep1", "rep2"], analysis_results, n_points=5)
        assert x.min() == -2.0
        assert x.max() ==  6.0
        assert len(x) == 5
        assert np.allclose(x, np.linspace(-2.0, 6.0, 5))

    def test_single_replicate(self):
        """Only one replicate: range = its own min/max."""
        analysis_results = {
            "only": {"zero_corrected_ppm": [10.0, 20.0]}
        }
        x = build_common_ppm_grid(["only"], analysis_results, n_points=3)
        assert x.min() == 10.0
        assert x.max() == 20.0
        assert len(x) == 3
        assert np.allclose(x, np.linspace(10.0, 20.0, 3))

    def test_multiple_replicates_different_lengths(self):
        """Different number of points per replicate."""
        analysis_results = {
            "a": {"zero_corrected_ppm": [1.0, 4.0]},
            "b": {"zero_corrected_ppm": [2.0, 3.0, 5.0]}
        }
        x = build_common_ppm_grid(["a", "b"], analysis_results, n_points=10)
        assert x.min() == 1.0
        assert x.max() == 5.0
        assert len(x) == 10

    def test_missing_zero_corrected_ppm(self):
        """Missing 'zero_corrected_ppm' raises ValueError."""
        analysis_results = {
            "bad": {"max_vals": [0.5]}  # no zero_corrected_ppm
        }
        with pytest.raises(ValueError, match="Missing 'zero_corrected_ppm'"):
            build_common_ppm_grid(["bad"], analysis_results)

    def test_empty_ppm_list(self):
        """Empty list raises ValueError."""
        analysis_results = {
            "rep": {"zero_corrected_ppm": []}
        }
        with pytest.raises(ValueError, match="empty"):
            build_common_ppm_grid(["rep"], analysis_results)

    def test_invalid_range(self):
        """If min == max (all same ppm), the grid would have zero range,
           but linspace(5.0, 5.0, 5) returns all 5.0s, which is fine.
           However, we check that min < max is enforced (if not, it raises
           because global_min >= global_max)."""
        analysis_results = {
            "a": {"zero_corrected_ppm": [5.0, 5.0]}
        }
        # This should actually succeed because min==max is allowed by linspace,
        # but our code currently raises if min >= max. We should test the
        # intended behaviour. If you later decide to allow min==max (grid of
        # constant value), you can adjust the code. For now, test the current
        # behaviour (raises ValueError).
        with pytest.raises(ValueError, match="Invalid ppm range"):
            build_common_ppm_grid(["a"], analysis_results)        

class TestCollectReplicateDifferences:
    def test_two_replicates_same(self):
        """Two identical replicates -> diff curves are identical, SEM=0."""
        x_data = np.array([0.0, 1.0, 2.0])
        sigmoid_corrected = np.array([0.8, 1.0, 0.9])
        A, gamma, y_min = 1.2, 1.0, 0.5
        analysis_results = {
            "rep1": {"zero_corrected_ppm": x_data.tolist(),
                     "sigmoid_corrected": sigmoid_corrected.tolist(),
                     "A": A, "gamma": gamma, "y_min": y_min},
            "rep2": {"zero_corrected_ppm": x_data.tolist(),
                     "sigmoid_corrected": sigmoid_corrected.tolist(),
                     "A": A, "gamma": gamma, "y_min": y_min},
        }
        x_common = np.linspace(0.0, 2.0, 20)
        result = collect_replicate_differences(
            ["rep1", "rep2"], analysis_results, x_common=x_common, compute_integrals=False
        )
        assert np.allclose(result["sem_diff"], 0.0)
        assert len(result["individual_diffs"]) == 2
        # The two diff curves should be identical
        np.testing.assert_allclose(result["individual_diffs"][0],
                                   result["individual_diffs"][1], rtol=1e-10)

    def test_single_replicate(self):
        """Single replicate: SEM = 0, mean = its diff."""
        x_data = np.array([0.0, 2.0])
        sigmoid_corrected = np.array([0.6, 0.8])
        A, gamma, y_min = 1.0, 0.5, 0.4
        analysis_results = {
            "rep": {"zero_corrected_ppm": x_data.tolist(),
                    "sigmoid_corrected": sigmoid_corrected.tolist(),
                    "A": A, "gamma": gamma, "y_min": y_min}
        }
        x_common = np.linspace(0.0, 2.0, 5)
        result = collect_replicate_differences(
            ["rep"], analysis_results, x_common=x_common, compute_integrals=False
        )
        assert np.allclose(result["sem_diff"], 0.0)
        assert result["mean_diff"].shape == (5,)

    def test_integrals_computation(self):
        """Integrals of mean_diff should be computed correctly."""
        x_data = np.array([0.0, 1.0])
        sigmoid_corrected = np.array([0.5, 0.5])
        A, gamma, y_min = 1.0, 10.0, 0.5   # Lorentzian nearly flat
        analysis_results = {
            "rep": {"zero_corrected_ppm": x_data.tolist(),
                    "sigmoid_corrected": sigmoid_corrected.tolist(),
                    "A": A, "gamma": gamma, "y_min": y_min}
        }
        x_common = np.linspace(0.0, 1.0, 100)
        result = collect_replicate_differences(
            ["rep"], analysis_results, x_common=x_common, compute_integrals=True
        )
        assert "integrals" in result
        # Verify that integral is exactly the mean of per‑replicate integrals?
        # Here only one replicate, so it's the same.
        # We'll just check type and shape.
        assert isinstance(result["integrals"], dict)
        # All regions completely inside [0,1]? Not all default regions lie there,
        # but those that overlap will have some value. We don't test specific values.

    def test_integrals_consistency(self):
        """Integral of mean difference == mean of per-replicate integrals (same grid)."""
        import copy

        # Common grid
        x_common = np.linspace(-5, 5, 100)

        # Two different replicates
        entry1 = {
            "zero_corrected_ppm": np.array([-3.0, 0.0, 4.0]),
            "sigmoid_corrected": np.array([0.6, 0.9, 0.7]),
            "A": 1.2, "gamma": 2.0, "y_min": 0.5,
        }
        entry2 = {
            "zero_corrected_ppm": np.array([-4.0, -1.0, 2.0, 5.0]),
            "sigmoid_corrected": np.array([0.5, 0.8, 1.0, 0.6]),
            "A": 1.3, "gamma": 1.5, "y_min": 0.4,
        }

        analysis_results = {"r1": entry1, "r2": entry2}
        keys = ["r1", "r2"]

        # Manually compute per-replicate diff curves and integrals
        per_replicate_integrals = []
        for k in keys:
            e = analysis_results[k]
            ppm = np.asarray(e["zero_corrected_ppm"])
            sigmoid_corrected = np.asarray(e["sigmoid_corrected"])
            A, gamma, y_min = e["A"], e["gamma"], e["y_min"]

            y_lor = constrained_lorentzian(x_common, A, gamma, y_min)
            spline = PchipInterpolator(ppm, sigmoid_corrected)
            y_spline = spline(x_common)
            diff = y_lor - y_spline
            integrals = compute_regions_integrals(x_common, diff)
            per_replicate_integrals.append(integrals)

        # Mean of per-replicate integrals
        regions = list(per_replicate_integrals[0].keys())
        mean_integrals_manual = {
            reg: float(np.mean([p[reg] for p in per_replicate_integrals]))
            for reg in regions
        }

        # Now use collect_replicate_differences
        result = collect_replicate_differences(
            keys, analysis_results, x_common=x_common, compute_integrals=True
        )

        # Compare
        for reg in regions:
            assert np.isclose(result["integrals"][reg], mean_integrals_manual[reg],
                            rtol=1e-10, atol=1e-15), \
                f"Region {reg}: expected {mean_integrals_manual[reg]}, got {result['integrals'][reg]}"        