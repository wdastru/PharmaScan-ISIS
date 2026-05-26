import numpy as np
import pytest
from reader_ng import find_maximum
from reader_ng import compute_regions_integrals, METABOLITE_REGIONS, N_POINTS_FIT

def test_find_maximum():
    arr = np.array([1, 3, 2, 5, 4])
    val, idx = find_maximum(arr)
    assert val == 5
    assert idx == 3

def test_find_maximum_with_limits():
    arr = np.array([1, 3, 5, 2, 4])
    val, idx = find_maximum(arr, start=1, end=4)  # indices 1,2,3 → values 3,5,2
    assert val == 5
    assert idx == 2
    
def test_compute_regions_integrals_constant_y():
    """With y = 1, the integral should equal the width of each region."""
    # Create x_fit covering the entire ppm range of interest (e.g., -10 to 10)
    x_fit = np.linspace(-10, 10, N_POINTS_FIT)
    y_fit = np.ones_like(x_fit)
    
    integrals = compute_regions_integrals(x_fit, y_fit)
    
    for region, bounds in METABOLITE_REGIONS.items():
        expected = bounds[1] - bounds[0]
        computed = integrals[region]
        # After boundary insertion, error is only due to interpolation between points.
        # Use a very tight tolerance.
        assert np.isclose(computed, expected, rtol=1e-6, atol=1e-9), \
            f"Region {region}: expected {expected}, got {computed}"

@pytest.mark.parametrize("region", list(METABOLITE_REGIONS.keys()))
def test_compute_regions_integrals_triangle(region):
    """For a triangle peaking at 1 inside the region, integral = width / 2."""
    bounds = METABOLITE_REGIONS[region]
    start, end = bounds
    # x_fit exactly covers the region
    x_fit = np.linspace(start, end, N_POINTS_FIT)
    centre = (start + end) / 2
    width = end - start
    # Triangle: 0 at boundaries, 1 at centre
    y_fit = 1.0 - np.abs(x_fit - centre) / (width / 2)
    
    integrals = compute_regions_integrals(x_fit, y_fit)
    expected = width / 2
    computed = integrals[region]
    assert np.isclose(computed, expected, rtol=1e-4, atol=1e-7), \
        f"Region {region}: expected {expected}, got {computed}"
    
    # Other regions must be zero because x_fit does not cover them
    for other in METABOLITE_REGIONS:
        if other != region:
            assert integrals[other] == 0.0, \
                f"Region {other} should be 0, got {integrals[other]}"
            
def test_compute_regions_integrals_oscillatory():
    """Compare integrals on an oscillatory function against a high‑resolution reference."""
    from scipy.integrate import simpson

    # Define an oscillatory, positive function that mimics a difference curve
    def oscillatory(x):
        return 0.5 + 0.3 * np.sin(2 * np.pi * x / 5) + 0.2 * np.cos(2 * np.pi * x / 2.5)
        # stays positive over [-10, 10] with min ~0.05

    # Full ppm range covering all regions
    x_full = np.linspace(-10, 10, 2000)   # dense grid for reference
    y_full = oscillatory(x_full)

    # Compute reference integrals using Simpson's rule on the dense grid
    reference_integrals = {}
    for region, (start, end) in METABOLITE_REGIONS.items():
        mask = (x_full >= start) & (x_full <= end)
        if np.any(mask):
            x_reg = x_full[mask]
            y_reg = y_full[mask]
            # Use Simpson if enough points, otherwise trapezoid
            if len(x_reg) > 3:
                ref = simpson(y_reg, x_reg)
            else:
                ref = np.trapezoid(y_reg, x_reg)
        else:
            ref = 0.0
        reference_integrals[region] = ref

    # Now run the actual function with the standard N_POINTS_FIT (200) grid
    x_fit = np.linspace(-10, 10, N_POINTS_FIT)
    y_fit = oscillatory(x_fit)
    computed_integrals = compute_regions_integrals(x_fit, y_fit)

    # Compare
    for region in METABOLITE_REGIONS:
        expected = reference_integrals[region]
        computed = computed_integrals[region]
        # Relative tolerance 1e-2 is safe because of coarser grid + boundary insertion
        assert np.isclose(computed, expected, rtol=1e-2, atol=1e-5), \
            f"Region {region}: expected {expected:.6f}, got {computed:.6f}"            