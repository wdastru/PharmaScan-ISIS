"""
constants.py
Global defaults and paths used across the saturation analysis package.
No side effects at import time.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Analysis / fitting defaults
# ----------------------------------------------------------------------
N_POINTS_FIT: int = 200
CACHE_VERSION: int = 2

# ----------------------------------------------------------------------
# Default metabolite regions (immutable template)
# ----------------------------------------------------------------------
DEFAULT_METABOLITE_REGIONS: dict[str, list[float]] = {
    "Glycolytic PMEs": [5.5, 9.0],
    "Pi": [4.3, 5.3],
    "PEP 1,3 BPG": [1.0, 4.3],
    "GAMMA-ATP": [-3.5, -1.3],
    "ALPHA,BETA-ADP": [-6, -3],
    "ALPHA-ATP": [-9, -6]
}

# ----------------------------------------------------------------------
# Directory paths (relative to the package root)
# ----------------------------------------------------------------------
_PACKAGE_ROOT = Path(__file__).resolve().parent      # saturation_analysis/

CACHE_DIR   = _PACKAGE_ROOT / "cache"
CONFIG_DIR  = _PACKAGE_ROOT / "configs"
OUTPUT_DIR  = _PACKAGE_ROOT / "output"

# ----------------------------------------------------------------------
# Default plot visibility dictionary
# ----------------------------------------------------------------------
def get_default_visibility() -> dict:
    """Return the default visibility dictionary used by all plotting functions."""
    return {
        "data": True,
        "spline": True,
        "lorentzian": True,
        "sigmoid": True,
        "difference": True,
        "regions": True,
        "corrected": True,
        "legend": {
            "z-spectra": True,
            "integrals": True,
        },
    }