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
CACHE_VERSION: int = 5

# ----------------------------------------------------------------------
# Default metabolite regions (immutable template)
# ----------------------------------------------------------------------
DEFAULT_METABOLITE_REGIONS: dict[str, dict] = {
    "Glycolytic PMEs": {"ppm": [5.5, 9.0],  "width": [1.0, None], "height": [0.0, None]},
    "Pi":              {"ppm": [4.3, 5.3],  "width": [1.0, None], "height": [0.0, None]},
    "PEP 1,3 BPG":     {"ppm": [1.0, 4.3],  "width": [1.0, None], "height": [0.0, None]},
    "GAMMA-ATP":       {"ppm": [-3.5, -1.3],"width": [1.0, None], "height": [0.0, None]},
    "ALPHA,BETA-ADP":  {"ppm": [-6, -3],    "width": [1.0, None], "height": [0.0, None]},
    "ALPHA-ATP":       {"ppm": [-9, -6],    "width": [1.0, None], "height": [0.0, None]},
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