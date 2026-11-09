import json
import numpy as np
from pathlib import Path

from termcolor import colored
from constants import OUTPUT_DIR

# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle methods / functions
        if callable(obj):
            return repr(obj)               # or str(obj)
        # Handle numpy arrays & scalars
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        # For any other object with a __dict__, try that
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        # Last resort
        try:
            return str(obj)
        except Exception:
            return f"<{type(obj).__name__}>"
                
def _safe_get_attr(obj, attr, default=None):
    """Return attribute value. If it's a callable (method), call it first."""
    val = getattr(obj, attr, default)
    if callable(val):
        try:
            return val()
        except TypeError:
            # Fallback if call fails
            return str(val) if default is None else default
    return val

def replace_uc_objects(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "uc" and value.__class__.__name__ == "unit_conversion":
                uc = value
                obj[key] = {
                    "size": uc._size,
                    "complex": uc._cplx,
                    "sw": uc._sw,
                    "obs_freq": uc._obs,
                    "carrier": uc._car,
                    "delta": uc._delta,
                    "first_ppm": uc._first,
                    # Use the safe getter for unit – it may be a method
                    "unit": _safe_get_attr(uc, "unit", "unknown"),
                    "ppm": list(uc.ppm) if hasattr(uc.ppm, '__iter__') else uc.ppm,
                    "hz": list(uc.hz) if hasattr(uc.hz, '__iter__') else uc.hz,
                }
            else:
                replace_uc_objects(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                replace_uc_objects(item)
    return obj

def _dedup_x_experiment(exp: dict) -> None:
    spline = exp.get("spline_fit_results", {})
    sig    = exp.get("sigmoidal_envelope_results", {})
    lor    = exp.get("lorentzian_envelope_results", {})

    # La griglia comune (200 pt) è la stessa in 4 posti: prendine una
    fit_x = exp.get("diff_x")
    if fit_x is None:
        fit_x = spline.get("x_fit")

    if fit_x is not None:
        exp["fit_x"] = fit_x

    exp.pop("diff_x", None)
    spline.pop("x", None)       # era = zero_corrected_ppm
    spline.pop("x_fit", None)   # era = fit_x
    sig.pop("x", None)
    lor.pop("x", None)

def _dedup_x_summary(summary: dict) -> None:
    spline = summary.get("spline_fit_results", {})
    lor    = summary.get("lorentzian_envelope_results", {})
    spline.pop("x", None)       # era = x_common
    spline.pop("x_fit", None)   # era = x_common
    lor.pop("x", None)          # era = x_common

def dedup_x(analysis_results: dict) -> None:
    for gd in analysis_results.get("groups", {}).values():
        for exp in gd.get("experiments", {}).values():
            _dedup_x_experiment(exp)
        if gd.get("summary"):
            _dedup_x_summary(gd["summary"])

def ensure_output_dir() -> None:
    """Crea la cartella degli output se non esiste."""
    OUTPUT_DIR.mkdir(exist_ok=True)

def save_analysis_results(config_name: str, analysis_results: dict) -> None:

    ensure_output_dir()
    analysis_results = replace_uc_objects(analysis_results)  # Convert uc objects to dicts for JSON serialization
    analysis_results["__schema__"] = "dedup_x_v1"
    dedup_x(analysis_results)
    version = f"-{analysis_results.get('__script_version__', '')}" if analysis_results.get('__script_version__') else ''
    with open(Path(OUTPUT_DIR / f"{config_name}{version}.json"), "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2, cls=SafeEncoder, ensure_ascii=False)   # indent for human readability


