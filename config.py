"""
config.py
Configuration loading/saving, merging, migration, and serialization helpers.
Uses constants from saturation_analysis.constants.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from termcolor import colored

from constants import (
    CONFIG_DIR,
    OUTPUT_DIR,
    DEFAULT_METABOLITE_REGIONS,
)

# ----------------------------------------------------------------------
# Metabolite regions (editable copy)
# ----------------------------------------------------------------------
# This is the single mutable copy used across the package.
# To reset to defaults, call reset_metabolite_regions().
METABOLITE_REGIONS = DEFAULT_METABOLITE_REGIONS.copy()

# ----------------------------------------------------------------------
# Directory helpers
# ----------------------------------------------------------------------
def ensure_config_dir() -> None:
    """Crea la cartella delle configurazioni se non esiste."""
    CONFIG_DIR.mkdir(exist_ok=True)

def ensure_output_dir() -> None:
    """Crea la cartella degli output se non esiste."""
    OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Utility: merge default values into a configuration dictionary
# ----------------------------------------------------------------------
def merge_config_defaults(defaults: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge `defaults` into `current`.
    For every key in `defaults` that is missing in `current`, add it.
    If both values are dicts, recurse.
    Otherwise, keep the `current` value.
    """
    merged = current.copy()
    for key, default_val in defaults.items():
        if key not in merged:
            merged[key] = default_val
        elif isinstance(default_val, dict) and isinstance(merged[key], dict):
            merged[key] = merge_config_defaults(default_val, merged[key])
    return merged

# ----------------------------------------------------------------------
# Config listing, loading, saving
# ----------------------------------------------------------------------
def list_configs() -> List[Path]:
    """Restituisce la lista dei file di configurazione (.json) nella cartella configs."""
    ensure_config_dir()
    return sorted(CONFIG_DIR.glob("*.json"))

def load_config(name: str) -> Dict[str, Any]:
    """Carica una configurazione per nome (senza estensione)."""
    config_path = CONFIG_DIR / f"{name}.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # --- Migration from old format ---
        if "groups" not in config:
            with_ref = config.get("with_ref", False)
            ref_count = config.get("multiple_amount_ref", 0) if with_ref else 0
            sample_count = config.get("multiple_amount", 0) if config.get("with_multiple", False) else 0
            folders = [Path(p) for p in config.get("folders", [])]
            topspin_folders = [Path(p) for p in config.get("topspin_folders", [])]
            groups = []
            idx = 0
            if with_ref and ref_count > 0:
                groups.append({
                    "label": "reference",
                    "is_reference": True,
                    "folders": folders[idx:idx+ref_count],
                    "topspin_folders": topspin_folders[idx:idx+ref_count]
                })
                idx += ref_count
            if sample_count > 0:
                groups.append({
                    "label": "sample",
                    "is_reference": False,
                    "folders": folders[idx:idx+sample_count],
                    "topspin_folders": topspin_folders[idx:idx+sample_count]
                })
            config["groups"] = groups
            # Remove old keys
            for old_key in ("with_ref", "with_multiple", "multiple_amount",
                            "multiple_amount_ref", "folders", "topspin_folders"):
                config.pop(old_key, None)
        else:
            # Ensure folders and files are Path objects
            for grp in config["groups"]:
                if "folders" in grp:
                    grp["folders"] = [Path(p) for p in grp.get("folders", [])]
                if "topspin_folders" in grp:
                    grp["topspin_folders"] = [Path(p) for p in grp.get("topspin_folders", [])]
                if "files" in grp:
                    grp["files"] = [Path(p) for p in grp.get("files", [])]
        return config
    except Exception as e:
        print(colored(
            f"Errore nel caricamento della configurazione '{name}': {e}", "red", attrs=["bold"])
        )
        return {}

def save_config(name: str, config: Dict[str, Any]) -> None:
    ensure_config_dir()
    config_path = CONFIG_DIR / f"{name}.json"
    to_save = config.copy()
    # Convert Paths to strings and ensure groups structure is clean
    if "groups" in to_save:
        to_save["groups"] = [
            {
                **grp, 
                "folders": [str(p) for p in grp.get("folders", [])],
                "topspin_folders": [str(p) for p in grp.get("topspin_folders", [])],
                "files": [str(p) for p in grp.get("files", [])],
            } 
            for grp in to_save["groups"]
        ]
    # Remove any leftover old keys (safety)
    for old in ("with_ref", "with_multiple", "multiple_amount",
                "multiple_amount_ref", "folders"):
        to_save.pop(old, None)
    try:
        with open(config_path, "w") as f:
            json.dump(to_save, f, indent=4)
        print(f"Configurazione salvata come '{name}'")
    except Exception as e:
        print(colored(
            f"Errore nel salvataggio: {e}", "red", attrs=["bold"])
        )

# ----------------------------------------------------------------------
# Interactive config selector
# ----------------------------------------------------------------------
def select_or_create_config() -> Tuple[str, Dict[str, Any]]:
    """Mostra config esistenti, permette di scegliere o crearne una nuova."""
    config_files = list_configs()
    print("\n--- Configurazioni disponibili ---")
    for i, cf in enumerate(config_files, 1):
        print(f"{i}. {cf.stem}")
    print(f"{len(config_files)+1}. Nuova configurazione")
    print(f"{len(config_files)+2}. Nessuna configurazione")
    
    while True:
        try:
            choice = input(f"\nScegli (1-{len(config_files)+2}): ").strip()
            idx = int(choice)
            if 1 <= idx <= len(config_files):
                name = config_files[idx-1].stem
                return name, load_config(name)
            elif idx == len(config_files) + 1:
                name = input("Inserisci un nome per la nuova configurazione: ").strip()
                if not name:
                    print("Nome non valido.")
                    continue
                # Verifica che non esista già
                if (CONFIG_DIR / f"{name}.json").exists():
                    print("Configurazione già esistente. Scegli un altro nome.")
                    continue
                return name, {}
            elif idx == len(config_files) + 2:
                return "", {}
            else:
                print("Scelta non valida.")
        except ValueError:
            print("Inserisci un numero valido.")

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

def save_analysis_results(config_name: str, analysis_results: dict) -> None:

    ensure_output_dir()
    analysis_results = replace_uc_objects(analysis_results)  # Convert uc objects to dicts for JSON serialization
    version = f"-{analysis_results.get('__script_version__', '')}" if analysis_results.get('__script_version__') else ''
    with open(Path(OUTPUT_DIR / f"{config_name}{version}.json"), "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2, cls=SafeEncoder, ensure_ascii=False)   # indent for human readability
