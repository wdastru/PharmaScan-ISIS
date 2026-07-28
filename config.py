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

import os

from constants import (
    CONFIG_DIR,
    OUTPUT_DIR,
    DEFAULT_METABOLITE_REGIONS,
)

from data_io import (
    select_experiment_folder, select_text_file
)

from utils import (
    ask_yes_no, ask_user_for_ppm_range, ask_int, ask_float, ask_choice
)

from constants import (
    get_default_visibility
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

def ensure_complete_config(config_name: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
    modified = False

    # ---------- Interactive creation of groups (if none exist) ----------
    if not config_data.get("groups"):
        print("No groups defined. Interactive setup.")
        with_ref = ask_yes_no("Include a reference group?", default=False)
        n_sample_groups = ask_int("Number of additional sample groups", min_val=0, default=1)
        groups = []

        # Helper to build a single group interactively
        def _create_group_interactively(label, is_ref):
            print(f"\n--- Group '{label}' ---")
            data_type = ask_choice(
                "Data source type",
                choices=["Bruker folders", "TopSpin folders", "Text (x/y) files"],
                default="Bruker folders"
            )
            if data_type == "Bruker folders":   # Paravision folders
                cnt = ask_int("Number of folders", min_val=1, default=1)
                folders = []
                for _ in range(cnt):
                    folders.append(select_experiment_folder())
                return {"label": label, "is_reference": is_ref, "folders": folders, "topspin_folders": [], "files": []}
            elif data_type == "TopSpin folders":    # TopSpin folders
                cnt = ask_int("Number of folders", min_val=1, default=1)
                topspin_folders = []
                for _ in range(cnt):
                    tmp_path: Path = select_experiment_folder()
                    topspin_folders.append(tmp_path)
                    #topspin_folders.append(select_experiment_folder())
                return {"label": label, "is_reference": is_ref, "folders": [], "topspin_folders": topspin_folders, "files": []}
            else:  # Text files
                cnt = ask_int("Number of text files", min_val=1, default=1)
                files = []
                BF1_values = []
                for _ in range(cnt):
                    files.append(select_text_file())
                    BF1_values.append(float(input("Enter BF1 value for this file (MHz): ")))
                return {"label": label, "is_reference": is_ref, "folders": [], "topspin_folders": [], "files": files, "BF1": BF1_values}

        if with_ref:
            ref_label = input("Label for reference group (default: reference): ").strip() or "reference"
            groups.append(_create_group_interactively(ref_label, True))

        for i in range(n_sample_groups):
            label = input(f"Label for sample group {i+1} (default: group{i+1}): ").strip() or f"group{i+1}"
            groups.append(_create_group_interactively(label, False))

        config_data["groups"] = groups
        modified = True

    else:
        # ---------- Ensure existing groups have required keys ----------
        for grp in config_data["groups"]:
            if "label" not in grp:
                grp["label"] = "group"
            if "is_reference" not in grp:
                grp["is_reference"] = False
            # Introduce the new key if missing
            if "files" not in grp:
                grp["files"] = []
                modified = True
            if "BF1" not in grp:
                grp["BF1"] = []
                modified = True

    # ---------- Fill missing data paths for any group ----------
    for grp in config_data["groups"]:
        if not grp.get("folders") and not grp.get("topspin_folders") and not grp.get("files"):
            # This group has no paths at all – prompt interactively
            print(f"\nGroup '{grp['label']}' has no data paths defined.")
            data_type = ask_choice(
                f"Data source type for '{grp['label']}'",
                choices=["Bruker folders", "TopSpin folders", "Text (x/y) files"],
                default="Bruker folders"
            )
            if data_type == "Bruker folders":
                cnt = ask_int("Number of folders", min_val=1, default=1)
                for _ in range(cnt):
                    grp.setdefault("folders", []).append(select_experiment_folder())
            elif data_type == "TopSpin folders":
                cnt = ask_int("Number of folders", min_val=1, default=1)
                for _ in range(cnt):
                    grp.setdefault("topspin_folders", []).append(select_experiment_folder())
            else:
                cnt = ask_int("Number of text files", min_val=1, default=1)
                for _ in range(cnt):
                    grp.setdefault("files", []).append(select_text_file())
                    grp.setdefault("BF1", []). append(float(input("Enter BF1 value for this file (MHz): ")))
            modified = True
        elif grp.get("folders") and grp.get("files"):
            # Both provided – warn but keep; the analysis will decide
            print(colored(
                f"Warning: Group '{grp['label']}' has both folders and files. "
                "Folders will be used for analysis.",
                "yellow"
            ))
        elif grp.get("files") and not grp.get("folders") and not grp.get("topspin_folders"):
            if "BF1" not in grp or len(grp["BF1"]) != len(grp["files"]):
                print(f"Group '{grp['label']}' has text files but missing or mismatched BF1 values.")
                bf1_values = []
                for f in grp["files"]:
                    bf1_values.append(float(input(f"Enter BF1 value for file '{f}' (MHz): ")))
                grp["BF1"] = bf1_values
                modified = True

    # ---------- Validate that all provided folder/file paths exist ----------
    for grp in config_data["groups"]:
        # Validate "folders" key
        if "folders" in grp:
            valid_folders = []
            for folder in grp["folders"]:
                if os.path.exists(folder):
                    valid_folders.append(folder)
                else:
                    print(colored(f"Folder not found: {folder}", "red"))
                    if ask_yes_no("Do you want to select a new folder?", default=True):
                        new_folder = select_experiment_folder()
                        if new_folder and os.path.exists(new_folder):
                            valid_folders.append(new_folder)
                            modified = True
                        else:
                            print(colored("New folder also invalid - skipping.", "yellow"))
                    else:
                        print(colored("Missing folder remains in the list.", "yellow"))
                        valid_folders.append(folder)

            grp["folders"] = valid_folders

        # Validate "topspin_folders" key (if present)
        if "topspin_folders" in grp:
            valid_tf = []
            for folder in grp["topspin_folders"]:
                if os.path.exists(folder):
                    valid_tf.append(folder)
                else:
                    print(colored(f"TopSpin folder not found: {folder}", "red"))
                    if ask_yes_no("Do you want to select a new folder?", default=True):
                        new_folder = select_experiment_folder()
                        if new_folder and os.path.exists(new_folder):
                            valid_tf.append(new_folder)
                            modified = True
                        else:
                            print(colored("New folder also invalid – skipping.", "yellow"))
                    else:
                        print(colored("Missing folder remains in the list.", "yellow"))
                        valid_folders.append(folder)
            grp["topspin_folders"] = valid_tf

        # Validate "files" key
        if "files" in grp:
            # Keep BF1 in sync if it exists (same length as files)
            has_bf1 = "BF1" in grp and len(grp["BF1"]) == len(grp["files"])
            valid_files = []
            valid_bf1 = [] if has_bf1 else None
            for i, file in enumerate(grp["files"]):
                if os.path.exists(file):
                    valid_files.append(file)
                    if has_bf1:
                        valid_bf1.append(grp["BF1"][i])
                else:
                    print(colored(f"File not found: {file}", "red"))
                    if ask_yes_no("Do you want to select a new file?", default=True):
                        new_file = select_text_file()
                        if new_file and os.path.exists(new_file):
                            valid_files.append(new_file)
                            if has_bf1:
                                new_bf1 = float(input(f"Enter BF1 value for '{new_file}' (MHz): "))
                                valid_bf1.append(new_bf1)
                            modified = True
                        else:
                            print(colored("New file also invalid – skipping.", "yellow"))
                    else:
                        print(colored("Missing folder remains in the list.", "yellow"))
                        valid_folders.append(folder)
            grp["files"] = valid_files
                    
    # ---------- ppm range handling ----------
    if config_data.get("start_ppm") is None or config_data.get("end_ppm") is None:
        config_data["ppm_missing"] = True
    else:
        config_data["ppm_missing"] = False

    # (If ppm_missing, the actual prompting occurs later during analysis,
    #  because we may need a spectrum to show. The flag is set here.)

    # ---------- lb handling ----------
    if config_data.get("lb_Hz") is None:
        config_data["lb_Hz"] = ask_float("Enter line broadening (lb) in Hz", default=0.005)
        modified = True

    # ---------- Plot visibility defaults ----------
    default_vis = get_default_visibility()
    if "plot_visibility" in config_data:
        current_vis = config_data["plot_visibility"]
        new_vis = merge_config_defaults(default_vis, current_vis)
        if new_vis != current_vis:
            config_data["plot_visibility"] = new_vis
            modified = True
    else:
        config_data["plot_visibility"] = default_vis
        modified = True

    # ---------- Metabolite regions defaults ----------
    if "metabolite_regions" not in config_data:
        config_data["metabolite_regions"] = DEFAULT_METABOLITE_REGIONS
        modified = True

    else:
        for name, region in config_data["metabolite_regions"].items():
            if isinstance(region, list):
                # Old style: just [start, end] ppm boundaries
                start, end = region[0], region[1]
                config_data["metabolite_regions"][name] = {
                    "ppm": [start, end],
                    "width": [1.0, None],   # default width
                    "height": [0.0, None]   # default height
                }
                modified = True
            elif isinstance(region, dict):
                # Possibly new style, but ensure required keys exist
                if "ppm" not in region:
                    raise ValueError(f"Metabolite region '{name}' missing 'ppm' key.")
                if "width" not in region:
                    region["width"] = [1.0, None]
                    modified = True
                if "height" not in region:
                    region["height"] = [0.0, None]
                    modified = True
            else:
                raise ValueError(f"Invalid metabolite region definition for '{name}'")
    
    # ---------- Save if modified ----------
    if modified and config_name:
        save_config(config_name, config_data)

    return config_data
