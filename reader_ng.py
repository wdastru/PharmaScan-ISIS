"""
reader_ng.py

Processing of Bruker NMR data for saturation transfer experiments.
Reads FIDs, applies corrections, displays spectra with interactive checkboxes,
allows the user to select a ppm range, and calculates maxima to generate
a saturation transfer curve.

Supports multiple groups (reference + any number of sample groups).
"""

import nmrglue as ng
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from matplotlib.figure import Figure
import re
import tkinter as tk
from tkinter import filedialog
from typing import List, Optional, Tuple, Dict, Any
from scipy.interpolate import PchipInterpolator, interp1d 
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy import stats
import json
from termcolor import colored
import os
import hashlib
from joblib import dump, load
import csv

print(f"Using nmrglue version: {ng.__version__}")

DEFAULT_METABOLITE_REGIONS: dict[str, List[float]] = {
    "Glycolytic PMEs": [5.5, 9.0],
    "Pi": [4.3, 5.3],
    "PEP 1,3 BPG": [1.0, 4.3],
    "GAMMA-ATP": [-3.5, -1.3],
    "ALPHA,BETA-ADP": [-6, -3],
    "ALPHA-ATP": [-9, -6]
}
METABOLITE_REGIONS = DEFAULT_METABOLITE_REGIONS.copy()
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
N_POINTS_FIT = 200

# ----------------------------------------------------------------------
# Utility functions for configuration migration and merging
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
# Cache management
# ----------------------------------------------------------------------
def _cache_path(config_name: str, config: Dict[str, Any]) -> Path:
    # Se il nome è vuoto (nessuna configurazione salvata), usa un hash
    if not config_name:
        key_str = json.dumps(_build_cache_key(config), sort_keys=True)
        name = hashlib.md5(key_str.encode()).hexdigest()[:8]
    else:
        # Pulisci il nome per evitare caratteri problematici
        name = "".join(c for c in config_name if c.isalnum() or c in " _-").rstrip()
    return CACHE_DIR / f"analysis_{name}.joblib"

def _build_cache_key(config: Dict[str, Any]) -> str:
    """
    Crea una chiave univoca basata sui parametri e sulle cartelle (compresa la data di modifica).
    """
    groups = config.get("groups", [])
    # Dati delle cartelle: percorso e timestamp dell'ultima modifica
    folder_info = []
    for grp in groups:
        for f in grp.get("folders", []):
            try:
                mtime = os.path.getmtime(f)
            except OSError:
                mtime = 0
            folder_info.append((str(f), mtime))
    key_data = {
        "groups": [{"label": g["label"], "is_reference": g.get("is_reference", False)}
                   for g in groups],
        "start_ppm": config.get("start_ppm"),
        "end_ppm": config.get("end_ppm"),
        "folders_info": folder_info,
        "metabolite_regions": METABOLITE_REGIONS,
        "use_extra_lorentzians": config.get("use_extra_lorentzians", False),
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()

def load_cache(config_name: str, config: Dict[str, Any]) -> Optional[dict]:
    cache_path = _cache_path(config_name, config)
    if not cache_path.exists():
        return None
    try:
        payload = load(cache_path)
        if payload.get("key") == _build_cache_key(config):
            return payload["analysis_results"]
        else:
            print(f"Cache obsoleta per '{config_name}'.")
            return None
    except Exception as e:
        print(colored(f"Errore cache per '{config_name}': {e}", "red", attrs=["bold"]))
        return None

def save_cache(config_name: str, config: Dict[str, Any], analysis_results: dict) -> None:
    cache_path = _cache_path(config_name, config)
    payload = {"key": _build_cache_key(config), "analysis_results": analysis_results}
    dump(payload, cache_path, compress=3)
    print(f"Cache salvata per '{config_name}' in {cache_path.name}")

# ----------------------------------------------------------------------
# Configuration handling
# ----------------------------------------------------------------------
CONFIG_DIR = Path(__file__).parent / "configs"
OUTPUT_DIR = Path(__file__).parent / "output"

def ensure_config_dir() -> None:
    """Crea la cartella delle configurazioni se non esiste."""
    CONFIG_DIR.mkdir(exist_ok=True)

def ensure_output_dir() -> None:
    """Crea la cartella degli output se non esiste."""
    OUTPUT_DIR.mkdir(exist_ok=True)

def get_default_visibility() -> Dict[str, bool]:
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
            groups = []
            idx = 0
            if with_ref and ref_count > 0:
                groups.append({
                    "label": "reference",
                    "is_reference": True,
                    "folders": folders[idx:idx+ref_count]
                })
                idx += ref_count
            if sample_count > 0:
                groups.append({
                    "label": "sample",
                    "is_reference": False,
                    "folders": folders[idx:idx+sample_count]
                })
            config["groups"] = groups
            # Remove old keys
            for old_key in ("with_ref", "with_multiple", "multiple_amount",
                            "multiple_amount_ref", "folders"):
                config.pop(old_key, None)
        else:
            # Ensure folders and files are Path objects
            for grp in config["groups"]:
                if "folders" in grp:
                    grp["folders"] = [Path(p) for p in grp.get("folders", [])]
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
# Utility functions
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

    with open(Path(OUTPUT_DIR / f"{config_name}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"Output values for {config_name}"])          # header
            for key, value in analysis_results.items():
                writer.writerow([key, value])          # csv module converts to string automatically

    analysis_results = replace_uc_objects(analysis_results)  # Convert uc objects to dicts for JSON serialization

    with open(Path(OUTPUT_DIR / f"{config_name}.json"), "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2, cls=SafeEncoder, ensure_ascii=False)   # indent for human readability

def find_methods(obj, path=""):
    """Return list of (path, object_repr) for any method or function found."""
    issues = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            current = f"{path}.{k}" if path else k
            if callable(v) and not isinstance(v, (type, type(None))):
                # Catch user-defined methods, built-in methods, functions, lambdas
                issues.append((current, repr(v)))
            else:
                issues.extend(find_methods(v, current))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            current = f"{path}[{i}]"
            if callable(v) and not isinstance(v, type):
                issues.append((current, repr(v)))
            else:
                issues.extend(find_methods(v, current))
    return issues

def find_maximum(arr: np.ndarray,
               start: Optional[int] = None,
               end: Optional[int] = None) -> Tuple[float, int]:
    """
    Find the maximum value and its index in a slice of the array.

    Parameters
    ----------
    arr : np.ndarray
        1D array to search.
    start : Optional[int], default None
        Start index (inclusive). If None, starts at 0.
    end : Optional[int], default None
        End index (exclusive). If None or > len(arr), uses len(arr).

    Returns
    -------
    Tuple[float, int]
        (maximum value, index of the maximum in the original array).
    """
    if start is None:
        start = 0
    if end is None or end > len(arr):
        end = len(arr)
    sub_arr = arr[start:end]
    max_val = sub_arr.max()
    max_idx = sub_arr.argmax() + start
    return float(max_val), int(max_idx)

def parameter_extract(file_path: Path, PARAMETER: str) -> List[float]:
    if not file_path.exists():
        raise FileNotFoundError(colored(
            f"{file_path} not found.", "red", attrs=["bold"])
        )
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    # Look for the header and also capture the block up to the next '##$'
    hdr_pattern = rf"##\${PARAMETER}=\s*\(\s*(?P<N>\d+)\s*\)\s*\n(?P<block>.*?)(?=\n##\$|\Z)"
    match = re.search(hdr_pattern, text, re.DOTALL)
    if not match:
        raise ValueError(colored(
            f"Header '##${PARAMETER}=( N )' non trovato in {file_path}.", "red", attrs=["bold"])
        )

    N = int(match.group("N"))
    block = match.group("block")
    print(f"{PARAMETER} dimension: {N}")

    # Extract numbers only from this block
    num_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    vals = re.findall(num_pattern, block)
    if len(vals) < N:
        raise ValueError(colored(
            f"Trovati solo {len(vals)} numeri nel blocco, attesi {N}.", "red", attrs=["bold"])
        )
    if len(vals) > N:
        print(colored(
            f"Attenzione: trovati {len(vals)} numeri nel blocco (attesi {N}), uso i primi {N}.", "yellow")
        )
    return [float(v) for v in vals[:N]]

def apply_phase(data: np.ndarray, p0: float, p1: float) -> np.ndarray:
    """
    Apply zero-order and first-order phase correction to a complex spectrum.

    Parameters
    ----------
    data : np.ndarray
        Complex spectrum.
    p0 : float
        Zero-order phase (degrees).
    p1 : float
        First-order phase (degrees).

    Returns
    -------
    np.ndarray
        Phase-corrected spectrum (complex).
    """
    return ng.proc_base.ps(data, p0=p0, p1=p1)

def show_phase(data: np.ndarray, p0: float, p1: float) -> None:
    """
    Display a plot of the real part of the spectrum after phase correction.

    Parameters
    ----------
    data : np.ndarray
        Complex spectrum.
    p0 : float
        Zero-order phase (degrees).
    p1 : float
        First-order phase (degrees).
    """
    phased = apply_phase(data, p0, p1)
    plt.plot(np.real(phased))
    plt.show()

def ppm_to_index(uc: Any, user_ppm: float) -> int:
    """
    Convert a ppm value to the nearest index on the frequency axis.
    If user_ppm is outside the spectrum, a warning is printed but the
    nearest valid index is still returned.

    Parameters
    ----------
    uc : nmrglue unit conversion object
        Obtained from uc_from_udic.
    user_ppm : float
        Desired ppm value.

    Returns
    -------
    int
        Corresponding index.
    """
    ppm_axis = uc.ppm_scale()
    if user_ppm < ppm_axis.min() or user_ppm > ppm_axis.max():
        print(colored(
            f"Warning: ppm {user_ppm} is outside the spectrum range "
            f"({ppm_axis.min():.2f} - {ppm_axis.max():.2f}). Using the nearest point.",
            'yellow'
        ))
    return int(np.abs(ppm_axis - user_ppm).argmin())

def compute_regions_integrals(x_fit: np.ndarray, y_fit: np.ndarray) -> Dict[str, float]:
    """
    Calcola gli integrali di regione per tutte le regioni definite in METABOLITE_REGIONS.
    Aggiunge i valori esatti dei bordi (start, end) all'array x per una maggiore precisione.

    Parameters
    ----------
    x_fit : np.ndarray
        Array delle ascisse (ppm) della curva fitted.
    y_fit : np.ndarray
        Array delle ordinate (intensità normalizzate) della curva fitted.

    Returns
    -------
    Dict[str, float]
        Dizionario con chiavi i nomi delle regioni e valori gli integrali calcolati.
    """
    integrals = {}
    for region, (start, end) in METABOLITE_REGIONS.items():
        # Trova i punti all'interno dell'intervallo
        mask = (x_fit >= start) & (x_fit <= end)
        if not np.any(mask):
            integrals[region] = 0.0
            continue

        # Estrai i punti interni
        x_inside = x_fit[mask]
        y_inside = y_fit[mask]

        # Aggiungi il bordo sinistro se non è già presente
        if start not in x_inside:
            y_start = np.interp(start, x_fit, y_fit)
            x_inside = np.concatenate(([start], x_inside))
            y_inside = np.concatenate(([y_start], y_inside))

        # Aggiungi il bordo destro se non è già presente
        if end not in x_inside:
            y_end = np.interp(end, x_fit, y_fit)
            x_inside = np.concatenate((x_inside, [end]))
            y_inside = np.concatenate((y_inside, [y_end]))

        # Calcola l'integrale con il metodo dei trapezi
        area = np.trapezoid(y_inside, x_inside)
        integrals[region] = area

    return integrals

def plot_data(
    x, 
    y, 
    x_fit, 
    y_fit, 
    y_std_data=None, 
    title="Max Values vs Saturation ppm",
    xlabel="Saturation ppm", 
    ylabel="Max Value",
    fit_label="", 
    invert_x=True, 
    add_lorentz=False, 
    lorentzian_envelope_results=None, 
    add_sigmoid=False, 
    sigmoidal_envelope_results=None,
    diff_x=None, 
    diff_y=None, 
    diff_label="Difference (Envelope - Spline)",
    visibility: Optional[Dict[str, bool]] = None,
    window_title: Optional[str] = None
) -> Figure:
    # Se non fornito, usa i default globali
    if visibility is None:
        visibility = get_default_visibility()
    
    fig = plt.figure(num=window_title, figsize=(8, 5))

    # Data points
    if visibility.get("data", True):
        if y_std_data is not None:
            plt.errorbar(x, y, yerr=np.array(y_std_data), fmt='o', color='b', label='Data')
        else:
            plt.plot(x, y, 'o', color='b', label='Data')
                
    # Spline fit
    if visibility.get("spline", True):
        plt.plot(x_fit, y_fit, 'r-', label=fit_label)

    # Lorentzian envelope
    if add_lorentz and lorentzian_envelope_results is not None and visibility.get("lorentzian", True):
        A = lorentzian_envelope_results.get("A")
        gamma = lorentzian_envelope_results.get("gamma")
        x_lor = lorentzian_envelope_results["x"]
        y_lor = lorentzian_envelope_results["y"]
        plt.plot(x_lor, y_lor, 'g--', linewidth=2,
                label=f'Lorentzian (A={A:.3f}, γ={gamma:.3f})')

    # Sigmoid envelope
    if add_sigmoid and sigmoidal_envelope_results is not None and visibility.get("sigmoid", True):
        L = sigmoidal_envelope_results.get("L")
        R = sigmoidal_envelope_results.get("R")
        tau = sigmoidal_envelope_results.get("tau")
        x_sig = sigmoidal_envelope_results["x"]
        y_sig = sigmoidal_envelope_results["y"]
        plt.plot(x_sig, y_sig, 'c--', linewidth=2,
                    label=f'Sigmoid (L={L:.2f}, R={R:.2f}, τ={tau:.3f})')                

    # Difference curve
    if diff_x is not None and diff_y is not None and visibility.get("difference", True):
        plt.plot(diff_x, diff_y, 'm-', linewidth=1.5, label=diff_label)

    # Metabolite regions
    if visibility.get("regions", True):
        ax = plt.gca()
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(METABOLITE_REGIONS))]
        for idx, (region_name, (start, end)) in enumerate(METABOLITE_REGIONS.items()):
            ax.axvspan(start, end, facecolor=colors[idx], alpha=0.25, edgecolor='none', label=region_name)
    
    if invert_x:
        plt.gca().invert_xaxis()
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if visibility["legend"].get("z-spectra", True):
        plt.legend()
    plt.show(block=False)
    return fig

def spline_fit(x, y, x_fit=None, n_points=N_POINTS_FIT) -> Dict[str, Any]:
    """
    Parameters
    ----------
    x, y : array-like
        Data points.
    x_fit : array-like, optional
        Pre‑computed x values for the fitted curve. If None, generate with np.linspace.
    n_points : int
        Only used if x_fit is None.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    fit_successful = False
    y_fit = None
    fit_label = ""
    try:
        spline = PchipInterpolator(x, y)
        if x_fit is None:
            x_fit = np.linspace(x.min(), x.max(), n_points)
        y_fit = spline(x_fit)
        fit_successful = True
        fit_label = "Spline Fit"
    except Exception as e:
        print(f"Spline fit failed: {e}")
    
    return {
            'x': x, 
            'y': y,
            'x_fit': x_fit, 
            'y_fit': y_fit,
            'fit_label': fit_label, 
            'fit_successful': fit_successful
        }    

def plot_spectra(title, spectra, n_exp, ppm_axis, sat_trans_hz, visibility=None, window_title=None) -> Figure:
    fig, ax = plt.subplots(num=window_title, figsize=(12, 6))
    lines = []
    labels = []
    for exp_idx in range(n_exp):
        line, = ax.plot(ppm_axis, np.real(spectra[exp_idx]),
                        label=f"{exp_idx:>2} : {sat_trans_hz[exp_idx]:.2f}",
                        alpha=0.7, linewidth=1.2)
        lines.append(line)
        labels.append(line.get_label())
    ax.invert_xaxis()
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    rax = fig.add_axes([0.80, 0.15, 0.19, 0.70])
    visibility_states = [l.get_visible() for l in lines]
    checks = CheckButtons(rax, labels, visibility_states)
    def _on_check(label):
        idx = labels.index(label)
        lines[idx].set_visible(not lines[idx].get_visible())
        fig.canvas.draw_idle()
    def _check_all(event):
        for i, line in enumerate(lines):
            if not line.get_visible():
                line.set_visible(True)
                checks.lines[i].set_visible(True)
        fig.canvas.draw_idle()
    def _uncheck_all(event):
        for i, line in enumerate(lines):
            if line.get_visible():
                line.set_visible(False)
                checks.lines[i].set_visible(False)
        fig.canvas.draw_idle()
    checks.on_clicked(_on_check)
    ax_all = fig.add_axes([0.80, 0.90, 0.09, 0.05])
    btn_all = Button(ax_all, "Check all")
    btn_all.on_clicked(_check_all)
    ax_none = fig.add_axes([0.90, 0.90, 0.09, 0.05])
    btn_none = Button(ax_none, "Uncheck all")
    btn_none.on_clicked(_uncheck_all)
    fig.tight_layout(rect=[0, 0, 0.80, 1])
    plt.show(block=False)
    return fig

def normalize_max_vals(max_vals, global_max, global_min):
    for i in range(len(max_vals)):
        max_vals[i] = (max_vals[i] - global_min) / (global_max - global_min) if global_max > global_min else 0.0
    return max_vals

def find_max_vals(spectra, start_idx, end_idx):
    max_vals: List[float] = []
    max_indexes: List[int] = []
    global_max: float = float('-inf')
    global_min: float = float('inf')
    val: float = 0.0
    idx: int = 0
    for exp_idx, spec in spectra.items():
        val, idx = find_maximum(spec, start=start_idx, end=end_idx)
        if val > global_max:
            global_max = val
        if val < global_min:
            global_min = val
        max_vals.append(val)
        max_indexes.append(idx)
    
    return max_vals, max_indexes, global_max, global_min

def ask_user_for_ppm_range(default_start=None, default_end=None) -> Tuple[float, float]:
    while True:
        try:
            start_prompt = "Enter the minimum ppm (start)"
            end_prompt = "Enter the maximum ppm (end)"
            if default_start is not None:
                start_prompt += f" (default {default_start})"
            if default_end is not None:
                end_prompt += f" (default {default_end})"
            start_input = input(f"{start_prompt}: ").strip()
            end_input = input(f"{end_prompt}: ").strip()
            start_ppm = float(start_input) if start_input else default_start
            end_ppm = float(end_input) if end_input else default_end
            if start_ppm is None or end_ppm is None:
                print("Inserire entrambi i valori.")
                continue
            if end_ppm <= start_ppm:
                print("end deve essere maggiore di start.")
                continue
            return start_ppm, end_ppm
        except ValueError:
            print("Inserire numeri validi.")

def correct_sat_frequencies(sat_trans_hz, max_indexes, work_offset_hz, uc, bf1):
    sat_trans_f1_ppm = [0.0] * len(sat_trans_hz)
    for i, (st_hz, idx) in enumerate(zip(sat_trans_hz, max_indexes)):
        delta = work_offset_hz[0] - uc.hz(idx)
        if st_hz != 0.0:
            sat_trans_hz[i] += delta
        sat_trans_f1_ppm[i] = sat_trans_hz[i] / bf1
    return sat_trans_f1_ppm

def ask_yes_no(prompt: str, default: Optional[bool] = None) -> bool:
    default_prompt = ""
    if default is True:
        default_prompt = " (Y/n)"
    elif default is False:
        default_prompt = " (y/N)"
    else:
        default_prompt = " (y/n)"
    while True:
        answer = input(prompt + default_prompt + ": ").strip().lower()
        if not answer and default is not None:
            return default
        if answer in ('y', 'yes'):
            return True
        if answer in ('n', 'no'):
            return False
        print("Rispondi con y o n.")

def ask_int(prompt: str, min_val: int = None, max_val: int = None, default: Optional[int] = None) -> int:
    default_prompt = f" (default {default})" if default is not None else ""
    while True:
        answer = input(f"{prompt}{default_prompt}: ").strip()
        if not answer and default is not None:
            return default
        try:
            value = int(answer)
            if min_val is not None and value < min_val:
                print(f"Valore deve essere >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Valore deve essere <= {max_val}")
                continue
            return value
        except ValueError:
            print("Inserire un numero intero.")

def ask_choice(prompt: str, choices: List[str], default: Optional[str] = None) -> str:
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    while True:
        ans = input(f"{prompt} (1-{len(choices)})" + (f" [{choices.index(default)+1}]" if default else "") + ": ").strip()
        if not ans and default:
            return default
        try:
            idx = int(ans) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            pass
        print("Invalid choice.")

# ----------------------------------------------------------------------
# Envelope fitting functions (unchanged)
# ----------------------------------------------------------------------
def constrained_lorentzian(x, A, gamma, y_min):
    if gamma == 0.0:
        return np.full_like(x, A)
    return A - (A - y_min) * gamma**2 / (gamma**2 + x**2)

def estimate_constrained_lorentzian(x_data, y_data):
    x = np.asarray(x_data)
    y = np.asarray(y_data)
    y_min = np.min(y)
    y_max = np.max(y)
    if y_max == y_min:
        return y_max, 0.0
    def error_for_A(A):
        if A < y_max:
            return np.inf
        gamma_max = np.inf
        for xi, yi in zip(x, y):
            if yi <= y_min:
                continue
            bound_sq = (A - yi) / (yi - y_min) * xi**2
            if bound_sq <= 0:
                return np.inf
            gamma_max = min(gamma_max, np.sqrt(bound_sq))
        if gamma_max <= 0.0:
            return np.inf
        def mse(gamma):
            if gamma == 0.0:
                y_pred = np.full_like(x, A)
            else:
                y_pred = A - (A - y_min) * gamma**2 / (gamma**2 + x**2)
            return np.sum((y_pred - y)**2)
        res = minimize_scalar(mse, bounds=(0.0, gamma_max), method='bounded')
        return res.fun
    upper_A = y_max + 5 * (y_max - y_min) if y_max > y_min else y_max + 1.0
    res_A = minimize_scalar(error_for_A, bounds=(y_max, upper_A), method='bounded')
    best_A = res_A.x
    gamma_max = np.inf
    for xi, yi in zip(x, y):
        if yi <= y_min:
            continue
        bound_sq = (best_A - yi) / (yi - y_min) * xi**2
        gamma_max = min(gamma_max, np.sqrt(bound_sq))
    def mse(gamma):
        if gamma == 0.0:
            y_pred = np.full_like(x, best_A)
        else:
            y_pred = best_A - (best_A - y_min) * gamma**2 / (gamma**2 + x**2)
        return np.sum((y_pred - y)**2)
    res_gamma = minimize_scalar(mse, bounds=(0.0, gamma_max), method='bounded')
    best_gamma = res_gamma.x
    return best_A, best_gamma

def constrained_sigmoid(x, L, R, tau, x0=0.0):
    return R + (L - R) / (1.0 + np.exp(-(x - x0) / tau))

def estimate_constrained_sigmoid(x_data, y_data, fix_center=True, x0_fixed=0.0):
    x = np.asarray(x_data)
    y = np.asarray(y_data)
    x0 = x0_fixed
    def solve_LR_for_tau(tau):
        z = 1.0 / (1.0 + np.exp(-(x - x0) / tau))
        def mse(params):
            L, R = params
            y_pred = L * z + R * (1 - z)
            return np.sum((y_pred - y)**2)
        constraints = []
        for i in range(len(x)):
            A_i = np.array([z[i], 1 - z[i]])
            b_i = y[i]
            constraints.append({'type': 'ineq', 'fun': lambda p, A=A_i, b=b_i: A[0]*p[0] + A[1]*p[1] - b})
        mask_left = x > x0
        mask_right = x < x0
        L0 = np.max(y[mask_right]) if np.any(mask_right) else np.max(y)
        R0 = np.max(y[mask_left]) if np.any(mask_left) else np.max(y)
        res = minimize(mse, [L0, R0], method='SLSQP', constraints=constraints,
                       bounds=[(0, None), (0, None)], options={'maxiter': 1000})
        if res.success:
            return res.x[0], res.x[1], res.fun
        else:
            L = R = np.max(y)
            return L, R, np.sum((np.full_like(y, L) - y)**2)
    def objective_tau(tau):
        if tau <= 0:
            return np.inf
        _, _, err = solve_LR_for_tau(tau)
        return err
    tau_min = 1e-6
    tau_max = np.ptp(x) * 10
    res_tau = minimize_scalar(objective_tau, bounds=(tau_min, tau_max), method='bounded')
    if res_tau.success:
        tau_opt = res_tau.x
    else:
        tau_opt = np.ptp(x) / 4
        print(colored(f"Warning: optimization for tau failed, using fallback tau={tau_opt:.4f}", "yellow", attrs=["bold"]))
    L_opt, R_opt, _ = solve_LR_for_tau(tau_opt)
    return L_opt, R_opt, tau_opt

# ----------------------------------------------------------------------
# Multi Lorentzian feature functions
# ----------------------------------------------------------------------

def lorentzian_peak(x, h, x0, w):
    """
    Lorentziana classica: h * w^2 / (w^2 + (x - x0)^2)
    h : ampiezza (positiva per CEST, negativa per NOE)
    x0: centro (ppm)
    w : semi-larghezza a metà altezza (ppm)
    """
    return h * w**2 / (w**2 + (x - x0)**2)

def fit_global_lorentzians(x_data, y_data, regions, center_init, baseline=1.0, fixed_width=0.2):
    """
    Fit simultaneo:
      - saturazione diretta (centro=0) -> (h_center, gamma)
      - una lorentziana per ogni regione (h, x0, w)
    
    Returns dict con 'center', 'extra', 'integrals_extra', 'y_total', 'y_center', etc.
    """
    def model(params, x):
        h_c, gamma = params[0], params[1]
        result = baseline - lorentzian_peak(x, h_c, 0, gamma)
        idx = 2
        for _ in region_list:
            h = params[idx]
            x0 = params[idx+1]
            w = params[idx+2]
            result += lorentzian_peak(x, h, x0, w) if w > 0 else 0
            idx += 3
        return result
    
    def mse(params):
        y_pred = model(params, x)
        return np.sum((y - y_pred)**2)
    
    def average_separation(values):
        """
        Returns the average difference between successive elements
        after sorting the input list in ascending order.

        Uses the fact that:
            sum of consecutive gaps = max - min
        so average gap = (max - min) / (n - 1)

        Parameters
        ----------
        values : list of numbers
            Input list (will be sorted internally).

        Returns
        -------
        float
            Average separation between successive sorted elements.

        Raises
        ------
        ValueError
            If the list contains fewer than 2 elements.
        """
        if len(values) < 2:
            raise ValueError("At least two values are required to calculate separation.")
        
        sorted_vals = sorted(values)
        total_range = sorted_vals[-1] - sorted_vals[0]
        return total_range / (len(sorted_vals) - 1)

    x = np.asarray(x_data)
    y = np.asarray(y_data)
    
    h0_c, gamma0 = center_init
    params_init = [h0_c, gamma0]        # altezza, larghezza della lorentziana centrale
    bounds = [(0, None), (0.05, 2.0)]   # altezza >= 0, larghezza > 0
    
    region_list = list(regions.items())
    for reg_name, (start, end) in region_list:
        x0_init = (start + end) / 2.0
        params_init += [0.0, x0_init, fixed_width]  # altezza, posizione e larghezza della lorentziana
        bounds += [(None, 0.0), (start, end), (average_separation(x), None)]    # altezza, posizione e larghezza della lorentziana (la minima larghezza è la separazione media tra i punti x)
    
    res = minimize(mse, params_init, bounds=bounds, method='L-BFGS-B')
    if not res.success:
        print(colored("Warning: global lorentzian fit did not converge", "yellow"))
    
    h_c_opt, gamma_opt = res.x[0], res.x[1]
    y_center = baseline - lorentzian_peak(x, h_c_opt, 0, gamma_opt)
    
    extra_results = {}
    integrals_extra = {}
    idx = 2
    for reg_name, (start, end) in region_list:
        h_opt = res.x[idx]
        x0_opt = res.x[idx+1]
        w_opt = res.x[idx+2]
        integral = np.pi * h_opt * w_opt
        y_peak = lorentzian_peak(x, h_opt, x0_opt, w_opt)
        extra_results[reg_name] = {'h': h_opt, 'x0': x0_opt, 'w': w_opt,
                                   'integral': integral, 'y': y_peak}
        integrals_extra[reg_name] = integral
        idx += 3
    
    y_extra_sum = np.zeros_like(x)
    for r in extra_results.values():
        y_extra_sum += r['y']
    y_total = y_center + y_extra_sum
    
    return {
        'center': {'h': h_c_opt, 'gamma': gamma_opt, 'baseline': baseline},
        'extra': extra_results,
        'integrals_extra': integrals_extra,
        'y_center': y_center,
        'y_extra_sum': y_extra_sum,
        'y_total': y_total,
        'x': x,
        'success': res.success
    }

def plot_lorentzian_decomposition(
        x_data, 
        y_data, 
        x_common, 
        L_main_y, 
        extra_lor_results, 
        title, 
        invert_x=True,
        window_title=None
    ):
    """
    Plot dei dati, lorentziana principale, lorentziane extra e somma.
    - x_data, y_data: punti originali (z-spectrum corretto)
    - x_common: griglia per le curve continue
    - L_main_y: valore della lorentziana principale su x_common
    - extra_lor_results: dict da fit_extra_lorentzians
    - title: titolo del plot
    """
    fig, ax = plt.subplots(num=window_title, figsize=(10,6))
    
    # Dati sperimentali
    ax.plot(x_data, y_data, 'o', color='k', label='Data (corrected)')
    
    # Lorentziana principale
    ax.plot(x_common, L_main_y, 'b-', linewidth=2, label='Lorentzian main')
    
    # Somma delle lorentziane extra
    sum_extra = np.zeros_like(x_common)
    for reg, res in extra_lor_results.items():
        y_peak = lorentzian_peak(x_common, res['h'], res['x0'], res['w'])
        sum_extra += y_peak
        # Plot singole lorentziane (opzionale, potrebbe essere confusionario, meglio usare tratti sottili)
        ax.plot(x_common, y_peak, '--', alpha=0.5, label=f"{reg} (h={res['h']:.2f})")
    
    # Totale
    total_y = L_main_y + sum_extra
    ax.plot(x_common, total_y, 'r-', linewidth=2, label='Total (main + extra)')
    
    if invert_x:
        ax.invert_xaxis()
    
    ax.set_xlabel('Saturation ppm')
    ax.set_ylabel('Normalized intensity')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show(block=False)
    return fig

# ----------------------------------------------------------------------
# Process a single z-spectrum to get integrals
# ----------------------------------------------------------------------
def process_zspectrum_and_integrals(max_vals, zero_corrected_ppm, use_extra_lorentzians=False) -> Dict[str, Any]:

    """Fit envelopes, spline, lorentzians, compute difference and integrals for one dataset."""
    
    # Sort
    combined = list(zip(zero_corrected_ppm, max_vals))
    combined.sort()
    zero_corrected_ppm, max_vals_sorted = zip(*combined)
    zero_corrected_ppm = list(zero_corrected_ppm)
    max_vals_sorted = list(max_vals_sorted)

    # Common grid
    x_common = np.linspace(min(zero_corrected_ppm), max(zero_corrected_ppm), N_POINTS_FIT)

    # Sigmoid envelope
    L, R, tau = estimate_constrained_sigmoid(zero_corrected_ppm, max_vals_sorted,
                                             fix_center=True, x0_fixed=0.0)
    y_sig = constrained_sigmoid(x_common, L, R, tau)
    sigmoid_env = {"L": L, "R": R, "tau": tau, "x": x_common, "y": y_sig,
                   "fit_label": f'Sigmoid (L={L:.3f}, R={R:.3f}, τ={tau:.3f})',
                   "fit_successful": True}

    # Correct with sigmoid
    linspace_indices = [np.argmin(np.abs(x_common - v)) for v in zero_corrected_ppm]
    sig_corrected = []
    for i, idx in enumerate(linspace_indices):
        env_val = sigmoid_env["y"][idx]
        if np.abs(env_val) < 1e-12:
            sig_corrected.append(max_vals_sorted[i])
        else:
            sig_corrected.append(max_vals_sorted[i] / env_val)

    # Lorentzian envelope on corrected data
    A, gamma = estimate_constrained_lorentzian(zero_corrected_ppm, sig_corrected)
    y_min = np.min(sig_corrected)
    y_lor = constrained_lorentzian(x_common, A, gamma, y_min)
    lor_env = {"A": A, "gamma": gamma, "x": x_common, "y": y_lor,
               "fit_label": f'Lorentzian (A={A:.3f}, γ={gamma:.3f})',
               "fit_successful": True}

    # Spline fit on corrected data
    spline_res = spline_fit(x=zero_corrected_ppm, y=sig_corrected, x_fit=x_common)
    if not spline_res.get("fit_successful", False):
        return {
            "integrals": {},
            "diff_x": None, "diff_y": None,
            "sigmoidal_envelope_results": sigmoid_env,
            "lorentzian_envelope_results": lor_env,
            "spline_fit_results": spline_res,
            # extra-lorentzians fitting
            "x_data": zero_corrected_ppm,
            "y_data": sig_corrected,
            "x_common": x_common,
            "extra_lorentzians_results": None,
            "integrals_extra": None,
            "lorentzian_fit": None
        }

    # Difference and integrals
    diff_y = lor_env["y"] - spline_res["y_fit"]
    integrals = compute_regions_integrals(x_common, diff_y)

    # Dizionario base da ritornare
    result = {
        "integrals": integrals,
        "diff_x": x_common,
        "diff_y": diff_y,
        "sigmoidal_envelope_results": sigmoid_env,
        "lorentzian_envelope_results": lor_env,
        "spline_fit_results": spline_res,
        # chiavi per la parte sperimentale
        "extra_lorentzians_results": None,
        "integrals_extra": None,
        "lorentzian_fit": None,
        "x_data": zero_corrected_ppm,
        "y_data": sig_corrected,
        "x_common": x_common
    }

    # --- Blocco SPERIMENTALE: solo se richiesto, non modifica i risultati sopra ---
    if use_extra_lorentzians:
        y_min_data = np.min(sig_corrected)
        h_center_init = A - y_min_data
        gamma_init = gamma
        lorentzian_fit = fit_global_lorentzians(
            zero_corrected_ppm, 
            sig_corrected,
            regions=METABOLITE_REGIONS,
            center_init=(h_center_init, gamma_init),
            baseline=1.0,
            fixed_width=0.2
        )
        result["lorentzian_fit"] = lorentzian_fit
        result["extra_lorentzians_results"] = lorentzian_fit["extra"]
        result["integrals_extra"] = lorentzian_fit["integrals_extra"]

    return result

# ----------------------------------------------------------------------
# Group statistics and p-values
# ----------------------------------------------------------------------
def _compute_group_stats(folder_keys: List[str], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    # Collect all integrals keys that appear in any entry (at least "integrals" is guaranteed)
    possible_keys = set()
    for key in folder_keys:
        entry = analysis_results.get(key, {})
        for k in ("integrals", "integrals_extra"):
            if k in entry:
                possible_keys.add(k)

    stats = {}
    for integrals_key in possible_keys:
        integrals_list = []
        for key in folder_keys:
            entry = analysis_results.get(key, {})
            integrals = entry.get(integrals_key)
            if integrals:
                integrals_list.append(integrals)
        if not integrals_list:
            stats[integrals_key] = {"mean": {}, "std": {}}
            continue
        regions = list(integrals_list[0].keys())
        mean_dict = {}
        std_dict = {}
        for reg in regions:
            vals = [d[reg] for d in integrals_list if reg in d]
            n = len(vals)
            if n > 1:
                mean_dict[reg] = float(np.mean(vals))
                std_dict[reg] = float(np.std(vals, ddof=1))
            elif n == 1:
                mean_dict[reg] = float(vals[0])
                std_dict[reg] = 0.0
            else:
                mean_dict[reg] = 0.0
                std_dict[reg] = 0.0
        stats[integrals_key] = {"mean": mean_dict, "std": std_dict}
    return stats

def _compute_pvalues(ref_keys: List[str], sample_keys: List[str],
                     analysis_results: Dict[str, Any], test='t-test') -> Dict[str, Optional[float]]:
    ref_integrals = []
    for key in ref_keys:
        integr = analysis_results.get(key, {}).get("integrals")
        if integr:
            ref_integrals.append(integr)
    sample_integrals = []
    for key in sample_keys:
        integr = analysis_results.get(key, {}).get("integrals")
        if integr:
            sample_integrals.append(integr)
    if not ref_integrals or not sample_integrals:
        print("Not enough data for p-value calculation.")
        return {}
    regions = list(ref_integrals[0].keys())
    pvalues = {}
    for reg in regions:
        ref_vals = [d[reg] for d in ref_integrals if reg in d]
        samp_vals = [d[reg] for d in sample_integrals if reg in d]
        if len(ref_vals) < 2 or len(samp_vals) < 2:
            pvalues[reg] = None
            continue
        if test == 't-test':
            _, p = stats.ttest_ind(ref_vals, samp_vals, equal_var=False)
        elif test == 'mann-whitney':
            _, p = stats.mannwhitneyu(ref_vals, samp_vals, alternative='two-sided')
        else:
            raise ValueError(f"Unknown test: {test}")
        pvalues[reg] = float(p)
    return pvalues

def plot_multigroup_integrals(group_stats, p_values, groups,
                              integrals_key,
                              title="Integrals by region",
                              ylabel="Integral (mean ± SD)",
                              figsize=(12, 6),
                              visibility=None,
                              window_title=None) -> Figure:
    """
    Bar chart of per-group mean integrals with error bars and significance stars.

    Parameters
    ----------
    group_stats : dict
        If integrals_key is None (default), expects:
            {group_label: {"mean": {region: val}, "std": {region: val}}}
        If integrals_key is a string (e.g. "integrals" or "integrals_extra"),
        expects:
            {group_label: {integrals_key: {"mean": ...}, "std": ...}}
    p_values : dict
        {sample_group_label: {region: p_value}}
    groups : list of dict
        Each dict must contain "label" and optionally "is_reference".
    integrals_key : str or None
        Key to select the sub-dictionary from each group's stats.
        If None, stats are taken directly from the group dictionary.
    """

    if visibility is None:
        visibility = get_default_visibility()
    if not group_stats:
        return None

    # Find the set of regions from the first group
    first_label = groups[0]["label"]
    regions = list(group_stats[first_label][integrals_key]["mean"].keys())
    n_regions = len(regions)
    n_groups = len(groups)
    x = np.arange(n_regions)
    bar_width = 0.8 / n_groups
    fig, ax = plt.subplots(num=window_title, figsize=figsize)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n_groups)]
    for i, grp in enumerate(groups):
        label = grp["label"]
        means = [group_stats[label][integrals_key]["mean"][reg] for reg in regions]
        stds  = [group_stats[label][integrals_key]["std"][reg] for reg in regions]
        offset = (i - n_groups/2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds, capsize=4,
               label=label, color=colors[i], edgecolor='black')
    significance = {0.001: '***', 0.01: '**', 0.05: '*'}
    ref_label = next((grp["label"] for grp in groups if grp.get("is_reference")), None)
    if ref_label and p_values:
        for i, grp in enumerate(groups):
            if grp["label"] == ref_label:
                continue
            p_vals = p_values.get(grp["label"], {})
            for j, reg in enumerate(regions):
                p = p_vals.get(reg)
                if p is None:
                    continue
                txt = None
                for thr in sorted(significance, reverse=True):
                    if p < thr:
                        txt = significance[thr]
                        break
                if txt is None:
                    # TODO: improve visibility of p values
                    #txt = f"p={p:.3f}"
                    pass
                ref_idx = next(k for k, g in enumerate(groups) if g["label"] == ref_label)
                y_ref = group_stats[ref_label][integrals_key]["mean"][reg] + group_stats[ref_label][integrals_key]["std"][reg]
                y_this = group_stats[grp["label"]][integrals_key]["mean"][reg] + group_stats[grp["label"]][integrals_key]["std"][reg]
                y_max = max(y_ref, y_this) * 1.05
                x_pos = x[j] + (i - n_groups/2 + 0.5) * bar_width
                ax.text(x_pos, y_max, txt, ha='center', va='bottom',
                        fontweight='bold', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if visibility["legend"].get("integrals", True):
        ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plt.show(block=False)
    return fig

def plot_group_folder_integrals(group_label, group_stats, per_folder_integrals,
                                integrals_key,
                                folder_names=None,
                                title=None, ylabel="Integrale",
                                figsize=(12, 6), visibility=None,
                                window_title=None) -> Figure:
    """
    Bar chart per un singolo gruppo: barre affiancate per ogni cartella
    (con colori e nomi) e barra della media ± SD.
    
    Parameters
    ----------
    group_stats : dict
        If integrals_key is None, expects {group_label: {"mean": {...}, "std": {...}}}
        If integrals_key is a string, expects {group_label: {integrals_key: {"mean": ..., "std": ...}}}
    per_folder_integrals : dict
        {group_label: {region: [val_folder1, val_folder2, ...]}}
    integrals_key : str or None
        Key to select the sub-dictionary from the group's stats.
    """
    if visibility is None:
        visibility = get_default_visibility()

    stats = group_stats.get(group_label, {}).get(integrals_key, {"mean": {}, "std": {}})
    
    folder_vals = per_folder_integrals.get(group_label, {})
    if not stats or not folder_vals:
        print(f"Dati insufficienti per il gruppo '{group_label}'")
        return None

    regions = list(stats["mean"].keys())
    means = [stats["mean"].get(r, 0.0) for r in regions]
    stds  = [stats["std"].get(r, 0.0) for r in regions]

    first_region = regions[0]
    n_folders = len(folder_vals.get(first_region, []))
    if n_folders == 0:
        print(f"Nessun integrale per cartella nel gruppo '{group_label}'")
        return None

    # Folder names
    if folder_names is None or len(folder_names) != n_folders:
        folder_names = [f"Cartella {i+1}" for i in range(n_folders)]

    # Bar layout (unchanged from original)
    available_width = 0.8
    gap_between_bars = 0.02
    extra_gap_before_avg = 0.05
    total_bars = n_folders + 1
    total_gaps = n_folders
    bar_width = (available_width - n_folders * gap_between_bars - extra_gap_before_avg) / total_bars
    if bar_width < 0.05:
        gap_between_bars = 0.01
        extra_gap_before_avg = 0.02
        bar_width = (available_width - n_folders * gap_between_bars - extra_gap_before_avg) / total_bars
        if bar_width < 0.03:
            bar_width = 0.03
    avg_bar_width = bar_width

    total_width = (n_folders * bar_width +
                   n_folders * gap_between_bars +
                   extra_gap_before_avg +
                   avg_bar_width)

    x = np.arange(len(regions))
    fig, ax = plt.subplots(num=window_title, figsize=figsize)

    cmap = plt.get_cmap('tab10')
    folder_colors = [cmap(i % 10) for i in range(n_folders)]
    color_avg = '#2c3e50'

    for j, reg in enumerate(regions):
        start_x = x[j] - total_width / 2
        vals = folder_vals.get(reg, [])

        # Folder bars
        for k, val in enumerate(vals):
            pos = start_x + k * (bar_width + gap_between_bars) + bar_width/2
            ax.bar(pos, val, bar_width,
                   color=folder_colors[k], edgecolor='black', linewidth=0.5,
                   label=folder_names[k] if j == 0 else "")

        # Average bar
        pos_avg = start_x + n_folders * (bar_width + gap_between_bars) + extra_gap_before_avg + avg_bar_width/2
        ax.bar(pos_avg, means[j], avg_bar_width,
               yerr=stds[j], capsize=4,
               color=color_avg, edgecolor='black', linewidth=0.8,
               label='Media ± SD' if j == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    if title is None:
        title = f"Integrali per regione – {group_label}"
    ax.set_title(title)
    if visibility["legend"].get("integrals", True):
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plt.show(block=False)
    return fig

# ----------------------------------------------------------------------
# Main interactive configuration setup
# ----------------------------------------------------------------------
def select_experiment_folder(title="Select a folder") -> Path:
    root = tk.Tk()
    root.withdraw()
    return Path(filedialog.askdirectory(title=title))

def extract_parameters(folder: Path) -> Tuple[List[float], List[float]]:
    method = folder / "method"
    sat_hz = parameter_extract(method, "PVM_SatTransFL")
    offset_hz = parameter_extract(method, "PVM_FrqWorkOffset")
    return sat_hz, offset_hz

def load_spectra(folder: Path):
    dic, data = ng.bruker.read(folder)
    udic = ng.bruker.guess_udic(dic, data)
    uc = ng.fileio.bruker.fileiobase.uc_from_udic(udic, dim=1)
    ppm_axis = uc.ppm_scale()
    n_exp = dic["acqu2s"]["TD"]
    bf1 = dic["acqus"]["BF1"]
    return dic, data, uc, ppm_axis, n_exp, bf1

def process_spectra(data: np.ndarray, dic: dict, n_exp: int):
    spectra: dict = {}
    for exp_idx in range(n_exp):
        fid = data[exp_idx, :]
        fid = ng.bruker.remove_digital_filter(dic, data=fid)
        fid_zf = ng.proc_base.zf_size(fid, size=2048)
        fid_apod = ng.proc_base.em(fid_zf, lb=0.005)
        spectrum = ng.proc_base.fft(fid_apod)
        spectrum_phased = ng.proc_autophase.autops(spectrum, fn="acme")
        spectrum_phased = spectrum_phased[::-1]
        spectra[exp_idx] = spectrum_phased
    return spectra

def select_text_file(title="Select a text data file (x/y columns)") -> Path:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Text files", "*.txt *.dat"), ("All files", "*.*")]
    )
    if not file_path:
        raise ValueError("No file selected.")
    return Path(file_path)

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
                choices=["Bruker folders", "Text (x/y) files"],
                default="Bruker folders"
            )
            if data_type == "Bruker folders":
                cnt = ask_int("Number of folders", min_val=1, default=1)
                folders = []
                for _ in range(cnt):
                    folders.append(select_experiment_folder())
                return {"label": label, "is_reference": is_ref, "folders": folders, "files": []}
            else:  # Text files
                cnt = ask_int("Number of text files", min_val=1, default=1)
                files = []
                BF1_values = []
                for _ in range(cnt):
                    files.append(select_text_file())
                    BF1_values.append(float(input("Enter BF1 value for this file (MHz): ")))
                return {"label": label, "is_reference": is_ref, "folders": [], "files": files, "BF1": BF1_values}

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
        if not grp.get("folders") and not grp.get("files"):
            # This group has no paths at all – prompt interactively
            print(f"\nGroup '{grp['label']}' has no data paths defined.")
            data_type = ask_choice(
                f"Data source type for '{grp['label']}'",
                choices=["Bruker folders", "Text (x/y) files"],
                default="Bruker folders"
            )
            if data_type == "Bruker folders":
                cnt = ask_int("Number of folders", min_val=1, default=1)
                for _ in range(cnt):
                    grp.setdefault("folders", []).append(select_experiment_folder())
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
        elif grp.get("files") and not grp.get("folders"):
            if "BF1" not in grp or len(grp["BF1"]) != len(grp["files"]):
                print(f"Group '{grp['label']}' has text files but missing or mismatched BF1 values.")
                bf1_values = []
                for f in grp["files"]:
                    bf1_values.append(float(input(f"Enter BF1 value for file '{f}' (MHz): ")))
                grp["BF1"] = bf1_values
                modified = True

    # ---------- ppm range handling ----------
    if config_data.get("start_ppm") is None or config_data.get("end_ppm") is None:
        config_data["ppm_missing"] = True
    else:
        config_data["ppm_missing"] = False

    # (If ppm_missing, the actual prompting occurs later during analysis,
    #  because we may need a spectrum to show. The flag is set here.)

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

    if "use_extra_lorentzians" not in config_data:
        config_data["use_extra_lorentzians"] = False
        modified = True
                    
    # ---------- Save if modified ----------
    if modified and config_name:
        save_config(config_name, config_data)

    return config_data

# ----------------------------------------------------------------------
# Core analysis routine
# ----------------------------------------------------------------------
def run_analysis(config_name: str, config: Dict[str, Any]) -> None:
    METABOLITE_REGIONS.clear()
    METABOLITE_REGIONS.update(config.get("metabolite_regions", DEFAULT_METABOLITE_REGIONS))
    plt.ion()

    groups = config["groups"]
    start_ppm = config.get("start_ppm")
    end_ppm = config.get("end_ppm")
    ppm_missing = config.get("ppm_missing", False)
    analysis_results: dict = {}

    # --- Cache ---
    cached = load_cache(config_name, config)
    use_cache = False
    if cached:
        use_cache = ask_yes_no(f"Cache valida trovata per '{config_name}'. Vuoi evitare il ricalcolo?", default=True)
    if cached and use_cache:
        analysis_results = cached

        # ------------------------------------------------------------
        # 1. Re‑plot Z‑spettri per i gruppi (media)
        # ------------------------------------------------------------
        for grp in groups:
            label = grp["label"]
            z = analysis_results.get(label, {})
            if "spline_fit_results" in z and z["spline_fit_results"].get("fit_successful"):
                fit = z["spline_fit_results"]
                plot_data(
                    fit["x"], fit["y"], fit["x_fit"], fit["y_fit"],
                    y_std_data=z.get("sd_max_vals"),
                    title=label, invert_x=True,
                    add_lorentz=True, lorentzian_envelope_results=z.get("lorentzian_envelope_results"),
                    add_sigmoid=True, sigmoidal_envelope_results=z.get("sigmoidal_envelope_results"),
                    diff_x=z.get("diff_x"), diff_y=z.get("diff_y"),
                    diff_label="Lorentzian envelope - Spline fit",
                    visibility=config.get("plot_visibility", get_default_visibility()),
                    window_title=f"Group {label} spline fit (da cache)"
                )

        # 1b. Plot aggiuntivi di decomposizione per le medie di gruppo
        use_extra_lor = config.get("use_extra_lorentzians", False)
        if use_extra_lor:
            for grp in groups:
                label = grp["label"]
                z = analysis_results.get(label, {})
                lorentzian_fit = z.get("lorentzian_fit")
                if lorentzian_fit is not None:
                    interp_center = interp1d(lorentzian_fit["x"], lorentzian_fit["y_center"],
                                                kind='linear', fill_value="extrapolate")
                    L_main_y_common = interp_center(z.get("x_common"))
                    plot_lorentzian_decomposition(
                        x_data=z.get("x_data"),
                        y_data=z.get("y_data"),
                        x_common=z.get("x_common"),
                        L_main_y=1+lorentzian_peak(z["x_common"], -z["lorentzian_fit"]["center"]["h"], 0, z["lorentzian_fit"]["center"]["gamma"]),
                        extra_lor_results=lorentzian_fit.get("extra"),
                        title=f"Lorentzian decomposition - {label} (average)",
                        invert_x=True,
                        window_title=f"Lorentzian decomposition - {label} (average)"
                    )                

        # ------------------------------------------------------------
        # 2. Re‑plot Z‑spettri per le singole cartelle
        # ------------------------------------------------------------
        folder_keys_per_group_cached = analysis_results.get("folder_keys_per_group", [])
        for grp_idx, keys in enumerate(folder_keys_per_group_cached):
            for key in keys:
                res = analysis_results.get(key, {})
                if "spline_fit_results" in res and res["spline_fit_results"].get("fit_successful"):
                    plot_data(
                        x=res["spline_fit_results"]["x"],
                        y=res["spline_fit_results"]["y"],
                        x_fit=res["spline_fit_results"]["x_fit"],
                        y_fit=res["spline_fit_results"]["y_fit"],
                        title=f"Spline: {key}",
                        invert_x=True,
                        add_lorentz=True,
                        lorentzian_envelope_results=res.get("lorentzian_envelope_results"),
                        add_sigmoid=True,
                        sigmoidal_envelope_results=res.get("sigmoidal_envelope_results"),
                        diff_x=res.get("diff_x"),
                        diff_y=res.get("diff_y"),
                        diff_label="Lorentzian envelope - Spline fit",
                        visibility=config.get("plot_visibility", get_default_visibility()),
                        window_title=f"Spline fit: {key} (da cache)"
                    )

        # 2b. Plot aggiuntivi di decomposizione multi‑lorentziana per singole cartelle (se presenti)
        use_extra_lor = config.get("use_extra_lorentzians", False)
        if use_extra_lor:
            for grp_idx, keys in enumerate(folder_keys_per_group_cached):
                for key in keys:
                    res = analysis_results.get(key, {})
                    lorentzian_fit = res.get("lorentzian_fit")
                    if lorentzian_fit is not None:
                        plot_lorentzian_decomposition(
                            x_data=res.get("x_data"),
                            y_data=res.get("y_data"),
                            x_common=res.get("x_common"),
                            L_main_y=1+lorentzian_peak(res["x_common"], -res["lorentzian_fit"]["center"]["h"], 0, res["lorentzian_fit"]["center"]["gamma"]),
                            extra_lor_results=lorentzian_fit.get("extra"),
                            title=f"Lorentzian decomposition - {key}",
                            invert_x=True,
                            window_title=f"Lorentzian decomposition - {key} (da cache)"
                        )

        # ------------------------------------------------------------
        # 3. Prepara i dati per i grafici a barre dei singoli gruppi
        # ------------------------------------------------------------
        per_folder_integrals = {}
        for grp_idx, grp in enumerate(groups):
            label = grp["label"]
            keys = folder_keys_per_group_cached[grp_idx] if grp_idx < len(folder_keys_per_group_cached) else []
            group_folder_integrals = {}
            for key in keys:
                integr = analysis_results.get(key, {}).get("integrals", {})
                for region, val in integr.items():
                    group_folder_integrals.setdefault(region, []).append(val)
            per_folder_integrals[label] = group_folder_integrals

        group_stats = analysis_results.get("group_stats", {})

        # ------------------------------------------------------------
        # 4. Grafico a barre per gruppo (cartelle + media)
        # ------------------------------------------------------------
        for grp_idx, grp in enumerate(groups):
            label = grp["label"]
            if label in group_stats and label in per_folder_integrals:
                plot_group_folder_integrals(
                    group_label=label,
                    group_stats=group_stats,
                    per_folder_integrals=per_folder_integrals,
                    folder_names=folder_keys_per_group_cached[grp_idx] if grp_idx < len(folder_keys_per_group_cached) else [],
                    visibility=config.get("plot_visibility", get_default_visibility()),
                    title=f"Spline - integrali per regione - {label} (da cache)",
                    window_title=f"Spline - integrali per regione - {label} (da cache)",
                    integrals_key="integrals"
                )
                if use_extra_lor:
                    plot_group_folder_integrals(
                        group_label=label,
                        group_stats=group_stats,
                        per_folder_integrals=per_folder_integrals,
                        folder_names=folder_keys_per_group_cached[grp_idx] if grp_idx < len(folder_keys_per_group_cached) else [],
                        visibility=config.get("plot_visibility", get_default_visibility()),
                        title=f"Lorentzian - integrali per regione - {label} (da cache)",
                        window_title=f"Lorentzian - integrali per regione - {label} (da cache)",
                        integrals_key="integrals_extra"
                    )

        # ------------------------------------------------------------
        # 5. Grafici multi‑gruppo con p‑value (già presente)
        # ------------------------------------------------------------
        pvals = analysis_results.get("p_values", {})
        plot_multigroup_integrals(group_stats, pvals, groups,
                                  visibility=config.get("plot_visibility", get_default_visibility()),
                                  title="Spline - integrali per regione (da cache)",
                                  window_title="Spline - integrali per regione (da cache)",
                                  integrals_key="integrals")
        if use_extra_lor:
            plot_multigroup_integrals(group_stats, pvals, groups,
                visibility=config.get("plot_visibility", get_default_visibility()),
                title="Lorentzian - integrali per regione (da cache)",
                window_title="Lorentzian - integrali per regione (da cache)",
                integrals_key="integrals_extra"
            )

        # --- Saving ---
        save_analysis_results(analysis_results=analysis_results, config_name=config_name)
    
        print("Press Enter to exit...")
        input()
        plt.close('all')
        return

    # --- Live analysis ---
    group_raw = [[] for _ in groups]
    group_meta = [{} for _ in groups]
    folder_keys_per_group = [[] for _ in groups]
    use_extra_lor = config.get("use_extra_lorentzians", False)
    file_data_raw = [ [] for _ in groups ]

    for grp_idx, grp in enumerate(groups):
        label = grp["label"]

        # Determine entry type for this group
        is_folder = bool(grp.get("folders"))
        is_file   = bool(grp.get("files"))
        if is_folder and is_file:
            print(colored(
                f"Warning: Group '{label}' has both folders and files. Only folders will be used.",
                "yellow"
            ))
            is_file = False
            entries = grp["folders"]
        elif is_folder:
            entries = grp["folders"]
        elif is_file:
            entries = grp["files"]
        else:
            pass
                    
        if is_folder:   # BRUKER data
            folders = entries
            for file in folders:
                base_name = f"{file.parent.name[:12]}…{file.parent.name[-12:]}-{file.stem}"
                folder_name_short = base_name
                counter = 1
                while folder_name_short in analysis_results:
                    folder_name_short = f"{base_name}_{counter}"
                    counter += 1
                analysis_results[folder_name_short] = {}
                folder_keys_per_group[grp_idx].append(folder_name_short)

                sat_trans_hz, work_offset_hz = extract_parameters(file)
                analysis_results[folder_name_short]["sat_trans_hz"] = sat_trans_hz
                analysis_results[folder_name_short]["work_offset_hz"] = work_offset_hz

                if group_meta[grp_idx].get("work_offset_hz") is None:
                    group_meta[grp_idx]["work_offset_hz"] = work_offset_hz
                else:
                    if group_meta[grp_idx]["work_offset_hz"] != work_offset_hz:
                        print(colored(
                            f"Error: different work_offset in group '{label}'", 
                            "red", 
                            attrs=["bold"])
                        )
                        #return

                dic, data, uc, ppm_axis, n_exp, bf1 = load_spectra(file)
                analysis_results[folder_name_short]["uc"] = uc
                analysis_results[folder_name_short]["bf1"] = bf1
                if group_meta[grp_idx].get("uc") is None:
                    group_meta[grp_idx]["uc"] = uc
                    group_meta[grp_idx]["bf1"] = bf1

                spectra = process_spectra(data, dic, n_exp)
                fig = plot_spectra(
                    title=f"{label} - {folder_name_short}",
                    spectra=spectra, n_exp=n_exp, ppm_axis=ppm_axis,
                    sat_trans_hz=sat_trans_hz,
                    visibility=config.get("plot_visibility", get_default_visibility()),
                    window_title=f"{label}: Spectra  for {folder_name_short}"
                )

                if ppm_missing and grp_idx == 0 and file == folders[0]:
                    plt.pause(0.05)
                    start_ppm, end_ppm = ask_user_for_ppm_range()
                    config["start_ppm"] = start_ppm
                    config["end_ppm"] = end_ppm
                    config["ppm_missing"] = False
                    ppm_missing = False
                    if config_name:
                        save_config(config_name, config)

                start_idx = ppm_to_index(uc, end_ppm)
                end_idx = ppm_to_index(uc, start_ppm)

                max_vals: List[float] = []
                max_indexes: List[int] = []
                global_max: float
                global_min: float
                max_vals, max_indexes, global_max, global_min = find_max_vals(spectra, start_idx, end_idx)
                max_vals = normalize_max_vals(max_vals=max_vals, global_max=global_max, global_min=global_min, )

                # Correct saturation frequencies
                zero_corrected_ppm: List[float] = correct_sat_frequencies(
                    sat_trans_hz, 
                    max_indexes,
                    work_offset_hz, 
                    uc, 
                    bf1
                )

                # --- Sort by ppm ---
                combined = list(zip(sat_trans_hz, max_indexes, max_vals, zero_corrected_ppm))
                combined.sort()
                sat_trans_hz[:], max_indexes[:], max_vals[:], zero_corrected_ppm[:] = zip(*combined)

                # --- Calculate integrals for this individual folder ---
                res = process_zspectrum_and_integrals(
                    max_vals, zero_corrected_ppm,
                    use_extra_lorentzians=use_extra_lor
                )
                analysis_results[folder_name_short].update(res)
            
                # After storing the results for the single folder, optionally plot it
                if res["spline_fit_results"].get("fit_successful", False):
                    analysis_results[folder_name_short].update({
                        "max_indexes": max_indexes,
                        "max_vals": max_vals
                    })
                    group_raw[grp_idx].append((max_indexes, max_vals, sat_trans_hz))

                    # --- Calculate integrals for this individual folder ---
                    res = process_zspectrum_and_integrals(
                        max_vals, 
                        zero_corrected_ppm,
                        use_extra_lorentzians=use_extra_lor
                    )
                    analysis_results[folder_name_short].update(res)
                    
                    # After storing the results for the single folder, optionally plot it
                    plot_data(
                        x=res["spline_fit_results"]["x"],
                        y=res["spline_fit_results"]["y"],
                        x_fit=res["spline_fit_results"]["x_fit"],
                        y_fit=res["spline_fit_results"]["y_fit"],
                        title=f" {label}: {folder_name_short}",
                        invert_x=True,
                        add_lorentz=True,
                        lorentzian_envelope_results=res["lorentzian_envelope_results"],
                        add_sigmoid=True,
                        sigmoidal_envelope_results=res["sigmoidal_envelope_results"],
                        diff_x=res["diff_x"],
                        diff_y=res["diff_y"],
                        diff_label="Lorentzian envelope - Spline fit",
                        visibility=config.get("plot_visibility", get_default_visibility()),
                        window_title=f" {label}: spline fit for {folder_name_short}"
                    )

                # Nuovo plot di decomposizione lorentziana
                if use_extra_lor and res.get("lorentzian_fit") is not None:
                    plot_lorentzian_decomposition(
                        x_data=res["x_data"],
                        y_data=res["y_data"],
                        x_common=res["x_common"],
                        L_main_y=1+lorentzian_peak(res["x_common"], -res["lorentzian_fit"]["center"]["h"], 0, res["lorentzian_fit"]["center"]["gamma"]),
                        extra_lor_results=res["extra_lorentzians_results"],
                        title=f"Lorentzian decomposition - {folder_name_short}",
                        invert_x=True,
                        window_title=f"Lorentzian decomposition - {folder_name_short}"
                    )
        else:
            files = entries
            for file_idx, file in enumerate(files):
                base_name = file.stem
                # Ensure unique key in analysis_results
                key = base_name
                counter = 1
                while key in analysis_results:
                    key = f"{base_name}_{counter}"
                    counter += 1
                analysis_results[key] = {}
                folder_keys_per_group[grp_idx].append(key)

                # --- Read x/y data from file ---
                try:
                    # Assume two columns: sat_trans_hz vs max_vals.
                    # Skip comments (lines starting with '#') and handle possible header.
                    data = np.loadtxt(file, comments='#')
                    if data.ndim != 2 or data.shape[1] < 2:
                        raise ValueError("File must contain at least two columns.")
                    sat_trans_hz = data[:, 0].tolist()
                    max_vals     = data[:, 1].tolist()
                    max_vals = normalize_max_vals(max_vals=max_vals, global_max=max(max_vals), global_min=min(max_vals))
                    
                    try:
                        value = config["groups"][grp_idx]["BF1"][file_idx]
                        group_meta[grp_idx]["bf1"] = value
                    except KeyError as e:
                        print(colored(f"Missing key {e} for {file}", "red", attrs=["bold"]))
                        continue
                    except (TypeError, IndexError) as e:
                        print(colored(f"Invalid structure (expected dict/list) for {file}: {e}", "red", attrs=["bold"]))
                        continue
                                        
                    zero_corrected_ppm = [sat_trans_hz[i] / group_meta[grp_idx]["bf1"] for i in range(len(sat_trans_hz))]
                except Exception as e:
                    print(colored(f"Error reading file {file}: {e}", "red", attrs=["bold"]))
                    continue

                max_indexes = [0] * len(max_vals)

                # --- Sort by ppm ---
                combined = list(zip(zero_corrected_ppm, max_vals, sat_trans_hz))
                combined.sort()
                zero_corrected_ppm[:], max_vals[:], sat_trans_hz[:] = zip(*combined)

                # --- Store raw data for later group averaging ---
                group_raw[grp_idx].append( (max_indexes, max_vals, sat_trans_hz) )

                # No max_indexes or sat_trans_hz needed; store what we have
                analysis_results[key].update({
                    "max_vals": max_vals,
                    "zero_corrected_ppm": zero_corrected_ppm
                })

                # --- Fit, integrate, and plot (common pipeline) ---
                res = process_zspectrum_and_integrals(
                    max_vals, 
                    zero_corrected_ppm,
                    use_extra_lorentzians=use_extra_lor
                )
                analysis_results[key].update(res)

                plot_data(
                    x=res["spline_fit_results"]["x"],
                    y=res["spline_fit_results"]["y"],
                    x_fit=res["spline_fit_results"]["x_fit"],
                    y_fit=res["spline_fit_results"]["y_fit"],
                    title=f"{label}: {key}",
                    invert_x=True,
                    add_lorentz=True,
                    lorentzian_envelope_results=res["lorentzian_envelope_results"],
                    add_sigmoid=True,
                    sigmoidal_envelope_results=res["sigmoidal_envelope_results"],
                    diff_x=res["diff_x"],
                    diff_y=res["diff_y"],
                    diff_label="Lorentzian envelope - Spline fit",
                    visibility=config.get("plot_visibility", get_default_visibility()),
                    window_title=f"{label}: spline fit for file {key}"
                )

        # --- Calculate the group average ---
        if group_raw[grp_idx]:
            idx_arr = np.array([d[0] for d in group_raw[grp_idx]])
            val_arr = np.array([d[1] for d in group_raw[grp_idx]])
            sat_arr = np.array([d[2] for d in group_raw[grp_idx]])
            n = len(group_raw[grp_idx])
            mean_max_idx = np.round(np.mean(idx_arr, axis=0)).tolist()
            mean_max_vals = np.mean(val_arr, axis=0).tolist()
            mean_sat = np.mean(sat_arr, axis=0).tolist()
            mean_zero_corrected_ppm: List[float] = [mean_sat[i] / group_meta[grp_idx]["bf1"] for i in range(len(mean_sat))]

            analysis_results[label] = {
                "max_indexes": mean_max_idx,
                "max_vals": mean_max_vals,
                "sat_trans_hz": mean_sat,
                "sd_max_indexes": np.std(idx_arr, axis=0, ddof=1).tolist() if n > 1 else [0]*len(idx_arr[0]),
                "sd_max_vals": np.std(val_arr, axis=0, ddof=1).tolist() if n > 1 else [0]*len(val_arr[0]),
                "sd_sat_trans_hz": np.std(sat_arr, axis=0, ddof=1).tolist() if n > 1 else [0]*len(sat_arr[0]),
                "bf1": group_meta[grp_idx]["bf1"],
            }

            # Fit and integrals for group average
            res_avg = process_zspectrum_and_integrals(
                mean_max_vals,
                mean_zero_corrected_ppm,
                use_extra_lorentzians=use_extra_lor
            )
            analysis_results[label].update(res_avg)

            # Plot group average
            if res_avg["spline_fit_results"].get("fit_successful", False):
                plot_data(
                    x=res_avg["spline_fit_results"]["x"],
                    y=res_avg["spline_fit_results"]["y"],
                    x_fit=res_avg["spline_fit_results"]["x_fit"],
                    y_fit=res_avg["spline_fit_results"]["y_fit"],
                    y_std_data=analysis_results[label].get("sd_max_vals"),
                    title=label, invert_x=True,
                    add_lorentz=True,
                    lorentzian_envelope_results=res_avg["lorentzian_envelope_results"],
                    add_sigmoid=True,
                    sigmoidal_envelope_results=res_avg["sigmoidal_envelope_results"],
                    diff_x=res_avg["diff_x"], diff_y=res_avg["diff_y"],
                    diff_label="Lorentzian envelope - Spline fit",
                    visibility=config.get("plot_visibility", get_default_visibility()),
                    window_title=f"Group {label} spline fit (average)"
                )
                
            # Nuovo plot di decomposizione lorentziana per la media del gruppo
            if use_extra_lor and res_avg.get("lorentzian_fit") is not None:
                interp_center = interp1d(res_avg["lorentzian_fit"]["x"], res_avg["lorentzian_fit"]["y_center"],
                                        kind='linear', fill_value="extrapolate")
                L_main_y_common = interp_center(res_avg["x_common"])
                plot_lorentzian_decomposition(
                    x_data=res_avg["x_data"],
                    y_data=res_avg["y_data"],
                    x_common=res_avg["x_common"],
                    L_main_y=1+lorentzian_peak(res_avg["x_common"], -res_avg["lorentzian_fit"]["center"]["h"], 0, res_avg["lorentzian_fit"]["center"]["gamma"]),
                    extra_lor_results=res_avg["extra_lorentzians_results"],
                    title=f"Lorentzian decomposition - {label} (average)",
                    invert_x=True,
                    window_title=f"Lorentzian decomposition - {label} (average)"
                )

    # ---- Statistics for the groups (mean ± std of per‑folder integrals) ----
    group_stats = {}
    per_folder_integrals = {}

    for grp_idx, grp in enumerate(groups):
        label = grp["label"]
        keys = folder_keys_per_group[grp_idx]
        group_stats[label] = _compute_group_stats(keys, analysis_results)
    
        # ---- Per‑folder integrals dictionary (per i grafici di gruppo) ----
        group_folder_integrals = {}
        for key in keys:
            integr = analysis_results.get(key, {}).get("integrals", {})
            for region, val in integr.items():
                group_folder_integrals.setdefault(region, []).append(val)

        per_folder_integrals[label] = group_folder_integrals    

    analysis_results["group_stats"] = group_stats

    # ---- p‑values (reference vs each sample) ----
    ref_label = None
    ref_keys = []
    for grp_idx, grp in enumerate(groups):
        if grp.get("is_reference", False):
            ref_label = grp["label"]
            ref_keys = folder_keys_per_group[grp_idx]
            break
    p_values = {}
    if ref_label and ref_keys:
        for grp_idx, grp in enumerate(groups):
            if grp["label"] == ref_label:
                continue
            other_keys = folder_keys_per_group[grp_idx]
            p_vals = _compute_pvalues(ref_keys, other_keys, analysis_results, test='t-test')
            p_values[grp["label"]] = p_vals
    analysis_results["p_values"] = p_values

    # Salvare in analysis_results la lista delle chiavi delle cartelle di ogni gruppo
    # da usare nel ramo cache
    analysis_results["folder_keys_per_group"] = folder_keys_per_group

    # ---- Save cache ----
    save_cache(config_name, config, analysis_results)
    
    # ---- Show multigroup bar plot ----
    plot_multigroup_integrals(
        group_stats, 
        p_values, 
        groups,
        visibility=config.get("plot_visibility", get_default_visibility()),
        title="Spline - integrali per regione (ricalcolati)",
        window_title="Spline - integrali per regione (ricalcolati)",
        integrals_key="integrals"
    )
    if use_extra_lor:
        plot_multigroup_integrals(
            group_stats, 
            p_values, 
            groups,
            visibility=config.get("plot_visibility", get_default_visibility()),
            title="Lorentzian - integrali per regione (ricalcolati)",
            window_title="Lorentzian - integrali per regione (ricalcolati)",
            integrals_key="integrals_extra"
        )

    # ---- Plot per gruppo con cartelle singole ----
    for grp_idx, grp in enumerate(groups):
        label = grp["label"]
        plot_group_folder_integrals(
            group_label=label,
            group_stats=group_stats,
            per_folder_integrals=per_folder_integrals,
            folder_names=folder_keys_per_group[grp_idx],   # <-- lista dei nomi brevi
            visibility=config.get("plot_visibility", get_default_visibility()),
            title=f"Spline - integrali per regione - {label} (ricalcolati)",
            window_title=f"Spline - integrali per regione - {label} (ricalcolati)",
            integrals_key="integrals"
        )
        if use_extra_lor:
            plot_group_folder_integrals(
                group_label=label,
                group_stats=group_stats,
                per_folder_integrals=per_folder_integrals,
                folder_names=folder_keys_per_group[grp_idx],   # <-- lista dei nomi brevi
                visibility=config.get("plot_visibility", get_default_visibility()),
                title=f"Lorentzian - integrali per regione - {label} (ricalcolati)",
                window_title=f"Lorentzian - integrali per regione - {label} (ricalcolati)",
                integrals_key="integrals_extra"
            )

    # Plot a seconda della modalità per la media del gruppo
    if use_extra_lor and res_avg.get("lorentzian_fit") is not None:
        interp_center = interp1d(res_avg["lorentzian_fit"]["x"], res_avg["lorentzian_fit"]["y_center"],
                                kind='linear', fill_value="extrapolate")
        L_main_y_common = interp_center(res_avg["x_common"])
        plot_lorentzian_decomposition(
            x_data=res_avg["x_data"],
            y_data=res_avg["y_data"],
            x_common=res_avg["x_common"],
            L_main_y=1+lorentzian_peak(res_avg["x_common"], -res_avg["lorentzian_fit"]["center"]["h"], 0, res_avg["lorentzian_fit"]["center"]["gamma"]),
            extra_lor_results=res_avg["extra_lorentzians_results"],
            title=f"Lorentzian decomposition - {label} (average)",
            invert_x=True,
            window_title=f"Lorentzian decomposition - {label} (average)"
        )
    else:
        plot_data(
            x=res_avg["spline_fit_results"]["x"],
            y=res_avg["spline_fit_results"]["y"],
            x_fit=res_avg["spline_fit_results"]["x_fit"],
            y_fit=res_avg["spline_fit_results"]["y_fit"],
            y_std_data=analysis_results[label].get("sd_max_vals"),
            title=label, invert_x=True,
            add_lorentz=True,
            lorentzian_envelope_results=res_avg["lorentzian_envelope_results"],
            add_sigmoid=True,
            sigmoidal_envelope_results=res_avg["sigmoidal_envelope_results"],
            diff_x=res_avg["diff_x"], diff_y=res_avg["diff_y"],
            diff_label="Lorentzian envelope - Spline fit",
            visibility=config.get("plot_visibility", get_default_visibility())
        )

    # --- Saving ---
    save_analysis_results(analysis_results=analysis_results, config_name=config_name)
    
    print("\nTutti i grafici sono stati creati. Premi Invio per uscire.")
    input()
    plt.close('all')

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> None:
    config_name, config_data = select_or_create_config()
    complete_config = ensure_complete_config(config_name, config_data)
    run_analysis(config_name, complete_config)

if __name__ == "__main__":
    main()