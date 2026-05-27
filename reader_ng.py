"""
reader_ng.py

Processing of Bruker NMR data for saturation transfer experiments.
Reads FIDs, applies corrections, displays spectra with interactive checkboxes,
allows the user to select a ppm range, and calculates maxima to generate
a saturation transfer curve.
"""

import nmrglue as ng
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import re
import tkinter as tk
from tkinter import filedialog
from typing import List, Optional, Tuple, Dict, Any
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy import stats
import json
from termcolor import colored
import os
import hashlib
from joblib import dump, load

print(f"Using nmrglue version: {ng.__version__}")

METABOLITE_REGIONS: dict[str, List[float]] = {
    "Glycolytic PMEs": [5.5, 9.0],
    #"G6P/G3P": [6.7, 7.0],
    #"3PG/F6P": [5.8, 6.7],
    "Pi": [4.3, 5.3],
    #"PDE": [3.5, 4.0],
    "PEP 1,3 BPG": [1.5, 3.5],
    "GAMMA-ATP": [-4, -1.3],
    "ALPHA-ATP": [-9, -6],
}
CACHE_DIR = Path(__file__).parent / "cache"          # cartella dedicata
CACHE_DIR.mkdir(exist_ok=True)
N_POINTS_FIT = 200

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
    folders = config.get("folders", [])
    # Dati delle cartelle: percorso e timestamp
    folder_info = []
    for f in folders:
        try:
            mtime = os.path.getmtime(f)
        except OSError:
            mtime = 0
        folder_info.append((str(f), mtime))
    # Parametri determinanti
    key_data = {
        "with_ref": config.get("with_ref"),
        "multiple_amount_ref": config.get("multiple_amount_ref"),
        "multiple_amount": config.get("multiple_amount"),
        "start_ppm": config.get("start_ppm"),
        "end_ppm": config.get("end_ppm"),
        "folders": folder_info,
        "metabolite_regions": METABOLITE_REGIONS,
    }
    # Serializza in modo stabile (ordinato)
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()

def load_cache(config_name: str, config: Dict[str, Any]) -> Optional[dict]:
    cache_path = _cache_path(config_name, config)
    if not cache_path.exists():
        return None
    try:
        payload = load(cache_path)
        current_key = _build_cache_key(config)
        if payload.get("key") == current_key:
            return payload["analysis_results"]
        else:
            print(f"Cache obsoleta per '{config_name}'.")
            return None
    except Exception as e:
        print(colored(
            f"Errore cache per '{config_name}': {e}", "red", attrs=["bold"])
        )
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

def ensure_config_dir() -> None:
    """Crea la cartella delle configurazioni se non esiste."""
    CONFIG_DIR.mkdir(exist_ok=True)

def get_default_visibility() -> Dict[str, bool]:
    return {
        "data": True,
        "spline": True,
        "lorentzian": True,
        "sigmoid": True,
        "difference": True,
        "regions": True,
        "corrected": True,
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
        # Converti i percorsi delle cartelle da stringhe a Path
        if "folders" in config:
            config["folders"] = [Path(p) for p in config["folders"]]
        return config
    except Exception as e:
        print(colored(
            f"Errore nel caricamento della configurazione '{name}': {e}", "red", attrs=["bold"])
        )
        return {}

def save_config(name: str, config: Dict[str, Any]) -> None:
    """Salva una configurazione per nome (senza estensione)."""
    ensure_config_dir()
    config_path = CONFIG_DIR / f"{name}.json"
    # Crea una copia convertendo i Path in stringhe
    to_save = config.copy()
    if "folders" in to_save:
        to_save["folders"] = [str(p) for p in to_save["folders"]]
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
    visibility: Optional[Dict[str, bool]] = None
) -> Figure:
    # Se non fornito, usa i default globali
    if visibility is None:
        visibility = get_default_visibility()
    
    fig = plt.figure(figsize=(8, 5))

    # Data points
    if visibility.get("data", True):
        if title in ("reference", "avg") and y_std_data is not None:
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

def plot_integrals_regions(
    data: Dict[str, Any],
    title: str = 'Intensità per regione',
    xlabel: str = 'Regione',
    ylabel: str = 'Intensità',
    figsize: tuple = (12, 6),
    show_values: bool = True,
    reference: Optional[bool] = False,
    multiple_amount_ref: Optional[int] = 0
) -> Figure:
    """
    Crea un grafico a barre (singolo o raggruppato) degli integrali di regione.
    """
    
    def check_integrals_keys_consistency(data):
        """
        data: multilevel dict
        returns: (is_consistent, integral, mismatches)
            integral: {key: value['integrals'] for every top-level key that has an 'integrals' sub-dict}
            mismatches: {key: {'missing': set, 'extra': set}} for keys whose integral keys differ from the first one
        """
        integral = {}
        integral_sets = {}

        for key, value in data.items():
            if isinstance(value, dict) and 'integrals' in value:
                integral[key] = value['integrals']          # store the whole inner dict
                integral_sets[key] = set(value['integrals'].keys())

        if not integral_sets:
            print("No 'integrals' entries found.")
            return True, integral, {}

        # Use the first entry as reference
        ref_key = next(iter(integral_sets))
        ref_set = integral_sets[ref_key]

        mismatches = {}
        for key, ks in integral_sets.items():
            if ks != ref_set:
                missing = ref_set - ks
                extra = ks - ref_set
                mismatches[key] = {'missing': missing, 'extra': extra}

        is_consistent = len(mismatches) == 0
        return is_consistent, integral, mismatches
    
    # Verifica che tutti i integrali abbiano le stesse chiavi

    is_consistent, integrals, mismatches = check_integrals_keys_consistency(data)
    if not is_consistent:
        print(f"Tutti i dizionari devono avere le stesse regioni.")
        print(f"{mismatches}")
        return None
    
    labels: List[str] = list(integrals.keys())
    n_series: int = len(labels)
    
    # Estrai le etichette comuni (regioni) dal primo dizionario
    region_labels: List[str] = list(integrals[labels[0]].keys())
    n_regions: int = len(region_labels)
    
    # Prepara i valori: matrice (n_series x n_regions)
    values_matrix: List = []
    for k, v in integrals.items():
        values_matrix.append([integrals[k][reg] for reg in region_labels])
    values_matrix = np.array(values_matrix)  # shape: (n_series, n_regions)

    # Crea il grafico
    fig_absolute: Figure
    ax_absolute: Axes
    fig_absolute, ax_absolute = plt.subplots(figsize=figsize)

    # Larghezza di ogni barra e posizioni
    bar_width: float = 0.8 / n_series if n_series > 1 else 0.6
    x = np.arange(n_regions)  # posizioni delle regioni sull'asse x

    # Colori
    colors = plt.cm.tab10(np.linspace(0, 1, n_series))
            
    # Etichette per la legenda (solo se più serie)
    if labels is None and n_series > 1:
        labels = [f'Gruppo {i+1}' for i in range(n_series)]
    elif labels is None:
        labels = ['']  # non serve legenda

    # Disegna le barre
    bars_list: List = []
    for i in range(n_series):
        offset: float = (i - n_series/2 + 0.5) * bar_width
        bars: plt.BarContainer = ax_absolute.bar(x + offset, values_matrix[i], width=bar_width,
                       label=labels[i] if n_series > 1 else None,
                       color=colors[i], edgecolor='black', linewidth=0.5)
        bars_list.append(bars)

    # Aggiungi valori sopra le barre se richiesto
    if show_values:
        for i, bars in enumerate(bars_list):
            for bar, val in zip(bars, values_matrix[i]):
                height: float = bar.get_height()
                y_text = height + 0.02 * values_matrix.max()
                # Posiziona il testo sopra la barra
                ax_absolute.text(
                    bar.get_x() + bar.get_width()/2, 
                    y_text,
                    f'{val:.5e}', 
                    ha='center', 
                    va='bottom', 
                    fontsize=8,
                    rotation=90
                )

    # Imposta etichette e titolo
    ax_absolute.set_title(title)
    ax_absolute.set_xlabel(xlabel)
    ax_absolute.set_ylabel(ylabel)
    ax_absolute.set_xticks(x, region_labels, rotation=45, ha='right')
    
    # Aggiungi legenda se più serie
    ax_absolute.legend()
    fig_absolute.tight_layout()

    if reference:

        title = "Intensità relative al riferimento"

        integral_list = list(integrals.values())
        integrals_referenced: List[Dict[str, float]] = []
        for integral in integral_list[multiple_amount_ref:-1]:
            d: Dict[str, float] = {}
            for key, integral in integral.items():
                d[key] = 100 * (integral / integrals['reference'][key] - 1)
            integrals_referenced.append(d)
        
        n_series -= (multiple_amount_ref + 1)

        # Prepara i valori: matrice (n_series x n_regions)
        values_matrix: List = []
        for d in integrals_referenced:
            values_matrix.append([d[reg] for reg in region_labels])
        values_matrix = np.array(values_matrix)  # shape: (n_series, n_regions)

        vmin = values_matrix.min()
        vmax = values_matrix.max()
        value_offset = 0.02 * (vmax - vmin)  # offset relativo all'altezza totale del grafico

        # Crea il grafico
        fig_referenced: Figure
        ax_referenced: Axes
        fig_referenced, ax_referenced = plt.subplots(figsize=figsize)

        # Larghezza di ogni barra e posizioni
        bar_width: float = 0.8 / n_series if n_series > 1 else 0.6
        x = np.arange(n_regions)  # posizioni delle regioni sull'asse x
        
        labels = labels[multiple_amount_ref:-1]
        # Etichette per la legenda (solo se più serie)
        if labels is None and n_series > 1:
            labels = [f'Gruppo {i+1}' for i in range(n_series)]
        elif labels is None:
            labels = ['']  # non serve legenda

        # Disegna le barre
        colors = colors[multiple_amount_ref:-1]
        bars_list: List = []
        for i in range(n_series):
            offset: float = (i - n_series/2 + 0.5) * bar_width
            bars: plt.BarContainer = ax_referenced.bar(x + offset, values_matrix[i], width=bar_width,
                        label=labels[i] if n_series > 1 else None,
                        color=colors[i], edgecolor='black', linewidth=0.5)
            bars_list.append(bars)


        # Aggiungi valori sopra le barre se richiesto
        if show_values:
            for i, bars in enumerate(bars_list):
                for bar, val in zip(bars, values_matrix[i]):
                    y_text: float = bar.get_height()
                    va: str
                    if val >= 0:
                        y_text += value_offset
                        va = 'bottom'   # allineamento verticale: testo sopra la barra
                    else:
                        y_text -= value_offset  # sotto la barra (valore negativo)
                        va = 'top'       # allineamento verticale: testo sotto (per non invadere)

                    # Posiziona il testo sopra la barra
                    ax_referenced.text(
                        bar.get_x() + bar.get_width()/2, 
                        y_text,
                        f'{val:.2f}', 
                        ha='center', 
                        va=va, 
                        fontsize=8, 
                        rotation=90
                    )

        # Imposta etichette e titolo
        ax_referenced.set_title(title)
        ax_referenced.set_xlabel(xlabel)
        ax_referenced.set_ylabel(ylabel)
        ax_referenced.set_xticks(x, region_labels, rotation=45, ha='right')
        
        # Aggiungi legenda se più serie
        ax_referenced.legend()
        fig_referenced.tight_layout()
        plt.show(block=False)
        return fig_absolute, fig_referenced
    
    plt.show(block=False)
    return fig_absolute

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
    """
    Legge i dati Bruker dalla cartella selezionata.

    Parameters
    ----------
    folder : Path
        Percorso della cartella dell'esperimento.

    Returns
    -------
    dic : dict
        Dizionario dei parametri Bruker.
    data : np.ndarray
        Array dei FID (2D: esperimenti x punti).
    uc : unit_conversion
        Oggetto per la conversione tra ppm, Hz e indici.
    ppm_axis : np.ndarray
        Array dei valori ppm corrispondenti agli indici.
    n_exp : int
        Numero di esperimenti (FID) acquisiti.
    bf1 : float
        Frequenza di base in MHz.
    """
    dic, data = ng.bruker.read(folder)
    udic = ng.bruker.guess_udic(dic, data)
    uc = ng.fileio.bruker.fileiobase.uc_from_udic(udic, dim=1)
    ppm_axis = uc.ppm_scale()
    n_exp = dic["acqu2s"]["TD"]
    bf1 = dic["acqus"]["BF1"]
    return dic, data, uc, ppm_axis, n_exp, bf1

def process_spectra(data: np.ndarray, dic: dict, n_exp: int):
    """
    Elabora ogni FID: rimozione filtro digitale, zero-filling, line broadening,
    FFT, correzione di fase automatica e inversione dell'asse.

    Parameters
    ----------
    data : np.ndarray
        Array 2D dei FID (righe = esperimenti).
    dic : dict
        Dizionario dei parametri Bruker.
    n_exp : int
        Numero di esperimenti.

    Returns
    -------
    spectra : Dict[int, np.ndarray]
        Dizionario con chiave indice esperimento e valore spettro complesso elaborato.
    """
    spectra = {}
    for exp_idx in range(n_exp):
        # ---- Select experiment folder        
        fid = data[exp_idx, :]
        # ---- Remove Bruker digital filter
        fid = ng.bruker.remove_digital_filter(dic, data=fid)
        # Zero-fill to 2048 points
        fid_zf = ng.proc_base.zf_size(fid, size=2048)
        # Line broadening (exponential)
        fid_apod = ng.proc_base.em(fid_zf, lb=0.005)
        # Fourier transform
        spectrum = ng.proc_base.fft(fid_apod)
        # Automatic phase correction (ACME method)
        spectrum_phased = ng.proc_autophase.autops(spectrum, fn="acme")
        # Invert the axis (left to right: decreasing ppm)
        spectrum_phased = spectrum_phased[::-1]
        spectra[exp_idx] = spectrum_phased
    return spectra

def plot_spectra(title, spectra, n_exp, ppm_axis, sat_trans_hz) -> Figure:
    """
    Crea un plot interattivo di tutti gli spettri con checkbox per mostrare/nascondere
    ogni traccia.

    Parameters
    ----------
    title : str
        Titolo del plot.
    spectra : Dict[int, np.ndarray]
        Dizionario degli spettri elaborati.
    n_exp : int
        Numero di esperimenti.
    ppm_axis : np.ndarray
        Asse dei ppm.
    sat_trans_hz : List[float]
        Frequenze di saturazione in Hz (usate come etichette).
    """

    #TODO: i check boxes non funzionano in multi-exp
    fig, ax = plt.subplots(figsize=(12, 6))
    lines = []
    labels = []
    for exp_idx in range(n_exp):
        line, = ax.plot(ppm_axis, np.real(spectra[exp_idx]),
                        label = f"{exp_idx:>2} : {sat_trans_hz[exp_idx]:.2f}",
                        alpha=0.7, linewidth=1.2)
        lines.append(line)
        labels.append(line.get_label())

    # Main plot settings
    ax.invert_xaxis()
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    # ---- CheckButtons panel ----
    rax = fig.add_axes([0.80, 0.15, 0.19, 0.70])
    visibility = [l.get_visible() for l in lines]
    checks = CheckButtons(rax, labels, visibility)

    # ----------------------------------------------------------------------
    # Callbacks for interactive widgets
    # to capture lines, labels, fig, checks
    # ----------------------------------------------------------------------
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

    # "Check all" button
    ax_all = fig.add_axes([0.80, 0.90, 0.09, 0.05])
    btn_all = Button(ax_all, "Check all")
    btn_all.on_clicked(_check_all)
    
    # "Uncheck all" button
    ax_none = fig.add_axes([0.90, 0.90, 0.09, 0.05])
    btn_none = Button(ax_none, "Uncheck all")
    btn_none.on_clicked(_uncheck_all)

    fig.tight_layout(rect=[0, 0, 0.80, 1])
    plt.show(block=False)
    return fig

def find_max_vals(spectra, start_idx, end_idx):
    """
    Trova il valore massimo in ogni spettro all'interno dell'intervallo di indici
    [start_idx, end_idx) e normalizza tutti i massimi rispetto al massimo globale.

    Parameters
    ----------
    spectra : Dict[int, np.ndarray]
        Dizionario degli spettri elaborati.
    start_idx : int
        Indice di inizio (inclusivo).
    end_idx : int
        Indice di fine (esclusivo).

    Returns
    -------
    max_vals : List[float]
        Lista con i valore massimo normalizzato.
    max_indexes : List[int]
        Lista con indice del massimo (originale).
    """
    # ---- Find maxima in the selected range ----
    max_vals: List[float] = []
    max_indexes: List[int] = []
    global_max: float = float('-inf')
    global_min: float = float('inf')
    for exp_idx, spec in spectra.items():
        val, idx = find_maximum(spec, start=start_idx, end=end_idx)
        if val > global_max:
            global_max = val
        if val < global_min:
            global_min = val
        max_vals.append(val)
        max_indexes.append(idx)
    # ---- Normalize max_vals ----
    for i in range(len(max_vals)):
        max_vals[i] = (max_vals[i] - global_min) / (global_max - global_min) if global_max > global_min else 0.0
    return max_vals, max_indexes

def ask_user_for_ppm_range(default_start=None, default_end=None) -> Tuple[float, float]:
    """
    Richiede all'utente di inserire l'intervallo in ppm (min e max).
    Se forniti default_start e default_end, li mostra come suggeriti.
    Ripete la richiesta finché non vengono forniti valori validi.
    Restituisce start_ppm, end_ppm.
    """
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

def correct_sat_frequencies(
    sat_trans_hz: List[float],
    max_indexes: List[int],
    work_offset_hz: List[float],
    uc: Any,
    bf1: float
) -> List[float]:
    """
    Corregge le frequenze di saturazione e le converte in ppm.

    Modifica *in-place* la lista sat_trans_hz:
    per ogni frequenza non nulla aggiunge il delta calcolato
    dallo scostamento del picco rispetto al work offset.

    Restituisce la lista delle frequenze corrette in ppm.
    """
    sat_trans_f1_ppm = [0.0] * len(sat_trans_hz)

    for i, (st_hz, idx) in enumerate(zip(sat_trans_hz, max_indexes)):
        delta = work_offset_hz[0] - uc.hz(idx)
        if st_hz != 0.0:
            sat_trans_hz[i] += delta
        sat_trans_f1_ppm[i] = sat_trans_hz[i] / bf1
    return sat_trans_f1_ppm

def ask_yes_no(prompt: str, default: Optional[bool] = None) -> bool:
    """
    Ask the user a yes/no question and return True for yes, False for no.
    default: default value if user presses Enter.
    """
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
    """
    Chiede all'utente di inserire un intero e lo restituisce.

    Parameters
    ----------
    prompt : str
        Il messaggio da mostrare all'utente.
    min_val : int, optional
        Se specificato, impone un valore minimo (incluso).
    max_val : int, optional
        Se specificato, impone un valore massimo (incluso).

    Returns
    -------
    int
        L'intero inserito dall'utente.
    """

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

# ----------------------------------------------------------------------
# Helper functions for statistics accumulation (refactored)
# ----------------------------------------------------------------------

def _init_stats_lists(length: int) -> Dict[str, List[float]]:
    """Create a dictionary of zero-filled lists for statistical accumulation.
    
    Returns lists for:
        - max_indexes, max_vals, sat_trans_hz (averages)
        - sd_max_indexes, sd_max_vals, sd_sat_trans_hz (for standard deviations)
    """
    return {
        "max_indexes": [0.0] * length,
        "max_vals": [0.0] * length,
        "sat_trans_hz": [0.0] * length,
        "sd_max_indexes": [0.0] * length,
        "sd_max_vals": [0.0] * length,
        "sd_sat_trans_hz": [0.0] * length,
    }

def _accumulate_averages(acc: Dict[str, List[float]], 
                         values: Tuple[List[int], List[float], List[float]], 
                         divisor: int) -> None:
    """Add weighted contributions to the average accumulators."""
    max_indexes, max_vals, sat_trans_hz = values
    for k, (i, v, st) in enumerate(zip(max_indexes, max_vals, sat_trans_hz)):
        acc["max_indexes"][k] += i / divisor
        acc["max_vals"][k] += v / divisor
        acc["sat_trans_hz"][k] += st / divisor

def _accumulate_squared_diffs(
    sd_acc: Dict[str, List[float]],
    values: Tuple[List[int], List[float], List[float]],
    avg: Dict[str, List[float]]
) -> None:
    """Sum squared differences from the mean for standard deviation."""
    max_indexes, max_vals, sat_trans_hz = values
    for j, (index, val, freq) in enumerate(zip(max_indexes, max_vals, sat_trans_hz)):
        sd_acc["sd_max_indexes"][j] += (index - avg["max_indexes"][j]) ** 2
        sd_acc["sd_max_vals"][j] += (val - avg["max_vals"][j]) ** 2
        sd_acc["sd_sat_trans_hz"][j] += (freq - avg["sat_trans_hz"][j]) ** 2

def _finalize_std_dev(sd_acc: Dict[str, List[float]], count: int) -> None:
    """Compute square root of variance (sample std. dev.)."""
    for key in ("sd_max_indexes", "sd_max_vals", "sd_sat_trans_hz"):
        for j in range(len(sd_acc[key])):
            sd_acc[key][j] = np.sqrt(sd_acc[key][j] / (count - 1)) if count > 1 else 0.0

# ----------------------------------------------------------------------
# Core logic: ensure complete config and run analysis
# ----------------------------------------------------------------------

def ensure_complete_config(config_name: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Rende completa la configurazione chiedendo i dati mancanti (cartelle, opzioni)."""

    modified = False  # <-- flag per tracciare modifiche

    # Se mancano le cartelle, chiedi tutto
    if not config_data.get("folders") or config_name == "":
        if config_name:
            print(f"Configurazione '{config_name}' senza cartelle definite. Procedura interattiva.")
        with_ref = ask_yes_no("Reference folder?", default=config_data.get("with_ref", False))
        multiple_amount_ref = ask_int("How many?", min_val=1, default=config_data.get("multiple_amount_ref", 1)) if with_ref else 0
        with_multiple = ask_yes_no("Multiple folders?", default=config_data.get("with_multiple", False))
        multiple_amount = ask_int("How many?", min_val=1, default=config_data.get("multiple_amount", 1)) if with_multiple else 1
        
        folders = []
        if with_ref:
            for _ in range(multiple_amount_ref):
                folders.append(select_experiment_folder(title="Select reference folder(s)"))
        for _ in range(multiple_amount):
            folders.append(select_experiment_folder(title="Select folder(s)"))
        
        config_data["with_ref"] = with_ref
        config_data["multiple_amount_ref"] = multiple_amount_ref
        config_data["with_multiple"] = with_multiple
        config_data["multiple_amount"] = multiple_amount
        config_data["folders"] = folders
        modified = True
    else:
        # Le cartelle ci sono già, usale così come sono
        pass
    
    # I ppm verranno chiesti durante l'analisi (serve uc). Segnaliamo se mancano.
    if config_data.get("start_ppm") is None or config_data.get("end_ppm") is None:
        config_data["ppm_missing"] = True
    else:
        config_data["ppm_missing"] = False
    
    # ========== merge / add plot_visibility ===============
    if "plot_visibility" in config_data:
        default_vis = get_default_visibility()
        current_vis = config_data["plot_visibility"]
        # add missing keys from default (e.g. when new plot elements are added)
        for key, val in default_vis.items():
            if key not in current_vis:
                current_vis[key] = val
                modified = True
        config_data["plot_visibility"] = current_vis
    else:
        config_data["plot_visibility"] = get_default_visibility()
        modified = True
    # ======================================================
                
    # Salva solo se ci sono state modifiche
    if modified:
        save_config(config_name, config_data)
        
    return config_data

def constrained_lorentzian(x, A, gamma, y_min):
    """Funzione di comodo per tracciare la curva ottimale."""
    if gamma == 0.0:
        return np.full_like(x, A)
    return A - (A - y_min) * gamma**2 / (gamma**2 + x**2)

def estimate_constrained_lorentzian(x_data, y_data):
    """
    Stima i parametri (A, gamma) per la Lorentziana:
        L(x) = A - (A - y_min) * gamma^2 / (gamma^2 + x^2)
    con i vincoli:
        - L(x_i) >= y_i  per ogni i
        - A >= max(y)
        - Il fondo del dip è fissato a y_min = min(y)
    e sceglie A e gamma che minimizzano la somma dei quadrati degli scarti.

    Restituisce (A, gamma).
    """
    x = np.asarray(x_data)
    y = np.asarray(y_data)

    y_min = np.min(y)
    y_max = np.max(y)

    # Se tutti i punti sono uguali, restituisci una curva piatta
    if y_max == y_min:
        return y_max, 0.0

    # Funzione errore per un dato A (gamma viene ottimizzato internamente)
    def error_for_A(A):
        if A < y_max:          # vincolo violato
            return np.inf

        # Calcola gamma_max(A) dai vincoli per ogni punto
        gamma_max = np.inf
        for xi, yi in zip(x, y):
            if yi <= y_min:    # questi punti sono sempre soddisfatti (toccano il fondo)
                continue
            # Vincolo: L(xi) = A - (A - y_min) * gamma^2/(gamma^2 + xi^2) >= yi
            # => gamma^2 <= (A - yi) / (yi - y_min) * xi^2
            # (derivazione nel testo)
            bound_sq = (A - yi) / (yi - y_min) * xi**2
            if bound_sq <= 0:
                # Il vincolo non può essere soddisfatto per questo A
                return np.inf
            gamma_max = min(gamma_max, np.sqrt(bound_sq))

        if gamma_max <= 0.0:
            return np.inf

        # Ora, fissato A, ottimizza gamma in [0, gamma_max] per minimizzare MSE
        def mse(gamma):
            if gamma == 0.0:
                y_pred = np.full_like(x, A)   # curva piatta
            else:
                y_pred = A - (A - y_min) * gamma**2 / (gamma**2 + x**2)
            return np.sum((y_pred - y)**2)

        # Ottimizzazione locale di gamma (un solo parametro)
        res = minimize_scalar(mse, bounds=(0.0, gamma_max), method='bounded')
        return res.fun   # restituisce l'errore minimo per questo A

    # Ottimizza A nell'intervallo [y_max, y_max + 3*(y_max - y_min)] 
    # (l'upper bound può essere ampio, per sicurezza)
    upper_A = y_max + 5 * (y_max - y_min) if y_max > y_min else y_max + 1.0
    res_A = minimize_scalar(error_for_A, bounds=(y_max, upper_A), method='bounded')
    best_A = res_A.x

    # Ricalcola il gamma ottimale per il miglior A
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
    """
    Sigmoide (logistica) per l'inviluppo superiore.
    Parametri:
        L : asintoto per x → +∞ (ppm piccoli, lato destro)
        R : asintoto per x → -∞ (ppm grandi, lato sinistro)
        tau : scala della pendenza (tau > 0)
        x0 : centro della transizione (default 0)
    """
    return R + (L - R) / (1.0 + np.exp(-(x - x0) / tau))

def estimate_constrained_sigmoid(x_data, y_data, fix_center=True, x0_fixed=0.0):
    """
    Stima i parametri (L, R, tau) per la sigmoide:
        S(x) = R + (L - R) / (1 + exp(-(x - x0)/tau))
    che soddisfa S(x_i) >= y_i per ogni i, minimizzando l'errore quadratico.

    Parametri
    ----------
    x_data, y_data : array-like
    fix_center : bool (True)
        Se True, centro fissato a x0_fixed.
    x0_fixed : float
        Centro se fix_center=True.

    Restituisce
    -----------
    (L, R, tau) se fix_center=True, altrimenti (L, R, tau, x0).
    """
    x = np.asarray(x_data)
    y = np.asarray(y_data)
    
    # Per semplicità assumiamo fix_center=True (adatto al tuo caso).
    x0 = x0_fixed

    # Funzione obiettivo per un dato tau: trova L,R ottimi che soddisfano i vincoli
    def solve_LR_for_tau(tau):
        # z_i = 1 / (1 + exp(-(x_i - x0)/tau))
        z = 1.0 / (1.0 + np.exp(-(x - x0) / tau))
        # S(x_i) = R + (L - R) * z_i = R*(1 - z_i) + L*z_i >= y_i
        # Vincoli lineari: L*z_i + R*(1-z_i) >= y_i  per ogni i
        # Inoltre L,R >= 0 (fisicamente i massimi non negativi)
        # Vogliamo minimizzare sum( (L*z_i + R*(1-z_i) - y_i)^2 )
        # Questo è un problema di ottimizzazione quadratica con vincoli lineari.
        # Possiamo risolverlo con scipy.optimize.minimize o con un metodo diretto.
        
        def mse(params):
            L, R = params
            y_pred = L * z + R * (1 - z)
            return np.sum((y_pred - y)**2)
        
        # Vincoli: per ogni i, L*z_i + R*(1-z_i) - y_i >= 0
        constraints = []
        for i in range(len(x)):
            # A_i * [L, R] >= b_i
            A_i = np.array([z[i], 1 - z[i]])
            b_i = y[i]
            constraints.append({'type': 'ineq', 'fun': lambda p, A=A_i, b=b_i: A[0]*p[0] + A[1]*p[1] - b})
        
        # Stima iniziale: prendi il massimo di y a sinistra e destra
        mask_left = x > x0
        mask_right = x < x0
        L0 = np.max(y[mask_right]) if np.any(mask_right) else np.max(y)
        R0 = np.max(y[mask_left]) if np.any(mask_left) else np.max(y)
        
        # Risolvi con vincoli
        res = minimize(mse, [L0, R0], method='SLSQP', constraints=constraints,
                       bounds=[(0, None), (0, None)], options={'maxiter': 1000})
        if res.success:
            L_opt, R_opt = res.x
            return L_opt, R_opt, res.fun
        else:
            # Fallback: usa il massimo assoluto per entrambi (curva piatta sopra i dati)
            L = R = np.max(y)
            return L, R, np.sum((np.full_like(y, L) - y)**2)

    # Ora ottimizziamo tau minimizzando l'errore con i migliori L,R
    def objective_tau(tau):
        if tau <= 0:
            return np.inf
        _, _, err = solve_LR_for_tau(tau)
        return err

    # Ricerca di tau ottimale in un intervallo ragionevole
    # tau può andare da un valore piccolo (transizione ripida) a grande (quasi lineare)
    tau_min = 1e-6
    tau_max = np.ptp(x) * 10  # 10 volte l'escursione in ppm
    res_tau = minimize_scalar(objective_tau, bounds=(tau_min, tau_max), method='bounded')
    
    if res_tau.success:
        tau_opt = res_tau.x
    else:
        tau_opt = np.ptp(x) / 4  # fallback
        print(colored(
            f"Warning: optimization for tau failed, using fallback tau={tau_opt:.4f}", "yellow", attrs=["bold"])
        )

    # Ricalcola L,R ottimi per il tau trovato
    L_opt, R_opt, _ = solve_LR_for_tau(tau_opt)
    
    return L_opt, R_opt, tau_opt

def _compute_integrals_stats(keys, analysis_results):
    """
    Compute mean and sample std of integrals per region for a list of folder keys.
    """
    integrals_list = []
    for key in keys:
        entry = analysis_results.get(key, {})
        integrals = entry.get("integrals", {})
        if integrals:                     # skip entries with no integrals
            integrals_list.append(integrals)

    if not integrals_list:
        return {"mean": {}, "std": {}}

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

    return {"mean": mean_dict, "std": std_dict}

def _compute_pvalues(ref_keys, sample_keys, analysis_results, test='t-test'):
    """
    Compute p-values comparing reference vs. sample integrals for each region.

    Parameters
    ----------
    ref_keys, sample_keys : List[str]
        Folder keys for the two groups.
    analysis_results : dict
    test : str
        't-test' or 'mann-whitney'.

    Returns
    -------
    dict
        {region: p_value} for regions present in both groups.
    """
    # Collect all non‑empty integral dicts from each group
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
        print("Not enough data for p‑value calculation.")
        return {}

    # Assume both groups have the same regions (already verified)
    regions = list(ref_integrals[0].keys())
    pvalues = {}

    for reg in regions:
        ref_vals = [d[reg] for d in ref_integrals if reg in d]
        samp_vals = [d[reg] for d in sample_integrals if reg in d]

        if len(ref_vals) < 2 or len(samp_vals) < 2:
            pvalues[reg] = None   # not enough data
            continue

        if test == 't-test':
            # Welch's t‑test (default)
            _, p = stats.ttest_ind(ref_vals, samp_vals, equal_var=False)
        elif test == 'mann-whitney':
            _, p = stats.mannwhitneyu(ref_vals, samp_vals, alternative='two-sided')
        else:
            raise ValueError(f"Unknown test: {test}")
        pvalues[reg] = float(p)

    return pvalues

def plot_grouped_comparison(
    ref_stats: dict,
    sample_stats: dict,
    pvalues: Optional[dict] = None,
    title: str = "Reference vs Sample – Integrals by Region",
    ylabel: str = "Integral (mean ± std)",
    figsize: tuple = (12, 6),
    significance_levels: Optional[dict] = None,
) -> Figure:
    """
    Grouped bar chart comparing reference and sample per‑region integrals.

    Parameters
    ----------
    ref_stats, sample_stats : dict
        Each must contain ``"mean"`` and ``"std"`` dicts keyed by region.
    pvalues : dict, optional
        p‑values keyed by region. If provided, significance stars are shown.
    title, ylabel, figsize : standard matplotlib parameters.
    significance_levels : dict, optional
        Mapping from p‑value threshold to annotation, e.g.
        {0.001: '***', 0.01: '**', 0.05: '*'}.
        Default: {0.001: '***', 0.01: '**', 0.05: '*'}.
    """
    if significance_levels is None:
        significance_levels = {0.001: '***', 0.01: '**', 0.05: '*'}

    # Gather regions that appear in both stats
    regions = [reg for reg in ref_stats["mean"] if reg in sample_stats["mean"]]
    if not regions:
        print("No common regions to plot.")
        return None

    n_regions = len(regions)
    x = np.arange(n_regions)
    bar_width = 0.35

    ref_means = [ref_stats["mean"][reg] for reg in regions]
    ref_stds  = [ref_stats["std"][reg] for reg in regions]
    samp_means = [sample_stats["mean"][reg] for reg in regions]
    samp_stds  = [sample_stats["std"][reg] for reg in regions]

    fig, ax = plt.subplots(figsize=figsize)

    bars_ref = ax.bar(
        x - bar_width/2, ref_means, bar_width,
        yerr=ref_stds, capsize=5,
        label="Reference", color="steelblue", edgecolor="black"
    )
    bars_samp = ax.bar(
        x + bar_width/2, samp_means, bar_width,
        yerr=samp_stds, capsize=5,
        label="Sample", color="darkorange", edgecolor="black"
    )

    # Add significance annotations
    if pvalues is not None:
        for i, reg in enumerate(regions):
            p = pvalues.get(reg)
            if p is None:
                continue
            # Determine star(s) based on thresholds
            txt = None
            for threshold in sorted(significance_levels, reverse=True):
                if p < threshold:
                    txt = significance_levels[threshold]
                    break
            if txt is None:
                txt = f"p={p:.3f}"

            # Position above the higher bar
            y_max = max(ref_means[i] + ref_stds[i], samp_means[i] + samp_stds[i])
            ax.text(x[i], y_max * 1.05, txt, ha='center', va='bottom',
                    fontweight='bold', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plt.show(block=True)
    return fig

def run_analysis(config_name: str, config: Dict[str, Any]) -> None:
    """Esegue l'analisi completa. Se i ppm mancano, li chiede usando la prima cartella."""

    plt.ion()   # <-- interactive mode ON
    folders: List[Path] = config["folders"]
    with_ref: bool = config.get("with_ref", False)
    with_multiple: bool = config.get("with_multiple", False)
    multiple_amount_ref: int = config.get("multiple_amount_ref", 0)
    multiple_amount: int = config.get("multiple_amount", 1)
    start_ppm: float = config.get("start_ppm")
    end_ppm: float = config.get("end_ppm")
    ppm_missing: bool = config.get("ppm_missing", False)
    analysis_results: dict = {}

    # ═══════════════════════════════════════════════════════════════
    #  🚀 PROVA A CARICARE DALLA CACHE
    # ═══════════════════════════════════════════════════════════════
    cached = load_cache(config_name, config)
    if cached is not None:
        use_cache = ask_yes_no(f"Cache valida trovata per '{config_name}'. Vuoi usarla?", default=True)
        if use_cache:
            analysis_results = cached
            # Ricrea i grafici degli Z-spettri
            for name, z in analysis_results.items():
                if "spline_fit_results" in z and z["spline_fit_results"]["fit_successful"]:
                    fit = z["spline_fit_results"]
                    plot_data(
                        fit["x"], 
                        fit["y"],
                        fit["x_fit"], 
                        fit["y_fit"],
                        y_std_data=z.get("sd_max_vals"),
                        title=name, 
                        invert_x=True,
                        add_lorentz=True,
                        lorentzian_envelope_results=z.get("lorentzian_envelope_results"),
                        add_sigmoid=True,
                        sigmoidal_envelope_results=z.get("sigmoidal_envelope_results"),
                        diff_x=z.get("diff_x"),
                        diff_y=z.get("diff_y"),
                        diff_label="Lorentzian envelope - Spline fit",
                        visibility=config.get("plot_visibility", get_default_visibility())
                    )
            # Grafico a barre degli integrali
            plot_integrals_regions(
                data=analysis_results,
                reference=with_ref,
                multiple_amount_ref=multiple_amount_ref if with_ref else 0
            )
            return   # Esce senza ricalcolare
        else:
            print("Cache ignorata. Ricalcolo in corso...")
    # ═══════════════════════════════════════════════════════════════

    # Initialize accumulators using helper
    ref_stats = None
    avg_stats = None
    ref_keys = []
    sample_keys = []
    ref_work_offset_hz = []
    avg_work_offset_hz = []
    
    for idx, folder in enumerate(folders):
        base_name = f"{folder.parent.name[:12]}…{folder.parent.name[-12:]}-{folder.stem}"
        folder_name_short = base_name
        counter = 1
        while folder_name_short in analysis_results:
            print(colored(
                f"Warning: folder name clash for '{base_name}'. Using '{folder_name_short}_{counter}' instead.",
                "yellow", attrs=["bold"]
            ))
            folder_name_short = f"{base_name}_{counter}"
            counter += 1

        analysis_results[folder_name_short] = {}

        if with_ref and idx < multiple_amount_ref:
            ref_keys.append(folder_name_short)
        if with_multiple and multiple_amount_ref <= idx < (multiple_amount_ref + multiple_amount):
            sample_keys.append(folder_name_short)        

        # ----------------------------------------------------------------------
        # Extract sat_trans_hz and work_offset_hz parameters from folder files
        # ----------------------------------------------------------------------
        try:
            sat_trans_hz, work_offset_hz = extract_parameters(folder)
            analysis_results[folder_name_short]["sat_trans_hz"] = sat_trans_hz
            analysis_results[folder_name_short]["work_offset_hz"] = work_offset_hz
        except (FileNotFoundError, ValueError) as e:
            print(colored(
                f"Errore in {folder}: {e}", "red", attrs=["bold"])
            )
            return
        
        if with_multiple:
            if idx < multiple_amount:
                if not avg_work_offset_hz:
                    avg_work_offset_hz = work_offset_hz
                elif avg_work_offset_hz != work_offset_hz:
                    print(colored(
                        f"Error: different work_offset_hz in sample folders.", "red", attrs=["bold"])
                    )

        if with_ref:
            if idx < multiple_amount_ref:
                if not ref_work_offset_hz:
                    ref_work_offset_hz = work_offset_hz
                elif ref_work_offset_hz != work_offset_hz:
                    print(colored(
                        f"Error: different work_offset_hz in reference folders.", "red", attrs=["bold"])
                    )
                
        # ----------------------------------------------------------------------
        # Load spectra
        # ----------------------------------------------------------------------
        dic, data, uc, ppm_axis, n_exp, bf1 = load_spectra(folder)
        analysis_results[folder_name_short]["uc"] = uc
        analysis_results[folder_name_short]["bf1"] = bf1
        if n_exp <= 0:
            print(f"Nessun esperimento in {folder}")
            return
        
        # ----------------------------------------------------------------------
        # Process and plot spectra
        # ----------------------------------------------------------------------
        spectra = process_spectra(data, dic, n_exp)
        fig: Figure = plot_spectra(title=f"Spectra - {folder_name_short}", spectra=spectra, n_exp=n_exp, ppm_axis=ppm_axis, sat_trans_hz=sat_trans_hz)
        
        # ----------------------------------------------------------------------
        # Se i ppm non sono ancora noti, chiediamo usando la prima cartella
        # ----------------------------------------------------------------------
        if ppm_missing and idx == 0:
            plt.pause(0.05) # <-- keep figures alive
            start_ppm, end_ppm = ask_user_for_ppm_range()
            # Aggiorna la configurazione
            config["start_ppm"] = start_ppm
            config["end_ppm"] = end_ppm
            config["ppm_missing"] = False
            ppm_missing = False
            if config_name:
                save_config(config_name, config)
        elif ppm_missing and idx > 0:
            print("Errore: ppm non definiti e non siamo alla prima cartella. Questo non dovrebbe accadere.")
            return
        
        start_idx = ppm_to_index(uc, end_ppm)
        end_idx = ppm_to_index(uc, start_ppm)
        if start_idx < 0 or end_idx < 0:
            print("Indici ppm non validi.")
            return
        
        # ----------------------------------------------------------------------
        # Find max values and indexes in the ppm range (the z-spectra)
        # ----------------------------------------------------------------------
        max_vals, max_indexes = find_max_vals(spectra, start_idx, end_idx)

        # ----------------------------------------------------------------------
        # Sort them all!!! 
        # ----------------------------------------------------------------------
        combined = list(zip(sat_trans_hz, max_indexes, max_vals))
        combined.sort()  # or sorted()
        sat_trans_hz[:], max_indexes[:], max_vals[:] = zip(*combined)

        analysis_results[folder_name_short].update({
            "max_indexes": max_indexes,
            "max_vals": max_vals, 
        })

        # ------------------------------------------------------------------
        # Accumulate averages for reference and multiple (sample) groups
        # ------------------------------------------------------------------
        num_sat = len(sat_trans_hz)
        if with_multiple:
            if avg_stats is None:
                avg_stats = _init_stats_lists(num_sat)
            # belong to multiple group?
            if multiple_amount_ref <= idx < (multiple_amount_ref + multiple_amount):
                if len(max_vals) == num_sat:
                    _accumulate_averages(avg_stats, (max_indexes, max_vals, sat_trans_hz), multiple_amount)
                    pass
                else:
                    print(f"{colored('Error', 'red', attrs=['bold'])}: number of saturation frequencies not matching number of experiments (max_values).")

        if with_ref:
            if ref_stats is None:
                ref_stats = _init_stats_lists(num_sat)
            
            # belong to reference group?
            if idx < multiple_amount_ref:
                if len(max_vals) == num_sat:
                    _accumulate_averages(ref_stats, (max_indexes, max_vals, sat_trans_hz), multiple_amount_ref)
                else:
                    print(f"{colored('Error', 'red', attrs=['bold'])}: number of saturation frequencies not matching number of experiments (max_values).")

    # ----------------------------------------------------------------------
    # After processing all folders, finalize averages and compute std dev
    # ----------------------------------------------------------------------
    if with_multiple and multiple_amount > 1 and avg_stats is not None:
        analysis_results["avg"] = {
            "max_indexes": [round(v) for v in avg_stats["max_indexes"]],
            "max_vals": avg_stats["max_vals"],
            "sat_trans_hz": avg_stats["sat_trans_hz"],
            "work_offset_hz": avg_work_offset_hz,
            # TODO: check if uc and bf1 are the same for the multiple expt 
            "uc": uc,
            "bf1": bf1
        }
        # Calculate standard deviations
        # We need the original data again – we stored them in analysis_results for each folder.
        # Instead of re-looping, we can loop over analysis_results items as before but now using helpers.
        # We'll do the squared differences accumulation in a second pass over the folder data.
        for idx, (k, v) in enumerate(analysis_results.items()):
            if multiple_amount_ref <= idx < (multiple_amount_ref + multiple_amount):
                _accumulate_squared_diffs(
                    avg_stats,
                    (v["max_indexes"], v["max_vals"], v["sat_trans_hz"]),
                    {"max_indexes": analysis_results["avg"]["max_indexes"],
                     "max_vals": analysis_results["avg"]["max_vals"],
                     "sat_trans_hz": analysis_results["avg"]["sat_trans_hz"]}
                )
        _finalize_std_dev(avg_stats, multiple_amount)
        analysis_results["avg"]["sd_max_indexes"] = avg_stats["sd_max_indexes"]
        analysis_results["avg"]["sd_max_vals"] = avg_stats["sd_max_vals"]
        analysis_results["avg"]["sd_sat_trans_hz"] = avg_stats["sd_sat_trans_hz"]

    if with_ref and ref_stats is not None:
        analysis_results["reference"] = {
            "max_indexes": [round(v) for v in ref_stats["max_indexes"]],
            "max_vals": ref_stats["max_vals"],
            "sat_trans_hz": ref_stats["sat_trans_hz"],
            "work_offset_hz": ref_work_offset_hz,
            # TODO: check if uc and bf1 are the same for the reference expt 
            "uc": uc,
            "bf1": bf1
        }
        for idx, (k, v) in enumerate(analysis_results.items()):
            if idx < multiple_amount_ref:
                _accumulate_squared_diffs(
                    ref_stats,
                    (v["max_indexes"], v["max_vals"], v["sat_trans_hz"]),
                    {"max_indexes": analysis_results["reference"]["max_indexes"],
                     "max_vals": analysis_results["reference"]["max_vals"],
                     "sat_trans_hz": analysis_results["reference"]["sat_trans_hz"]}
                )
        _finalize_std_dev(ref_stats, multiple_amount_ref)
        analysis_results["reference"]["sd_max_indexes"] = ref_stats["sd_max_indexes"]
        analysis_results["reference"]["sd_max_vals"] = ref_stats["sd_max_vals"]
        analysis_results["reference"]["sd_sat_trans_hz"] = ref_stats["sd_sat_trans_hz"]
    
    # ----------------------------------------------------------------------
    # Correct saturation frequencies, fit z-spectra and calculate integrals
    # ----------------------------------------------------------------------
    for name, z in analysis_results.items():
        if len(z["sat_trans_hz"]) == len(z["max_vals"]):
            zero_corrected_ppm  = correct_sat_frequencies(
                z["sat_trans_hz"], 
                z["max_indexes"], 
                z["work_offset_hz"], 
                z["uc"], 
                z["bf1"]
            )

            # ----------------------------------------------------------------------
            # correct_sat_frequencies(...) può in linea di principio cambiare 
            # l'ordine dei dati. Riordina dopo la correzione in modo che 
            # zero_corrected_ppm sia crescente
            # ----------------------------------------------------------------------
            combined = list(zip(zero_corrected_ppm, z["sat_trans_hz"], z["max_indexes"], z["max_vals"]))
            combined.sort()   # ordina per zero_corrected_ppm (primo elemento)
            (zero_corrected_ppm[:],
            z["sat_trans_hz"][:],
            z["max_indexes"][:],
            z["max_vals"][:]) = zip(*combined)            

            # Common x grid for all curves
            x_common = np.linspace(np.min(zero_corrected_ppm), np.max(zero_corrected_ppm), N_POINTS_FIT)
            
            # ------------------- Sigmoid upper envelope ----------------------
            # Stima con centro fissato a 0 (modifica se vuoi centro libero)
            L, R, tau = estimate_constrained_sigmoid(x_data=zero_corrected_ppm, y_data=z["max_vals"], fix_center=True, x0_fixed=0.0)

            # For each value in zero_corrected_ppm, find the index in x_common where the
            # element is closest to that value.
            # np.abs(x_common - val) computes the absolute differences between all points
            # in x_common and the current val.
            # np.argmin returns the position (index) of the smallest difference, i.e.,
            # the closest match.
            # The list comprehension collects these indices for all 19 elements of
            # zero_corrected_ppm.           
            linspace_indices = [np.argmin(np.abs(x_common - val)) for val in zero_corrected_ppm]

            y_sig = constrained_sigmoid(x_common, L, R, tau, x0=0.0)
            sigmoidal_envelope_results =  {
                "L": L,
                "R": R,
                "tau": tau,
                "x": x_common,
                "y": y_sig,
                "fit_label": f'Sigmoid (L={L:.3f}, R={R:.3f}, τ={tau:.3f})',
                "fit_successful": True,
            }
            analysis_results[name]["sigmoidal_envelope_results"] = sigmoidal_envelope_results

            # ----------------- Sigmoid correct z-specra data -----------------
            # Create a new list of sigmoid_corrected_max_vals
            sigmoid_corrected_max_vals: list[float] = [0.0] * len(z["max_vals"])

            # Correct the max_vals values
            for i, (idx, val) in enumerate(zip(linspace_indices,analysis_results[name]["max_vals"])):
                envelope_val = analysis_results[name]["sigmoidal_envelope_results"]["y"][idx]
                if np.abs(envelope_val) < 1e-12:
                    # If envelope is essentially zero, skip correction or 
                    # set to original val. Emit a warning and fall back to 
                    # uncorrected value
                    print(colored(
                        f"Warning: Sigmoid envelope near zero at index {idx} "
                        f"(ppm ~{zero_corrected_ppm[i]:.3f}). Using uncorrected value.", "yellow", attrs=["bold"])
                    )
                    sigmoid_corrected_max_vals[i] = val  # or some other fallback

                else:
                    sigmoid_corrected_max_vals[i] = val / envelope_val
            analysis_results[name]["sigmoid_corrected_max_vals"] = sigmoid_corrected_max_vals

            # ------------------- Lorentzian upper envelope -------------------
            A, gamma = estimate_constrained_lorentzian(x_data=zero_corrected_ppm, y_data=z["sigmoid_corrected_max_vals"])
            y_min = np.min(z["sigmoid_corrected_max_vals"])
            y_lor = constrained_lorentzian(x_common, A, gamma, y_min)
            lorentzian_envelope_results =  {
                "A": A,
                "gamma": gamma,
                "x": x_common,
                "y": y_lor,
                "fit_label": f'Lorentzian (A={A:.3f}, γ={gamma:.3f})',
                "fit_successful": True,
            }
            analysis_results[name]["lorentzian_envelope_results"] = lorentzian_envelope_results

            # ------------------- Spline fit ----------------------
            spline_fit_results = spline_fit(x=zero_corrected_ppm, y=z["sigmoid_corrected_max_vals"], x_fit=x_common)

            # Early exit se il fit non ha successo
            if not spline_fit_results.get("fit_successful", False):
                print(f"Spline fit fallito per {name}")
                analysis_results[name]["spline_fit_results"] = spline_fit_results  # salviamo comunque per eventuale debug
                analysis_results[name]["diff_x"] = None
                analysis_results[name]["diff_y"] = None
                analysis_results[name]["integrals"] = {}
                continue

            # --- Fit riuscito: salva i risultati ---
            analysis_results[name]["spline_fit_results"] = spline_fit_results

            # --- Calcolo della curva differenza (Lorentzian envelope - spline fit) ---
            # Poiché abbiamo creato x_common = np.linspace(...) e sia lorentzian che spline usano lo stesso x_common,
            # non serve il controllo di lunghezza: sono identici per costruzione.
            diff_x = x_common
            diff_y = lorentzian_envelope_results["y"] - spline_fit_results["y_fit"]

            analysis_results[name]["diff_x"] = diff_x
            analysis_results[name]["diff_y"] = diff_y

            # --- Plot ---
            plot_data(
                x=spline_fit_results["x"],   # qui stai passando i punti originali ordinati
                y=spline_fit_results["y"],
                x_fit=spline_fit_results["x_fit"],
                y_fit=spline_fit_results["y_fit"],
                add_lorentz=True,
                lorentzian_envelope_results=lorentzian_envelope_results,
                add_sigmoid=True,
                sigmoidal_envelope_results=sigmoidal_envelope_results,
                y_std_data=analysis_results[name].get("sd_max_vals") if name in ("reference", "avg") else None,
                title=name,
                invert_x=True,
                diff_x=diff_x,
                diff_y=diff_y,
                diff_label="Lorentzian envelope - Spline fit",
                visibility=config.get("plot_visibility", get_default_visibility())
            )

            # --- Calcolo integrali ---
            integrals = compute_regions_integrals(diff_x, diff_y)
            analysis_results[name]["integrals"] = integrals

        else:
            print(f"Numero di frequenze di saturazione non corrispondente per {name}")
    
    if with_ref and ref_keys:
        analysis_results["reference_integrals_stats"] = _compute_integrals_stats(
            ref_keys, analysis_results
        )

    if with_multiple and sample_keys:
        analysis_results["sample_integrals_stats"] = _compute_integrals_stats(
            sample_keys, analysis_results
        )

    if with_ref and with_multiple and ref_keys and sample_keys:
        analysis_results["p_values"] = _compute_pvalues(
            ref_keys, sample_keys, analysis_results, test='t-test'
        )

    # ═══════════════════════════════════════════════════════════════
    #  💾 SALVA I RISULTATI NELLA CACHE
    # ═══════════════════════════════════════════════════════════════
    save_cache(config_name, config, analysis_results)

    # ----------------------------------------------------------------------
    # Plot integrals
    # ----------------------------------------------------------------------
    plot_integrals_regions(
        data=analysis_results,
        reference=with_ref,
        multiple_amount_ref=multiple_amount_ref if with_ref else 0
    )

    # ----------------------------------------------------------------------
    # Grouped comparison plot (reference vs sample)
    # ----------------------------------------------------------------------
    if with_ref and with_multiple:
        ref_stats = analysis_results.get("reference_integrals_stats")
        sample_stats = analysis_results.get("sample_integrals_stats")
        pvals = analysis_results.get("p_values")
        if ref_stats and sample_stats:
            plot_grouped_comparison(
                ref_stats, sample_stats, pvals,
                title="Reference vs Sample - Integrals by Region"
            )    

def main() -> None:
    config_name, config_data = select_or_create_config()
    complete_config = ensure_complete_config(config_name, config_data)
    run_analysis(config_name, complete_config)

if __name__ == "__main__":
    main()