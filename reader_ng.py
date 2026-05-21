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
import json
from termcolor import colored
import os
import hashlib
from joblib import dump, load

METABOLITE_REGIONS: dict[str, List[float]] = {
    "SP": [5.6, 7.5],
    #"G6P/G3P": [6.7, 7.0],
    #"3PG/F6P": [5.8, 6.7],
    "Pi": [4.5, 5.2],
    "PDE": [3.5, 4.0],
    "PEP": [1.5, 3.0],
    "GAMMA-ATP": [-3.5, -1.5],
    "ALPHA-ATP": [-8.5, -7.5],
}
CACHE_DIR = Path(__file__).parent / "cache"          # cartella dedicata
CACHE_DIR.mkdir(exist_ok=True)

def _cache_path(config_name: str, config: Dict[str, Any]) -> Path:
    # Se il nome è vuoto (nessuna configurazione salvata), usa un hash
    if not config_name:
        import hashlib
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
        if payload["key"] == current_key:
            print(f"Cache valida per '{config_name}', caricamento immediato.")
            return payload["analysis_results"]
        else:
            print(f"Cache obsoleta per '{config_name}'.")
            return None
    except Exception as e:
        print(f"Errore cache per '{config_name}': {e}")
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
        print(f"Errore nel caricamento della configurazione '{name}': {e}")
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
        print(f"Errore nel salvataggio: {e}")

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
    """
    Extract numerical values following a Bruker parameter header in a text file
    (e.g., method).

    Expected format:
        ##$PARAMETER= ( N )
        val1 val2 ... valN

    Parameters
    ----------
    file_path : Path
        Path to the file (e.g., method).
    PARAMETER : str
        Parameter name (e.g., "PVM_SatTransFL").

    Returns
    -------
    List[float]
        List of N numerical values.

    Raises
    ------
    ValueError
        If the header is not found or if fewer numbers than N are extracted.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found.")
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    # Look for the parameter header: ##$PARAMETER= ( N )
    hdr_pattern = rf"##\${PARAMETER}=\(\s*(?P<N>\d+)\s*\)"
    hdr_match = re.search(hdr_pattern, text)
    if not hdr_match:
        raise ValueError(f"Header '##${PARAMETER}=( N )' non trovato in {file_path}.")
    N = int(hdr_match.group("N"))
    print(f"{PARAMETER} dimension: {N}")

    # The rest of the text after the header
    start_pos = hdr_match.end()
    tail = text[start_pos:]

    # Extract all numbers (integers or floats, including scientific notation)
    num_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    vals = re.findall(num_pattern, tail)
    if len(vals) < N:
        raise ValueError(f"Trovati solo {len(vals)} numeri, attesi {N}.")
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
    return int(np.abs(ppm_axis - user_ppm).argmin())

def compute_regions_integrals(x_fit: np.ndarray, y_fit: np.ndarray) -> Dict[str, float]:
    """
    Calcola gli integrali di regione per tutte le regioni definite in METABOLITE_REGIONS.

    Parameters
    ----------
    x_fit : np.ndarray
        Array delle asycisse (ppm) della curva fitted.
    y_fit : np.ndarray
        Array delle ordinate (intensità normalizzate) della curva fitted.

    Returns
    -------
    Dict[str, float]
        Dizionario con chiavi i nomi delle regioni e valori gli integrali calcolati.
    """
    def _region_integral(bounds, x_fit, y_fit):
        mask = (x_fit >= bounds[0]) & (x_fit <= bounds[1])
        if not np.any(mask):
            return 0.0
        bottom_area = np.trapezoid(y_fit[mask], x_fit[mask])
        return bottom_area
    
    return {region: _region_integral(bounds, x_fit, y_fit) for region, bounds in METABOLITE_REGIONS.items()}

def plot_data_with_spline(x, y, x_fit, y_fit, y_std_data = None, title="Max Values vs Saturation ppm",
                          xlabel="Saturation ppm", ylabel="Max Value",
                          fit_label="", invert_x=True) -> Figure:
    """
    Plot data points and a spline fit through them.

    Parameters
    ----------
    x : array-like
        x-coordinates of the data points.
    y : array-like
        y-coordinates of the data points.
    x_fit : array-like
        x-coordinates of the fitted curve.
    y_fit : array-like
        y-coordinates of the fitted curve.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    fit_label : str, optional
        Label for the fitted curve (used in legend).
    invert_x : bool, default True
        If True, invert the x-axis (useful for ppm scales where high values are on the left).
    """
    # Create the plot
    fig = plt.figure(figsize=(8, 5))

    if title in ("reference", "avg") and y_std_data is not None:
        plt.errorbar(x, y, yerr=np.array(y_std_data), fmt='o', color='b', label='Data')
    else:
        plt.plot(x, y, 'o', color='b', label='Data')
    plt.plot(x_fit, y_fit, 'r-', label=fit_label)
    if invert_x:
        plt.gca().invert_xaxis()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show(block=False)
    return fig

def fit_curve(x, y, n_points=200) -> Dict[str, Any]:
    """
    Esegue lo spline fit dei dati.

    Parameters
    ----------
    x, y : array-like
        Dati originali.
    n_points : int
        Numero di punti per la curva fitted.

    Returns
    -------
    dict con chiavi:
        'x_sorted' : np.ndarray
        'y_sorted' : np.ndarray
        'x_fit' : np.ndarray o None se fit fallito
        'y_fit' : np.ndarray o None se fit fallito
        'fit_label' : str (vuota se fallito)
        'fit_successful' : bool
    """
    x = np.asarray(x)
    y = np.asarray(y)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    fit_successful = False
    x_fit = None
    y_fit = None
    fit_label = ""
    try:
        spline = PchipInterpolator(x_sorted, y_sorted)
        fit_successful = True
        x_fit = np.linspace(x_sorted.min(), x_sorted.max(), n_points)
        y_fit = spline(x_fit)
        fit_label = f"Spline Fit"
    except ImportError:
        print("scipy not available - cannot perform spline fit.")
    except Exception as e:
        print(f"Spline fit failed: {e}")
    return {
        'x_sorted': x_sorted, 'y_sorted': y_sorted,
        'x_fit': x_fit, 'y_fit': y_fit,
        'fit_label': fit_label, 'fit_successful': fit_successful
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
        plt.show(block=True)
        return fig_absolute, fig_referenced
    
    plt.show(block=True)
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
    Elabora ogni FID: rimozione filtro digitale, zero‑filling, line broadening,
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
                        label=f"{sat_trans_hz[exp_idx]:.2f}",
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
    global_max = 0.0
    for exp_idx, spec in spectra.items():
        val, idx = find_maximum(spec, start=start_idx, end=end_idx)
        if val > global_max:
            global_max = val
        max_vals.append(val)
        max_indexes.append(idx)
    # ---- Normalize max_vals ----
    for i in range(len(max_vals)):
        max_vals[i] /= global_max
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

    Modifica *in‑place* la lista sat_trans_hz:
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


def fit_saturation_curve(x_ppm: List[float], y_vals: List[float]) -> Dict[str, Any]:
    """
    Esegue il fit della curva di saturazione usando i parametri standard.

    Utilizza fit_curve con smoothing=0.02 e 200 punti di campionamento.
    """
    return fit_curve(x=x_ppm, y=y_vals, n_points=200)

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
    """Create a dictionary of zero‑filled lists for statistical accumulation.
    
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
    else:
        # Le cartelle ci sono già, usale così come sono
        pass
    
    # I ppm verranno chiesti durante l'analisi (serve uc). Segnaliamo se mancano.
    if config_data.get("start_ppm") is None or config_data.get("end_ppm") is None:
        config_data["ppm_missing"] = True
    else:
        config_data["ppm_missing"] = False
    
    # Salva subito le eventuali modifiche (cartelle, opzioni)
    save_config(config_name, config_data)
    return config_data

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
        analysis_results = cached
        for name, z in analysis_results.items():
            if "fit_result" in z and z["fit_result"]["fit_successful"]:
                fit = z["fit_result"]
                plot_data_with_spline(
                    fit["x_sorted"], fit["y_sorted"],
                    fit["x_fit"], fit["y_fit"],
                    y_std_data=z.get("sd_max_vals"),
                    title=name, invert_x=True
                )
        plot_integrals_regions(
            data=analysis_results,
            reference=with_ref,
            multiple_amount_ref=multiple_amount_ref if with_ref else 0
        )
        return   # <-- esce dopo che il grafico a barre è stato chiuso
    # ═══════════════════════════════════════════════════════════════

    # Initialize accumulators using helper
    ref_stats = None
    avg_stats = None
    ref_work_offset_hz = []
    avg_work_offset_hz = []
    
    for idx, folder in enumerate(folders):
        folder_name_short = f"{folder.parent.name[:12]}…{folder.parent.name[-12:]}-{folder.stem}"
        analysis_results[folder_name_short] = {}

        # ----------------------------------------------------------------------
        # Extract sat_trans_hz and work_offset_hz parameters from folder files
        # ----------------------------------------------------------------------
        try:
            sat_trans_hz, work_offset_hz = extract_parameters(folder)
            analysis_results[folder_name_short]["sat_trans_hz"] = sat_trans_hz
            analysis_results[folder_name_short]["work_offset_hz"] = work_offset_hz
        except (FileNotFoundError, ValueError) as e:
            print(f"Errore in {folder}: {e}")
            return
        
        if with_multiple:
            if idx < multiple_amount:
                if not avg_work_offset_hz:
                    avg_work_offset_hz = work_offset_hz
                elif avg_work_offset_hz != work_offset_hz:
                    print(f"{colored('Error', 'red', attrs=['bold'])}: different work_offset_hz in sample folders.")

        if with_ref:
            if idx < multiple_amount_ref:
                if not ref_work_offset_hz:
                    ref_work_offset_hz = work_offset_hz
                elif ref_work_offset_hz != work_offset_hz:
                    print(f"{colored('Error', 'red', attrs=['bold'])}: different work_offset_hz in reference folders.")
                
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
            corrected_ppm  = correct_sat_frequencies(
                z["sat_trans_hz"], 
                z["max_indexes"], 
                z["work_offset_hz"], 
                z["uc"], 
                z["bf1"]
            )
            fit_result = fit_saturation_curve(corrected_ppm, z["max_vals"])
            if fit_result["fit_successful"]:
                analysis_results[name]["fit_result"] = fit_result
                plot_data_with_spline(
                    analysis_results[name]["fit_result"]["x_sorted"],  
                    analysis_results[name]["fit_result"]["y_sorted"],
                    analysis_results[name]["fit_result"]["x_fit"], 
                    analysis_results[name]["fit_result"]["y_fit"],
                    y_std_data=analysis_results[name].get("sd_max_vals") if name in ("reference", "avg") else None,
                    title=name, invert_x=True
                )

                # ----------------------------------------------------------
                # Calculate integrals
                # ----------------------------------------------------------
                integrals = compute_regions_integrals(
                    z["fit_result"]["x_fit"], 
                    z["fit_result"]["y_fit"]
                )
                analysis_results[name]["integrals"] = integrals
                
            else:
                print(f"Fit fallito per {name}")
                analysis_results[name]["integrals"] = {}
        else:
            print(f"Numero di frequenze di saturazione non corrispondente per {name}")
    
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

def main() -> None:
    config_name, config_data = select_or_create_config()
    complete_config = ensure_complete_config(config_name, config_data)
    run_analysis(config_name, complete_config)

if __name__ == "__main__":
    main()