"""
reader_ng.py

Processing of Bruker NMR data for saturation transfer experiments.
Reads FIDs, applies corrections, displays spectra with interactive checkboxes,
allows the user to select a ppm range, and calculates maxima to generate
a saturation transfer curve.
"""

import nmrglue as ng
from nmrglue.fileio.fileiobase import unit_conversion
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import re
import tkinter as tk
from tkinter import filedialog
from typing import List, Optional, Tuple, Dict, Any, Union
from scipy.interpolate import UnivariateSpline

REGIONS: dict[str, List[float]] = {
    "G6P/G3P": [6.7, 7.0],
    "3PG/F6P": [5.8, 6.7],
    "Pi": [4.5, 5.2],
    "PDE": [3.5, 4.0],
    "PEP": [1.5, 3.0],
    "GAMMA-ATP": [-3.5, -1.5],
    "ALPHA-ATP": [-8.5, -7.5],
}

# ----------------------------------------------------------------------
# GLOBALS
# ----------------------------------------------------------------------
start_ppm: float|None = None
end_ppm: float|None = None

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def find_maxima(arr: np.ndarray,
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
        raise FileNotFoundError(f"{file_path} file not found.")

    text = file_path.read_text(encoding="utf-8", errors="ignore")

    # Look for the parameter header: ##$PARAMETER= ( N )
    hdr_pattern = rf"##\${PARAMETER}=\(\s*(?P<N>\d+)\s*\)"
    hdr_match = re.search(hdr_pattern, text)
    if not hdr_match:
        raise ValueError(f"Parameter header '##${PARAMETER}=( N )' not found in {file_path} file.")
    N = int(hdr_match.group("N"))
    print(f"{PARAMETER} dimension: {N}")

    # The rest of the text after the header
    start_pos = hdr_match.end()
    tail = text[start_pos:]

    # Extract all numbers (integers or floats, including scientific notation)
    num_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    vals = re.findall(num_pattern, tail)
    print(f"Found {len(vals)} numerical values for {PARAMETER}")

    if len(vals) < N:
        raise ValueError(f"Found only {len(vals)} numbers after {PARAMETER} header, expected {N}.")

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
    ppm_axis = uc.ppm_scale()          # full ppm array
    index = int(np.abs(ppm_axis - user_ppm).argmin())
    return index

def compute_regions_integrals(x_fit: np.ndarray, y_fit: np.ndarray) -> Dict[str, float]:
    """
    Calcola gli integrali di regione per tutte le regioni definite in REGIONS.

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

    def _region_integral(bounds, x_fit, y_fit) -> float:
        """
        Calcola l'integrale di (1 - y) nell'intervallo [bounds[0], bounds[1]].

        Parameters
        ----------
        bounds : list o tupla di due float
            Limiti inferiore e superiore dell'intervallo in ppm.
        x_fit : np.ndarray
            Array delle ascisse.
        y_fit : np.ndarray
            Array delle ordinate.

        Returns
        -------
        float
            Valore dell'integrale.
        """
        mask: np.ndarray = (x_fit >= bounds[0]) & (x_fit <= bounds[1])
        if not np.any(mask):
            return 0.0
        return np.trapezoid(1 - y_fit[mask], x_fit[mask])

    region_integrals: Dict[str, float] = {
        region_name:
        _region_integral(bounds = region_bounds, x_fit = x_fit, y_fit = y_fit) for region_name, region_bounds in REGIONS.items()
                              }
    
    return region_integrals

def plot_data_with_spline(
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray],
        x_fit: Union[List[float], np.ndarray],
        y_fit: Union[List[float], np.ndarray],
        title: str = "Max Values vs Saturation ppm",
        xlabel: str = "Saturation ppm",
        ylabel: str = "Max Value",
        fit_label: str = "",
        invert_x: bool = True
    ) -> Figure:
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
    fig: Figure = plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'o', color='b', label='Data')
    plt.plot(x_fit, y_fit, 'r-', label = fit_label)

    if invert_x:
        plt.gca().invert_xaxis()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    return fig

def fit_curve(x: Union[List[float], np.ndarray],
              y: Union[List[float], np.ndarray],
              smoothing: float = 0.0,
              n_points: int = 200) -> Dict[str, Any]:
    """
    Esegue lo spline fit dei dati.

    Parameters
    ----------
    x, y : array-like
        Dati originali.
    smoothing : float
        Fattore di smoothing per la spline (0 = interpolazione esatta).
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
        spline = UnivariateSpline(x_sorted, y_sorted, s=smoothing)
        fit_successful = True
        x_fit = np.linspace(x_sorted.min(), x_sorted.max(), n_points)
        y_fit = spline(x_fit)
        fit_label = f"Spline (s={smoothing})"
    except ImportError:
        print("scipy not available - cannot perform spline fit.")
    except Exception as e:
        print(f"Spline fit failed: {e}")

    return {
        'x_sorted': x_sorted,
        'y_sorted': y_sorted,
        'x_fit': x_fit,
        'y_fit': y_fit,
        'fit_label': fit_label,
        'fit_successful': fit_successful
    }

def plot_integrals_regions(
    integrals: Union[Dict[str, float], List[Dict[str, float]]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    title: str = 'Intensità per regione',
    xlabel: str = 'Regione',
    ylabel: str = 'Intensità',
    figsize: tuple = (12, 6),
    show_values: bool = True,
    reference: Optional[bool] = False
) -> Figure:
    """
    Crea un grafico a barre (singolo o raggruppato) degli integrali di regione.
    """
    # Determina se è una lista o un dizionario singolo
    is_list: bool = isinstance(integrals, list)
    
    if not is_list:
        # Caso singolo dizionario (comportamento originale)
        integrals = [integrals]  # Converti in lista per unificare la logica

    # Verifica che tutti i dizionari abbiano le stesse chiavi
    if len(integrals) > 1:
        keys_set: List[set[str]] = [set(d.keys()) for d in integrals]
        if not all(k == keys_set[0] for k in keys_set):
            raise ValueError("Tutti i dizionari devono avere le stesse chiavi (regioni).")
    
    # Estrai le etichette comuni (regioni) dal primo dizionario
    region_labels: List[str] = list(integrals[0].keys())
    n_regions: int = len(region_labels)
    
    n_series: int = len(integrals)

    # Prepara i valori: matrice (n_series x n_regions)
    values_matrix: List = []
    for d in integrals:
        values_matrix.append([d[reg] for reg in region_labels])
    values_matrix = np.array(values_matrix)  # shape: (n_series, n_regions)

    # Crea il grafico
    fig_absolute: Figure
    ax_absolute: Axes
    fig_absolute, ax_absolute = plt.subplots(figsize=figsize)

    # Larghezza di ogni barra e posizioni
    bar_width: float = 0.8 / n_series if n_series > 1 else 0.6
    x = np.arange(n_regions)  # posizioni delle regioni sull'asse x
    
    # Colori
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_series))
    else:
        # Se i colori sono meno delle serie, ripeti
        if len(colors) < n_series:
            colors = colors * (n_series // len(colors) + 1)
        colors = colors[:n_series]

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
                    f'{val:.2e}', 
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

        title = "Intensità relative alla prima serie (100% = riferimento)"

        labels = labels[1:]
        colors = colors[1:]

        integrals_referenced: List[Dict[str, float]] = []
        if reference:
            for index, integral in enumerate(integrals[1:]):
                d: Dict[str, float] = {}
                for key, integral in integral.items():
                    d[key] = 100 * (integral / integrals[0][key] -1)
                integrals_referenced.append(d)
        
        n_series -= 1

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
        
        # Colori
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, n_series))
        else:
            # Se i colori sono meno delle serie, ripeti
            if len(colors) < n_series:
                colors = colors * (n_series // len(colors) + 1)
            colors = colors[:n_series]

        # Etichette per la legenda (solo se più serie)
        if labels is None and n_series > 1:
            labels = [f'Gruppo {i+1}' for i in range(n_series)]
        elif labels is None:
            labels = ['']  # non serve legenda

        # Disegna le barre
        bars_list: List = []
        if reference:
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
    
    if reference:
        return fig_absolute, fig_referenced
    else:
        return fig_absolute

def select_experiment_folder(title: str = "Select a folder") -> Path:
    """Mostra una finestra di dialogo e restituisce il percorso della cartella selezionata."""
    root = tk.Tk()
    root.withdraw()
    folder = Path(filedialog.askdirectory(title=title))
    return folder

def extract_parameters(folder: Path) -> Tuple[List[float], List[float]]:
    """Estrae PVM_SatTransFL e PVM_FrqWorkOffset dal file method."""
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
    dic: dict
    data: np.ndarray
    dic, data = ng.bruker.read(folder)
    udic: dict = ng.bruker.guess_udic(dic, data)
    uc: unit_conversion = ng.fileio.bruker.fileiobase.uc_from_udic(udic, dim=1)
    ppm_axis: np.ndarray = uc.ppm_scale()
    n_exp: int = dic["acqu2s"]["TD"]
    bf1: float = dic["acqus"]["BF1"]

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
    spectra: Dict[int, np.ndarray] = {}

    for exp_idx in range(n_exp):
        # ---- Select experiment folder
        fid: np.ndarray = data[exp_idx , :]

        # ---- Remove Bruker digital filter
        fid = ng.bruker.remove_digital_filter(dic, data=fid)

        # Zero-fill to 2048 points
        fid_zf: np.ndarray = ng.proc_base.zf_size(fid, size=2048)

        # Line broadening (exponential)
        fid_apod: np.ndarray = ng.proc_base.em(fid_zf, lb=0.005)
        
        # Fourier transform
        spectrum: np.ndarray = ng.proc_base.fft(fid_apod)

        # Automatic phase correction (ACME method)
        spectrum_phased: np.ndarray = ng.proc_autophase.autops(spectrum, fn="acme")

        # Invert the axis (left to right: decreasing ppm)
        spectrum_phased = spectrum_phased[::-1]
        spectra[exp_idx] = spectrum_phased
    
    return spectra

def plot_spectra(spectra, n_exp, ppm_axis, sat_trans_hz) -> Figure:
    """
    Crea un plot interattivo di tutti gli spettri con checkbox per mostrare/nascondere
    ogni traccia.

    Parameters
    ----------
    spectra : Dict[int, np.ndarray]
        Dizionario degli spettri elaborati.
    n_exp : int
        Numero di esperimenti.
    ppm_axis : np.ndarray
        Asse dei ppm.
    sat_trans_hz : List[float]
        Frequenze di saturazione in Hz (usate come etichette).
    """
    lines: List[plt.Line2D] = []
    labels: List[str] = []
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(12, 6))
    for exp_idx in range(n_exp):
        # Plot the real part
        line, = ax.plot(
            ppm_axis,
            np.real(spectra[exp_idx]),
            label=f"{sat_trans_hz[exp_idx]:.2f}",
            alpha=0.7,
            linewidth=1.2,
        )
        lines.append(line)
        labels.append(line.get_label())

    # Main plot settings
    ax.invert_xaxis()                           # decreasing ppm from left to right
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.grid(True, alpha=0.3)

    # ---- CheckButtons panel ----
    rax = fig.add_axes([0.80, 0.15, 0.19, 0.70])   # [left, bottom, width, height]
    visibility = [l.get_visible() for l in lines]
    checks = CheckButtons(rax, labels, visibility)

    # ----------------------------------------------------------------------
    # Callbacks for interactive widgets
    # to capture lines, labels, fig, checks
    # ----------------------------------------------------------------------
    def _on_check(label: str) -> None:
        idx = labels.index(label)
        lines[idx].set_visible(not lines[idx].get_visible())
        fig.canvas.draw_idle()

    def _check_all(event: Any) -> None:
        for i, l in enumerate(lines):
            if not l.get_visible():
                checks.set_active(i)
        plt.draw()

    def _uncheck_all(event: Any) -> None:
        for i, l in enumerate(lines):
            if l.get_visible():
                checks.set_active(i)
        plt.draw()

    checks.on_clicked(_on_check)

    # "Check all" button
    ax_all = fig.add_axes([0.80, 0.90, 0.09, 0.05])
    btn_all = Button(ax_all, "Check all")
    btn_all.on_clicked(_check_all)

    # "Uncheck all" button
    ax_none = fig.add_axes([0.90, 0.90, 0.09, 0.05])
    btn_none = Button(ax_none, "Uncheck all")
    btn_none.on_clicked(_uncheck_all)

    fig.tight_layout(rect=[0, 0, 0.80, 1])       # leave space on the right for widgets

    # Show the figure (non-blocking)
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
    max_vals : Dict[int, float]
        Dizionario con chiave indice esperimento e valore massimo normalizzato.
    max_indexes : Dict[int, int]
        Dizionario con chiave indice esperimento e indice del massimo (originale).
    """
    # ---- Find maxima in the selected range ----
    max_vals: Dict[int, float] = {}
    max_indexes: Dict[int, int] = {}
    global_max_value: float = 0.0
    for exp_idx in spectra:
        val: float
        idx: int
        val, idx = find_maxima(spectra[exp_idx], start=start_idx, end=end_idx)
        if val > global_max_value:
            global_max_value = val
        max_vals[exp_idx] = val
        max_indexes[exp_idx] = idx

    # ---- Normalize max_vals ----
    for exp_idx , val in max_vals.items():
        max_vals[exp_idx ] /= global_max_value

    return max_vals, max_indexes

def ask_user_for_ppm_range(uc) -> Tuple[int, int]:
    """
    Richiede all'utente di inserire l'intervallo in ppm (min e max).
    Ripete la richiesta finché non vengono forniti valori validi.
    Restituisce gli indici start_idx, end_idx corrispondenti.
    """
    while True:
        try:
            start_ppm = float(input("Enter the minimum ppm (start): "))
            end_ppm = float(input("Enter the maximum ppm (end): "))

            if end_ppm <= start_ppm:
                print("Invalid range: end must be greater than start. Please try again.")
                continue

            print(f"Selected range: [{start_ppm}, {end_ppm}]")  # end-exclusive
            return start_ppm, end_ppm

        except ValueError as e:
            print(f"Input error: {e}. Please enter valid numbers.")
        except Exception as e:
            print(f"Unexpected error: {e}. Please try again.")
            
def correct_sat_freq(sat_trans_hz, max_vals, max_indexes, work_offset_hz, uc, bf1):
    """
    Calcola le frequenze di saturazione corrette (in ppm) e restituisce il fit.
    NOTA: la lista sat_trans_hz viene modificata in-place con le frequenze corrette in Hz.
    """    
    sat_trans_f1_ppm: List[float] = [0.0] * len(sat_trans_hz)
    for exp_idx in max_vals:
        # Calculate the offset from the reference            
        delta: float = work_offset_hz[0] - uc.hz(max_indexes[exp_idx])
        sat_trans_hz[exp_idx] += delta  # modifica voluta
        sat_trans_f1_ppm[exp_idx] = sat_trans_hz[exp_idx] / bf1   # convert to ppm
    
    return fit_curve(
        x=sat_trans_f1_ppm,
        y=list(max_vals.values()),
        smoothing=0.0,
        n_points=200,
    )

def ask_yes_no(prompt):
    """Ask the user a yes/no question and return True for yes, False for no."""
    while True:
        answer = input(prompt + " (y/n): ").strip().lower()
        if answer in ('y'):
            return True
        if answer in ('n'):
            return False
        print("Please answer with y or n.")

def ask_int(prompt: str, min_val: int = None, max_val: int = None) -> int:
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
    while True:
        answer = input(f"{prompt} ").strip()
        try:
            value = int(answer)
            if min_val is not None and value < min_val:
                print(f"Errore: il valore deve essere >= {min_val}.")
                continue
            if max_val is not None and value > max_val:
                print(f"Errore: il valore deve essere <= {max_val}.")
                continue
            return value
        except ValueError:
            print("Errore: inserire un numero intero valido.")

def main() -> None:
    """
    Main routine of the script.

    - Prompts the user to select a Bruker experiment folder.
    - Reads the 'method' file to obtain saturation frequencies and offset.
    - Loads and processes all FIDs (digital filter removal, zero-filling,
      line broadening, FFT, automatic phase correction, axis inversion).
    - Displays all spectra with an interactive checkbox panel.
    - Asks the user for a ppm range of interest.
    - Finds the maximum intensity within that range for each spectrum.
    - Corrects the saturation frequencies using the frequency offset.
    - Plots the maximum values against the corrected saturation ppm.
    """
    global start_ppm, end_ppm
    
    folders: List[Path] = []
    region_integrals_dict: Dict[str, Dict[str, float]] = {}
    
    with_ref: bool = ask_yes_no("Reference folder?")
    with_multiple: bool = ask_yes_no("Multiple folders?")
    multiple_amount: int = ask_int("How many?") if with_multiple else 1

    # ---- Select reference folder ----
    if with_ref:
        folder_ref: Path = select_experiment_folder(title="Select reference folder")
        folders.append(folder_ref)
    # ---- Select experiment folder ----
    for n in range(multiple_amount):
        folder: Path = select_experiment_folder(title="Select a folder")
        folders.append(folder)
    
    for folder in folders:
        folder_name_short: str = f"…{folder.parent.name[-32:]}"
        # ---- Extract parameters from method file ----
        sat_trans_hz: List[float]
        work_offset_hz: List[float]
        try:
            sat_trans_hz, work_offset_hz = extract_parameters(folder=folder)
        except FileNotFoundError as e:
            print(f"Errore: {e}")
            return
        except ValueError as e:  # se parameter_extract solleva ValueError per altri motivi
            print(f"Errore nel formato dei parametri: {e}")
            return

        # ---- Read Bruker data ----
        dic: Dict
        data: np.ndarray
        uc: unit_conversion
        ppm_axis: np.ndarray
        n_exp: int
        bf1: float
        (dic, data, uc, ppm_axis, n_exp, bf1) = load_spectra(folder=folder)

        # ---- Verifica che ci siano esperimenti ----
        if n_exp <= 0:
            print("Errore: nessun esperimento trovato (n_exp = 0). Impossibile proseguire.")
            return
        
        # ---- Process data ----
        spectra: Dict
        spectra = process_spectra(data=data, dic=dic, n_exp=n_exp)

        # ---- User input for ppm range ----
        start_idx: int
        end_idx: int
        if not start_ppm or not end_ppm:
            # ---- Plot spectra ----
            spectra_fig: Figure = plot_spectra(
                spectra=spectra, 
                n_exp=n_exp, 
                ppm_axis=ppm_axis, 
                sat_trans_hz=sat_trans_hz
            )
            (start_ppm, end_ppm) = ask_user_for_ppm_range(uc=uc)
            start_idx = ppm_to_index(uc=uc, user_ppm=end_ppm)
            end_idx = ppm_to_index(uc=uc, user_ppm=start_ppm)
            if start_idx < 0 or end_idx < 0:
                print("Invalid range: end and start indexes have to be both non-negative. Exiting.")
                return
            plt.close(spectra_fig)
            
        # ---- Find normalized maxima along the spectra ----
        max_vals: Dict[int, float]
        max_indexes: Dict[int, int]
        (max_vals, max_indexes) = find_max_vals(spectra=spectra, start_idx=start_idx, end_idx=end_idx)
        
        spline_fig: Figure
        #integrals_fig: Figure

        # ---- Correct saturation frequencies and final plot ----
        if len(sat_trans_hz) == len(max_vals):
            
            fit_result: Dict[str, Any] = correct_sat_freq(
                sat_trans_hz=sat_trans_hz, 
                max_vals=max_vals,
                max_indexes=max_indexes,
                work_offset_hz=work_offset_hz,
                uc=uc,
                bf1=bf1,
            )
            
            if fit_result["fit_successful"]:
                spline_fig = plot_data_with_spline(
                    x=fit_result["x_sorted"],
                    y=fit_result["y_sorted"],
                    x_fit=fit_result["x_fit"],
                    y_fit=fit_result["y_fit"],
                    title=folder_name_short,
                    xlabel="Saturation ppm",
                    ylabel="Max Value",
                    invert_x=True
                )
                region_integrals: Dict[str, float] = compute_regions_integrals(x_fit=fit_result["x_fit"], y_fit=fit_result["y_fit"])
                region_integrals_dict[folder_name_short] = region_integrals
                #integrals_fig = plot_integrals_regions(integrals=region_integrals, labels=folder)

        else:
            print("Number of saturation frequencies does not match number of processed series; skipping plot.")

        #plt.close(spline_fig)
        #plt.close(integrals_fig)

    plot_integrals_regions(
        integrals=list(region_integrals_dict.values()),
        labels=list(region_integrals_dict.keys()),
        reference=with_ref
    )

if __name__ == "__main__":
    main()