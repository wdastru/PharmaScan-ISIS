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
from matplotlib.lines import Line2D
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
        print(f"{file_path} file not found. Aborting.")
        exit(1)

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

def compute_regions_integrals(x_fit, y_fit) -> Dict[str, float]:
    def region_integral(bounds, x_fit, y_fit) -> float:
        integral: float = 0.0
        for i, x in enumerate(x_fit):
            if x < bounds[0] or x > bounds[1]:
                continue
            else:
                integral += (1 - y_fit[i])
        return integral

    region_integrals: Dict[str, float] = {
        region_name:
        region_integral(bounds = region_bounds, x_fit = x_fit, y_fit = y_fit) for region_name, region_bounds in REGIONS.items()
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
    ) -> Dict[str, float]:
    """
    Plot data points and a spline fit through them.

    Parameters
    ----------
    x : array-like
        x-coordinates of the data points.
    y : array-like
        y-coordinates of the data points.
    smoothing : float, default 0.0
        Smoothing factor for the spline:
        - 0.0 → exact interpolation (passes through all points)
        - >0  → smoothing spline; larger values give a smoother curve.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    invert_x : bool, default True
        If True, invert the x-axis (useful for ppm scales where high values are on the left).

    Notes
    -----
    If `scipy.interpolate.UnivariateSpline` is available, a spline fit is attempted.
    If it fails or scipy is not installed, a quadratic polynomial fit is used as fallback.
    The fitted curve is displayed together with the original data points.
    """

    # Create the plot
    plt.figure(figsize=(8, 5))
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

def plot_integrals_regions(integrals):
    # Estrai etichette e valori
    labels: List[str] = list(integrals.keys())
    values: List[float] = list(integrals.values())

    # Crea il grafico a barre
    plt.figure(figsize=(10, 6))
    bars: plt.BarContainer = plt.bar(labels, values, color='skyblue', edgecolor='black')

    # Aggiungi titolo ed etichette assi
    plt.title('Intensità per regione')
    plt.xlabel('Regione')
    plt.ylabel('Intensità')

    # Ruota le etichette sull'asse x per leggibilità (opzionale)
    plt.xticks(rotation=45, ha='right')

    # Aggiungi i valori sopra le barre (opzionale)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 * max(values),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Mostra il grafico
    plt.tight_layout()  # Per evitare sovrapposizioni
    plt.show()  

def select_experiment_folder() -> Path:
    """Mostra una finestra di dialogo e restituisce il percorso della cartella selezionata."""
    root = tk.Tk()
    root.withdraw()
    folder = Path(filedialog.askdirectory(title="Select a folder"))
    return folder

def extract_parameters(folder: Path) -> Tuple[List[float], List[float]]:
    """Estrae PVM_SatTransFL e PVM_FrqWorkOffset dal file method."""
    method = folder / "method"
    sat_hz = parameter_extract(method, "PVM_SatTransFL")
    offset_hz = parameter_extract(method, "PVM_FrqWorkOffset")
    return sat_hz, offset_hz

def load_spectra(folder: Path):
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

def plot_spectra(spectra, n_exp, ppm_axis, sat_trans_hz):
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

    # ---- CheckButtons panel ----
    rax = fig.add_axes([0.80, 0.15, 0.19, 0.70])   # [left, bottom, width, height]
    visibility = [l.get_visible() for l in lines]
    checks = CheckButtons(rax, labels, visibility)
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

def find_max_vals(spectra, start_idx, end_idx):
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

            end_idx = ppm_to_index(uc=uc, user_ppm=start_ppm)
            start_idx = ppm_to_index(uc=uc, user_ppm=end_ppm)

            if start_idx < 0 or end_idx <= start_idx:
                print("Invalid range: end must be greater than start and both non-negative. Please try again.")
                continue

            print(f"Selected range: [{start_idx}, {end_idx}]")  # end-exclusive
            return start_idx, end_idx

        except ValueError as e:
            print(f"Input error: {e}. Please enter valid numbers.")
        except Exception as e:
            print(f"Unexpected error: {e}. Please try again.")
            
def correct_sat_freq(sat_trans_hz, max_vals, max_indexes, work_offset_hz, uc, bf1):
    sat_trans_f1_ppm: List[float] = [0.0] * len(sat_trans_hz)
    for exp_idx in max_vals:
        # Calculate the offset from the reference            
        delta: float = work_offset_hz[0] - uc.hz(max_indexes[exp_idx])
        sat_trans_hz[exp_idx] += delta
        sat_trans_f1_ppm[exp_idx] = sat_trans_hz[exp_idx] / bf1   # convert to ppm
    
    return fit_curve(
        x=sat_trans_f1_ppm,
        y=list(max_vals.values()),
        smoothing=0.0,
        n_points=200,
    )

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
    # ---- Select experiment folder ----
    folder: Path = select_experiment_folder()
    
    # ---- Extract parameters from method file ----
    sat_trans_hz: float
    work_offset_hz: float
    (sat_trans_hz, work_offset_hz) = extract_parameters(folder=folder)

    # ---- Read Bruker data ----
    dic: Dict
    data: np.ndarray
    uc: unit_conversion
    ppm_axis: np.ndarray
    n_exp: int
    bf1: float
    (dic, data, uc, ppm_axis, n_exp, bf1) = load_spectra(folder=folder)

    # ---- Process data ----
    spectra: Dict
    spectra = process_spectra(data=data, dic=dic, n_exp=n_exp)

    # ---- Plot spectra ----
    plot_spectra(spectra=spectra, n_exp=n_exp, ppm_axis=ppm_axis, sat_trans_hz=sat_trans_hz)

    # ---- User input for ppm range ----
    start_idx: int
    end_idx: int
    (start_idx, end_idx) = ask_user_for_ppm_range(uc=uc)

    # ---- Find normalized maxima along the spectra ----
    max_vals: Dict[int, float]
    max_indexes: Dict[int, float]
    (max_vals, max_indexes) = find_max_vals(spectra=spectra, start_idx=start_idx, end_idx=end_idx)

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
            plot_data_with_spline(
                x=fit_result["x_sorted"],
                y=fit_result["y_sorted"],
                x_fit=fit_result["x_fit"],
                y_fit=fit_result["y_fit"],
                title="Max Values vs Saturation ppm",
                xlabel="Saturation ppm",
                ylabel="Max Value",
                invert_x=True
            )
            region_integrals = compute_regions_integrals(x_fit=fit_result["x_fit"], y_fit=fit_result["y_fit"])
            plot_integrals_regions(region_integrals)

    else:
        print("Number of saturation frequencies does not match number of processed series; skipping plot.")

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()    