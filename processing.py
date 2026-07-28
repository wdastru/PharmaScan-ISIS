"""
processing.py
Elaborazione dei dati: ricerca massimi, correzione frequenze,
integrali, pipeline z-spectrum, statistiche di gruppo.
"""

import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy import stats
from typing import List, Optional, Tuple, Dict, Any
from termcolor import colored

from constants import N_POINTS_FIT
from config import METABOLITE_REGIONS
from fitting import (
    spline_fit,
    constrained_lorentzian,
    constrained_sigmoid,
    estimate_constrained_lorentzian,
    estimate_constrained_sigmoid,
)

# ----------------------------------------------------------------------
# Ricerca massimi e normalizzazione
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

def normalize_max_vals(max_vals, global_max, global_min):
    for i in range(len(max_vals)):
        max_vals[i] = (max_vals[i] - global_min) / (global_max - global_min) if global_max > global_min else 0.0
    return max_vals

# ----------------------------------------------------------------------
# Conversione ppm / indice
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Correzione frequenze di saturazione
# ----------------------------------------------------------------------
def replace_zero_delta(sat_trans_hz, delta):
    """
    Replaces the delta value at the position where sat_trans_hz is 0.0
    with the mean of delta values corresponding to the smallest positive
    and smallest negative sat_trans_hz values.
    """
    if len(sat_trans_hz) != len(delta):
        raise ValueError("Both lists must have the same length")
    
    # Find the index of 0.0
    zero_idx = sat_trans_hz.index(0.0)
    
    # Find smallest positive and smallest negative values (excluding 0.0)
    min_positive_val = float('inf')
    min_negative_val = float('-inf')
    min_positive_idx = None
    min_negative_idx = None
    
    for idx, val in enumerate(sat_trans_hz):
        if idx == zero_idx:
            continue
        if val > 0 and val < min_positive_val:
            min_positive_val = val
            min_positive_idx = idx
        elif val < 0 and val > min_negative_val:
            min_negative_val = val
            min_negative_idx = idx
    
    # Calculate the mean of their corresponding delta values
    mean_delta = (delta[min_positive_idx] + delta[min_negative_idx]) / 2
    
    # Create a new delta list with the replacement
    new_delta = delta.copy()
    new_delta[zero_idx] = mean_delta

    new_delta = [d - mean_delta for d in new_delta]
    
    return new_delta

def correct_sat_frequencies(sat_trans_hz, max_indexes, work_offset_hz, uc, bf1):
    sat_trans_f1_ppm = [0.0] * len(sat_trans_hz)

    freq = [0.0] * len(sat_trans_hz)
    delta = [0.0] * len(sat_trans_hz)
    for i, (st_hz, idx) in enumerate(zip(sat_trans_hz, max_indexes)):
        freq[i] = uc.hz(idx)
        delta[i] = work_offset_hz[0] - freq[i]
  
    delta = replace_zero_delta(sat_trans_hz, delta)

    for i, (st_hz, idx) in enumerate(zip(sat_trans_hz, max_indexes)):
        sat_trans_hz[i] += delta[i]

    sat_trans_f1_ppm = [f / bf1 for f in sat_trans_hz]

    return sat_trans_f1_ppm

# ----------------------------------------------------------------------
# Integrali sulle regioni
# ----------------------------------------------------------------------
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
    
    for name, region in METABOLITE_REGIONS.items():

        if region["ppm"] is not None and len(region["ppm"]) == 2:
            start, end = region["ppm"]
        else:
            print(colored(f"Region {name} is not defined properly. Aborting.", "red"))
            exit(1)

        # Trova i punti all'interno dell'intervallo
        mask = (x_fit >= start) & (x_fit <= end)
        if not np.any(mask):
            integrals[name] = 0.0
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
        integrals[name] = area

    return integrals

# ----------------------------------------------------------------------
# Pipeline per un singolo z-spectrum
# ----------------------------------------------------------------------
def process_zspectrum_and_integrals(max_vals, zero_corrected_ppm) -> Dict[str, Any]:
    """Fit envelopes, spline, compute difference and integrals for one dataset."""
    
    # 2. Sort
    combined = list(zip(zero_corrected_ppm, max_vals))
    combined.sort()
    zero_corrected_ppm, max_vals_sorted = zip(*combined)
    zero_corrected_ppm = list(zero_corrected_ppm)
    max_vals_sorted = list(max_vals_sorted)

    # 3. Common grid
    x_common = np.linspace(min(zero_corrected_ppm), max(zero_corrected_ppm), N_POINTS_FIT)

    # 4. Sigmoid envelope
    L, R, tau = estimate_constrained_sigmoid(zero_corrected_ppm, max_vals_sorted,
                                             fix_center=True, x0_fixed=0.0)
    y_sig = constrained_sigmoid(x_common, L, R, tau)
    sigmoid_env = {"L": L, "R": R, "tau": tau, "x": x_common, "y": y_sig,
                   "fit_label": f'Sigmoid (L={L:.3f}, R={R:.3f}, τ={tau:.3f})',
                   "fit_successful": True}

    # 5. Correct with sigmoid
    linspace_indices = [np.argmin(np.abs(x_common - v)) for v in zero_corrected_ppm]
    sigmoid_corrected = []
    for i, idx in enumerate(linspace_indices):
        env_val = sigmoid_env["y"][idx]
        if np.abs(env_val) < 1e-12:
            sigmoid_corrected.append(max_vals_sorted[i])
        else:
            sigmoid_corrected.append(max_vals_sorted[i] / env_val)

    # 6. Lorentzian envelope on corrected data
    A, gamma = estimate_constrained_lorentzian(zero_corrected_ppm, sigmoid_corrected)
    y_min = np.min(sigmoid_corrected)
    y_lor = constrained_lorentzian(x_common, A, gamma, y_min)
    lor_env = {"A": A, "gamma": gamma, "x": x_common, "y": y_lor,
               "fit_label": f'Lorentzian (A={A:.3f}, γ={gamma:.3f})',
               "fit_successful": True}

    # 7. Spline fit on corrected data
    spline_res = spline_fit(x=zero_corrected_ppm, y=sigmoid_corrected, x_fit=x_common)
    if not spline_res.get("fit_successful", False):
        return {"integrals": {},
                "diff_x": None, "diff_y": None,
                "sigmoidal_envelope_results": sigmoid_env,
                "lorentzian_envelope_results": lor_env,
                "spline_fit_results": spline_res}

    # 8. Difference and integrals
    diff_y = lor_env["y"] - spline_res["y_fit"]
    integrals = compute_regions_integrals(x_common, diff_y)

    return {
        "integrals": integrals,
        "diff_x": x_common,
        "diff_y": diff_y,
        "sigmoidal_envelope_results": sigmoid_env,
        "lorentzian_envelope_results": lor_env,
        "spline_fit_results": spline_res,
        # ---- intermediate data for later averaging ----
        "zero_corrected_ppm": zero_corrected_ppm,
        "sigmoid_corrected": sigmoid_corrected,
        "A": A,
        "gamma": gamma,
        "y_min": y_min,
    }

# ----------------------------------------------------------------------
# Statistiche di gruppo e p-value
# ----------------------------------------------------------------------
def _compute_group_stats(folder_keys: List[str], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    integrals_list = []
    for key in folder_keys:
        entry = analysis_results.get(key, {})
        integrals = entry.get("integrals", {})
        if integrals:
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

# ----------------------------------------------------------------------
# Differenze replicate e griglia comune
# ----------------------------------------------------------------------
def build_common_ppm_grid(
    folder_keys: List[str],
    analysis_results: Dict[str, Any],
    n_points: int = N_POINTS_FIT
) -> np.ndarray:
    """
    Build a common ppm axis covering all replicates in a group.

    The function retrieves the 'zero_corrected_ppm' array from each
    replicate's entry in analysis_results, finds the global minimum
    and maximum, and returns a linearly spaced grid with `n_points`.

    Parameters
    ----------
    folder_keys : List[str]
        Keys into `analysis_results` corresponding to the replicates
        (e.g., 'folder_name_short' for Bruker data or 'file_stem' for text files).
    analysis_results : Dict[str, Any]
        The main results dictionary. Each replicate must already contain
        the key 'zero_corrected_ppm' (list of floats).
    n_points : int, optional
        Number of points in the output grid (default = N_POINTS_FIT).

    Returns
    -------
    x_common : np.ndarray
        1D array of ppm values from the overall min to the overall max,
        equally spaced.

    Raises
    ------
    ValueError
        If any replicate lacks 'zero_corrected_ppm' or if the range is invalid.
    """
    global_min = float('inf')
    global_max = -float('inf')

    for key in folder_keys:
        entry = analysis_results.get(key, {})
        ppm_list = entry.get("zero_corrected_ppm")
        if ppm_list is None:
            raise ValueError(
                f"Missing 'zero_corrected_ppm' for replicate '{key}' in analysis_results."
            )
        # Handle both list and numpy array
        ppm_arr = np.asarray(ppm_list)
        if ppm_arr.size == 0:
            raise ValueError(f"'zero_corrected_ppm' for '{key}' is empty.")
        global_min = min(global_min, ppm_arr.min())
        global_max = max(global_max, ppm_arr.max())

    if global_min >= global_max:
        raise ValueError(
            f"Invalid ppm range: min={global_min}, max={global_max}."
        )

    return np.linspace(global_min, global_max, n_points)

def collect_replicate_differences(
    folder_keys: List[str],
    analysis_results: Dict[str, Any],
    x_common: Optional[np.ndarray] = None,
    n_points: int = N_POINTS_FIT,
    compute_integrals: bool = True,
) -> Dict[str, Any]:
    """
    Reconstruct the difference curve (Lorentzian – spline) for each replicate
    on a common ppm grid, then compute the mean and SEM.

    Parameters
    ----------
    folder_keys : List[str]
        Keys for individual replicates in analysis_results.
    analysis_results : Dict[str, Any]
        Must contain, for each key, 'zero_corrected_ppm', 'sig_corrected',
        'A', 'gamma', 'y_min'.
    x_common : np.ndarray, optional
        Pre‑built common grid. If None, built automatically via
        build_common_ppm_grid().
    n_points : int
        Used only if x_common is None.
    compute_integrals : bool
        If True, calculate integrals of the mean difference curve.

    Returns
    -------
    dict with keys:
        "x_common" : np.ndarray
        "mean_diff" : np.ndarray
        "sem_diff" : np.ndarray
        "individual_diffs" : list of np.ndarray
        "integrals" : dict (if compute_integrals=True)
    """

    if x_common is None:
        x_common = build_common_ppm_grid(folder_keys, analysis_results, n_points=n_points)

    diffs = []
    for key in folder_keys:
        entry = analysis_results[key]
        ppm = np.asarray(entry["zero_corrected_ppm"])
        sigmoid_corrected = np.asarray(entry["sigmoid_corrected"])
        A = entry["A"]
        gamma = entry["gamma"]
        y_min = entry["y_min"]

        # Lorentzian envelope
        y_lor = constrained_lorentzian(x_common, A, gamma, y_min)

        # Re‑fit spline on the replicate's corrected data
        spline = PchipInterpolator(ppm, sigmoid_corrected)
        y_spline = spline(x_common)

        diff = y_lor - y_spline
        diffs.append(diff)

    diffs_arr = np.array(diffs)  # shape (n_replicates, n_points)
    mean_diff = np.mean(diffs_arr, axis=0)
    n = len(diffs)
    if n > 1:
        sem_diff = np.std(diffs_arr, axis=0, ddof=1) / np.sqrt(n)
    else:
        sem_diff = np.zeros_like(mean_diff)

    result = {
        "x_common": x_common,
        "mean_diff": mean_diff,
        "sem_diff": sem_diff,
        "individual_diffs": [d for d in diffs],  # list of arrays
    }
    if compute_integrals:
        result["integrals"] = compute_regions_integrals(x_common, mean_diff)

    return result

# ----------------------------------------------------------------------
# Elaborazione spettri (da FID a spettro phased)
# ----------------------------------------------------------------------
def process_spectra(data: np.ndarray, dic: dict, n_exp: int, lb_Hz: float = 0.005):

    def _next_power_of_2(n):
        if n < 1:
            return 1
        return 1 << (n - 1).bit_length()

    lb = lb_Hz / dic.copy().get("acqus", {}).get("SW_h", 1.0)
    spectra: dict = {}
    for exp_idx in range(n_exp):
        fid = data[exp_idx, :]
        fid = ng.bruker.remove_digital_filter(dic, data=fid)
        fid_zf = ng.proc_base.zf_size(fid, size=_next_power_of_2(len(fid)))
        fid_apod = ng.proc_base.em(fid_zf, lb)
        spectrum = ng.proc_base.fft(fid_apod)
        spectrum_phased = ng.proc_autophase.autops(spectrum, fn="acme")
        spectrum_phased = spectrum_phased[::-1]
        spectra[exp_idx] = spectrum_phased
    return spectra

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

