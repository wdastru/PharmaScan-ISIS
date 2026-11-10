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
    Replaces the delta value at the position whose sat_trans_hz is
    closest to 0.0 with the mean of delta values corresponding to the
    smallest positive and smallest negative sat_trans_hz values.
    """
    if len(sat_trans_hz) != len(delta):
        raise ValueError("Both lists must have the same length")
    
    # Find the index of the value closest to 0.0. Exact 0.0 is not
    # guaranteed: the on-resonance frequency depends on how the
    # acquisition parameters express the offset, and floating-point
    # round-trips can leave it as e.g. 1e-15 or 1e-3.
    abs_vals = np.abs(np.asarray(sat_trans_hz, dtype=float))
    zero_idx = int(np.argmin(abs_vals))
    if abs_vals[zero_idx] > 1e-6:
        print(colored(
            f"Warning: no saturation frequency near 0 Hz "
            f"(closest is {sat_trans_hz[zero_idx]:.6f} Hz). "
            f"Using the closest point as the on-resonance reference.",
            "yellow"
        ))

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
    zero_corrected_ppm, y_raw_sorted = zip(*combined)
    zero_corrected_ppm = list(zero_corrected_ppm)
    y_raw_sorted = list(y_raw_sorted)

    # 3. Common grid
    x_fit_ppm = np.linspace(min(zero_corrected_ppm), max(zero_corrected_ppm), N_POINTS_FIT)

    # 4. Sigmoid envelope
    L, R, tau = estimate_constrained_sigmoid(zero_corrected_ppm, y_raw_sorted,
                                             fix_center=True, x0_fixed=0.0)
    y_sigmoid = constrained_sigmoid(x_fit_ppm, L, R, tau)
    sigmoid_env = {"L": L, "R": R, "tau": tau, "x": x_fit_ppm, "y": y_sigmoid,
                   "fit_label": f'Sigmoid (L={L:.3f}, R={R:.3f}, τ={tau:.3f})',
                   "fit_successful": True}

    # 5. Correct with sigmoid
    # Evaluate the sigmoid envelope EXACTLY at the experimental ppm points,
    # instead of sampling it on x_fit_ppm (200 points) and mapping each data
    # point to the nearest grid node. The previous approach introduced a
    # quantization error: when the number of saturation frequencies exceeded
    # N_POINTS_FIT, several data points shared the same envelope value, so
    # y_corrected was not exactly max_vals / sigmoid(ppm).
    #
    # Using the analytical formula at the actual data points guarantees that
    # y_corrected[i] == max_vals_sorted[i] / constrained_sigmoid(ppm_i, L, R, tau)
    # for every i, which is what a manual recomputation would produce.
    env_at_data = constrained_sigmoid(
        np.asarray(zero_corrected_ppm), L, R, tau, x0=0.0
    )
    y_corrected = [
        y_raw_sorted[i] / env_at_data[i]
        if abs(env_at_data[i]) > 1e-12
        else y_raw_sorted[i]
        for i in range(len(y_raw_sorted))
    ]

    # 6. Lorentzian envelope on corrected data
    A, gamma = estimate_constrained_lorentzian(zero_corrected_ppm, y_corrected)
    y_min = np.min(y_corrected)
    y_lorentzian = constrained_lorentzian(x_fit_ppm, A, gamma, y_min)
    lor_env = {"A": A, "gamma": gamma, "x": x_fit_ppm, "y": y_lorentzian,
               "fit_label": f'Lorentzian (A={A:.3f}, γ={gamma:.3f})',
               "fit_successful": True}

    # 7. Spline fit on corrected data
    spline_res = spline_fit(x=zero_corrected_ppm, y=y_corrected, x_fit=x_fit_ppm)
    if not spline_res.get("fit_successful", False):
        return {"integrals": {},
                "diff_x": None, "diff_y": None,
                "sigmoidal_envelope_results": sigmoid_env,
                "lorentzian_envelope_results": lor_env,
                "spline_fit_results": spline_res}

    # 8. Difference and integrals
    y_diff = lor_env["y"] - spline_res["y_fit"]
    integrals = compute_regions_integrals(x_fit_ppm, y_diff)

    return {
        "integrals": integrals,
        "diff_x": x_fit_ppm,
        "diff_y": y_diff,
        "sigmoidal_envelope_results": sigmoid_env,
        "lorentzian_envelope_results": lor_env,
        "spline_fit_results": spline_res,
        # ---- intermediate data for later averaging ----
        "zero_corrected_ppm": zero_corrected_ppm,
        "sigmoid_corrected": y_corrected,
        "A": A,
        "gamma": gamma,
        "y_min": y_min,
    }

# ----------------------------------------------------------------------
# Statistiche di gruppo e p-value
# ----------------------------------------------------------------------
def _compute_group_stats(experiments: Dict[str, Any]) -> Dict[str, Any]:
    """Statistiche (mean/std) degli integrali su tutti gli esperimenti di un gruppo."""
    integrals_list = [
        exp["integrals"]
        for exp in experiments.values()
        if exp.get("integrals")
    ]
    if not integrals_list:
        return {"mean": {}, "std": {}}
    regions = list(integrals_list[0].keys())
    mean_dict, std_dict = {}, {}
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
    return {"integrals_mean": mean_dict, "integrals_std": std_dict}

def _compute_pvalues(
    ref_experiments: Dict[str, Any],
    sample_experiments: Dict[str, Any],
    test: str = 't-test',
) -> Dict[str, Optional[float]]:
    ref_integrals = [
        exp["integrals"] for exp in ref_experiments.values() if exp.get("integrals")
    ]
    sample_integrals = [
        exp["integrals"] for exp in sample_experiments.values() if exp.get("integrals")
    ]
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
    experiments: Dict[str, Any],
    n_points: int = N_POINTS_FIT,
) -> np.ndarray:
    """
    Build a common ppm axis covering all experiments in a group.

    Parameters
    ----------
    experiments : Dict[str, Any]
        Dizionario degli esperimenti del gruppo. Ogni esperimento deve
        contenere la chiave 'zero_corrected_ppm' (lista di float).
    n_points : int, optional
        Numero di punti della griglia (default = N_POINTS_FIT).

    Returns
    -------
    x_fit_ppm : np.ndarray
        1D array di ppm da min globale a max globale, equispaziato.

    Raises
    ------
    ValueError
        Se un esperimento manca di 'zero_corrected_ppm' o se il range è invalido.
    """

    global_min = float('inf')
    global_max = -float('inf')
    for key, entry in experiments.items():
        ppm_list = entry.get("zero_corrected_ppm")
        if ppm_list is None:
            raise ValueError(f"Missing 'zero_corrected_ppm' for experiment '{key}'.")
        ppm_arr = np.asarray(ppm_list)
        if ppm_arr.size == 0:
            raise ValueError(f"'zero_corrected_ppm' for '{key}' is empty.")
        global_min = min(global_min, ppm_arr.min())
        global_max = max(global_max, ppm_arr.max())
    if global_min >= global_max:
        raise ValueError(f"Invalid ppm range: min={global_min}, max={global_max}.")
    
    return np.linspace(global_min, global_max, n_points)

def collect_replicate_differences(
    experiments: Dict[str, Any],
    x_fit_ppm: Optional[np.ndarray] = None,
    n_points: int = N_POINTS_FIT,
    compute_integrals: bool = True,
) -> Dict[str, Any]:
    """
    Reconstruct the difference curve (Lorentzian – spline) for each replicate
    on a common ppm grid, then compute the mean and SEM.

    Also returns the mean ± SEM of the Lorentzian envelope, of the spline fit,
    and of the sigmoid-corrected z-spectrum, all sampled on the same x_fit_ppm
    grid. These are the quantities to be used when plotting group-level
    z-spectra and fits, so that everything stays on the corrected scale.

    Returns
    -------
    dict with keys:
        "x_fit_ppm" : np.ndarray
        "y_diff_mean", "sem_diff" : np.ndarray
        "mean_lorentzian", "mean_spline" : np.ndarray
        "mean_corrected", "sem_corrected" : np.ndarray
        "individual_diffs" : list of np.ndarray
        "integrals" : dict (if compute_integrals=True)
    """

    if x_fit_ppm is None:
        x_fit_ppm = build_common_ppm_grid(experiments, n_points=n_points)

    y_diff_common, y_lorentzian_common, y_spline_common, y_corrected_common = [], [], [], []
    for key, entry in experiments.items():
        ppm = np.asarray(entry["zero_corrected_ppm"])
        sig = np.asarray(entry["sigmoid_corrected"])
        A, gamma, y_min = entry["A"], entry["gamma"], entry["y_min"]

        y_corrected_common.append(PchipInterpolator(ppm, sig)(x_fit_ppm))
        y_lorentzian_common.append(constrained_lorentzian(x_fit_ppm, A, gamma, y_min))
        y_spline_common.append(PchipInterpolator(ppm, sig)(x_fit_ppm))
        y_diff_common.append(y_lorentzian_common[-1] - y_spline_common[-1])

    y_diff_arr      = np.array(y_diff_common)
    y_lorentzian_arr = np.array(y_lorentzian_common)
    y_spline_arr    = np.array(y_spline_common)
    y_corrected_arr  = np.array(y_corrected_common)
    n = len(y_diff_common)

    def _mean_sem(a):
        m = np.mean(a, axis=0)
        s = np.std(a, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(m)
        return m, s

    y_diff_mean,       y_diff_sem       = _mean_sem(y_diff_arr)
    y_lorentzian_mean, _              = _mean_sem(y_lorentzian_arr)
    y_spline_mean,     _              = _mean_sem(y_spline_arr)
    mean_corrected,  sem_corrected  = _mean_sem(y_corrected_arr)

    # punti dati originali (media sui replicati)
    first_key = next(iter(experiments))
    x_zero_corrected_ppm = np.asarray(experiments[first_key]["zero_corrected_ppm"])
    y_corrected_at_data = []
    for entry in experiments.values():
        ppm = np.asarray(entry["zero_corrected_ppm"])
        sc  = np.asarray(entry["sigmoid_corrected"])
        y_corrected_at_data.append(sc if np.array_equal(ppm, x_zero_corrected_ppm)
                                 else np.interp(x_zero_corrected_ppm, ppm, sc))
    y_corrected_mean, y_corrected_sem = _mean_sem(np.array(y_corrected_at_data))

    result = {
        "fit_x":                    x_fit_ppm,
        "y_diff_mean":              y_diff_mean,
        "y_diff_sem":               y_diff_sem,
        "y_lorentzian_mean":        y_lorentzian_mean,
        "y_spline_mean":            y_spline_mean,
        "y_corrected_mean":         mean_corrected,
        "y_corrected_sem":          sem_corrected,
        "x_data":                   x_zero_corrected_ppm,
        "y_corrected_at_data_mean": y_corrected_mean,
        "y_corrected_at_data_sem":  y_corrected_sem,
    }
    if compute_integrals:
        result["integrals_of_mean_curve"] = compute_regions_integrals(x_fit_ppm, y_diff_mean)

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

