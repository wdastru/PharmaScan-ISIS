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
import re
import tkinter as tk
from tkinter import filedialog
from typing import List, Optional, Tuple, Dict, Any

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def findMaxima(arr: np.ndarray,
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
    print(f"Found {len(vals)} numerical values after for {PARAMETER}")

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

def main() -> None:
    """
    Main routine of the script.

    - Prompts the user to select a Bruker experiment folder.
    - Reads the 'method' file to obtain saturation frequencies and offset.
    - Loads and processes all FIDs (digital filter removal, zero‑filling,
      line broadening, FFT, automatic phase correction, axis inversion).
    - Displays all spectra with an interactive checkbox panel.
    - Asks the user for a ppm range of interest.
    - Finds the maximum intensity within that range for each spectrum.
    - Corrects the saturation frequencies using the frequency offset.
    - Plots the maximum values against the corrected saturation ppm.
    """
    # ---- Select experiment folder ----
    root = tk.Tk()
    root.withdraw()                     # hide the main Tk window
    full = Path(filedialog.askdirectory(title="Select a folder"))
    method = full / "method"

    # ---- Extract parameters from method file ----
    sat_trans_fl = parameter_extract(method, "PVM_SatTransFL")   # Hz
    frq_work_offset_hz = parameter_extract(method, "PVM_FrqWorkOffset")  # Hz

    # ---- Read Bruker data ----
    dic, data = ng.bruker.read(full)
    udic = ng.bruker.guess_udic(dic, data)
    uc = ng.fileio.bruker.fileiobase.uc_from_udic(udic, dim=1)
    ppm_axis = uc.ppm_scale()
    n_exp = dic["acqu2s"]["TD"]
    bf1 = dic["acqus"]["BF1"]

    # ---- Process spectra and create plot ----
    fig, ax = plt.subplots(figsize=(12, 6))
    lines: List[plt.Line2D] = []
    labels: List[str] = []
    loaded: Dict[int, np.ndarray] = {}

    for i in range(n_exp):
        # ---- Select experiment folder
        fid = data[i, :]

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
        loaded[i] = spectrum_phased

        # Plot the real part
        line, = ax.plot(
            ppm_axis,
            np.real(spectrum_phased),
            label=f"{sat_trans_fl[i]:.2f}",
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
    def on_check(label: str) -> None:
        idx = labels.index(label)
        lines[idx].set_visible(not lines[idx].get_visible())
        fig.canvas.draw_idle()

    def check_all(event: Any) -> None:
        for i, l in enumerate(lines):
            if not l.get_visible():
                checks.set_active(i)
        plt.draw()

    def uncheck_all(event: Any) -> None:
        for i, l in enumerate(lines):
            if l.get_visible():
                checks.set_active(i)
        plt.draw()

    # ---- CheckButtons panel ----
    rax = fig.add_axes([0.80, 0.15, 0.19, 0.70])   # [left, bottom, width, height]
    visibility = [l.get_visible() for l in lines]
    checks = CheckButtons(rax, labels, visibility)
    checks.on_clicked(on_check)

    # "Check all" button
    ax_all = fig.add_axes([0.80, 0.90, 0.09, 0.05])
    btn_all = Button(ax_all, "Check all")
    btn_all.on_clicked(check_all)

    # "Uncheck all" button
    ax_none = fig.add_axes([0.90, 0.90, 0.09, 0.05])
    btn_none = Button(ax_none, "Uncheck all")
    btn_none.on_clicked(uncheck_all)

    fig.tight_layout(rect=[0, 0, 0.80, 1])       # leave space on the right for widgets

    # Show the figure (non-blocking)
    plt.show(block=False)

    # ---- User input for ppm range ----
    try:
        start_ppm = float(input("Enter the minimum ppm (start): "))
        end_ppm = float(input("Enter the maximum ppm (end): "))

        end_idx = ppm_to_index(uc=uc, user_ppm=start_ppm)
        start_idx = ppm_to_index(uc=uc, user_ppm=end_ppm)

        if start_idx < 0 or end_idx <= start_idx:
            raise ValueError("Invalid range: end must be greater than start and both non-negative.")

        print(f"Selected range: [{start_idx}, {end_idx}]")  # end-exclusive
    except ValueError as e:
        print(f"Error: {e}")
        return

    # ---- Find maxima in the selected range ----
    max_vals: Dict[int, float] = {}
    max_indexes: Dict[int, int] = {}
    for p in loaded:
        val, idx = findMaxima(loaded[p], start=start_idx, end=end_idx)
        max_vals[p] = val
        max_indexes[p] = idx

    # ---- Correct saturation frequencies and final plot ----
    if len(sat_trans_fl) == len(max_vals):
        sat_trans_f1_ppm: List[float] = [0.0] * len(sat_trans_fl)
        for p in max_vals:
            # Calculate the offset from the reference            
            delta = frq_work_offset_hz[0] - uc.hz(max_indexes[p])
            sat_trans_fl[p] += delta
            sat_trans_f1_ppm[p] = sat_trans_fl[p] / bf1   # convert to ppm
            print(f"{sat_trans_f1_ppm[p]}\t{max_vals[p]}")

        # Prepare data for fitting
        x_vals = np.array(sat_trans_f1_ppm)
        y_vals = np.array(list(max_vals.values()))

        # Sort by x (important for plotting the curve)
        sort_idx = np.argsort(x_vals)
        x_sorted = x_vals[sort_idx]
        y_sorted = y_vals[sort_idx]

        # ---- Attempt spline interpolation / smoothing (requires scipy) ----
        fit_successful = False
        try:
            from scipy.interpolate import UnivariateSpline

            # ---------- ADJUST SMOOTHING HERE ----------
            # smoothing = 0   -> exact interpolation (passes through all points)
            # smoothing > 0   -> smoothing spline; larger value = smoother curve
            smoothing = 0.0   # <-- change this value to tune smoothness
            # -------------------------------------------

            spline = UnivariateSpline(x_sorted, y_sorted, s=smoothing)
            fit_successful = True

            # Generate smooth curve for plotting
            x_fit = np.linspace(x_sorted.min(), x_sorted.max(), 200)
            y_fit = spline(x_fit)

            print(f"\nSpline fit (smoothing factor = {smoothing}) completed.")

        except ImportError:
            print("\nscipy not installed – cannot perform spline fit.")
        except Exception as e:
            print(f"\nSpline fit failed: {e}")

        # ---- Final plot with data and fitted curve ----
        plt.figure(figsize=(8, 5))
        plt.plot(x_sorted, y_sorted, 'o', color='b', label='Data')

        if fit_successful or ('y_fit' in locals() and y_fit is not None):
            plt.plot(x_fit, y_fit, 'r-', label='Spline')

        plt.gca().invert_xaxis()
        plt.title("Max Values vs Saturation ppm")
        plt.xlabel("Saturation ppm")
        plt.ylabel("Max Value")
        plt.grid(True)
        if fit_successful or ('y_fit' in locals() and y_fit is not None):
            plt.legend()
        plt.show(block=True)

    else:
        print("Number of saturation frequencies does not match number of processed series; skipping plot.")

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()    