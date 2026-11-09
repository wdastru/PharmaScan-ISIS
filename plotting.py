"""
plotting.py
Tutte le funzioni di visualizzazione grafica.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, Button
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from typing import List, Optional, Dict, Any
from termcolor import colored

from constants import get_default_visibility
from config import METABOLITE_REGIONS

def group_color(group, all_groups):
    """Colore coerente per ogni gruppo.
       Reference = nero. Poi group 1 = blu, group 2 = arancione, poi verde, rosso, viola, marrone."""
    if group.get("is_reference", False):
        return "black"
    # trova la posizione di questo gruppo tra i NON-reference
    non_refs = [g for g in all_groups if not g.get("is_reference", False)]
    idx = 0
    for k, g in enumerate(non_refs):
        if g is group or g.get("label") == group.get("label"):
            idx = k
            break
    palette = ["blue", "orange", "green", "red", "purple", "brown"]
    return palette[idx % len(palette)]

def plot_z_spectrum(
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
    window_title: Optional[str] = None,
    color: Optional[str] = None,
) -> Figure:
    # Se non fornito, usa i default globali
    if visibility is None:
        visibility = get_default_visibility()
            # Colore del gruppo: se fornito lo uso per dati e spline,
    # altrimenti mantengo i vecchi colori (dati blu, spline rossa)
    data_color = color if color is not None else "b"
    spline_color = color if color is not None else "r"
    
    fig = plt.figure(num=window_title, figsize=(8, 5))

    # Data points
    if visibility.get("data", True):
        if y_std_data is not None:
            plt.errorbar(x, y, yerr=np.array(y_std_data), fmt='o', color=data_color, label='Data')
        else:
            plt.plot(x, y, 'o', color=data_color, label='Data')
                
    # Spline fit
    if visibility.get("spline", True):
        plt.plot(x_fit, y_fit, '-', color=spline_color, label=fit_label)

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
        for idx, (name, region) in enumerate(METABOLITE_REGIONS.items()):
            if region["ppm"] is not None and len(region["ppm"]) == 2:
                start, end = region["ppm"]
                ax.axvspan(start, end, facecolor=colors[idx], alpha=0.25, edgecolor='none', label=name)
            else:
                # Do not terminate the whole application from a plotting
                # helper: skip the malformed region and keep drawing the rest.
                print(colored(
                    f"Warning: region '{name}' is not defined properly; skipping.",
                    "yellow"
                ))
                continue
    
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

def plot_spectra(title, spectra, n_exp, ppm_axis, sat_trans_hz, visibility=None, window_title=None) -> Figure:
    fig, ax = plt.subplots(num=window_title, figsize=(12, 6))
    lines:List[Line2D] = []
    labels = []
    for exp_idx in range(n_exp):
        line: Line2D = ax.plot(ppm_axis, np.real(spectra[exp_idx]),
                        label=f"{exp_idx:>2} : {sat_trans_hz[exp_idx]:.2f}",
                        alpha=0.7, linewidth=1.2)[0]
        lines.append(line)
        labels.append(line.get_label())
    ax.invert_xaxis()
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    rax = fig.add_axes([0.80, 0.15, 0.19, 0.70])
    visibility_states: List = [line.get_visible() for line in lines]
    checks = CheckButtons(rax, labels, visibility_states)
    _silent_update = False   # flag to block callback during programmatic changes

    def _on_check(label):
        nonlocal _silent_update          # needed because we're in a nested function
        if _silent_update:
            return                        # skip if we're doing a programmatic update
        idx = labels.index(label)
        lines[idx].set_visible(not lines[idx].get_visible())
        fig.canvas.draw_idle()

    def _check_all(event):
        nonlocal _silent_update
        _silent_update = True             # block callbacks
        for i, line in enumerate(lines):
            if not line.get_visible():
                line.set_visible(True)
                checks.set_active(i, True)
        _silent_update = False            # re-enable
        fig.canvas.draw_idle()

    def _uncheck_all(event):
        nonlocal _silent_update
        _silent_update = True
        for i, line in enumerate(lines):
            if line.get_visible():
                line.set_visible(False)
                checks.set_active(i, False)
        _silent_update = False
        fig.canvas.draw_idle()

    checks.on_clicked(_on_check)
    ax_all = fig.add_axes([0.80, 0.90, 0.09, 0.05])
    btn_all = Button(ax_all, "Check all")
    btn_all.on_clicked(_check_all)
    ax_none = fig.add_axes([0.90, 0.90, 0.09, 0.05])
    btn_none = Button(ax_none, "Uncheck all")
    btn_none.on_clicked(_uncheck_all)
    fig.tight_layout(rect=[0, 0, 0.80, 1])
    # Keep strong references to the widgets on the figure itself.
    # Without this, `checks`, `btn_all`, and `btn_none` are only referenced
    # by local variables; once this function returns they get garbage
    # collected, their callbacks are disconnected, and the checkboxes/
    # buttons stop responding even though they're still visible on screen.
    fig._widgets = (checks, btn_all, btn_none)
    plt.show(block=False)
    return fig

def plot_groups_comparison(group_stats, p_values, groups,
                              title="Groups comparison",
                              ylabel="Integral (mean ± SD)",
                              figsize=(12, 6),
                              visibility=None,
                              window_title=None) -> Figure:
    if visibility is None:
        visibility = get_default_visibility()
    if not group_stats:
        return None
    first_label = groups[0]["label"]
    regions = list(group_stats[first_label]["mean"].keys())
    n_regions = len(regions)
    n_groups = len(groups)
    x = np.arange(n_regions)
    bar_width = 0.8 / n_groups
    fig, ax = plt.subplots(num=window_title, figsize=figsize)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n_groups)]
    for i, grp in enumerate(groups):
        label = grp["label"]
        means = [group_stats[label]["mean"][reg] for reg in regions]
        stds  = [group_stats[label]["std"][reg] for reg in regions]
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
                # Iterate from the most stringent threshold (0.001 -> '***')
                # to the least (0.05 -> '*'). The FIRST match is the correct
                # significance label. Without `reverse=False` (the default),
                # any p < 0.05 would be labelled '*' and '**'/'***' would
                # never appear.
                for thr in sorted(significance):     # [0.001, 0.01, 0.05]
                    if p < thr:
                        txt = significance[thr]
                        break
                if txt is None:
                    # TODO: improve visibility of p values
                    #txt = f"p={p:.3f}"
                    pass
                ref_idx = next(k for k, g in enumerate(groups) if g["label"] == ref_label)
                y_ref = group_stats[ref_label]["mean"][reg] + group_stats[ref_label]["std"][reg]
                y_this = group_stats[grp["label"]]["mean"][reg] + group_stats[grp["label"]]["std"][reg]
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

def plot_group_breakdown(group_label, group_stats, per_folder_integrals,
                                folder_names=None,
                                title=None, ylabel="Integrale",
                                figsize=(12, 6), visibility=None,
                                window_title=None) -> Figure:
    """
    Bar chart per un singolo gruppo: barre affiancate per ogni cartella
    (con colori e nomi) e barra della media ± SD.
    """
    if visibility is None:
        visibility = get_default_visibility()

    stats = group_stats.get(group_label)
    folder_vals = per_folder_integrals.get(group_label, {})

    if not stats or not folder_vals:
        print(f"Dati insufficienti per il gruppo '{group_label}'")
        return None

    regions = list(stats["mean"].keys())
    means = [stats["mean"][r] for r in regions]
    stds  = [stats["std"][r] for r in regions]

    first_region = regions[0]
    n_folders = len(folder_vals.get(first_region, []))
    if n_folders == 0:
        print(f"Nessun integrale per cartella nel gruppo '{group_label}'")
        return None

    # Usa i nomi passati o genera nomi di default
    if folder_names is None or len(folder_names) != n_folders:
        folder_names = [f"Cartella {i+1}" for i in range(n_folders)]

    # Spazio totale e calcolo larghezze (come prima)
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
    avg_bar_width = bar_width  # mantieni stessa larghezza per allineamento esatto

    total_width = (n_folders * bar_width +
                   n_folders * gap_between_bars +
                   extra_gap_before_avg +
                   avg_bar_width)

    x = np.arange(len(regions))

    fig, ax = plt.subplots(num=window_title, figsize=figsize)

    # Colormap per le cartelle
    cmap = plt.get_cmap('tab10')
    folder_colors = [cmap(i % 10) for i in range(n_folders)]
    color_avg = '#2c3e50'  # blu scuro per la media

    for j, reg in enumerate(regions):
        start_x = x[j] - total_width / 2
        vals = folder_vals.get(reg, [])

        # Barre delle cartelle
        for k, val in enumerate(vals):
            pos = start_x + k * (bar_width + gap_between_bars) + bar_width/2
            ax.bar(pos, val, bar_width,
                   color=folder_colors[k], edgecolor='black', linewidth=0.5,
                   label=folder_names[k] if j == 0 else "")

        # Barra della media
        pos_avg = start_x + n_folders * (bar_width + gap_between_bars) + extra_gap_before_avg + avg_bar_width/2
        ax.bar(pos_avg, means[j], avg_bar_width,
               yerr=stds[j], capsize=4,
               color=color_avg, edgecolor='black', linewidth=0.8,
               label='Media ± SD' if j == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    if title is None:
        title = f"{group_label} group breakdown"
    ax.set_title(title)
    if visibility["legend"].get("integrals", True):
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plt.show(block=False)
    return fig

def plot_group_difference(
    group_label: str,
    group_data: Dict[str, Any],
    visibility: Optional[Dict[str, bool]] = None,
    window_title: Optional[str] = None,
) -> Figure:
    """
    Plot the group-averaged z-spectrum, Lorentzian envelope, spline fit,
    and mean difference curve ± SEM, all on the common ppm grid.

    Data points default to the sigmoid-corrected z-spectrum (mean ± SEM).
    Set visibility["corrected"] = False to fall back to raw mean (uncorrected).
    """
    if visibility is None:
        visibility = get_default_visibility()

    fig, ax = plt.subplots(num=window_title, figsize=(8, 5))

    # X-axis for all corrected quantities: the common grid.
    x_common = np.array(group_data["x_common"])

    # --- Data points ---
    if visibility.get("data", True):

        use_corrected = (
            visibility.get("corrected", True)
            and "mean_corrected_at_data" in group_data
            and "sem_corrected_at_data" in group_data
        )
        if use_corrected:
            x_pts = np.array(group_data["x_data"])
            y_pts = np.array(group_data["mean_corrected_at_data"])
            yerr_pts = np.array(group_data["sem_corrected_at_data"])
            ax.errorbar(x_pts, y_pts, yerr=yerr_pts, fmt='o', color='b',
                        markersize=5,
                        label='Z-spectrum corrected (mean ± SEM)')
        elif "sat_trans_hz" in group_data and "max_vals" in group_data:
            # Fallback: raw mean z-spectrum, uncorrected.
            raw_ppm = np.array(group_data["sat_trans_hz"]) / group_data["bf1"]
            raw_vals = np.array(group_data["max_vals"])
            raw_sd = np.array(group_data.get("sd_max_vals", [0] * len(raw_vals)))
            ax.errorbar(raw_ppm, raw_vals, yerr=raw_sd, fmt='o', color='b',
                        label='Raw mean (uncorrected)')

    # --- Spline fit (mean of per-replicate splines, on corrected data) ---
    if visibility.get("spline", True):
        sp = group_data.get("spline_fit_results", {})
        if sp.get("fit_successful") and sp.get("y_fit") is not None:
            ax.plot(sp["x_fit"], sp["y_fit"], 'r-', linewidth=1.5,
                    label=sp.get("fit_label", "Spline fit"))

    # --- Lorentzian envelope (mean of per-replicate envelopes) ---
    if visibility.get("lorentzian", True):
        lor = group_data.get("lorentzian_envelope_results")
        if lor and lor.get("y") is not None:
            ax.plot(lor["x"], lor["y"], 'g--', linewidth=2,
                    label=lor.get("fit_label", "Lorentzian envelope"))

    # --- Mean difference curve ± SEM ---
    mean_diff = np.array(group_data["mean_diff_y"])
    sem_diff = np.array(group_data["sem_diff_y"])

    if visibility.get("difference", True):
        ax.plot(x_common, mean_diff, 'm-', linewidth=1.5,
                label='Mean diff (Lor−Spline)')
        ax.fill_between(x_common, mean_diff - sem_diff, mean_diff + sem_diff,
                        color='m', alpha=0.2, label='± SEM')

    # --- Metabolite regions ---
    if visibility.get("regions", True):
        cmap = plt.get_cmap('tab10')
        for idx, (name, region) in enumerate(METABOLITE_REGIONS.items()):
            if region["ppm"] is not None and len(region["ppm"]) == 2:
                start, end = region["ppm"]
                ax.axvspan(start, end, facecolor=cmap(idx % 10),
                           alpha=0.25, edgecolor='none', label=name)
            else:
                # Same as plot_z_spectrum: do not abort the app from a
                # plotting helper. Skip the malformed region.
                print(colored(
                    f"Warning: region '{name}' is not defined properly; skipping.",
                    "yellow"
                ))
                continue

    ax.invert_xaxis()
    ax.set_xlabel("Saturation ppm")
    ax.set_ylabel("Intensity / Difference")
    ax.set_title(f"Group {group_label} – Averaged Difference")
    ax.grid(True)
    if visibility["legend"].get("z-spectra", True):
        ax.legend()
    plt.show(block=False)
    return fig

def plot_all_z_spectra(
    groups,
    analysis_results,
    folder_keys_per_group,
    visibility=None,
    title=None,
    window_title=None,
):
    """
    Overlay of the group-level 'Averaged Difference' panels.

    For every group, replicates the same visual elements as
    plot_group_difference -- experimental corrected points ± SEM, mean
    spline, mean Lorentzian, mean difference ± SEM -- on a single shared
    axis, each group drawn in its own color.

    Two legends are shown:
      - main legend: one entry per group (color)
      - style legend: one entry per visual element (points, spline,
        Lorentzian, mean difference)

    Parameters
    ----------
    groups, analysis_results :
        Same semantics as in plot_group_difference.
    folder_keys_per_group :
        Unused; kept only for backward compatibility with the caller.
    visibility : dict, optional
        Toggles: "data", "spline", "lorentzian", "difference", "regions",
        and "legend" -> "z-spectra".
    """
    if visibility is None:
        visibility = get_default_visibility()

    show_data       = visibility.get("data", True)
    show_spline     = visibility.get("spline", True)
    show_lorentzian = visibility.get("lorentzian", True)
    show_diff       = visibility.get("difference", True)
    show_regions    = visibility.get("regions", True)
    show_legend     = visibility.get("legend", {}).get("z-spectra", True)

    fig, ax = plt.subplots(figsize=(10, 6), num=window_title)
    cmap = plt.get_cmap("tab10")

    for grp in groups:
        label = grp["label"]
        color = group_color(grp, groups)
        grp_data = analysis_results.get(label, {})
        if not grp_data:
            continue

        x_common = grp_data.get("x_common")
        if x_common is None:
            continue
        x_common = np.asarray(x_common)

        # --- 1) Experimental corrected points ± SEM ---
        x_pts = grp_data.get("x_data")
        y_pts = grp_data.get("mean_corrected_at_data")
        y_err = grp_data.get("sem_corrected_at_data")
        if show_data and x_pts is not None and y_pts is not None:
            ax.errorbar(
                np.asarray(x_pts), np.asarray(y_pts),
                yerr=np.asarray(y_err) if y_err is not None else None,
                fmt='o', color=color, markersize=4, alpha=0.5,
                capsize=2, elinewidth=0.8,
                # No label: style entries live in the style legend below
            )

        # --- 2) Spline fit (mean of replicates) ---
        #     The spline carries the per-group label so it populates
        #     the MAIN legend (one entry per group).
        sp = grp_data.get("spline_fit_results", {})
        if show_spline and sp.get("fit_successful") and sp.get("y_fit") is not None:
            ax.plot(sp["x_fit"], sp["y_fit"], "-",
                    color=color, linewidth=2.0, label=label)

        # --- 3) Lorentzian envelope (mean of replicates) ---
        lor = grp_data.get("lorentzian_envelope_results")
        if show_lorentzian and lor and lor.get("y") is not None:
            ax.plot(lor["x"], lor["y"], "--",
                    color=color, linewidth=1.5, alpha=0.8)

        # --- 4) Mean difference curve ± SEM ---
        mean_diff = grp_data.get("mean_diff_y")
        sem_diff  = grp_data.get("sem_diff_y")
        if show_diff and mean_diff is not None:
            mean_diff_arr = np.asarray(mean_diff)
            ax.plot(x_common, mean_diff_arr, ":",
                    color=color, linewidth=2.0)
            if sem_diff is not None:
                sem_diff_arr = np.asarray(sem_diff)
                ax.fill_between(
                    x_common,
                    mean_diff_arr - sem_diff_arr,
                    mean_diff_arr + sem_diff_arr,
                    color=color, alpha=0.10, linewidth=0,
                )

    # --- Metabolite regions ---
    if show_regions:
        for idx, (name, region) in enumerate(METABOLITE_REGIONS.items()):
            if region.get("ppm") and len(region["ppm"]) == 2:
                start, end = region["ppm"]
                ax.axvspan(start, end, facecolor=cmap(idx % 10),
                           alpha=0.15, edgecolor="none")

    ax.invert_xaxis()
    ax.set_xlabel("Saturation ppm")
    ax.set_ylabel("Intensity / Difference")
    ax.set_title(title or "All groups – Averaged Difference")
    ax.grid(True, alpha=0.3)

    if show_legend:
        # MAIN legend: one entry per group (via the spline lines)
        leg_main = ax.legend(loc="upper right", title="Groups")
        ax.add_artist(leg_main)   # keep it when we build the second legend

        # STYLE legend: one entry per visual element (gray, no group color)
        style_handles = [
            Line2D([0], [0], color="k", marker="o", linestyle="",
                   markersize=6, alpha=0.6,
                   label="Experimental (mean ± SEM)"),
            Line2D([0], [0], color="k", linewidth=2.0,
                   label="Spline (mean)"),
            Line2D([0], [0], color="k", linewidth=1.5, linestyle="--",
                   label="Lorentzian (mean)"),
            Line2D([0], [0], color="k", linewidth=2.0, linestyle=":",
                   label="Mean diff (Lor − Spline)"),
        ]
        ax.legend(handles=style_handles, loc="lower left", title="Style")

    fig.tight_layout()
    plt.show(block=False)
    return fig