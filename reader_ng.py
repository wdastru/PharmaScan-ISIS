"""
reader_ng.py

Processing of Bruker NMR data for saturation transfer experiments.
Reads FIDs, applies corrections, displays spectra with interactive checkboxes,
allows the user to select a ppm range, and calculates maxima to generate
a saturation transfer curve.

Supports multiple groups (reference + any number of sample groups).
"""

import copy
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from termcolor import colored

from constants import (
    CACHE_DIR, get_default_visibility
)

from export import ( 
    save_analysis_results, SafeEncoder
)

from config import (
    METABOLITE_REGIONS, _resolve_group_source, save_config, select_or_create_config,
    ensure_complete_config,
)

from plotting import (
    plot_group_difference, plot_group_breakdown, plot_spectra,
        plot_groups_comparison, plot_z_spectrum, plot_all_z_spectra, group_color
)

from cache import (
    load_cache, save_cache
)

from data_io import (
    extract_parameters, load_spectra
)

from processing import (
    process_spectra, ppm_to_index, find_max_vals, normalize_max_vals,
    correct_sat_frequencies, process_zspectrum_and_integrals,
    collect_replicate_differences, _compute_group_stats, _compute_pvalues
)

from utils import (
    ask_user_for_ppm_range, ask_yes_no, get_git_hash
)

from theme import apply_theme

print(f"Using nmrglue version: {ng.__version__}")

CACHE_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Core analysis routine
# ----------------------------------------------------------------------
def run_analysis(config_name: str, config: Dict[str, Any]) -> None:
    METABOLITE_REGIONS.clear()
    METABOLITE_REGIONS.update(copy.deepcopy(config["metabolite_regions"]))
    plt.ion()

    groups = config["groups"]
    lb_Hz = config["lb_Hz"]
    start_ppm = config.get("start_ppm")
    end_ppm = config.get("end_ppm")
    ppm_missing = config.get("ppm_missing", False)

    # Struttura di lavoro annidata: gruppo -> esperimenti + summary + p_values
    groups_data: Dict[str, Dict[str, Any]] = {
        grp["label"]: {
            "is_reference": grp.get("is_reference", False),
            "experiments":  {},
            "summary":      None,
            "p_values":     None,
        }
        for grp in groups
    }
    # Metadati per-gruppo (solo per i controlli di consistenza di work_offset_hz)
    group_meta: Dict[str, Dict[str, Any]] = {grp["label"]: {} for grp in groups}

    # Contenitore finale, assemblato alla fine del ramo live
    analysis_results: dict = {}

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    cached = load_cache(config_name, config)
    use_cache = False
    if cached:
        use_cache = ask_yes_no(
            f"Cache valida trovata per '{config_name}'. Vuoi evitare il ricalcolo?",
            default=True,
        )

    if cached and use_cache:
        analysis_results = cached
        cached_groups = analysis_results.get("groups", {})

        # 1. Group-difference plots
        for label, gd in cached_groups.items():
            if gd.get("summary"):
                plot_group_difference(
                    group_label=label,
                    group_data=gd["summary"],
                    visibility=config.get("plot_visibility", get_default_visibility()),
                    window_title=f"Group {label} – Averaged Difference (da cache)",
                    color=group_color(label=label, all_groups=groups)
                )

        # 2. Z-spectra per esperimento
        for label, gd in cached_groups.items():
            for key, res in gd.get("experiments", {}).items():
                if res.get("spline_fit_results", {}).get("fit_successful"):
                    plot_z_spectrum(
                        x=res["zero_corrected_ppm"],
                        y=res["sigmoid_corrected"],
                        x_fit=res["fit_x"],
                        y_fit=res["spline_fit_results"]["y_fit"],
                        title=f"Spline: {key}",
                        invert_x=True,
                        add_lorentz=True,
                        lorentzian_envelope_results=res.get("lorentzian_envelope_results"),
                        add_sigmoid=True,
                        sigmoidal_envelope_results=res.get("sigmoidal_envelope_results"),
                        diff_x=res["fit_x"],
                        diff_y=res["diff_y"],
                        diff_label="Lorentzian envelope - Spline fit",
                        visibility=config.get("plot_visibility", get_default_visibility()),
                        color=group_color(label=label, all_groups=groups),
                        window_title=f"Spline fit: {key} (da cache)"
                    )

        # 3. Recupera group_stats (top-level)
        group_stats = analysis_results.get("group_stats", {})

        # 4. Barre per gruppo
        for label, gd in cached_groups.items():
            if not gd.get("experiments"):
                continue
            per_folder = {}
            for exp in gd["experiments"].values():
                for region, val in exp.get("integrals", {}).items():
                    per_folder.setdefault(region, []).append(val)
            plot_group_breakdown(
                group_label=label,
                group_stats=group_stats,
                per_folder_integrals={label: per_folder},
                folder_names=list(gd["experiments"].keys()),
                visibility=config.get("plot_visibility", get_default_visibility()),
                title=f"{label} group breakdown (da cache)",
                window_title=f"{label} group breakdown (da cache)",
            )

        # 5. Grafico multi-gruppo con p-values
        p_values_for_plot = {
            lbl: gd["p_values"]
            for lbl, gd in cached_groups.items()
            if not gd["is_reference"] and gd.get("p_values")
        }
        plot_groups_comparison(
            group_stats, p_values_for_plot, groups,
            visibility=config.get("plot_visibility", get_default_visibility()),
            title="Groups comparison (da cache)",
            window_title="Groups comparison (da cache)",
        )

        # ------------------------------------------------------------
        # 6. Tutti gli Z-spectra raggruppati (da cache)
        # ------------------------------------------------------------
        title_zs = "All groups - Averaged Difference (da cache)"
        plot_all_z_spectra(
            all_groups=groups,
            analysis_results=analysis_results,
            visibility=config.get("plot_visibility", get_default_visibility()),
            title=title_zs,
            window_title=title_zs,
        )

        # --- Saving ---
        save_analysis_results(analysis_results=analysis_results, config_name=config_name)
        print("Press Enter to exit...")
        input()
        plt.close("all")
        return

    # ------------------------------------------------------------------
    # Live analysis
    # ------------------------------------------------------------------
    for grp_idx, grp in enumerate(groups):
        label = grp["label"]
        source_type, entries = _resolve_group_source(grp)
        experiments = groups_data[label]["experiments"]

        # ---------- PV (Paravision Bruker) ----------
        if source_type == "pv":

            folders = entries
            group_spectra_to_plot = []  # Accumulator for this group's spectra
            for folder in entries:
                base_name = f"{folder.parent.name[:12]}…{folder.parent.name[-12:]}-{folder.stem}"
                folder_name_short = base_name
                counter = 1
                while folder_name_short in experiments:
                    folder_name_short = f"{base_name}_{counter}"
                    counter += 1

                sat_trans_hz, work_offset_hz = extract_parameters(folder, "method")

                if group_meta[label].get("work_offset_hz") is None:
                    group_meta[label]["work_offset_hz"] = work_offset_hz
                else:
                    # Compare with tolerance: SFO1 values differ only in the
                    # last few decimal digits between replicates of the same
                    # experiment (float parsing noise, rtol ~ 1e-8). Exact `!=`
                    # was triggering false positives. rtol=1e-6 is ~30x above
                    # the observed noise.
                    if not np.allclose(
                        group_meta[label]["work_offset_hz"], work_offset_hz,
                        rtol=1e-6, atol=1e-6
                    ):
                        print(colored(
                            f"Warning: different work_offset in group '{label}' "
                            f"(prev={group_meta[label]['work_offset_hz']}, "
                            f"new={work_offset_hz})",
                            "yellow"
                        ))

                dic, data, uc, ppm_axis, n_exp, bf1 = load_spectra(folder)

                spectra = process_spectra(data, dic, n_exp, lb_Hz=lb_Hz)
                
                # Accumulate for the group-level plot
                current_exp_spectra = []
                for exp_idx in range(n_exp):
                    current_exp_spectra.append({
                        'ppm_axis': ppm_axis,
                        'spectrum': spectra[exp_idx],
                        'label': f"{exp_idx:>2} : {sat_trans_hz[exp_idx]:.2f}"
                    })
                group_spectra_to_plot.append({
                    'name': folder_name_short,
                    'spectra': current_exp_spectra
                })
                if ppm_missing and grp_idx == 0 and folder == folders[0]:
                    # Show a temporary plot of the first folder so the user
                    # can select the ppm range interactively.
                    plot_spectra(
                        title=f"{label} - Select ppm range",
                        spectra_data=group_spectra_to_plot,
                        visibility=config.get("plot_visibility", get_default_visibility()),
                        window_title=f"{label}: Select ppm range"
                    )
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

                max_vals, max_indexes, global_max, global_min = find_max_vals(
                    spectra, start_idx, end_idx
                )
                max_vals = normalize_max_vals(max_vals, global_max, global_min)

                zero_corrected_ppm = correct_sat_frequencies(
                    sat_trans_hz, max_indexes, work_offset_hz, uc, bf1
                )

                combined = list(zip(sat_trans_hz, max_indexes, max_vals, zero_corrected_ppm))
                combined.sort()
                sat_trans_hz[:], max_indexes[:], max_vals[:], zero_corrected_ppm[:] = zip(*combined)

                res = process_zspectrum_and_integrals(max_vals, zero_corrected_ppm)

                experiments[folder_name_short] = {
                    "source_type":  "pv",
                    "sat_trans_hz": sat_trans_hz,
                    "bf1":          bf1,
                    "max_vals":     max_vals,
                }
                experiments[folder_name_short].update(res)

                plot_z_spectrum(
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
                    color=group_color(label=label, all_groups=groups),
                    window_title=f" {label}: spline fit for {folder_name_short}"
                )
            
            # After all folders of this PV group, plot all spectra together
            if group_spectra_to_plot:
                plot_spectra(
                    title=f"{label} - All Spectra",
                    spectra_data=group_spectra_to_plot,
                    visibility=config.get("plot_visibility", get_default_visibility()),
                    window_title=f"{label}: All Spectra"
                )
        elif source_type == "ts":   # Topspin BRUKER data
            folders = entries
            group_spectra_to_plot = []  # Accumulator for this group's spectra
            for folder in folders:
                base_name = f"{folder.parent.name[:12]}…{folder.parent.name[-12:]}-{folder.stem}"
                folder_name_short = base_name
                counter = 1
                while folder_name_short in experiments:
                    folder_name_short = f"{base_name}_{counter}"
                    counter += 1

                work_offset_hz = extract_parameters(folder, "acqu2")
                sat_trans_hz = extract_parameters(folder, "fq2list")

                if group_meta[label].get("work_offset_hz") is None:
                    group_meta[label]["work_offset_hz"] = work_offset_hz
                else:
                    # Compare with tolerance: SFO1 values differ only in the
                    # last few decimal digits between replicates of the same
                    # experiment (float parsing noise, rtol ~ 1e-8). Exact `!=`
                    # was triggering false positives. rtol=1e-6 is ~30x above
                    # the observed noise.
                    if not np.allclose(
                        group_meta[label]["work_offset_hz"], work_offset_hz,
                        rtol=1e-6, atol=1e-6
                    ):
                        print(colored(
                            f"Warning: different work_offset in group '{label}' "
                            f"(prev={group_meta[label]['work_offset_hz']}, "
                            f"new={work_offset_hz})",
                            "yellow"
                        ))

                dic, data, uc, ppm_axis, n_exp, bf1 = load_spectra(folder)

                spectra = process_spectra(data, dic, n_exp, lb_Hz=lb_Hz)
                
                # Accumulate for the group-level plot
                current_exp_spectra = []
                for exp_idx in range(n_exp):
                    current_exp_spectra.append({
                        'ppm_axis': ppm_axis,
                        'spectrum': spectra[exp_idx],
                        'label': f"{exp_idx:>2} : {sat_trans_hz[exp_idx]:.2f}"
                    })
                group_spectra_to_plot.append({
                    'name': folder_name_short,
                    'spectra': current_exp_spectra
                })

                if ppm_missing and grp_idx == 0 and folder == folders[0]:
                    # Show a temporary plot of the first folder so the user
                    # can select the ppm range interactively.
                    plot_spectra(
                        title=f"{label} - Select ppm range",
                        spectra_data=group_spectra_to_plot,
                        visibility=config.get("plot_visibility", get_default_visibility()),
                        window_title=f"{label}: Select ppm range"
                    )
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

                max_vals, max_indexes, global_max, global_min = find_max_vals(
                    spectra, start_idx, end_idx
                )
                max_vals = normalize_max_vals(max_vals, global_max, global_min)

                zero_corrected_ppm = correct_sat_frequencies(
                    sat_trans_hz, max_indexes, work_offset_hz, uc, bf1
                )

                combined = list(zip(sat_trans_hz, max_indexes, max_vals, zero_corrected_ppm))
                combined.sort()
                sat_trans_hz[:], max_indexes[:], max_vals[:], zero_corrected_ppm[:] = zip(*combined)

                res = process_zspectrum_and_integrals(max_vals, zero_corrected_ppm)

                experiments[folder_name_short] = {
                    "source_type":  "ts",
                    "sat_trans_hz": sat_trans_hz,
                    "bf1":          bf1,
                    "max_vals":     max_vals,
                }
                experiments[folder_name_short].update(res)

                plot_z_spectrum(
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
                    color=group_color(label=label, all_groups=groups),
                    window_title=f" {label}: spline fit for {folder_name_short}"
                )
            
            # After all folders of this TS group, plot all spectra together
            if group_spectra_to_plot:
                plot_spectra(
                    title=f"{label} - All Spectra",
                    spectra_data=group_spectra_to_plot,
                    visibility=config.get("plot_visibility", get_default_visibility()),
                    window_title=f"{label}: All Spectra"
                )
        else:   # source_type == "text": txt data files (sat_trans_hz vs max_vals)
            files = entries
            for file_idx, folder in enumerate(files):
                base_name = folder.stem
                key = base_name
                counter = 1
                while key in experiments:
                    key = f"{base_name}_{counter}"
                    counter += 1

                try:
                    data = np.loadtxt(folder, comments="#")
                    if data.ndim != 2 or data.shape[1] < 2:
                        raise ValueError("File must contain at least two columns.")
                    sat_trans_hz = data[:, 0].tolist()
                    max_vals     = data[:, 1].tolist()
                    max_vals = normalize_max_vals(
                        max_vals, max(max_vals), min(max_vals)
                    )

                    try:
                        bf1_file = config["groups"][grp_idx]["BF1"][file_idx]
                    except (KeyError, TypeError, IndexError) as e:
                        print(colored(
                            f"Missing/invalid BF1 for {folder}: {e}",
                            "red", attrs=["bold"],
                        ))
                        continue

                    zero_corrected_ppm = [st / bf1_file for st in sat_trans_hz]
                except Exception as e:
                    print(colored(
                        f"Error reading file {folder}: {e}",
                        "red", attrs=["bold"],
                    ))
                    continue

                combined = list(zip(zero_corrected_ppm, max_vals, sat_trans_hz))
                combined.sort()
                zero_corrected_ppm[:], max_vals[:], sat_trans_hz[:] = zip(*combined)

                res = process_zspectrum_and_integrals(max_vals, zero_corrected_ppm)

                experiments[key] = {
                    "source_type":  "text",
                    "sat_trans_hz": sat_trans_hz,
                    "bf1":          bf1_file,
                    "max_vals":     max_vals,
                }
                experiments[key].update(res)

                plot_z_spectrum(
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
                    color=group_color(label=label, all_groups=groups),
                    window_title=f"{label}: spline fit for file {key}"
                )

        # ---------- Summary del gruppo ----------
        if experiments:
            val_arr = np.array([e["max_vals"]     for e in experiments.values()])
            sat_arr = np.array([e["sat_trans_hz"] for e in experiments.values()])
            n = len(experiments)
            mean_max_vals = np.mean(val_arr, axis=0).tolist()
            mean_sat      = np.mean(sat_arr, axis=0).tolist()
            y_raw_sd = (
                np.std(val_arr, axis=0, ddof=1).tolist() if n > 1
                else [0] * len(val_arr[0])
            )

            coll_res = collect_replicate_differences(experiments, compute_integrals=True)
            fit_x = coll_res["fit_x"]
            first_bf1 = next(iter(experiments.values())).get("bf1")

            groups_data[label]["summary"] = {
                "y_raw_mean":    mean_max_vals,
                "x_raw_mean_hz": mean_sat,
                "y_raw_sd":      y_raw_sd,
                "spline_fit_results": {
                    "x_fit": fit_x,
                    "y_fit": coll_res["y_spline_mean"],
                    "fit_label": "Spline fit (mean of replicates, corrected)",
                    "fit_successful": True,
                },
                "lorentzian_envelope_results": {
                    "x": fit_x,
                    "y": coll_res["y_lorentzian_mean"],
                    "fit_label": "Lorentzian envelope (mean of replicates)",
                    "fit_successful": True,
                },
                "x_data":                    coll_res["x_data"],
                "y_corrected_at_data_mean":  coll_res["y_corrected_at_data_mean"],
                "y_corrected_at_data_sem":   coll_res["y_corrected_at_data_sem"],
                "bf1":       first_bf1,
                "fit_x":     fit_x,
                "y_diff_mean": coll_res["y_diff_mean"],
                "y_diff_sem":  coll_res["y_diff_sem"],
                "integrals_of_mean_curve": coll_res["integrals_of_mean_curve"],
            }

            plot_group_difference(
                group_label=label,
                group_data=groups_data[label]["summary"],
                visibility=config.get("plot_visibility", get_default_visibility()),
                window_title=f"Group {label} - Averaged Difference",
                color=group_color(label=label, all_groups=groups)
            )

    # ------------------------------------------------------------------
    # Statistiche finale
    # ------------------------------------------------------------------
    group_stats = {
        label: _compute_group_stats(gd["experiments"])
        for label, gd in groups_data.items()
        if gd["experiments"]
    }

    ref_label = next(
        (lbl for lbl, gd in groups_data.items() if gd["is_reference"]),
        None,
    )
    if ref_label:
        ref_experiments = groups_data[ref_label]["experiments"]
        for label, gd in groups_data.items():
            if label == ref_label or not gd["experiments"]:
                continue
            gd["p_values"] = _compute_pvalues(
                ref_experiments, gd["experiments"], test="t-test"
            )

    # ------------------------------------------------------------------
    # Assemblaggio finale + plot di riepilogo + save
    # ------------------------------------------------------------------
    analysis_results["groups"] = groups_data
    analysis_results["group_stats"] = group_stats
    analysis_results["__script_version__"] = get_git_hash(short=True)

    p_values_for_plot = {
        lbl: gd["p_values"]
        for lbl, gd in groups_data.items()
        if not gd["is_reference"] and gd.get("p_values")
    }
    plot_groups_comparison(
        group_stats, p_values_for_plot, groups,
        visibility=config.get("plot_visibility", get_default_visibility()),
        title="Groups comparison (ricalcolati)",
        window_title="Groups comparison (ricalcolati)",
    )

    for label, gd in groups_data.items():
        if not gd["experiments"]:
            continue
        per_folder = {}
        for exp in gd["experiments"].values():
            for region, val in exp.get("integrals", {}).items():
                per_folder.setdefault(region, []).append(val)
        plot_group_breakdown(
            group_label=label,
            group_stats=group_stats,
            per_folder_integrals={label: per_folder},
            folder_names=list(gd["experiments"].keys()),
            visibility=config.get("plot_visibility", get_default_visibility()),
            title=f"{label} group breakdown (ricalcolati)",
            window_title=f"{label} group breakdown (ricalcolati)",
        )

    # --- Saving ---
    # ---- Tutti gli Z-spectra raggruppati (ricalcolati) ----
    title_zs = "All groups - Averaged Difference (ricalcolati)"
    plot_all_z_spectra(
        groups=groups,
        analysis_results=analysis_results,
        visibility=config.get("plot_visibility", get_default_visibility()),
        title=title_zs,
        window_title=title_zs,
    )
    analysis_results["__script_version__"] = get_git_hash(short=True)
    save_cache(config_name, config, analysis_results)
    save_analysis_results(analysis_results=analysis_results, config_name=config_name)

    print("\nTutti i grafici sono stati creati. Premi Invio per uscire.")
    input()
    plt.close("all")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> None:
    # Apply the global matplotlib theme BEFORE any plotting.
    # This affects every figure created afterwards (both the live branch
    # and the cache branch), because it sets plt.rcParams.
    apply_theme()

    config_name, config_data = select_or_create_config()
    complete_config = ensure_complete_config(config_name, config_data)
    run_analysis(config_name, complete_config)

if __name__ == "__main__":
    main()
