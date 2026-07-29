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
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from termcolor import colored

from constants import (
    DEFAULT_METABOLITE_REGIONS, 
    CACHE_DIR, get_default_visibility
)

from config import (
    METABOLITE_REGIONS, save_config, select_or_create_config, 
    save_analysis_results, ensure_complete_config
)

from plotting import (
    plot_group_difference, plot_group_breakdown, plot_spectra,
    plot_groups_comparison, plot_z_spectrum
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

print(f"Using nmrglue version: {ng.__version__}")

CACHE_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Core analysis routine
# ----------------------------------------------------------------------
def run_analysis(config_name: str, config: Dict[str, Any]) -> None:
    METABOLITE_REGIONS.clear()
    METABOLITE_REGIONS.update(config.get("metabolite_regions", DEFAULT_METABOLITE_REGIONS))
    plt.ion()

    groups = config["groups"]
    lb_Hz = config["lb_Hz"]
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
        # 1. Plot group average difference curves (da cache)
        # ------------------------------------------------------------
        for grp in groups:
            label = grp["label"]
            grp_data = analysis_results.get(label, {})
            if grp_data:   # any data at all – plot_group_difference will handle missing keys gracefully
                plot_group_difference(
                    group_label=label,
                    group_data=grp_data,
                    visibility=config.get("plot_visibility", get_default_visibility()),
                    window_title=f"Group {label} – Averaged Difference (da cache)"
                )
                
        # ------------------------------------------------------------
        # 2. Re‑plot Z‑spettri per le singole cartelle
        # ------------------------------------------------------------
        folder_keys_per_group_cached = analysis_results.get("folder_keys_per_group", [])
        for grp_idx, keys in enumerate(folder_keys_per_group_cached):
            for key in keys:
                res = analysis_results.get(key, {})
                if "spline_fit_results" in res and res["spline_fit_results"].get("fit_successful"):
                    plot_z_spectrum(
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
                title=f"{label} group breakdown (da cache)"
                plot_group_breakdown(
                    group_label=label,
                    group_stats=group_stats,
                    per_folder_integrals=per_folder_integrals,
                    folder_names=folder_keys_per_group_cached[grp_idx] if grp_idx < len(folder_keys_per_group_cached) else [],
                    visibility=config.get("plot_visibility", get_default_visibility()),
                    title=title,
                    window_title=title
                )

        # ------------------------------------------------------------
        # 5. Grafico multi‑gruppo con p‑value (già presente)
        # ------------------------------------------------------------
        pvals = analysis_results.get("p_values", {})
        title="Groups comparison (da cache)"
        plot_groups_comparison(group_stats, pvals, groups,
                                  visibility=config.get("plot_visibility", get_default_visibility()),
                                  title=title,
                                  window_title=title)

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
    file_data_raw = [ [] for _ in groups ]

    for grp_idx, grp in enumerate(groups):
        label = grp["label"]

        # Determine entry type for this group
        is_folder = bool(grp.get("folders"))
        is_topspin_folder = bool(grp.get("topspin_folders"))
        is_file   = bool(grp.get("files"))
        if is_folder and is_file: # TODO: handle topspin folders if needed
            print(colored(
                f"Warning: Group '{label}' has both folders and files. Only folders will be used.",
                "yellow"
            ))
            is_file = False
            entries = grp["folders"]
        elif is_folder:
            entries = grp["folders"]
        elif is_topspin_folder:
            entries = grp["topspin_folders"]
        elif is_file:
            entries = grp["files"]
        else:
            pass
                    
        if is_folder:   # BRUKER data
            folders = entries
            for folder in folders:
                base_name = f"{folder.parent.name[:12]}…{folder.parent.name[-12:]}-{folder.stem}"
                folder_name_short = base_name
                counter = 1
                while folder_name_short in analysis_results:
                    folder_name_short = f"{base_name}_{counter}"
                    counter += 1
                analysis_results[folder_name_short] = {}
                folder_keys_per_group[grp_idx].append(folder_name_short)

                sat_trans_hz, work_offset_hz = extract_parameters(folder, "method")
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

                dic, data, uc, ppm_axis, n_exp, bf1 = load_spectra(folder)
                analysis_results[folder_name_short]["uc"] = uc
                analysis_results[folder_name_short]["bf1"] = bf1
                if group_meta[grp_idx].get("uc") is None:
                    group_meta[grp_idx]["uc"] = uc
                    group_meta[grp_idx]["bf1"] = bf1

                spectra = process_spectra(data, dic, n_exp, lb_Hz=lb_Hz)
                fig = plot_spectra(
                    title=f"{label} - {folder_name_short}",
                    spectra=spectra, n_exp=n_exp, ppm_axis=ppm_axis,
                    sat_trans_hz=sat_trans_hz,
                    visibility=config.get("plot_visibility", get_default_visibility()),
                    window_title=f"{label}: Spectra  for {folder_name_short}"
                )

                if ppm_missing and grp_idx == 0 and folder == folders[0]:
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

                res = process_zspectrum_and_integrals(max_vals, zero_corrected_ppm)
                analysis_results[folder_name_short].update({
                    "max_indexes": max_indexes,
                    "max_vals": max_vals,
                })
                analysis_results[folder_name_short].update(res)
                group_raw[grp_idx].append((max_indexes, max_vals, sat_trans_hz))

                # --- Calculate integrals for this individual folder ---
                
                # After storing the results for the single folder, optionally plot it
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
                    window_title=f" {label}: spline fit for {folder_name_short}"
                )
        elif is_topspin_folder:   # BRUKER data spectroscopy
            folders = entries
            for folder in folders:
                base_name = f"{folder.parent.name[:12]}…{folder.parent.name[-12:]}-{folder.stem}"
                folder_name_short = base_name
                counter = 1
                while folder_name_short in analysis_results:
                    folder_name_short = f"{base_name}_{counter}"
                    counter += 1
                analysis_results[folder_name_short] = {}
                folder_keys_per_group[grp_idx].append(folder_name_short)

                work_offset_hz = extract_parameters(folder, "acqu2")
                sat_trans_hz = extract_parameters(folder, "fq2list")

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

                dic, data, uc, ppm_axis, n_exp, bf1 = load_spectra(folder)
                analysis_results[folder_name_short]["uc"] = uc
                analysis_results[folder_name_short]["bf1"] = bf1
                if group_meta[grp_idx].get("uc") is None:
                    group_meta[grp_idx]["uc"] = uc
                    group_meta[grp_idx]["bf1"] = bf1

                spectra = process_spectra(data, dic, n_exp, lb_Hz=lb_Hz)
                fig = plot_spectra(
                    title=f"{label} - {folder_name_short}",
                    spectra=spectra, n_exp=n_exp, ppm_axis=ppm_axis,
                    sat_trans_hz=sat_trans_hz,
                    visibility=config.get("plot_visibility", get_default_visibility()),
                    window_title=f"{label}: Spectra  for {folder_name_short}"
                )

                if ppm_missing and grp_idx == 0 and folder == folders[0]:
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

                res = process_zspectrum_and_integrals(max_vals, zero_corrected_ppm)
                analysis_results[folder_name_short].update({
                    "max_indexes": max_indexes,
                    "max_vals": max_vals,
                })
                analysis_results[folder_name_short].update(res)
                group_raw[grp_idx].append((max_indexes, max_vals, sat_trans_hz))

                # --- Calculate integrals for this individual folder ---
                
                # After storing the results for the single folder, optionally plot it
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
                    window_title=f" {label}: spline fit for {folder_name_short}"
                )
        else:   # txt data files (sat_trans_hz vs max_vals) 
            files = entries
            for file_idx, folder in enumerate(files):
                base_name = folder.stem
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
                    data = np.loadtxt(folder, comments='#')
                    if data.ndim != 2 or data.shape[1] < 2:
                        raise ValueError("File must contain at least two columns.")
                    sat_trans_hz = data[:, 0].tolist()
                    max_vals     = data[:, 1].tolist()
                    max_vals = normalize_max_vals(max_vals=max_vals, global_max=max(max_vals), global_min=min(max_vals))
                    
                    try:
                        value = config["groups"][grp_idx]["BF1"][file_idx]
                        group_meta[grp_idx]["bf1"] = value
                    except KeyError as e:
                        print(colored(f"Missing key {e} for {folder}", "red", attrs=["bold"]))
                        continue
                    except (TypeError, IndexError) as e:
                        print(colored(f"Invalid structure (expected dict/list) for {folder}: {e}", "red", attrs=["bold"]))
                        continue
                                        
                    zero_corrected_ppm = [sat_trans_hz[i] / group_meta[grp_idx]["bf1"] for i in range(len(sat_trans_hz))]
                except Exception as e:
                    print(colored(f"Error reading file {folder}: {e}", "red", attrs=["bold"]))
                    continue

                max_indexes = [0] * len(max_vals)

                # --- Sort by ppm ---
                combined = list(zip(zero_corrected_ppm, max_vals, sat_trans_hz))
                combined.sort()
                zero_corrected_ppm[:], max_vals[:], sat_trans_hz[:] = zip(*combined)

                res = process_zspectrum_and_integrals(max_vals, zero_corrected_ppm)
                # No max_indexes, zero_corrected_ppm or sat_trans_hz needed; store what we have
                analysis_results[key].update({
                    "max_vals": max_vals,
                })
                analysis_results[key].update(res)
                # --- Store raw data for later group averaging ---
                group_raw[grp_idx].append((max_indexes, max_vals, sat_trans_hz))

                # --- Fit, integrate, and plot (common pipeline) ---
                res = process_zspectrum_and_integrals(
                    max_vals, 
                    zero_corrected_ppm
                )
                analysis_results[key].update(res)

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
                    window_title=f"{label}: spline fit for file {key}"
                )

        # --- Build averaged difference curves for the group ---
        if group_raw[grp_idx]:
            keys = folder_keys_per_group[grp_idx]

            # --- Raw data for plotting (mean of max_vals) ---
            val_arr = np.array([d[1] for d in group_raw[grp_idx]])
            sat_arr = np.array([d[2] for d in group_raw[grp_idx]])
            n = len(group_raw[grp_idx])
            mean_max_vals = np.mean(val_arr, axis=0).tolist()
            mean_sat = np.mean(sat_arr, axis=0).tolist()
            sd_max_vals = np.std(val_arr, axis=0, ddof=1).tolist() if n > 1 else [0]*len(val_arr[0])

            # Ppm axis for the raw data (mean saturation freq / BF1)
            mean_zero_ppm = [mean_sat[i] / group_meta[grp_idx]["bf1"] for i in range(len(mean_sat))]

            # --- Collect per-replicate differences on a common grid ---
            coll_res = collect_replicate_differences(
                keys, analysis_results, compute_integrals=True
            )
            x_common = coll_res["x_common"]
            mean_diff = coll_res["mean_diff"]
            sem_diff = coll_res["sem_diff"]
            group_integrals = coll_res["integrals"]

            # --- Store under the group label ---
            analysis_results[label] = {
                "max_indexes": np.round(np.mean(np.array([d[0] for d in group_raw[grp_idx]]), axis=0)).tolist(),
                "max_vals": mean_max_vals,
                "sat_trans_hz": mean_sat,
                "sd_max_vals": sd_max_vals,
                "bf1": group_meta[grp_idx]["bf1"],
                "x_common": x_common,
                "mean_diff_y": mean_diff,
                "sem_diff_y": sem_diff,
                "integrals": group_integrals,   # exactly matches mean of per-replicate integrals
                "per_replicate_diffs": coll_res["individual_diffs"],  # optional
            }

            plot_group_difference(
                group_label=label,
                group_data=analysis_results[label],
                visibility=config.get("plot_visibility", get_default_visibility()),
                window_title=f"Group {label} - Averaged Difference"
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

    # ---- Show multigroup bar plot ----
    title="Groups comparison (ricalcolati)"
    plot_groups_comparison(group_stats, p_values, groups,
                              visibility=config.get("plot_visibility", get_default_visibility()),
                              title=title,
                              window_title=title)

    # ---- Plot per gruppo con cartelle singole ----
    for grp_idx, grp in enumerate(groups):
        label = grp["label"]
        title: str = f"{label} group breakdown (ricalcolati)"
        plot_group_breakdown(
            group_label=label,
            group_stats=group_stats,
            per_folder_integrals=per_folder_integrals,
            folder_names=folder_keys_per_group[grp_idx],   # <-- lista dei nomi brevi
            visibility=config.get("plot_visibility", get_default_visibility()),
            title=title,
            window_title=title
        )

    # --- Saving ---
    analysis_results["__script_version__"] = get_git_hash(short=True)
    save_cache(config_name, config, analysis_results)
    save_analysis_results(analysis_results=analysis_results, config_name=config_name)
    
    #print("\nTutti i grafici sono stati creati.")
    #plt.show(block=True)
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