"""
data_io.py
Funzioni per l'input/output: selezione cartelle/file, estrazione parametri,
caricamento spettri Bruker/TopSpin, elaborazione FID e correzione di fase.
"""

import nmrglue as ng
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import re
import tkinter as tk
from tkinter import filedialog
from typing import List, Optional, Tuple, Any

from termcolor import colored

def select_experiment_folder(title="Select a folder") -> Path:
    root = tk.Tk()
    root.withdraw()
    return Path(filedialog.askdirectory(title=title))

def select_text_file(title="Select a text data file (x/y columns)") -> Path:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Text files", "*.txt *.dat"), ("All files", "*.*")]
    )
    if not file_path:
        raise ValueError("No file selected.")
    return Path(file_path)

def parameter_extract(file_path: Path, PARAMETER: str = None) -> List[float]:

    def _read_fq2list(filename):
        with open(filename, encoding="utf-8") as f:
            return [float(line.strip()) for line in f if line.strip()]

    if not file_path.exists():
        raise FileNotFoundError(colored(
            f"{file_path} not found.", "red", attrs=["bold"])
        )
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    if PARAMETER is not None:
        # Look for the header and also capture the block up to the next '##$'
        hdr_pattern = rf"##\${PARAMETER}=(?:\s*\(\s*(?P<N>\d+)\s*\)\s*\n(?P<block>.*?)(?=\r?\n##\$|\Z)|\s*(?P<value>[^\r\n]*))"
        match = re.search(hdr_pattern, text, re.DOTALL)
        if not match:
            raise ValueError(colored(
                f"Header '##${PARAMETER}=( N )' non trovato in {file_path}.", "red", attrs=["bold"])
            )
        else:

            N_str = match.group("N")      # None if not matched
            block = match.group("block")  # None if not matched
            val   = match.group("value")  # None if not matched

            if val is not None:
                print(f"{PARAMETER} value: {val}")
                return [float(val)]
            else:

                if match.group("N") is not None:
                    N = int(N_str)

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
    else:
        return _read_fq2list(file_path)
        
def extract_parameters(folder: Path, filename: str) -> Tuple[List[float], List[float]]:
    file = folder / filename
    if filename == "method":
        sat_hz = parameter_extract(file, "PVM_SatTransFL")
        offset_hz = parameter_extract(file, "PVM_FrqWorkOffset")
        return sat_hz, offset_hz
    elif filename == "acqu2":
        offset_hz = parameter_extract(file, "SFO1")
        return offset_hz
    elif filename == "fq2list":
        sat_hz = parameter_extract(file)
        return sat_hz
    else:
        raise ValueError(f"Unknown parameter file: {filename}")

def load_spectra(folder: Path):
    dic, data = ng.bruker.read(folder)
    udic = ng.bruker.guess_udic(dic, data)
    uc = ng.fileio.bruker.fileiobase.uc_from_udic(udic, dim=1)
    ppm_axis = uc.ppm_scale()
    n_exp = dic["acqu2s"]["TD"]
    bf1 = dic["acqus"]["BF1"]
    return dic, data, uc, ppm_axis, n_exp, bf1

