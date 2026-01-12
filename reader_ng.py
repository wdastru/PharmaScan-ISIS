import nmrglue as ng
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import re

def findMaxima(arr, start=None, end=None):
    """
    Return the maximum value in arr[start:end].
    If start/end are None, use the full array.
    """
    if start is None:
        start = 0
    if end is None or end > len(arr):
        end = len(arr)
    
    sub_arr = arr[start:end]
    max_val = sub_arr.max()
    max_idx = sub_arr.argmax() + start  # adjust index to original array
    return max_val, max_idx

def parameter_extract(file_path: Path, PARAMETER: str) -> list[float]:
    if not file_path.exists():
        print(f"{file_path} file not found. Aborting.")
        exit(1)

    text = file_path.read_text(encoding="utf-8", errors="ignore")

    hdr_pattern = rf"##\${PARAMETER}=\(\s*(?P<N>\d+)\s*\)"
    hdr_match = re.search(hdr_pattern, text)
    if not hdr_match:
        raise ValueError(f"Parameter header '##${PARAMETER}=( N )' not found in {file_path} file.")
    else:
        N = int(hdr_match.group("N"))

    print(f"{PARAMETER} dimension: {N}")

    start_pos = hdr_match.end() # position atfer the match
    tail = text[start_pos:]     # the rest of the text after the match

    num_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    vals = re.findall(num_pattern, tail)

    print(f"Found {len(vals)} numerical values after for {PARAMETER}")

    if len(vals) < N:
        raise ValueError(f"Found only {len(vals)} numbers after {PARAMETER} header, expected {N}.")

    return [float(v) for v in vals[:N]]

def apply_phase(data, p0, p1):
    return ng.proc_base.ps(data, p0=p0, p1=p1)

def show_phase(data, p0, p1):
    phased = apply_phase(data, p0, p1)
    plt.plot(np.real(phased))
    plt.show()

base = Path(".")
patient = "20251209_124252_31P_localizzato_muscolo_topo5F_Rotenone_cinetica_5uM_1_48"
expt = "13"

path = base / patient / expt
method = path / "method"
sat_trans_fl = parameter_extract(method, "PVM_SatTransFL")

dic, data = ng.bruker.read(path)

n_exp = dic["acqu2s"]["TD"]

with open("parameters.txt", "w", encoding="utf-8") as f:
    try:
        json.dump(dic, f, indent=4)
    except Exception as e:
        print(f"🚫 Failed to save parameters.txt: {e}")

plt.figure(figsize=(12, 6))

loaded = {}     # {proc: np.ndarray}

for i in range(n_exp):
    fid = data[i, :]
    fid = ng.bruker.remove_digital_filter(dic, data=fid)
    lb = 0.005  # line broadening in Hz
    fid_apod = ng.proc_base.em(fid, lb=lb)
    spectrum = ng.proc_base.fft(fid_apod)
    spectrum_phased = ng.proc_autophase.autops(spectrum, fn="acme")
    loaded[i] = spectrum_phased
    plt.plot(np.real(spectrum_phased), label=f"{sat_trans_fl[i]}", alpha=0.7)

plt.xlabel("Points")
plt.ylabel("Intensity")
plt.grid(True, alpha=0.3)
plt.legend(ncol=3, fontsize=9)
plt.tight_layout()
plt.show(block=False)

# Ask user for min and max index
try:
    start_idx = int(input("Enter the minimum index (start): "))
    end_idx = int(input("Enter the maximum index (end): "))

    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError("Invalid range: end must be greater than start and both non-negative.")

    print(f"Selected range: [{start_idx}, {end_idx}]")  # end-exclusive
except ValueError as e:
    print(f"Error: {e}")

max_vals = {}
max_indexes = {}
for p, a in loaded.items():
    max_vals[p], max_indexes[p] = findMaxima(loaded[p], start=start_idx, end=end_idx)

for p, max_val in max_vals.items():
    print(f"proc {p}: max = {max_val} at index {max_indexes[p]}")

if len(sat_trans_fl) == len(max_vals):

    plt.figure(figsize=(8, 5))
    plt.plot(list(sat_trans_fl), list(max_vals.values()), marker='o', linestyle='None', color='b')
    plt.title("Max Values vs Saturation Frequencies")
    plt.xlabel("Saturation Frequencies")
    plt.ylabel("Max Value")
    plt.grid(True)
    plt.show(block=True)
else:
    print("Number of saturation frequencies does not match number of processed series; skipping plot.")    