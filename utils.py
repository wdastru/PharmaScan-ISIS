"""
utils.py
Utility generali: funzioni interattive (ask_*), get_git_hash, find_methods.
"""

import os
import subprocess
from typing import List, Optional, Tuple

def ask_yes_no(prompt: str, default: Optional[bool] = None) -> bool:
    default_prompt = ""
    if default is True:
        default_prompt = " (Y/n)"
    elif default is False:
        default_prompt = " (y/N)"
    else:
        default_prompt = " (y/n)"
    while True:
        answer = input(prompt + default_prompt + ": ").strip().lower()
        if not answer and default is not None:
            return default
        if answer in ('y', 'yes'):
            return True
        if answer in ('n', 'no'):
            return False
        print("Rispondi con y o n.")

def ask_int(prompt: str, min_val: int = None, max_val: int = None, default: Optional[int] = None) -> int:
    default_prompt = f" (default {default})" if default is not None else ""
    while True:
        answer = input(f"{prompt}{default_prompt}: ").strip()
        if not answer and default is not None:
            return default
        try:
            value = int(answer)
            if min_val is not None and value < min_val:
                print(f"Valore deve essere >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Valore deve essere <= {max_val}")
                continue
            return value
        except ValueError:
            print("Inserire un numero intero.")

def ask_float(prompt: str, min_val: float = None, max_val: float = None, default: Optional[float] = None) -> float:
    default_prompt = f" (default {default})" if default is not None else ""
    while True:
        answer = input(f"{prompt}{default_prompt}: ").strip()
        if not answer and default is not None:
            return default
        try:
            value = float(answer)
            if min_val is not None and value < min_val:
                print(f"Valore deve essere >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Valore deve essere <= {max_val}")
                continue
            return value
        except ValueError:
            print("Inserire un numero.")

def ask_choice(prompt: str, choices: List[str], default: Optional[str] = None) -> str:
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    while True:
        ans = input(f"{prompt} (1-{len(choices)})" + (f" [{choices.index(default)+1}]" if default else "") + ": ").strip()
        if not ans and default:
            return default
        try:
            idx = int(ans) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            pass
        print("Invalid choice.")

def ask_user_for_ppm_range(default_start=None, default_end=None) -> Tuple[float, float]:
    while True:
        try:
            start_prompt = "Enter the minimum ppm (start)"
            end_prompt = "Enter the maximum ppm (end)"
            if default_start is not None:
                start_prompt += f" (default {default_start})"
            if default_end is not None:
                end_prompt += f" (default {default_end})"

            start_ppm = ask_float(start_prompt, default=default_start)
            end_ppm = ask_float(end_prompt, default=default_end)
            
            if start_ppm is None or end_ppm is None:
                print("Inserire entrambi i valori.")
                continue
            if end_ppm <= start_ppm:
                print("end deve essere maggiore di start.")
                continue
            return start_ppm, end_ppm
        except ValueError:
            print("Inserire numeri validi.")

def get_git_hash(short=True):
    """
    Return 'abc1234' for a clean working tree, or 'abc1234-dirty'
    if there are uncommitted changes.
    """
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        cmd = ["git", "rev-parse"]
        if short:
            cmd.append("--short")
        cmd.append("HEAD")
        hash_ = subprocess.check_output(
            cmd, cwd=base, stderr=subprocess.DEVNULL
        ).strip().decode()

        # Check for uncommitted changes
        dirty = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=base,
            stderr=subprocess.DEVNULL
        )
        if dirty != 0:
            hash_ += "-dirty"
        return hash_
    except Exception:
        return "unknown"

def find_methods(obj, path=""):
    """Return list of (path, object_repr) for any method or function found."""
    issues = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            current = f"{path}.{k}" if path else k
            if callable(v) and not isinstance(v, (type, type(None))):
                # Catch user-defined methods, built-in methods, functions, lambdas
                issues.append((current, repr(v)))
            else:
                issues.extend(find_methods(v, current))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            current = f"{path}[{i}]"
            if callable(v) and not isinstance(v, type):
                issues.append((current, repr(v)))
            else:
                issues.extend(find_methods(v, current))
    return issues

