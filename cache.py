"""
cache.py
Caching delle analisi con joblib.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, Any

from joblib import dump, load
from termcolor import colored

from constants import CACHE_DIR, CACHE_VERSION
from config import METABOLITE_REGIONS

def _cache_path(config_name: str, config: Dict[str, Any]) -> Path:
    # Se il nome è vuoto (nessuna configurazione salvata), usa un hash
    if not config_name:
        key_str = json.dumps(_build_cache_key(config), sort_keys=True)
        name = hashlib.md5(key_str.encode()).hexdigest()[:8]
    else:
        # Pulisci il nome per evitare caratteri problematici
        name = "".join(c for c in config_name if c.isalnum() or c in " _-").rstrip()
    return CACHE_DIR / f"analysis_{name}.joblib"

def _build_cache_key(config: Dict[str, Any]) -> str:
    """
    Crea una chiave univoca basata sui parametri e sulle cartelle (compresa la data di modifica).
    """
    groups = config.get("groups", [])
    # Dati delle cartelle: percorso e timestamp dell'ultima modifica
    folder_info = []
    for grp in groups:
        for f in grp.get("folders", []):
            try:
                mtime = os.path.getmtime(f)
            except OSError:
                mtime = 0
            folder_info.append((str(f), mtime))
    key_data = {
        "cache_version": CACHE_VERSION,
        "groups": [{"label": g["label"], "is_reference": g.get("is_reference", False)}
                   for g in groups],
        "start_ppm": config.get("start_ppm"),
        "end_ppm": config.get("end_ppm"),
        "folders_info": folder_info,
        "metabolite_regions": METABOLITE_REGIONS,
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()

def load_cache(config_name: str, config: Dict[str, Any]) -> Optional[dict]:
    cache_path = _cache_path(config_name, config)
    if not cache_path.exists():
        return None
    try:
        payload = load(cache_path)
        if payload.get("key") == _build_cache_key(config):
            return payload["analysis_results"]
        else:
            print(f"Cache obsoleta per '{config_name}'.")
            return None
    except Exception as e:
        print(colored(f"Errore cache per '{config_name}': {e}", "red", attrs=["bold"]))
        return None

def save_cache(config_name: str, config: Dict[str, Any], analysis_results: dict) -> None:
    cache_path = _cache_path(config_name, config)
    payload = {"key": _build_cache_key(config), "analysis_results": analysis_results}
    dump(payload, cache_path, compress=3)
    print(f"Cache salvata per '{config_name}' in {cache_path.name}")
