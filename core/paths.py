"""Emplacements de travail de HarnessOpt (cache, exports, réglages).

Ce module remplace les chemins Windows écrits en dur un peu partout dans
l'application (``C:\\Temp\\HarnessOpt_cache``, ``C:\\Users\\...\\Desktop\\...``).
Il expose un cache utilisable sur n'importe quelle machine et un petit fichier
de réglages JSON qui mémorise, d'une session à l'autre, ce que l'utilisateur a
saisi dans l'interface (dossier STL, dossier des clamps, point A/B, etc.).

Ordre de résolution du dossier de cache :

1. la variable d'environnement ``HARNESSOPT_CACHE`` si elle est définie ;
2. ``%LOCALAPPDATA%\\HarnessOpt\\cache`` sous Windows ;
3. ``~/.cache/harnessopt`` ailleurs (Linux, macOS).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

APP_NAME = "HarnessOpt"

#: Sous-dossiers toujours présents dans le cache.
CACHE_SUBFOLDERS = (
    "stl",
    "color",
    "fusion",
    "sphere_generations",
    "graphs",
    "paths",
    "runs",
)


def _default_cache_root() -> Path:
    env = os.environ.get("HARNESSOPT_CACHE")
    if env:
        return Path(env).expanduser()

    if sys.platform.startswith("win"):
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / APP_NAME / "cache"
        return Path.home() / "AppData" / "Local" / APP_NAME / "cache"

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / APP_NAME.lower()
    return Path.home() / ".cache" / APP_NAME.lower()


def _default_config_root() -> Path:
    env = os.environ.get("HARNESSOPT_CONFIG")
    if env:
        return Path(env).expanduser()

    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / APP_NAME
        return Path.home() / "AppData" / "Roaming" / APP_NAME

    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / APP_NAME.lower()
    return Path.home() / ".config" / APP_NAME.lower()


BASE_CACHE: Path = _default_cache_root()
CONFIG_ROOT: Path = _default_config_root()
SETTINGS_PATH: Path = CONFIG_ROOT / "settings.json"

STL_DIR: Path = BASE_CACHE / "stl"
COLOR_DIR: Path = BASE_CACHE / "color"
FUSION_DIR: Path = BASE_CACHE / "fusion"
RUNS_DIR: Path = BASE_CACHE / "runs"

#: Maillage fusionné de l'environnement, produit par l'étape d'extraction et
#: relu par les agents.
FUSED_MESH_PATH: Path = FUSION_DIR / "clipped_obstacles.vtk"

#: Table « face -> famille de couleur DMU » produite en même temps que la
#: fusion (voir :mod:`core.routing_rules`). Absente = clearance uniforme.
FACE_FAMILY_PATH: Path = FUSION_DIR / "face_families.npz"


def ensure_cache_folders(root: Path | str | None = None) -> Path:
    """Crée l'arborescence de cache et renvoie sa racine.

    En cas de dossier non inscriptible (poste verrouillé, lecteur réseau
    indisponible), on bascule sur un dossier temporaire plutôt que de faire
    planter le démarrage de l'application.
    """
    target = Path(root) if root is not None else BASE_CACHE
    try:
        for sub in ("",) + CACHE_SUBFOLDERS:
            (target / sub).mkdir(parents=True, exist_ok=True)
        return target
    except OSError:
        fallback = Path(tempfile.gettempdir()) / f"{APP_NAME}_cache"
        for sub in ("",) + CACHE_SUBFOLDERS:
            (fallback / sub).mkdir(parents=True, exist_ok=True)
        return fallback


def cache_has_content(root: Path | str | None = None) -> bool:
    """Indique si le cache contient des données d'une session précédente."""
    target = Path(root) if root is not None else BASE_CACHE
    if not target.exists():
        return False
    for _dirpath, _dirnames, filenames in os.walk(target):
        if filenames:
            return True
    return False


def cache_size_bytes(root: Path | str | None = None) -> int:
    """Taille cumulée du cache, en octets (0 s'il n'existe pas)."""
    target = Path(root) if root is not None else BASE_CACHE
    total = 0
    if not target.exists():
        return 0
    for dirpath, _dirnames, filenames in os.walk(target):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                continue
    return total


def clear_cache(root: Path | str | None = None) -> Path:
    """Vide le cache puis recrée l'arborescence vide."""
    target = Path(root) if root is not None else BASE_CACHE
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    return ensure_cache_folders(target)


def load_settings() -> dict:
    """Relit les réglages utilisateur (dict vide si absent ou illisible)."""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict) -> bool:
    """Enregistre les réglages utilisateur. Renvoie False en cas d'échec."""
    try:
        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2, ensure_ascii=False)
        tmp.replace(SETTINGS_PATH)
        return True
    except (OSError, TypeError, ValueError):
        return False


def update_settings(**values) -> dict:
    """Fusionne ``values`` dans les réglages existants et les enregistre."""
    settings = load_settings()
    settings.update({k: v for k, v in values.items() if v is not None})
    save_settings(settings)
    return settings


def human_size(num_bytes: int) -> str:
    """Formate une taille en octets pour affichage dans l'IHM."""
    size = float(num_bytes)
    for unit in ("o", "Ko", "Mo", "Go"):
        if size < 1024.0 or unit == "Go":
            return f"{size:.0f} {unit}" if unit == "o" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} Go"
