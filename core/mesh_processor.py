"""Étape 1 du cycle de routage : extraction des STL puis fusion du DMU.

Le travail est exécuté dans un processus séparé (voir ``extraction_worker``) et
communique avec l'interface par une ``multiprocessing.Queue``.

Deux modes :

* **Export CATIA** : on demande à CATIA d'exporter les pièces en STL. Ce mode
  nécessite Windows + CATIA + le module ``core.catia_handler`` (propre au poste
  de travail, non versionné). L'import est volontairement *paresseux* pour que
  tout le reste de l'application fonctionne sans CATIA.
* **STL existants** : on lit un dossier de STL déjà exportés. Ce mode n'a
  besoin d'aucune dépendance CATIA et fonctionne sur n'importe quelle machine.

En plus du maillage fusionné, l'étape produit une table « face -> famille de
couleur DMU » (``face_families.npz``) : c'est elle qui permet ensuite
d'appliquer une distance de sécurité différente selon la nature de la pièce
survolée (structure, air chaud, hydraulique haute pression, ...).
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import numpy as np
import pyvista as pv

from core.mesh_model import MeshModel
from core.paths import BASE_CACHE, FACE_FAMILY_PATH, FUSED_MESH_PATH, ensure_cache_folders

#: Nombre de STL fusionnés d'un bloc. Fusionner par paquets évite à la fois le
#: coût quadratique d'un ``merge`` par fichier et un pic mémoire sur les DMU
#: comportant plusieurs milliers de pièces.
_MERGE_CHUNK = 64


class CatiaUnavailable(RuntimeError):
    """Levée quand l'export CATIA est demandé mais indisponible sur le poste."""


def _import_catia_exporter():
    """Importe l'exportateur CATIA à la demande.

    L'import reste tardif pour que l'application démarre même si le connecteur
    a été retiré du poste. Le connecteur lui-même s'importe sur toutes les
    plateformes : c'est à l'appel qu'il signale l'absence de pywin32 ou de
    CATIA, avec un message distinct dans chaque cas.
    """
    try:
        from core.catia_handler import run_catia_export_via_vba  # type: ignore
    except ImportError as exc:
        raise CatiaUnavailable(
            "Le connecteur CATIA (core/catia_handler.py) est introuvable sur "
            "ce poste.\n\n"
            "Choisissez « Un dossier de fichiers STL déjà exportés » pour "
            "travailler à partir de STL existants."
        ) from exc
    return run_catia_export_via_vba


def _init_com():
    """Initialise COM sous Windows ; sans effet ailleurs."""
    try:
        import pythoncom  # type: ignore
    except ImportError:
        return None
    pythoncom.CoInitialize()
    return pythoncom


def _list_stl(directory: str) -> list[str]:
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    return [os.path.join(directory, n) for n in names if n.lower().endswith(".stl")]


def fuse_stl_folder(stl_paths, family_of_part=None, progress=None):
    """Fusionne des STL en un seul maillage et trace l'origine de chaque face.

    Args:
        stl_paths: chemins des fichiers ``.stl`` à fusionner.
        family_of_part: dict ``nom de fichier -> famille de couleur DMU``.
            Utilisé pour construire la table face -> famille.
        progress: callback ``(index, total)`` appelé pendant la fusion.

    Returns:
        ``(merged_mesh, bounds, face_family_codes, family_names)`` où
        ``face_family_codes`` est un tableau d'entiers de longueur ``n_cells``
        indexant ``family_names``.
    """
    family_of_part = family_of_part or {}
    family_names: list[str] = []
    family_index: dict[str, int] = {}

    def code_for(name: str) -> int:
        family = family_of_part.get(name, "standard")
        if family not in family_index:
            family_index[family] = len(family_names)
            family_names.append(family)
        return family_index[family]

    chunk: list[pv.PolyData] = []
    merged: pv.PolyData | None = None
    codes: list[np.ndarray] = []
    total = len(stl_paths)

    def flush():
        nonlocal merged, chunk
        if not chunk:
            return
        block = chunk[0] if len(chunk) == 1 else chunk[0].merge(chunk[1:])
        merged = block if merged is None else merged.merge(block)
        chunk = []

    for i, path in enumerate(stl_paths):
        try:
            mesh = pv.read(path).triangulate()
        except Exception as exc:  # STL corrompu : on continue sans le bloquer
            print(f"⚠️ STL ignoré ({os.path.basename(path)}) : {exc}")
            continue

        n_cells = int(mesh.n_cells)
        if n_cells == 0:
            continue

        codes.append(np.full(n_cells, code_for(os.path.basename(path)), dtype=np.int16))
        chunk.append(mesh)
        if len(chunk) >= _MERGE_CHUNK:
            flush()

        if progress is not None and (i % 10 == 0 or i == total - 1):
            progress(i + 1, total)

    flush()

    if merged is None or merged.n_points == 0:
        return None, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), np.zeros(0, dtype=np.int16), []

    face_codes = np.concatenate(codes) if codes else np.zeros(0, dtype=np.int16)
    return merged, tuple(float(b) for b in merged.bounds), face_codes, family_names


def extraction_worker(target_dir, use_catia, exclude_filter, res_queue):
    """Point d'entrée du processus d'extraction (voir le contrôleur)."""
    com = _init_com()
    try:
        ensure_cache_folders()
        model = MeshModel(BASE_CACHE)

        def send_progress(curr, tot, msg, pct):
            res_queue.put(("UPDATE", msg, curr, tot, pct))

        actual_stl_dir = str(model.stl_folder) if use_catia else target_dir

        if use_catia:
            start_pct, range_pct = 0.60, 0.35
            send_progress(0, 1, "🛰️ Export CATIA en cours...", 0.05)
            run_catia_export_via_vba = _import_catia_exporter()
            try:
                exported_dir = run_catia_export_via_vba(exclude_filter)
            except RuntimeError as exc:
                # CATIA absent, non lancé, ou macro en échec : ce sont des
                # situations attendues, pas des bogues. On les remonte telles
                # quelles à l'utilisateur plutôt qu'avec une pile d'appels.
                raise CatiaUnavailable(str(exc)) from exc
            if exported_dir:
                # On relit le dossier réellement écrit par la macro plutôt que
                # de le supposer : les deux modules ne peuvent plus diverger.
                actual_stl_dir = str(exported_dir)
        else:
            start_pct, range_pct = 0.05, 0.90
            send_progress(0, 1, "📂 Lecture du dossier STL...", 0.05)

            if not os.path.isdir(actual_stl_dir):
                res_queue.put(("ERROR", f"❌ Le dossier spécifié n'existe pas :\n{actual_stl_dir}"))
                return

            if not _list_stl(actual_stl_dir):
                res_queue.put((
                    "ERROR",
                    "⚠️ Dossier vide ou invalide\n\n"
                    "Aucun fichier '.stl' n'a été trouvé dans le répertoire :\n"
                    f"📁 {actual_stl_dir}\n\n"
                    "Vérifiez le chemin, ou lancez un export depuis CATIA.",
                ))
                return

        df = model.run_color_analysis(actual_stl_dir, progress_callback=send_progress)
        family_of_part = dict(zip(df["Part Number"], df["Color"])) if df is not None else {}

        send_progress(0, 1, "📐 Fusion des maillages en cours...", start_pct)
        stl_paths = _list_stl(actual_stl_dir)
        total_stls = len(stl_paths)

        def on_merge_progress(done, tot):
            pct = start_pct + (done / max(1, tot)) * range_pct
            send_progress(done, tot, f"📐 Fusion ({done}/{tot})...", pct)

        merged_mesh, bounds, face_codes, family_names = fuse_stl_folder(
            stl_paths, family_of_part=family_of_part, progress=on_merge_progress
        )

        if merged_mesh is None:
            res_queue.put(("ERROR", "❌ La fusion n'a produit aucune géométrie exploitable."))
            return

        FUSED_MESH_PATH.parent.mkdir(parents=True, exist_ok=True)
        merged_mesh.save(str(FUSED_MESH_PATH))
        print(f"✅ Maillage fusionné généré : {FUSED_MESH_PATH}")

        # Table face -> famille de couleur : c'est elle qui autorise une
        # distance de sécurité différente selon la pièce survolée.
        np.savez_compressed(
            str(FACE_FAMILY_PATH),
            face_family=face_codes,
            family_names=np.array(family_names, dtype=object),
        )

        meta_path = Path(BASE_CACHE) / "color" / "extraction_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "stl_dir": actual_stl_dir,
                    "n_parts": total_stls,
                    "n_cells": int(merged_mesh.n_cells),
                    "bounds": list(bounds),
                    "families": family_names,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

        res_queue.put(("UPDATE", "⚙️ Finalisation de l'extraction...", 100, 100, 1.0))
        res_queue.put((
            "SUCCESS",
            df,
            int(merged_mesh.n_points),
            total_stls,
            bounds,
            str(FUSED_MESH_PATH),
        ))

    except CatiaUnavailable as exc:
        res_queue.put(("ERROR", str(exc)))
    except Exception as exc:
        res_queue.put(("ERROR", f"Une erreur critique est survenue :\n{exc}\n\n{traceback.format_exc()}"))
    finally:
        if com is not None:
            com.CoUninitialize()
