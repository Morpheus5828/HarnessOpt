import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyvista as pv
from scipy.spatial import cKDTree
from datetime import datetime
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import config


class Part:
    def __init__(self, name: str, path: str, color):
        self.name = name
        self.path = str(path)
        self.color = color
        # Chargement immédiat du maillage individuel
        self.mesh = pv.read(self.path)


class MeshesFusion:
    def __init__(self, folder_path: str, part_color_path: str | None = None, fusion_mesh_path: str | None = None,
                 progress_callback=None):
        """
        Initialise le moteur de fusion.
        progress_callback: function(current, total, message, percentage)
        """
        self.part_color_path = str(part_color_path)
        self.folder_path = Path(folder_path)
        self.parts: list[Part] = []
        self.mesh = None
        self.progress_callback = progress_callback

        if fusion_mesh_path is not None:
            self.mesh = pv.read(str(fusion_mesh_path))

        self._attribute_surface_mesh()

    def _attribute_surface_mesh(self):
        if not Path(self.part_color_path).exists():
            return

        part_color = pd.read_excel(self.part_color_path)

        tasks = []
        for _, row in part_color.iterrows():
            if pd.isna(row.get("Part Number")): continue

            part_name = str(row["Part Number"])
            filename = part_name if part_name.lower().endswith(".stl") else f"{part_name}.stl"
            path = self.folder_path / filename

            if path.exists():
                tasks.append((part_name, str(path), row.get("Color")))

        total = len(tasks)
        if total == 0: return

        loaded_parts = [None] * total
        completed_count = 0

        def load_single_stl(index, name, path_str, color):
            try:
                return index, Part(name, path_str, color)
            except Exception as e:
                print(f"Erreur chargement {name}: {e}")
                return index, None

        max_threads = min(32, (os.cpu_count() or 4) * 2)

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(load_single_stl, i, n, p, c) for i, (n, p, c) in enumerate(tasks)]

            for future in as_completed(futures):
                idx, part_obj = future.result()
                if part_obj:
                    loaded_parts[idx] = part_obj

                completed_count += 1

                if self.progress_callback and completed_count % 5 == 0:
                    pct = (completed_count / total) * 0.4
                    self.progress_callback(completed_count, total, f"📦 Chargement STLs: {completed_count}/{total}", pct)

        self.parts = [p for p in loaded_parts if p is not None]

        if self.progress_callback:
            self.progress_callback(total, total, f"📦 Chargement STLs terminé", 0.4)

    def compute_fusion(self):
        """Fusionne tous les maillages chargés en un seul bloc."""
        if not self.parts:
            return None

        meshes = [p.mesh for p in self.parts if p.mesh is not None]

        # --- MISE À JOUR UI (45%) ---
        if self.progress_callback:
            self.progress_callback(len(meshes), len(meshes), "🧬 Fusion PyVista en cours...", 0.45)

        # Fusion des points et surfaces (merge_points=True pour nettoyer les doublons)
        self.mesh = pv.merge(meshes, merge_points=True)
        return self.mesh

    def save(self, output_dir: str | Path, meta_filename: str = "metadata.json"):
        """Sauvegarde le mesh global et les pièces individuelles avec suivi UI."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        parts_dir = output_dir / "parts"
        parts_dir.mkdir(exist_ok=True)

        parts_meta = []
        total_parts = len(self.parts)

        # --- SAUVEGARDE DES PIÈCES (90% -> 98%) ---
        for i, part in enumerate(self.parts):
            stl_filename = f"{part.name}.stl" if not part.name.lower().endswith(".stl") else part.name
            rel_path = Path("parts") / stl_filename
            dest_path = output_dir / rel_path

            if part.mesh:
                part.mesh.save(str(dest_path))
            else:
                shutil.copy(part.path, dest_path)

            parts_meta.append({
                "name": part.name,
                "path": str(rel_path),
                "color": part.color
            })

            if self.progress_callback and i % 5 == 0:
                pct = 0.9 + (i / total_parts) * 0.08
                self.progress_callback(i, total_parts, f"💾 Sauvegarde STLs: {i}/{total_parts}", pct)

        # --- SAUVEGARDE DU MESH GLOBAL VTP (98% -> 100%) ---
        fusion_filename = "fusion_main.vtp"
        if self.mesh:
            if self.progress_callback:
                self.progress_callback(total_parts, total_parts, "💾 Finalisation du fichier VTP...", 0.98)
            # Le format VTP est beaucoup plus rapide et léger pour le chargement ultérieur
            self.mesh.save(str(output_dir / fusion_filename))
        else:
            fusion_filename = None

        # Sauvegarde des Metadata
        metadata = {
            "folder_path": str(self.folder_path),
            "part_color_path": str(self.part_color_path),
            "fusion_mesh_file": fusion_filename,
            "parts": parts_meta,
            "timestamp": datetime.now().isoformat()
        }

        with open(output_dir / meta_filename, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        if self.progress_callback:
            self.progress_callback(total_parts, total_parts, "✅ Sauvegarde terminée !", 1.0)

    @classmethod
    def load(cls, saved_dir: str | Path, meta_filename: str = "metadata.json", max_workers: int = 20,
             load_parts: bool = True):
        """Charge une session de fusion précédente (Mode Light ou Full)."""
        saved_dir = Path(saved_dir)
        meta_path = saved_dir / meta_filename
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata introuvable: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        obj = cls.__new__(cls)
        obj.folder_path = Path(meta.get("folder_path"))
        obj.part_color_path = meta.get("part_color_path")
        obj.progress_callback = None  # Pas de callback sur un load statique classmethod
        obj.parts = []
        obj.mesh = None

        if load_parts:
            parts_data = meta.get("parts", [])
            loaded_parts = [None] * len(parts_data)

            def load_single_part(index, p_data):
                try:
                    rel_path = p_data.get("path")
                    if not rel_path: return index, None
                    part_path = str(saved_dir / rel_path)
                    return index, Part(p_data["name"], part_path, p_data.get("color"))
                except:
                    return index, None

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(load_single_part, i, p) for i, p in enumerate(parts_data)]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Chargement Session"):
                    idx, p_obj = future.result()
                    if p_obj: loaded_parts[idx] = p_obj

            obj.parts = [p for p in loaded_parts if p is not None]

        fusion_file = meta.get("fusion_mesh_file")
        if fusion_file and (saved_dir / fusion_file).exists():
            obj.mesh = pv.read(str(saved_dir / fusion_file))

        return obj

    def get_fixation_point(self, pts, radius: float = None):
        """Récupère les points proches de la structure pour le routage."""
        if radius is None:
            radius = getattr(config, "FIXATION_RADIUS", 100.0)

        struct_points = [p.mesh.points for p in self.parts if p.mesh and str(p.color).lower() == "structure"]
        if not struct_points: return []

        all_struct_pts = np.vstack(struct_points)
        tree = cKDTree(all_struct_pts)
        indices = tree.query_ball_point(pts, r=radius)
        return [pts[i] for i, idx_list in enumerate(indices) if len(idx_list) > 0]
