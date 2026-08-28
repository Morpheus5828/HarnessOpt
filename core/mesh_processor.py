import os
import json
import pythoncom
import concurrent.futures
import traceback
import pyvista as pv
from pathlib import Path

from core.mesh_model import MeshModel
from core.catia_handler import BASE_CACHE
from core.catia_handler import run_catia_export_via_vba


def extraction_worker(target_dir, use_catia, exclude_filter, res_queue):
    try:
        pythoncom.CoInitialize()
        model = MeshModel(BASE_CACHE)

        def send_progress(curr, tot, msg, pct):
            res_queue.put(("UPDATE", msg, curr, tot, pct))

        actual_stl_dir = str(model.stl_folder) if use_catia else target_dir

        if use_catia:
            start_pct = 0.60
            range_pct = 0.35
            send_progress(0, 1, "🛰️ Export CATIA...", 0.05)

            run_catia_export_via_vba(exclude_filter)
            df = model.run_color_analysis(actual_stl_dir, progress_callback=send_progress)
        else:
            start_pct = 0.05
            range_pct = 0.90
            send_progress(0, 1, "📂 Lecture des fichiers locaux et Analyse ATA...", 0.05)

            if not os.path.exists(actual_stl_dir):
                res_queue.put(("ERROR", f"❌ Le dossier spécifié n'existe pas :\n{actual_stl_dir}"))
                return

            stl_files = [f for f in os.listdir(actual_stl_dir) if f.lower().endswith(".stl")]

            if len(stl_files) == 0:
                joli_message = (
                    f"⚠️ Dossier vide ou invalide\n\n"
                    f"Aucun fichier '.stl' n'a été trouvé dans le répertoire :\n"
                    f"📁 {actual_stl_dir}\n\n"
                    f"Veuillez vérifier votre chemin ou exporter depuis CATIA."
                )
                res_queue.put(("ERROR", joli_message))
                return

            df = model.run_color_analysis(actual_stl_dir, progress_callback=send_progress)

        # =========================================================
        # CALCUL DES BBOX ET FUSION VTK (CORRECTION ICI)
        # =========================================================
        send_progress(0, 1, "📐 Fusion des maillages en cours...", start_pct)

        glob_xmin, glob_xmax = float('inf'), float('-inf')
        glob_ymin, glob_ymax = float('inf'), float('-inf')
        glob_zmin, glob_zmax = float('inf'), float('-inf')

        valid_stls = [os.path.join(actual_stl_dir, f) for f in os.listdir(actual_stl_dir) if f.lower().endswith(".stl")]
        total_stls = len(valid_stls)

        merged_mesh = pv.PolyData()

        # On fusionne tous les STLs en un seul bloc pour l'environnement des Agents
        for i, stl in enumerate(valid_stls):
            try:
                mesh = pv.read(stl)
                merged_mesh = merged_mesh.merge(mesh)
                b = mesh.bounds
                glob_xmin, glob_xmax = min(glob_xmin, b[0]), max(glob_xmax, b[1])
                glob_ymin, glob_ymax = min(glob_ymin, b[2]), max(glob_ymax, b[3])
                glob_zmin, glob_zmax = min(glob_zmin, b[4]), max(glob_zmax, b[5])
            except Exception as e:
                print(f"⚠️ Erreur Fusion sur un STL : {e}")

            if i % 10 == 0 or i == total_stls - 1:
                current_pct = start_pct + ((i + 1) / total_stls) * range_pct
                send_progress(i + 1, total_stls, f"📐 Fusion ({i + 1}/{total_stls})...", current_pct)

        bounds = (glob_xmin, glob_xmax, glob_ymin, glob_ymax, glob_zmin, glob_zmax)

        # 🔥 SAUVEGARDE DU FICHIER VTK POUR L'AGENT
        fusion_dir = os.path.join(BASE_CACHE, "fusion")
        os.makedirs(fusion_dir, exist_ok=True)
        fusion_path = os.path.join(fusion_dir, "clipped_obstacles.vtk")

        if merged_mesh.n_points > 0:
            merged_mesh.save(fusion_path)
            print(f"✅ Maillage fusionné généré avec succès : {fusion_path}")

        meta_path = Path(BASE_CACHE) / "color" / "extraction_meta.json"
        with open(meta_path, "w") as f:
            json.dump({"stl_dir": actual_stl_dir}, f)

        res_queue.put(("UPDATE", "⚙️ Finalisation de l'extraction...", 100, 100, 1.0))
        res_queue.put(("SUCCESS", df, 0, len(valid_stls), bounds, fusion_path))

    except Exception as e:
        res_queue.put(("ERROR", f"Une erreur critique est survenue :\n{str(e)}\n\n{traceback.format_exc()}"))
    finally:
        pythoncom.CoUninitialize()
