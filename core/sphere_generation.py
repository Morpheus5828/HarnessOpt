import os
import sys
import glob
import json
import joblib
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

from core.mesh_fusion import MeshesFusion

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

from core.HS9019 import Rules
from config import *


def export_html_plotly(stl: pv.PolyData, points: list, out_html: str):
    import plotly.graph_objects as go
    tri = stl.triangulate()
    faces = tri.faces.reshape((-1, 4))[:, 1:4]
    verts = tri.points

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            opacity=0.35, name="STL"))

    P = np.asarray(points, dtype=float)
    hover_text = [f"x={x:.3f}<br>y={y:.3f}<br>z={z:.3f}" for x, y, z in P]
    fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                               name="valid_points", text=hover_text, hoverinfo="text",
                               marker=dict(size=4)))

    fig.update_layout(title="Sphere generation points", scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    print(f"[HTML] Exported -> {out_html}")


def create_point(
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
        dx: float,
        dy: float,
        dz: float,
        device: str = "cuda"
):
    eps_x, eps_y, eps_z = dx * 0.5, dy * 0.5, dz * 0.5

    xs = np.arange(x_min, x_max + eps_x, dx, dtype=np.float32)
    ys = np.arange(y_min, y_max + eps_y, dy, dtype=np.float32)
    zs = np.arange(z_min, z_max + eps_z, dz, dtype=np.float32)

    nx, ny, nz = len(xs), len(ys), len(zs)
    total = nx * ny * nz

    print(f"XS shape: {xs.shape}")
    print(f"YS shape: {ys.shape}")
    print(f"ZS shape: {zs.shape}")
    print(f"[Grid spacing] dx={dx:.3f} mm, dy={dy:.3f} mm, dz={dz:.3f} mm")
    print(f"[Grid size] nx={nx}, ny={ny}, nz={nz} -> total={total:,} points sur CPU")

    grid = np.meshgrid(xs, ys, zs, indexing="ij")
    all_points = np.stack(grid, axis=-1).reshape(-1, 3)

    return all_points, (nx, ny, nz)


def _generate_sphere(pts, pl, radius):
    spheris = pv.Sphere(center=pts, radius=radius)
    pl.add_mesh(spheris)


def _is_valid_point(pts, mesh_fusion, radius, hs_rules):
    return pts if not _clash(pts, mesh_fusion, radius, hs_rules) else None


def _get_distance(stl, pts):
    implicit = pv.ImplicitPolyDataDistance(stl)
    return abs(implicit.evaluate_function(pts))


# 🟢 MODIFICATION : Remplacement du CellLocator VTK par l'outil de distance natif PyVista
def build_implicit_dist(poly: "pv.PolyData"):
    if not isinstance(poly, pv.PolyData):
        poly = poly.extract_surface(algorithm='dataset_surface')
    return pv.ImplicitPolyDataDistance(poly)


# 🟢 MODIFICATION : Utilisation de evaluate_function sans pointeurs C++
def distance_point_surface(implicit_dist, pts):
    return abs(implicit_dist.evaluate_function(np.asarray(pts)))


# 🟢 MODIFICATION : Signature mise à jour pour prendre l'implicit_dist PyVista
def _clash_polydata(pts, radius, hs_rules, implicit_dist):
    d_surf = distance_point_surface(implicit_dist, pts)
    if d_surf - radius <= hs_rules.DISTANCE_WITH_STRUCTURE:
        return True
    return False


# 🟢 MODIFICATION : Enchaînement propre
def _is_valid_point_polydata(pts, radius, hs_rules, implicit_dist):
    if not _clash_polydata(pts, radius, hs_rules, implicit_dist):
        return pts
    return None


# =========================================================
# 💡 ANALYSE DE CLASH OPTIMISÉE POUR TOUTES LES COULEURS ATA
# =========================================================
def _clash(pts, mesh_fusion, radius, hs_rules):
    # 1. Vérification de la distance minimale globale avec la structure
    tree = cKDTree(mesh_fusion.mesh.points)
    d_kdtree, idx = tree.query(pts)
    if d_kdtree - radius <= hs_rules.DISTANCE_WITH_STRUCTURE:
        return True

    # 2. Boucle d'isolation par règles métier ATA sur chaque part du DMU
    for part in mesh_fusion.parts:
        distance = _get_distance(part.mesh, pts)

        # Isolation Conditionnelle : Ventilation & Circuit d'Air (U212)
        if part.color == "ecs_air_circuit":
            if distance < hs_rules.DISTANCE_WITH_VENTILATION_REFRIGERANT:
                return True

        # Isolation Conditionnelle : Lignes d'Air Chaud (U29 - Haute Pression)
        elif part.color == "high_pressure_system":
            if distance < hs_rules.DISTANCE_HOT_AIR_LINES:
                return True

        # Isolation Conditionnelle : Hydraulique Haute Pression (U29 - Retour / Aspiration / Azote)
        elif part.color in ["return_system", "air_azote", "suction"]:
            if distance < getattr(hs_rules, "DISTANCE_WITH_HIGH_PRESSURE_HYDRAULIC_LINE", 25.0):
                return True

        # Isolation Conditionnelle : Circuit Carburant (U28 - Fuel)
        elif part.color == "fuel":
            if distance < getattr(hs_rules, "DISTANCE_WITH_FUEL_SYSTEM", 50.0):
                return True

        # Isolation Conditionnelle : Circuits d'Air Froid et Spécifiques (U214 / U215)
        elif part.color in ["ecs_cold_circuit", "p3_circuit"]:
            if distance < getattr(hs_rules, "DISTANCE_WITH_COLD_CIRCUIT", 20.0):
                return True


def create_obb_mesh(center, dist_axis, med_axis, thru_axis, dist_min, dist_max, med_min, med_max, thru_min, thru_max):
    cube = pv.Cube()
    lengths = [dist_max - dist_min, med_max - med_min, thru_max - thru_min]
    local_center_offset = np.array(
        [(dist_min + dist_max) / 2.0, (med_min + med_max) / 2.0, (thru_min + thru_max) / 2.0])
    cube.scale(lengths, inplace=True)
    cube.points += local_center_offset
    rot_matrix = np.eye(4)
    rot_matrix[0:3, 0], rot_matrix[0:3, 1], rot_matrix[0:3, 2] = dist_axis, med_axis, thru_axis
    cube.transform(rot_matrix, inplace=True)
    cube.points += center
    return cube


def compute_clamp_points(mesh, preds_ml, tube_radius, clamp_margin=0.5):
    pts = np.array(mesh.points)
    if len(pts) == 0:
        return []

    center = np.mean(pts, axis=0)

    cov = np.cov(pts.T)
    _, evecs = np.linalg.eigh(cov)
    axis_row = evecs[:, 2]

    mesh_with_normals = mesh.compute_normals(cell_normals=True, point_normals=False)
    normals = mesh_with_normals.cell_data['Normals']

    dots = np.abs(np.dot(normals, axis_row))
    lateral_normals = normals[dots < 0.2]

    if len(lateral_normals) > 0:
        cov_norm = np.cov(lateral_normals.T)
        _, eig_norm = np.linalg.eigh(cov_norm)
        axis_thru = eig_norm[:, 2]
    else:
        axis_thru = evecs[:, 0]

    axis_normal = np.cross(axis_row, axis_thru)
    axis_normal /= np.linalg.norm(axis_normal)

    proj_normal = np.dot(pts - center, axis_normal)
    max_proj = np.max(proj_normal)
    min_proj = np.min(proj_normal)

    if abs(min_proj) <= abs(max_proj):
        axis_normal = -axis_normal
        max_proj = abs(min_proj)
    else:
        max_proj = abs(max_proj)

    offset_distance = max_proj + tube_radius + clamp_margin
    routing_center = center + offset_distance * axis_normal

    proj_row = np.dot(pts - center, axis_row)
    row_min, row_max = np.min(proj_row), np.max(proj_row)
    total_row_length = row_max - row_min
    unit_width = total_row_length / preds_ml

    proj_thru = np.dot(pts - center, axis_thru)
    thru_min, thru_max = np.min(proj_thru), np.max(proj_thru)
    slot_depth = thru_max - thru_min
    offset_ecart = (slot_depth / 2.0) + tube_radius + clamp_margin

    points_in_out = []
    for i in range(preds_ml):
        slot_center_local = row_min + (unit_width / 2.0) + (i * unit_width)
        local_center = routing_center + axis_row * slot_center_local

        pt_in = local_center - axis_thru * offset_ecart
        pt_out = local_center + axis_thru * offset_ecart
        points_in_out.extend([pt_in, pt_out])

    return points_in_out


def load_predictions(filepath):
    annotations = {}
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    fname, val = line.strip().split(":")
                    try:
                        annotations[fname] = int(val)
                    except ValueError:
                        pass
    return annotations


def extract_smart_features(mesh):
    n_pts = mesh.n_points
    n_faces = mesh.n_cells
    vol = mesh.volume
    area = mesh.area
    compacity = area / (vol ** (2 / 3)) if vol > 0 else 0
    pts = mesh.points
    center = np.mean(pts, axis=0)
    cov = np.cov((pts - center).T)
    eigenvalues, _ = np.linalg.eigh(cov)
    true_thickness = np.sqrt(np.abs(eigenvalues[0]))
    true_height = np.sqrt(np.abs(eigenvalues[1]))
    true_length = np.sqrt(np.abs(eigenvalues[2]))
    elongation = true_length / true_thickness if true_thickness > 0.001 else 0
    return np.array([n_pts, n_faces, vol, area, compacity, true_length, true_height, elongation]).reshape(1, -1)


def run_color_analysis(self, target_dir, progress_callback=None):
    stl_stems = [p for p in os.listdir(target_dir) if p.lower().endswith('.stl')]
    total = len(stl_stems)
    if total == 0:
        raise ValueError("Aucun fichier STL trouvé dans le dossier cible.")

    all_parts, all_color, all_presence = [], [], []

    for i, stl in enumerate(stl_stems):
        part_num = stl.split(".")[0].split("_")[0]
        current_color = "standard"

        if len(part_num) > 1:
            c2 = part_num[1]

            if c2 == "1":
                current_color = "equipement"

            elif c2 == "2" and len(part_num) > 2:
                c3 = part_num[2]

                if c3 == "1" and len(part_num) > 3:
                    c4 = part_num[3]
                    if c4 == "1":
                        current_color = "ecs_air_circuit"
                    elif c4 == "4":
                        current_color = "p3_circuit"
                    elif c4 == "5":
                        current_color = "ecs_cold_circuit"
                    elif c4 == "6":
                        current_color = "equipement"

                elif c3 == "8":
                    current_color = "fuel"

                elif c3 == "9" and len(part_num) > 3:
                    c4 = part_num[3]
                    if c4 == "1":
                        current_color = "high_pressure_system"
                    elif c4 == "2":
                        current_color = "return_system"
                    elif c4 == "3":
                        current_color = "air_azote"
                    elif c4 == "4":
                        current_color = "suction"

            elif c2 == "5":
                current_color = "structure"

            elif c2 == "6" and len(part_num) > 2:
                c3 = part_num[2]
                if c3 == "7":
                    current_color = "flight_control_system-fcs"

        part_num_lower = part_num.lower()
        if "insonorisation" in part_num_lower:
            current_color = "insonorisation"
        elif "copper" in part_num_lower or "foil" in part_num_lower:
            current_color = "copper_foils"
        elif "mecanical" in part_num_lower:
            current_color = "mecanical_installation"

        all_parts.append(stl)
        all_color.append(current_color)
        all_presence.append(True)

        if progress_callback and i % 20 == 0:
            prog_pct = (i / total) * 0.4
            progress_callback(i, total, f"🎨 Analyse des couleurs: {i}/{total}", prog_pct)

    df = pd.DataFrame({"Part Number": all_parts, "Color": all_color, "Presence": all_presence})
    self.color_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(str(self.color_path), index=False)
    return df


def add_clamps_connection(clamps_folder, radius, bbox=None, clamp_margin=1.0):
    clamps_points = []
    model = joblib.load(MODEL_PATH)

    stl_files = []
    stl_files.extend(glob.glob(os.path.join(clamps_folder, "*.stl")))
    stl_files.extend(glob.glob(os.path.join(clamps_folder, "*.STL")))
    stl_files = sorted(list(set(stl_files)))

    print(f"\n🔍 {len(stl_files)} fichiers de clamps trouvés dans le dossier.")

    for filepath in stl_files:
        mesh = pv.read(filepath)

        if bbox is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = bbox
            center = mesh.center
            if not (x_min <= center[0] <= x_max and
                    y_min <= center[1] <= y_max and
                    z_min <= center[2] <= z_max):
                continue

        X_new = extract_smart_features(mesh)
        preds_ml = int(model.predict(X_new)[0])

        if preds_ml <= 0:
            print(
                f"⚠️ [IA] Le modèle a prédit 0 passage pour {os.path.basename(filepath)} (Maillage ouvert ?). Forçage à 1.")
            preds_ml = 1
        elif preds_ml > 20:
            print(f"🚨 [IA] Prédiction absurde ({preds_ml} passages) pour {os.path.basename(filepath)}. Forçage à 1.")
            preds_ml = 1
        else:
            print(f"✅ [IA] {preds_ml} passage(s) détecté(s) pour {os.path.basename(filepath)}.")

        points = compute_clamp_points(mesh, preds_ml, radius, clamp_margin)

        for i in range(0, len(points), 2):
            segment = {
                "in": points[i],
                "out": points[i + 1]
            }
            clamps_points.append(segment)

    return clamps_points


def point_generator(
        mesh_fusion,
        all_points,
        radius,
        n_jobs=16,
        clamps_folder=None,
        bbox=None,
        clamp_margin=1.0
):
    print(f"🛠️ DEBUG - Valeur reçue pour clamps_folder : '{clamps_folder}'")
    clamps_points = []

    if clamps_folder is not None and os.path.exists(clamps_folder.strip()):
        dossier_propre = clamps_folder.strip()
        print(f"🔩 Intégration des clamps activée. Chargement depuis : {dossier_propre}")

        raw_clamps = add_clamps_connection(dossier_propre, radius, bbox, clamp_margin)

        if raw_clamps:
            print("🛡️ Filtrage anti-clash des terminaux de clamps...")

            # 🟢 MODIFICATION : Plus de StaticCellLocator ni de vtk.reference.
            # Utilisation de la distance implicite absolue de PyVista.
            implicit_dist = pv.ImplicitPolyDataDistance(mesh_fusion.mesh)
            contact_tolerance = 0.5  # Équivalent de la racine carrée de 0.5**2

            for seg in raw_clamps:
                d_in = abs(implicit_dist.evaluate_function(np.asarray(seg["in"], dtype=np.float32)))
                d_out = abs(implicit_dist.evaluate_function(np.asarray(seg["out"], dtype=np.float32)))

                if d_in > contact_tolerance and d_out > contact_tolerance:
                    clamps_points.append(seg)

            print(f"✅ Clamps valides conservés : {len(clamps_points)} / {len(raw_clamps)}")

        if len(clamps_points) > 0:
            try:
                cache_dir = os.path.join(r"C:\Temp\HarnessOpt_cache", "sphere_generations")
                os.makedirs(cache_dir, exist_ok=True)
                clamps_out_path = os.path.join(cache_dir, "clamps_only.npz")
                np.savez_compressed(clamps_out_path, clamp_points=np.array(clamps_points, dtype=object))
                print(f"💾 [EXCEPTIONNEL] Fichier de débogage des clamps sauvegardé ici : {clamps_out_path}")
            except Exception as e:
                print(f"⚠️ Erreur lors de la sauvegarde des clamps en npz : {e}")
    else:
        print("⏭️ Intégration des clamps désactivée ou dossier introuvable.")

    print(f"📌 Points de clamps finaux validés : {len(clamps_points)}")
    print(
        f"Lancement du filtrage zone [0 - {DISTANCE_MAX_WITH_STRUCTURE}mm] sur la grille de {len(all_points):,} points...")

    total_input = len(all_points)
    hs_rules = Rules()
    pts = np.asarray(all_points, dtype=np.float32)

    tree = cKDTree(mesh_fusion.mesh.points)
    distances, _ = tree.query(pts, k=1, workers=n_jobs)

    dist_min = radius + hs_rules.DISTANCE_WITH_STRUCTURE
    mask_valid = (distances >= dist_min) & (distances <= DISTANCE_MAX_WITH_STRUCTURE)

    nb_trop_loin = np.sum(distances > DISTANCE_MAX_WITH_STRUCTURE)
    nb_collision = np.sum(distances < dist_min)
    del tree

    crit_colors = ["ecs_air_circuit", "high_pressure_system", "return_system", "air_azote", "suction", "fuel",
                   "ecs_cold_circuit", "p3_circuit"]
    relevant_parts = [p for p in mesh_fusion.parts if p.color in crit_colors and p.mesh.n_points > 0]

    if len(relevant_parts) > 0:
        for part in relevant_parts:
            idx_active = np.where(mask_valid)[0]
            if len(idx_active) == 0: break

            part_tree = cKDTree(part.mesh.points)
            d_part, _ = part_tree.query(pts[idx_active], k=1, workers=n_jobs)

            if part.color == "ecs_air_circuit":
                threshold = hs_rules.DISTANCE_WITH_VENTILATION_REFRIGERANT
            elif part.color == "high_pressure_system":
                threshold = hs_rules.DISTANCE_HOT_AIR_LINES
            elif part.color in ["return_system", "air_azote", "suction"]:
                threshold = getattr(hs_rules, "DISTANCE_WITH_HIGH_PRESSURE_HYDRAULIC_LINE", 25.0)
            elif part.color == "fuel":
                threshold = getattr(hs_rules, "DISTANCE_WITH_FUEL_SYSTEM", 50.0)
            elif part.color in ["ecs_cold_circuit", "p3_circuit"]:
                threshold = getattr(hs_rules, "DISTANCE_WITH_COLD_CIRCUIT", 20.0)
            else:
                threshold = 0.0

            mask_valid[idx_active] &= (d_part >= threshold)

    valid_points = pts[mask_valid]

    print("\n" + "=" * 40)
    print(f"BILAN DU FILTRAGE (Zone {0}-{DISTANCE_MAX_WITH_STRUCTURE}mm)")
    print(f"Points initiaux (grille) : {total_input:,}")
    print(f"Éliminés (Collision)     : {nb_collision:,}")
    print(f"Éliminés (>{DISTANCE_MAX_WITH_STRUCTURE}mm)        : {nb_trop_loin:,}")
    print(f"Points de clamps forcés  : {len(clamps_points)} (clash testé séparément)")
    print(f"Points conservés (grille): {len(valid_points):,} ({(len(valid_points) / total_input) * 100:.2f}%)")
    print("=" * 40 + "\n")

    return {
        "grid_points": valid_points.tolist(),
        "clamp_points": clamps_points
    }


def export_points_to_stl_spheres(points, radius, filename="points_spheres.stl"):
    import pyvista as pv
    import numpy as np

    print(f"Génération du maillage STL ({len(points)} points)...")
    cloud = pv.PolyData(np.asarray(points))
    sphere = pv.Sphere(radius=radius, theta_resolution=8, phi_resolution=8)
    mesh_spheres = cloud.glyph(geom=sphere, orient=False, scale=False)
    mesh_spheres.save(filename)
    print(f"Succès ! Fichier STL enregistré : {filename}")
