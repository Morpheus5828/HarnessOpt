


import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
import vtk
import time


def get_rotation_matrix_from_vectors(vec1, vec2):
    """
    Calcule la matrice de rotation 3x3 pour aligner vec1 sur vec2 (Formule de Rodrigues).
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-6:
        if c > 0:
            return np.eye(3)
        else:
            axis = np.cross(a, [1, 0, 0])
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(a, [0, 1, 0])
            axis = axis / np.linalg.norm(axis)
            return 2 * np.outer(axis, axis) - np.eye(3)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def trouver_normale_base_independante_orientation(mesh):
    """
    Identifie la face plane la plus grande du STL pour servir de base de contact.
    """
    mesh_avec_surfaces = mesh.compute_cell_sizes()
    mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)

    surfaces = mesh_avec_surfaces.cell_data["Area"]
    normales = mesh.cell_normals

    normales_arrondies = np.round(normales, decimals=4)
    normales_uniques, indices_uniques, inverses = np.unique(
        normales_arrondies, axis=0, return_index=True, return_inverse=True
    )

    surfaces_par_normale = np.zeros(len(normales_uniques))
    for i in range(len(surfaces)):
        surfaces_par_normale[inverses[i]] += surfaces[i]

    idx_max_surface = np.argmax(surfaces_par_normale)
    normale_base_detectee = normales_uniques[idx_max_surface]
    return normale_base_detectee


def injecter_geometrie_fixations_dans_chemins(all_variants_optimized, valid_fixations, mesh_global, tube_radius):
    """
    Pour chaque variante, remplace les points d'origine trop proches d'un crabe
    par le triplet parfait : [Point Entrée -> Centre -> Point Sortie].
    Vérifie dynamiquement la validité anti-clash de CHAQUE configuration retenue.
    """
    import vtk
    chemins_ajustes = []

    # Instanciation unique de l'outil de distance VTK
    implicit_dist = None
    if mesh_global is not None:
        implicit_dist = vtk.vtkImplicitPolyDataDistance()
        implicit_dist.SetInput(mesh_global)

    for variant_idx, pts in enumerate(all_variants_optimized):
        # Récupérer les fixations de cette variante spécifique
        fixations_locales = [f for f in valid_fixations if f["variant_index"] == variant_idx]

        if not fixations_locales:
            chemins_ajustes.append(pts)
            continue

        # Trier par index décroissant pour éviter le décalage des slices lors des mutations
        fixations_locales = sorted(fixations_locales, key=lambda x: x["point_index"], reverse=True)
        new_pts = list(pts)

        for fix in fixations_locales:
            j = fix["point_index"]
            center = np.array(fix["center"])
            pt_entree_orig = np.array(fix["pt_entree"])
            pt_sortie_orig = np.array(fix["pt_sortie"])
            dx = fix["half_length"]

            # Fenêtre de recherche adaptative à la densité des points
            indices_a_supprimer = []
            marge_indices = max(5, int((dx * 1.5) / 1.0))

            # 🟢 CORRECTION ICI : Suppression du walrus operator problématique
            fenetre_recherche = range(max(0, j - marge_indices), min(len(new_pts), j + marge_indices + 1))

            for k in fenetre_recherche:
                dist = np.linalg.norm(np.array(new_pts[k]) - center)
                if dist <= (dx * 1.2):
                    indices_a_supprimer.append(k)

            if j not in indices_a_supprimer and j < len(new_pts):
                indices_a_supprimer.append(j)

            indices_a_supprimer = sorted(list(set(indices_a_supprimer)))
            start_idx = indices_a_supprimer[0]
            end_idx = indices_a_supprimer[-1]

            # Extraction des nœuds de raccordement extérieurs
            pt_avant = np.array(new_pts[start_idx - 1]) if start_idx > 0 else None
            pt_apres = np.array(new_pts[end_idx + 1]) if (end_idx + 1) < len(new_pts) else None

            # 💾 PLAN DE SECOURS ULTIME : On sauvegarde la géométrie brute initiale du graphe
            points_origine_intacts = new_pts[start_idx:end_idx + 1]

            # Initialisation par défaut (si tout échoue, on ne modifie rien pour rester SAFE)
            triplet_retenu = points_origine_intacts
            solution_safe_trouvee = False

            if implicit_dist is not None:
                vec_in = pt_entree_orig - center
                vec_out = pt_sortie_orig - center
                scale_factors = [1.0, 0.75, 0.50, 0.25]

                # 1️⃣ NIVEAU 1 : TEST DES TRIPLETS AVEC RÉDUCTION D'ENTRAXE (100% à 25%)
                for scale in scale_factors:
                    test_in = center + vec_in * scale
                    test_out = center + vec_out * scale

                    sub_segments_test = []
                    if pt_avant is not None:
                        sub_segments_test.append(np.linspace(pt_avant, test_in, num=6))

                    sub_segments_test.append(np.linspace(test_in, center, num=5))
                    sub_segments_test.append(np.linspace(center, test_out, num=5)[1:])

                    if pt_apres is not None:
                        sub_segments_test.append(np.linspace(test_out, pt_apres, num=6)[1:])

                    points_enveloppe_complete = np.vstack(sub_segments_test)

                    clash_detecte = False
                    for p in points_enveloppe_complete:
                        if abs(implicit_dist.EvaluateFunction(p)) <= tube_radius:
                            clash_detecte = True
                            break

                    if not clash_detecte:
                        triplet_retenu = [test_in.tolist(), center.tolist(), test_out.tolist()]
                        solution_safe_trouvee = True
                        if scale < 1.0:
                            print(
                                f"⚠️ [FIX INJECTION] Entraxe collier {j} réduit à {scale * 100}% pour éviter une collision avec la structure.")
                        break

                # 2️⃣ NIVEAU 2 : TEST DU NŒUD CENTRAL PUR SI LES TRIPLETS ONT ÉCHOUÉ
                if not solution_safe_trouvee:
                    sub_segments_center = []
                    if pt_avant is not None:
                        sub_segments_center.append(np.linspace(pt_avant, center, num=6))
                    if pt_apres is not None:
                        sub_segments_center.append(np.linspace(center, pt_apres, num=6)[1:])

                    if sub_segments_center:
                        points_enveloppe_center = np.vstack(sub_segments_center)
                        clash_center = False
                        for p in points_enveloppe_center:
                            if abs(implicit_dist.EvaluateFunction(p)) <= tube_radius:
                                clash_center = True
                                break

                        if not clash_center:
                            triplet_retenu = [center.tolist()]
                            solution_safe_trouvee = True
                            print(
                                f"🚨 [FIX INJECTION] Triplets en clash au collier {j} ➔ Repli sur le nœud central pur (Vérifié SAFE).")

                # 3️⃣ NIVEAU 3 : REJET TOTAL SI TOUT CLASH (ON GARDE LE GRAPHE INTACT)
                if not solution_safe_trouvee:
                    print(
                        f"🛑 [FIX INJECTION] Risque de clash critique détecté au collier {j} ! Alignement annulé pour protéger le câble.")
                    triplet_retenu = points_origine_intacts

            # ✂️ SUBSTITUTION CONTRÔLÉE ET SÉCURISÉE
            new_pts[start_idx:end_idx + 1] = triplet_retenu

        chemins_ajustes.append(new_pts)

    return chemins_ajustes


def add_fixation_points_v4(all_variant, mesh_global, chemin_stl, rayon_harnais, tolerance_contact=0.5):
    print(f"📦 Préparation géométrique de la fixation : {chemin_stl}")
    clamp_base = pv.read(chemin_stl)

    # --- ALIGNEMENT INITIAL DU STL ---
    nb_detectee = trouver_normale_base_independante_orientation(clamp_base)
    R_3x3 = get_rotation_matrix_from_vectors(nb_detectee, [0, 0, -1])
    T_rot = np.eye(4)
    T_rot[0:3, 0:3] = R_3x3
    clamp_base.transform(T_rot, inplace=True)

    bounds = clamp_base.bounds
    cx, cy, z_contact = (bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, bounds[4]
    clamp_base.translate([-cx, -cy, -z_contact], inplace=True)

    new_bounds = clamp_base.bounds
    dx, dy = (new_bounds[1] - new_bounds[0]) / 2.0, (new_bounds[3] - new_bounds[2]) / 2.0

    longueur_collier = 2.0 * dx
    print(f"📐 Longueur cible du collier calculée : {longueur_collier:.2f} mm")

    OFFSET_Z_CENTRE = rayon_harnais - 2.0
    OFFSET_Z_CARDINAUX = rayon_harnais + 2.0
    cz_cardinaux = new_bounds[5] + OFFSET_Z_CARDINAUX
    cz_centre = new_bounds[5] + OFFSET_Z_CENTRE

    local_center = np.array([0.0, 0.0, cz_centre, 1.0])
    local_cardinals = np.array([
        [dx, 0.0, cz_cardinaux, 1.0],  # Sortie (+X)
        [-dx, 0.0, cz_cardinaux, 1.0],  # Entrée (-X)
        [0.0, dy, cz_cardinaux, 1.0],
        [0.0, -dy, cz_cardinaux, 1.0],
    ])
    local_base_corners = np.array([
        [dx, dy, 0.0, 1.0], [-dx, dy, 0.0, 1.0],
        [dx, -dy, 0.0, 1.0], [-dx, -dy, 0.0, 1.0]
    ])

    mesh_global.compute_normals(cell_normals=True, point_normals=False, inplace=True)

    # Instance de distance VTK pour la vérification de la semelle
    implicit_dist = vtk.vtkImplicitPolyDataDistance()
    implicit_dist.SetInput(mesh_global)

    valid_fixations = []

    # Compteurs de debug mis à jour avec le nouveau filtre
    stats = {
        "total_fenetres": 0,
        "rej_trop_court": 0,
        "rej_rectitude": 0,
        "rej_distance": 0,
        "rej_parallele": 0,
        "rej_semelle": 0,
        "rej_clash_solide": 0  # 🟢 Nouveau compteur
    }

    for variant_idx, pts in enumerate(all_variant):
        pts = np.array(pts)
        n_pts = len(pts)
        print(f"\n🚀 Analyse de la Variante n°{variant_idx} | Contient {n_pts} points.")

        dernier_index_k_valide = -1

        for j in range(n_pts):
            if j <= dernier_index_k_valide:
                continue

            stats["total_fenetres"] += 1

            # 1. Recherche de l'index 'k' pour la portion de 5 cm
            cum_dist = 0.0
            k = j
            while k < n_pts - 1 and cum_dist < longueur_collier:
                cum_dist += np.linalg.norm(pts[k + 1] - pts[k])
                k += 1

            if cum_dist < longueur_collier * 0.95:
                stats["rej_trop_court"] += 1
                break

                # 2. VÉRIFICATION DE RECTITUDE
            dist_directe = np.linalg.norm(pts[k] - pts[j])
            if dist_directe < longueur_collier * 0.95:
                stats["rej_rectitude"] += 1
                continue

            # 3. CALCUL DU CENTRE ET PROJECTION
            pt_centre_chemin = (pts[j] + pts[k]) / 2.0
            c_id, pt_impact = mesh_global.find_closest_cell(pt_centre_chemin, return_closest_point=True)
            if c_id == -1:
                continue

            normale_face = mesh_global.cell_normals[c_id]

            # Vérification distance au mur
            distance_au_mur = np.linalg.norm(pt_impact - pt_centre_chemin)
            limit_dist = rayon_harnais + 2.0
            if distance_au_mur > limit_dist:
                stats["rej_distance"] += 1
                continue

            # 4. VÉRIFICATION DU PARALLÉLISME
            direction_cable = (pts[k] - pts[j]) / dist_directe
            dot_product = np.abs(np.dot(direction_cable, normale_face))

            if dot_product > 0.15:
                stats["rej_parallele"] += 1
                continue

            # --- CONSTRUCTION DU REPERE LOCAL T ---
            z_axis_local = normale_face / np.linalg.norm(normale_face)
            tangent_projected = direction_cable - np.dot(direction_cable, z_axis_local) * z_axis_local
            if np.linalg.norm(tangent_projected) < 1e-6:
                continue
            x_axis_local = tangent_projected / np.linalg.norm(tangent_projected)
            y_axis_local = np.cross(z_axis_local, x_axis_local)

            # 🎯 RECALAGE : On applique un micro-décollage (+0.05 mm) pour immuniser le contact du bruit numérique
            pt_contact_securise = pt_impact + z_axis_local * 0.05

            T = np.eye(4)
            T[0:3, 0], T[0:3, 1], T[0:3, 2], T[0:3, 3] = x_axis_local, y_axis_local, z_axis_local, pt_contact_securise

            # 5. VÉRIFICATION DE LA SEMELLE (Planéité locale)
            base_pts_global = [(T @ coin)[:3] for coin in local_base_corners]
            base_in_full_contact = True

            for bp in base_pts_global:
                if abs(implicit_dist.EvaluateFunction(bp)) > tolerance_contact:
                    base_in_full_contact = False
                    break

            if not base_in_full_contact:
                stats["rej_semelle"] += 1
                continue

            # --- 🟢 GENERATION ET TEST DU SOLIDE (ANTI-CLASH DU COLLIER COMPLET) ---
            clamp_instance = clamp_base.copy()
            clamp_instance.transform(T, inplace=True)

            # On force le format PolyData propre pour l'analyse d'intersection
            if not isinstance(clamp_instance, pv.PolyData):
                clamp_instance = clamp_instance.extract_surface(algorithm='dataset_surface')

            # Test d'interférence solide de la pièce complète
            _, solid_collisions = clamp_instance.collision(mesh_global)

            if solid_collisions > 0:
                stats["rej_clash_solide"] += 1
                print(
                    f"  ❌ [REJET SOLIDE] Fenêtre [{j}->{k}] : Le corps de la fixation traverse la structure ({solid_collisions} facettes).")
                continue

            # --- VALUATION EN CAS DE SUCCÈS ---
            glob_center = (T @ local_center)[:3]
            glob_cards = [(T @ cp)[:3] for cp in local_cardinals]

            valid_fixations.append({
                "mesh": clamp_instance,
                "center": glob_center,
                "pt_entree": glob_cards[1],
                "pt_sortie": glob_cards[0],
                "half_length": dx,
                "variant_index": variant_idx,
                "point_index": (j + k) // 2,
            })

            print(f"  ✅ [VALIDÉ] Fixation ajoutée avec succès au point {(j + k) // 2} !")
            dernier_index_k_valide = k

    # Rapport final étendu
    print("\n" + "=" * 50)
    print("📊 RAPPORT DE DEBUGGING DES FIXATIONS")
    print("=" * 50)
    print(f" Portion(s) de câble testée(s) au total : {stats['total_fenetres']}")
    print(f" ❌ Rejetées car fin de câble trop courte : {stats['rej_trop_court']}")
    print(f" ❌ Rejetées car trop en zigzag/courbe    : {stats['rej_rectitude']}")
    print(f" ❌ Rejetées car trop LOIN de la tôle    : {stats['rej_distance']}")
    print(f" ❌ Rejetées car NON PARALLÈLES (Tordu)  : {stats['rej_parallele']}")
    print(f" ❌ Rejetées car tôle courbe sous semelle: {stats['rej_semelle']}")
    print(f" ❌ Rejetées car INTERFÉRENCE CORPS (CAO) : {stats['rej_clash_solide']}")  # 🟢 Nouveau
    print(f" 🎉 Fixations validées au final         : {len(valid_fixations)}")
    print("=" * 50 + "\n")

    # Sauvegarde automatique du fichier de cache
    try:
        from core.catia_handler import BASE_CACHE
        import os
        import pickle

        fixations_dir = os.path.join(str(BASE_CACHE), "graphs")
        os.makedirs(fixations_dir, exist_ok=True)
        out_pickle = os.path.join(fixations_dir, "detected_fixations.pickle")
        with open(out_pickle, "wb") as f:
            pickle.dump(valid_fixations, f)
        print(f"💾 [CACHE] {len(valid_fixations)} fixations enregistrées avec succès dans : {out_pickle}")
    except Exception as e:
        print(f"⚠️ Attention : Échec de l'écriture du cache des fixations : {e}")

    return valid_fixations

def resample_path_parametric(pts, mesh_global, tube_radius, step_mm=8.0):
    """Prend un chemin de points bruts hachés par les voxels, calcule sa distance

    cumulée et redistribue de nouveaux nœuds propres tous les 'step_mm'.
    Vérifie ensuite que la peau du nouveau tube ne touche pas le décor.
    """
    pts = np.array(pts)
    if len(pts) <= 2:
        return pts

    # 1. Calcul des distances cumulées le long de la ligne brisée d'origine
    segments_dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum_dist_orig = np.concatenate(([0], np.cumsum(segments_dists)))
    total_length = cum_dist_orig[-1]

    # 2. Détermination du nombre de points idéal basé sur le pas en mm
    num_points = max(4, int(total_length / step_mm))

    # 3. Génération de la nouvelle répartition uniforme (Linspace sur la distance)
    cum_dist_new = np.linspace(0, total_length, num_points)

    # 4. Interpolation 3D pour obtenir les nouvelles coordonnées lisses
    pts_resampled = np.zeros((num_points, 3))
    for i in range(3):
        pts_resampled[:, i] = np.interp(
            cum_dist_new, cum_dist_orig, pts[:, i]
        )

    # 5. CONTRÔLE DE SÉCURITÉ : Est-ce que ce lissage coupe un virage dans la tôle ?
    if mesh_global is not None:
        implicit_dist = vtk.vtkImplicitPolyDataDistance()
        implicit_dist.SetInput(mesh_global)

        # Génération du tube virtuel sur la trajectoire interpolée
        test_line = pv.MultipleLines(points=pts_resampled)
        test_tube = test_line.tube(radius=tube_radius, n_sides=8, capping=True)

        clash_detecte = False
        for v in test_tube.points:
            if implicit_dist.EvaluateFunction(v) <= 0.1:
                clash_detecte = True
                break

        # 🛡️ Si l'interpolation pure crée un clash (virage trop serré coupé),
        # on augmente la densité de points pour coller plus fidèlement au graphe d'origine
        if clash_detecte:
            # On double la résolution locale pour épouser le contour sans collision
            num_points_safe = num_points * 2
            cum_dist_safe = np.linspace(0, total_length, num_points_safe)
            pts_safe = np.zeros((num_points_safe, 3))
            for i in range(3):
                pts_safe[:, i] = np.interp(
                    cum_dist_safe, cum_dist_orig, pts[:, i]
                )
            return pts_safe

    return pts_resampled


import numpy as np
import pyvista as pv
import vtk


def corriger_courbures_trajectoire(pts, valid_fixations, variant_idx, mesh_global, tube_radius, iterations=15, alpha=0.4):
    """
    Applique un lissage laplacien itératif contrôlé par analyse relative
    pour harmoniser les transitions sans se faire bloquer par la proximité des tôles.
    """
    pts = np.array(pts, dtype=float)
    if len(pts) <= 2:
        return pts

    implicit_dist = None
    if mesh_global is not None:
        implicit_dist = vtk.vtkImplicitPolyDataDistance()
        implicit_dist.SetInput(mesh_global)

    fixations_locales = [f for f in valid_fixations if f["variant_index"] == variant_idx]
    points_verrouilles = []
    for fix in fixations_locales:
        points_verrouilles.extend([np.array(fix["center"]), np.array(fix["pt_entree"]), np.array(fix["pt_sortie"])])

    new_pts = np.copy(pts)

    for _ in range(iterations):
        for i in range(1, len(new_pts) - 1):
            p_prev = new_pts[i - 1]
            p_curr = new_pts[i]
            p_next = new_pts[i + 1]

            est_verrouille = False
            for p_v in points_verrouilles:
                if np.linalg.norm(p_curr - p_v) < 0.5:
                    est_verrouille = True
                    break
            if est_verrouille:
                continue

            p_ideal = (p_prev + p_next) / 2.0
            p_candidat = p_curr + alpha * (p_ideal - p_curr)

            if implicit_dist is not None:
                # Analyse comparative avant / après micro-déplacement
                t_line_old = pv.MultipleLines(points=np.vstack([p_prev, p_curr, p_next]))
                t_solid_old = t_line_old.tube(radius=tube_radius, n_sides=8, capping=True)
                min_dist_old = min([implicit_dist.EvaluateFunction(v) for v in t_solid_old.points])

                t_line_new = pv.MultipleLines(points=np.vstack([p_prev, p_candidat, p_next]))
                t_solid_new = t_line_new.tube(radius=tube_radius, n_sides=8, capping=True)
                min_dist_new = min([implicit_dist.EvaluateFunction(v) for v in t_solid_new.points])

                if min_dist_new >= (min_dist_old - 0.05) or min_dist_new > 0.1:
                    new_pts[i] = p_candidat
            else:
                new_pts[i] = p_candidat

    return new_pts


def densifier_et_arrondir_coudes(pts, valid_fixations, variant_idx, mesh_global, tube_radius, angle_seuil_deg=145):
    """
    Détecte les virages brusques et insère un congé de Bézier (Bezier Fillet).
    Utilise un contrôle de collision relatif pour ne pas bloquer le lissage
    dans les zones confinées ou proches des parois.
    """
    import vtk
    pts = np.array(pts, dtype=float)
    if len(pts) <= 2:
        return pts

    implicit_dist = None
    if mesh_global is not None:
        implicit_dist = vtk.vtkImplicitPolyDataDistance()
        implicit_dist.SetInput(mesh_global)

    fixations_locales = [f for f in valid_fixations if f["variant_index"] == variant_idx]
    points_verrouilles = []
    for fix in fixations_locales:
        points_verrouilles.extend([np.array(fix["center"]), np.array(fix["pt_entree"]), np.array(fix["pt_sortie"])])

    new_pts = [pts[0]]

    for i in range(1, len(pts) - 1):
        p_prev = pts[i - 1]
        p_curr = pts[i]
        p_next = pts[i + 1]

        est_verrouille = False
        for p_v in points_verrouilles:
            if np.linalg.norm(p_curr - p_v) < 1.0:
                est_verrouille = True
                break
        if est_verrouille:
            new_pts.append(p_curr)
            continue

        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-4 or n2 < 1e-4:
            new_pts.append(p_curr)
            continue

        dot_prod = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
        angle_interne = 180.0 - np.degrees(np.arccos(dot_prod))

        if angle_interne < angle_seuil_deg:
            marge_recul = min(12.0, n1 * 0.4, n2 * 0.4)

            A = p_curr - (v1 / n1) * marge_recul
            B = p_curr
            C = p_curr + (v2 / n2) * marge_recul

            t_vals = np.linspace(0, 1, num=5)
            bezier_nodes = np.array([(1 - t)**2 * A + 2 * (1 - t) * t * B + t**2 * C for t in t_vals])

            # 🎯 CALCUL DU CONTROLE DE COLLISION RELATIF
            if implicit_dist is not None:
                # 1. Évaluation du pire enfoncement du coude tranchant actuel
                t_line_old = pv.MultipleLines(points=np.vstack([p_prev, p_curr, p_next]))
                t_solid_old = t_line_old.tube(radius=tube_radius, n_sides=8, capping=True)
                min_dist_old = min([implicit_dist.EvaluateFunction(v) for v in t_solid_old.points])

                # 2. Évaluation du pire enfoncement du nouveau congé lisse proposé
                t_line_new = pv.MultipleLines(points=bezier_nodes)
                t_solid_new = t_line_new.tube(radius=tube_radius, n_sides=8, capping=True)
                min_dist_new = min([implicit_dist.EvaluateFunction(v) for v in t_solid_new.points])

                # Validation : Si l'arrondi n'aggrave pas la pénétration (avec marge d'approximation)
                if min_dist_new >= (min_dist_old - 0.05) or min_dist_new > 0.1:
                    new_pts.extend(bezier_nodes.tolist())
                    continue
            else:
                new_pts.extend(bezier_nodes.tolist())
                continue

        new_pts.append(p_curr)

    new_pts.append(pts[-1])

    unique_pts = [new_pts[0]]
    for p in new_pts[1:]:
        if np.linalg.norm(np.array(p) - np.array(unique_pts[-1])) > 0.1:
            unique_pts.append(p)

    return np.array(unique_pts)


import numpy as np
import pyvista as pv
import vtk
import time


def animer_lissage_harnais_3d(pts_bruts, valid_fixations, mesh_global, tube_radius, max_frames=100, alpha=0.20):
    """
    Version TURBO-BOOSTÉE de l'optimiseur par agent.
    La convergence est violente et instantanée.
    """
    print("⚡ Moteur IA en mode TURBO ENGAGÉ. Attention les yeux...")

    # 1. Configuration du Plotter
    pl = pv.Plotter(title="HarnessOpt - Ultra Fast RL Engine")
    pl.set_background("white")

    # 2. Obstacles
    if mesh_global is not None:
        if not isinstance(mesh_global, pv.PolyData):
            mesh_global = mesh_global.extract_surface(algorithm='dataset_surface')
        pl.add_mesh(mesh_global, color="lightgray", opacity=0.12, style="wireframe")

    # 3. Fixations
    if valid_fixations:
        fix_blocks = pv.MultiBlock([f["mesh"] for f in valid_fixations]).combine()
        pl.add_mesh(fix_blocks, color="dodgerblue", opacity=0.9)

    # 4. Verrouillage
    points_verrouilles = []
    for fix in valid_fixations:
        points_verrouilles.extend([np.array(fix["center"]), np.array(fix["pt_entree"]), np.array(fix["pt_sortie"])])

    implicit_dist = None
    if mesh_global is not None:
        implicit_dist = vtk.vtkImplicitPolyDataDistance()
        implicit_dist.SetInput(mesh_global)

    state = {
        "path": np.array(pts_bruts, dtype=float).copy(),
        "frame": 0,
        "is_running": False,
        "last_collisions": 0,
        "best_score": -999999.0
    }

    # Récompense de l'Agent
    def evaluer_politique_trajectoire(points):
        score_courbure = 0.0
        for i in range(1, len(points) - 1):
            if implicit_dist is not None:
                if implicit_dist.EvaluateFunction(points[i]) < tube_radius:
                    return -999999.0  # Crash tôle = Rejet immédiat
            milieu_virtuel = 0.5 * (points[i - 1] + points[i + 1])
            score_courbure += np.linalg.norm(points[i] - milieu_virtuel) ** 2
        return -score_courbure

    state["best_score"] = evaluer_politique_trajectoire(state["path"])

    # Init Graphique
    line_mesh = pv.MultipleLines(points=state["path"])
    tube_mesh = line_mesh.tube(radius=tube_radius, n_sides=16, capping=True)  # n_sides=16 pour gratter des FPS
    pl.add_mesh(tube_mesh, color="red", opacity=0.5, name="harness_volume")
    pl.add_mesh(line_mesh, color="black", line_width=3, name="harness_axis")

    # =========================================================================
    # 🧠 UN PAS D'APPRENTISSAGE (OPTIMISÉ)
    # =========================================================================
    def calculer_un_pas_agent_rl():
        current_path = state["path"]
        score_actuel = state["best_score"]

        # Le bruit se réduit au fil des frames pour fignoler la trajectoire
        sigma = max(0.005, 0.3 * (1 - state["frame"] / max_frames))

        bruit = np.random.normal(0, sigma, current_path.shape)
        for axe in range(3):
            bruit[:, axe] = np.convolve(bruit[:, axe], np.ones(5) / 5, mode='same')

        nouveau_path = np.copy(current_path)

        for i in range(1, len(current_path) - 1):
            if any(np.linalg.norm(current_path[i] - p_v) < 0.6 for p_v in points_verrouilles):
                continue
            milieu = 0.5 * (current_path[i - 1] + current_path[i + 1])
            nouveau_path[i] = current_path[i] + alpha * (milieu - current_path[i]) + bruit[i]

        nouveau_score = evaluer_politique_trajectoire(nouveau_path)
        if nouveau_score > score_actuel:
            state["path"] = nouveau_path
            state["best_score"] = nouveau_score
            return True
        return False

    # =========================================================================
    # 🚀 BOUCLE SANS BRIDE (NO SLEEP)
    # =========================================================================
        # =========================================================================
        # 🚀 BOUCLE SANS BRIDE (NO SLEEP) - VERSION CORRIGÉE
        # =========================================================================
        def demarrer_lissage_automatique(value=None):
            if state["is_running"]:
                return

            state["is_running"] = True

            while state["frame"] < max_frames:
                # Sécurité anti-crash : on vérifie si l'utilisateur a fermé la fenêtre
                if pl.closed or pl.renderer is None:
                    break

                state["frame"] += 1

                # --- 100 CALCULS INTENSIFS EN COULISSES ---
                for _ in range(100):
                    calculer_un_pas_agent_rl()

                # Rendu VTK ultra-rapide
                new_line = pv.MultipleLines(points=state["path"])
                new_tube = new_line.tube(radius=tube_radius, n_sides=16, capping=True)

                # Check collisions uniquement une frame sur 20
                if state["frame"] % 20 == 0 or state["frame"] == max_frames:
                    if mesh_global is not None:
                        _, state["last_collisions"] = new_tube.collision(mesh_global)

                couleur_harnais = "dodgerblue" if state["best_score"] > -10.0 else "orange"

                pl.add_mesh(new_tube, color=couleur_harnais, opacity=0.5, name="harness_volume")
                pl.add_mesh(new_line, color="black", line_width=3, name="harness_axis")

                pl.add_text(
                    f"⚡ MODE TURBO - Itération: {state['frame']} / {max_frames}\n"
                    f"📐 Score Courbure : {state['best_score']:.2f}\n"
                    f"💥 Interférences : {state['last_collisions']}",
                    position="upper_left", color="darkred", font_size=12, name="dashboard"
                )

                pl.update()

                if pl.iren is not None:
                    pl.iren.process_events()

            print(f"🎉 Fini ! Score final : {state['best_score']:.2f}")
            state["is_running"] = False
