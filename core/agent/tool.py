import numpy as np
import trimesh
import pyvista as pv
from core.agent.config import *
from core.agent.config import _CRABE_GEOMETRY_CACHE

SENSOR_COUNT_DEFAULT = 14


def get_rotation_matrix_from_vectors(vec1, vec2):
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-6:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))


# =========================================================================
# 🦀 LOGIQUE DES CRABES
# =========================================================================

def load_crabe_clamp(stl_path, max_check_points=400):
    mesh_clamp = trimesh.load_mesh(stl_path, force='mesh')

    normals_rounded = np.round(mesh_clamp.face_normals, 4)
    areas = mesh_clamp.area_faces
    uniq_normals, inverse = np.unique(normals_rounded, axis=0, return_inverse=True)
    area_sum = np.zeros(len(uniq_normals))
    np.add.at(area_sum, inverse, areas)
    base_normal = uniq_normals[np.argmax(area_sum)]

    R = get_rotation_matrix_from_vectors(base_normal, np.array([0.0, 0.0, -1.0]))
    verts = mesh_clamp.vertices @ R.T
    bmin, bmax = verts.min(axis=0), verts.max(axis=0)
    cx, cy, z_contact = (bmin[0] + bmax[0]) / 2.0, (bmin[1] + bmax[1]) / 2.0, bmin[2]
    verts = verts - np.array([cx, cy, z_contact])
    bmin, bmax = verts.min(axis=0), verts.max(axis=0)

    check_points = np.asarray(verts, dtype=np.float32)
    if len(check_points) > max_check_points:
        rng = np.random.default_rng(0)
        check_points = check_points[rng.choice(len(check_points), max_check_points, replace=False)]

    return {"dx": float((bmax[0] - bmin[0]) / 2.0), "dy": float((bmax[1] - bmin[1]) / 2.0),
            "height": float(bmax[2]),
            "check_points": check_points,
            # Géométrie complète dans le repère local du crabe : plan de
            # contact en z = 0, corps vers les z positifs. ``check_points`` en
            # est un échantillon aléatoire, inutilisable pour un affichage —
            # d'où ces deux tableaux, qui permettent de dessiner le crabe
            # exactement là où le test de collision le place.
            "vertices": np.asarray(verts, dtype=np.float32),
            "faces": np.asarray(mesh_clamp.faces, dtype=np.int64)}


def crabe_transform(surface_position, x_axis, y_axis, normal):
    """Repère du crabe posé : matrice de rotation et origine.

    C'est la convention de :func:`is_crabe_clash_free` — la même doit servir à
    l'affichage, sans quoi on dessinerait le crabe ailleurs que là où sa
    collision a été vérifiée.
    """
    rotation = np.stack([x_axis, y_axis, normal], axis=1)
    return rotation, np.asarray(surface_position, dtype=np.float64)


def crabe_world_vertices(crabe, crabe_geometry):
    """Sommets du crabe posé, dans le repère de la maquette.

    Renvoie ``None`` si la géométrie n'est pas disponible : sans modèle STL
    chargeable, il n'y a rien à dessiner — et rien n'a été posé non plus.
    """
    if crabe_geometry is None:
        return None
    vertices = crabe_geometry.get("vertices")
    if vertices is None or not len(vertices):
        return None

    seat = crabe.get("surface_position")
    if seat is None:
        seat = crabe.get("position")
    if seat is None:
        return None

    rotation, origin = crabe_transform(
        seat, crabe["x_axis"], crabe["y_axis"], crabe["normal"]
    )
    return origin + vertices @ rotation.T


def get_crabe_geometry(stl_path):
    if not stl_path:
        return None
    if stl_path in _CRABE_GEOMETRY_CACHE:
        return _CRABE_GEOMETRY_CACHE[stl_path]
    try:
        geometry = load_crabe_clamp(stl_path)
        print(f"🦀 Clip crabe chargé : dx={geometry['dx']:.1f}mm, dy={geometry['dy']:.1f}mm, "
              f"hauteur={geometry['height']:.1f}mm ({stl_path})")
    except Exception as e:
        print(f"⚠️ Impossible de charger le clip crabe '{stl_path}' : {e}")
        geometry = None
    _CRABE_GEOMETRY_CACHE[stl_path] = geometry
    return geometry


# =========================================================================
# 🪮 LOGIQUE DES CLAMPS (PEIGNES)
# =========================================================================

def extraire_geometrie_pure_clamp(mesh_trimesh):
    """
    Extrait les axes parfaits d'après les plans de la CAO (normales)
    et trouve les encoches via l'analyse de la matière.
    Spécifique aux peignes (Clamps). Adapté pour Trimesh.
    """
    vertices = mesh_trimesh.vertices
    areas = mesh_trimesh.area_faces
    normals = mesh_trimesh.face_normals

    # 1. Extraction ultra-robuste des axes CAO via la matrice de covariance
    C = np.zeros((3, 3))
    for n, a in zip(normals, areas):
        C += a * np.outer(n, n)

    _, eigenvectors = np.linalg.eigh(C)
    R_local = eigenvectors

    proj = vertices @ R_local
    extents = np.max(proj, axis=0) - np.min(proj, axis=0)

    idx_long = np.argmax(extents)
    idx_dos = np.argmin(extents)
    idx_haut = 3 - idx_long - idx_dos

    vec_long = R_local[:, idx_long]
    vec_haut = R_local[:, idx_haut]
    vec_dos = R_local[:, idx_dos]

    # Orientation vers le haut des dents
    coords_H = vertices @ vec_haut
    if np.mean(coords_H) > (np.min(coords_H) + np.max(coords_H)) / 2.0:
        vec_haut = -vec_haut
        coords_H = vertices @ vec_haut

    coords_L = vertices @ vec_long

    # 2. Détection Topologique des encoches
    h_min, h_max = np.min(coords_H), np.max(coords_H)

    threshold_H = h_max - 0.4 * (h_max - h_min)
    mask_top = coords_H > threshold_H
    top_L = coords_L[mask_top]

    L_min, L_max = np.min(coords_L), np.max(coords_L)
    nb_bins = 200
    hist, _ = np.histogram(top_L, bins=nb_bins, range=(L_min, L_max))

    is_matter = hist > 0

    gaps_centers_L = []
    in_gap = False
    gap_start_idx = 0

    for i in range(nb_bins):
        if not is_matter[i] and not in_gap:
            in_gap = True
            gap_start_idx = i
        elif is_matter[i] and in_gap:
            in_gap = False
            gap_end_idx = i - 1
            if (gap_end_idx - gap_start_idx) >= (nb_bins * 0.01):
                center_idx = (gap_start_idx + gap_end_idx) / 2.0
                center_coord = L_min + (center_idx / nb_bins) * (L_max - L_min)
                gaps_centers_L.append(center_coord)

    if not gaps_centers_L:
        gaps_centers_L = [(L_min + L_max) / 2.0]

    hauteur_segment = extents[idx_haut] * 0.90

    coords_D = vertices @ vec_dos
    d_min = np.min(coords_D)
    d_max = np.max(coords_D)

    h_center = (h_min + h_max) / 2.0

    return gaps_centers_L, vec_long, vec_haut, vec_dos, d_min, d_max, h_center, hauteur_segment


def placer_lignes_dans_encoches_clamp(mesh_trimesh, transform_globale):
    """
    Place les lignes avec un décalage de +2mm devant les encoches trouvées
    pour les fixations de type CLAMP (peigne).
    Retourne les segments (p_in, p_out, center).
    """
    gaps_L_coords, vec_L, vec_H, vec_D, d_min, d_max, h_center, h_segment = extraire_geometrie_pure_clamp(mesh_trimesh)

    R_glob = transform_globale[:3, :3]
    T_glob = transform_globale[:3, 3]

    passages = []

    # Décalage de 2mm à l'avant du peigne
    d_placement = d_min - 2.0

    for pos_L in gaps_L_coords:
        current_center_local = (pos_L * vec_L) + (h_center * vec_H) + (d_placement * vec_D)

        p_bottom_local = current_center_local - (h_segment / 2.0) * vec_H
        p_top_local = current_center_local + (h_segment / 2.0) * vec_H

        p_bottom_global = (R_glob @ p_bottom_local) + T_glob
        p_top_global = (R_glob @ p_top_local) + T_glob
        center_global = (R_glob @ current_center_local) + T_glob

        passages.append({
            'p_in': p_bottom_global.tolist(),
            'p_out': p_top_global.tolist(),
            'center': center_global.tolist()
        })

    return passages, len(gaps_L_coords)


# =========================================================================
# UTILITAIRES ET CHEMINS
# =========================================================================

def compute_full_tangents(waypoints):
    tangents = np.zeros_like(waypoints)
    if len(waypoints) >= 3:
        tangents[1:-1] = waypoints[2:] - waypoints[:-2]
    if len(waypoints) >= 2:
        tangents[0] = waypoints[1] - waypoints[0]
        tangents[-1] = waypoints[-1] - waypoints[-2]
    return tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8)


def _interp_point_at_arclength(waypoints, cum_length, target):
    out = np.empty(3, dtype=np.float32)
    for k in range(3):
        out[k] = np.interp(target, cum_length, waypoints[:, k])
    return out


def evaluate_crabe_alignment(waypoints, local_pq, mesh, dx, dy, normal_cos_threshold,
                             surface_tolerance, straightness_tolerance, max_clearance=None):
    n = len(waypoints)
    tangents = compute_full_tangents(waypoints)

    surface_pts, center_dist, center_face = local_pq.on_surface(waypoints)
    normal = mesh.face_normals[center_face]

    if max_clearance is None:
        max_clearance = np.inf

    x_axis = tangents - np.einsum('ij,ij->i', tangents, normal)[:, None] * normal
    x_norm = np.linalg.norm(x_axis, axis=1, keepdims=True)
    x_axis = x_axis / (x_norm + 1e-8)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis, axis=1, keepdims=True) + 1e-8)

    corners = [waypoints + x_axis * dx, waypoints - x_axis * dx,
               waypoints + y_axis * dy, waypoints - y_axis * dy]
    min_cos = np.ones(n, dtype=np.float32)
    max_corner_dev = np.zeros(n, dtype=np.float32)
    for c in corners:
        _, c_dist, c_face = local_pq.on_surface(c)
        c_normal = mesh.face_normals[c_face]
        cos_sim = np.einsum('ij,ij->i', normal, c_normal)
        min_cos = np.minimum(min_cos, cos_sim)
        max_corner_dev = np.maximum(max_corner_dev, np.abs(c_dist - center_dist))

    planar_ok = (min_cos >= normal_cos_threshold) & (max_corner_dev < surface_tolerance) & (
            center_dist <= max_clearance)

    seg_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1) if n > 1 else np.array([])
    cum_length = np.insert(np.cumsum(seg_lengths), 0, 0.0) if len(seg_lengths) > 0 else np.zeros(n)
    straight_ok = np.zeros(n, dtype=bool)
    for i in range(n):
        target_in, target_out = cum_length[i] - dx, cum_length[i] + dx
        if target_in < 0 or target_out > cum_length[-1]:
            continue
        real_in = _interp_point_at_arclength(waypoints, cum_length, target_in)
        real_out = _interp_point_at_arclength(waypoints, cum_length, target_out)
        theo_in = waypoints[i] - x_axis[i] * dx
        theo_out = waypoints[i] + x_axis[i] * dx
        if (np.linalg.norm(real_in - theo_in) < straightness_tolerance and
                np.linalg.norm(real_out - theo_out) < straightness_tolerance):
            straight_ok[i] = True

    eligible = planar_ok & straight_ok
    diag = {
        "points": int(n),
        "hauteur_ok": int(np.sum(center_dist <= max_clearance)),
        "plan_ok": int(np.sum((min_cos >= normal_cos_threshold) & (max_corner_dev < surface_tolerance))),
        "droit_ok": int(np.sum(straight_ok)),
        "eligibles": int(np.sum(eligible)),
    }
    return eligible, min_cos.astype(np.float32), x_axis, y_axis, normal, surface_pts, diag


def is_crabe_clash_free(surface_position, x_axis, y_axis, normal, crabe_geometry, local_pq, mesh,
                        tolerance=0.5):
    check_points = crabe_geometry.get("check_points")
    if check_points is None or len(check_points) == 0:
        return True
    R = np.stack([x_axis, y_axis, normal], axis=1)
    world_pts = surface_position + check_points @ R.T
    closest, dist, faces = local_pq.on_surface(world_pts)
    inside = np.einsum('ij,ij->i', world_pts - closest, mesh.face_normals[faces]) < 0
    return not np.any(inside & (dist > tolerance))


def place_crabes_greedy(waypoints, eligible, min_spacing, existing_positions=None, is_valid=None):
    seg_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1) if len(waypoints) > 1 else np.array([])
    cum_length = np.insert(np.cumsum(seg_lengths), 0, 0.0) if len(seg_lengths) > 0 else np.zeros(len(waypoints))
    placed_idx = []

    # Le point de départ (A) compte comme un ancrage initial
    last_anchor_len = 0.0

    for i in range(len(waypoints)):
        # 1. Vérifier si l'on passe sur une fixation CAO scannée (p_in / p_out)
        is_existing_anchor = False
        if existing_positions is not None and len(existing_positions) > 0:
            dist_to_existing = np.linalg.norm(existing_positions - waypoints[i], axis=1)
            if np.min(dist_to_existing) < 30.0:  # Rayon d'exclusion (3 cm autour de la fixation)
                is_existing_anchor = True

        # Si on survole une fixation existante, ON NE POSE RIEN et on réinitialise le compteur des 25 cm
        if is_existing_anchor:
            last_anchor_len = cum_length[i]
            continue

        # 2. Règle des 25 cm : Si on a parcouru >= min_spacing depuis le dernier ancrage
        if eligible[i] and (cum_length[i] - last_anchor_len) >= min_spacing:
            if is_valid is not None and not is_valid(i):
                continue

            placed_idx.append(i)
            # On valide la pose du crabe, il devient notre nouvel ancrage
            last_anchor_len = cum_length[i]

    return placed_idx


def compute_crabes(waypoints, local_pq, mesh, crabe_geometry, normal_cos_threshold, surface_tolerance,
                   straightness_tolerance, min_spacing, max_clearance=None, clash_tolerance=0.5,
                   existing_positions=None):
    if crabe_geometry is None or len(waypoints) < 2:
        return [], np.zeros(len(waypoints), dtype=bool), {}

    dx, dy = crabe_geometry["dx"], crabe_geometry["dy"]
    eligible, min_cos, x_axis, y_axis, normal, surface_pts, diag = evaluate_crabe_alignment(
        waypoints, local_pq, mesh, dx, dy, normal_cos_threshold, surface_tolerance, straightness_tolerance,
        max_clearance=max_clearance
    )
    # Écart de parallélisme entre l'embase du crabe et la structure, en degrés :
    # c'est la mesure directe de « chaque crabe doit être posé à plat ».
    tilt_deg = np.degrees(np.arccos(np.clip(min_cos, -1.0, 1.0)))
    clash_rejects = [0]

    def _clash_ok(i):
        ok = is_crabe_clash_free(surface_pts[i], x_axis[i], y_axis[i], normal[i],
                                 crabe_geometry, local_pq, mesh, tolerance=clash_tolerance)
        if not ok:
            clash_rejects[0] += 1
        return ok

    placed_idx = place_crabes_greedy(waypoints, eligible, min_spacing,
                                     existing_positions=existing_positions,
                                     is_valid=_clash_ok)
    diag["clash_rejets"] = clash_rejects[0]

    seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1) if len(waypoints) > 1 else np.zeros(0)
    cum_arc = np.concatenate([[0.0], np.cumsum(seg)])

    crabes = []
    for i in placed_idx:
        crabes.append({
            # Index et abscisse curviligne du point porteur : c'est ce qui
            # permet de vérifier la règle des 250 mm sans recalculer la
            # trajectoire (voir core.routing_rules).
            "index": int(i),
            "arc_mm": float(cum_arc[i]) if i < len(cum_arc) else 0.0,
            "tilt_deg": float(tilt_deg[i]),
            "position": waypoints[i].astype(np.float32),
            "surface_position": np.asarray(surface_pts[i], dtype=np.float32),
            "x_axis": x_axis[i].astype(np.float32),
            "y_axis": y_axis[i].astype(np.float32),
            "normal": normal[i].astype(np.float32),
        })
    return crabes, eligible, diag


def resample_curve(points, num_points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return points

    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(segment_lengths), 0, 0.0)
    total_dist = cum_dist[-1]

    if total_dist <= 1e-6:
        return np.tile(points[0], (num_points, 1))

    new_dist = np.linspace(0.0, total_dist, int(num_points), dtype=np.float32)

    new_points = np.zeros((num_points, 3), dtype=np.float32)
    for i in range(3):
        new_points[:, i] = np.interp(new_dist, cum_dist, points[:, i])

    return new_points


def generate_dense_waypoints(start, goal, num_points, mesh=None, intermediate_points=None,
                             on_warning=None):
    """Chemin de départ suivant la surface du maillage (géodésique).

    Args:
        on_warning: appelé avec un message quand un tronçon géodésique échoue.
            Sur une maquette faite de pièces disjointes — le cas courant — il
            n'existe aucun chemin d'arêtes entre deux pièces séparées, et le
            tronçon se réduit à une ligne droite. Ce repli était silencieux :
            l'utilisateur croyait obtenir une géodésique et recevait une
            corde. Voir :mod:`core.path_planner` pour la recherche dans
            l'espace libre, qui n'a pas cette limite.
    """
    num_points = int(max(2, num_points))
    failures = [0]
    if mesh is None:
        return np.linspace(start, goal, num_points, dtype=np.float32)

    try:
        # 1. Conversion Trimesh -> PyVista PolyData
        pad = np.full((len(mesh.faces), 1), 3, dtype=np.int64)
        faces_pv = np.hstack((pad, mesh.faces)).flatten()
        pv_mesh = pv.PolyData(mesh.vertices, faces_pv)

        # 2. Construction et tri des étapes (A -> Fixations -> B)
        full_path_nodes = [start]
        if intermediate_points is not None and len(intermediate_points) > 0:
            # On trie les points intermédiaires selon leur avancée de A vers B
            vec_AB = goal - start
            dir_AB = vec_AB / (np.linalg.norm(vec_AB) + 1e-8)
            projs = [np.dot(np.array(p) - start, dir_AB) for p in intermediate_points]

            # Tri des points en fonction de leur projection
            sorted_pts = [p for _, p in sorted(zip(projs, intermediate_points))]
            full_path_nodes.extend(sorted_pts)

        full_path_nodes.append(goal)

        # 3. Calcul géodésique tronçon par tronçon (Avec filet de sécurité local)
        all_path_points = []
        for i in range(len(full_path_nodes) - 1):
            s_pt = full_path_nodes[i]
            e_pt = full_path_nodes[i + 1]

            # 🔥 HEURISTIQUE ANTI-RAMPAGE : Si on relie deux boules d'une MÊME encoche,
            # on force une LIGNE DROITE EN L'AIR au lieu de coller à la surface du peigne !
            is_air_gap = False
            if intermediate_points is not None and len(intermediate_points) > 0:
                dist_nodes = np.linalg.norm(s_pt - e_pt)
                if dist_nodes < 150.0:  # Les points d'un peigne sont forcément proches
                    # On vérifie que les deux points sont bien nos boules intermédiaires
                    s_is_inter = any(np.linalg.norm(s_pt - np.array(p)) < 1e-3 for p in intermediate_points)
                    e_is_inter = any(np.linalg.norm(e_pt - np.array(p)) < 1e-3 for p in intermediate_points)
                    if s_is_inter and e_is_inter:
                        is_air_gap = True

            if is_air_gap:
                # Saut direct et tout droit dans l'encoche, on évite le calcul géodésique !
                nb_steps = max(5, int(np.linalg.norm(s_pt - e_pt) / 2))
                segment_pts = np.linspace(s_pt, e_pt, nb_steps, dtype=np.float32)
            else:
                _, s_idx = mesh.kdtree.query(s_pt)
                _, e_idx = mesh.kdtree.query(e_pt)

                try:
                    # On essaie de coller à la surface pour le reste du trajet
                    geo_path = pv_mesh.geodesic(s_idx, e_idx)
                    segment_pts = geo_path.points
                except Exception:
                    # Surface coupée entre les deux extrémités : pont droit.
                    failures[0] += 1
                    segment_pts = np.linspace(s_pt, e_pt, 10, dtype=np.float32)

            # On évite de dupliquer le point de jonction entre les segments
            if len(all_path_points) > 0:
                all_path_points.extend(segment_pts[1:])
            else:
                all_path_points.extend(segment_pts)

        if failures[0] and on_warning is not None:
            on_warning(
                f"{failures[0]} tronçon(s) géodésique(s) impossible(s) : la maquette est "
                "faite de pièces disjointes. Ces tronçons sont de simples lignes droites, "
                "collées à la structure. Préférez une recherche dans l'espace libre."
            )

        # 4. Ré-échantillonnage global
        return resample_curve(np.array(all_path_points), num_points)

    except Exception as e:
        print(f"⚠️ Échec global géodésique ({e}), repli sur ligne droite euclidienne A->B.")
        return np.linspace(start, goal, num_points, dtype=np.float32)


def snap_mandatory_points(waypoints, targets, used=None):
    """Ramène le tracé exactement sur les points de passage imposés.

    Une attraction par récompense ne suffit pas quand l'utilisateur a demandé
    d'emprunter une fixation : elle *incite* le câble à s'en approcher, elle ne
    garantit rien. Un peigne, lui, impose au câble de traverser son encoche —
    à quelques millimètres près, il n'y passe pas.

    On force donc le point le plus proche de chaque passage à coïncider avec
    lui, et on renvoie les indices concernés pour que l'agent cesse de les
    déplacer. Les deux extrémités sont exclues : elles appartiennent aux
    équipements et ne se négocient pas non plus.

    Args:
        waypoints: trajectoire ``(n, 3)``, modifiée sur place.
        targets: points imposés ``(m, 3)``, ou ``None``.
        used: indices déjà réservés, à ne pas réattribuer.

    Returns:
        L'ensemble des indices verrouillés.
    """
    if targets is None or len(targets) == 0 or len(waypoints) < 3:
        return set()

    locked = set(used or ())
    for target in np.asarray(targets, dtype=np.float32):
        distances = np.linalg.norm(waypoints - target, axis=1)
        # Extrémités et points déjà attribués : hors jeu. Deux passages
        # distincts ne peuvent pas partager le même point du câble.
        distances[0] = np.inf
        distances[-1] = np.inf
        for index in locked:
            if 0 <= index < len(distances):
                distances[index] = np.inf
        if not np.isfinite(distances).any():
            break
        index = int(np.argmin(distances))
        waypoints[index] = target
        locked.add(index)
    return locked


def get_segment_test_points(wps, steps=10):
    if len(wps) < 2:
        return wps
    t_vals = np.linspace(0.0, 1.0, steps)[:, None, None]
    return (wps[:-1] + t_vals * (wps[1:] - wps[:-1])).reshape(-1, 3)


def count_segment_face_crossings(waypoints, mesh):
    n_seg = len(waypoints) - 1
    crossing_mask = np.zeros(max(n_seg, 0), dtype=bool)
    if n_seg < 1:
        return 0, crossing_mask

    origins = np.asarray(waypoints[:-1], dtype=np.float64)
    vecs = np.asarray(waypoints[1:], dtype=np.float64) - origins
    lengths = np.linalg.norm(vecs, axis=1)
    valid = lengths > 1e-9
    if not np.any(valid):
        return 0, crossing_mask
    dirs = np.zeros_like(vecs)
    dirs[valid] = vecs[valid] / lengths[valid, None]

    try:
        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=origins[valid], ray_directions=dirs[valid], multiple_hits=True
        )
        if len(index_ray) > 0:
            valid_idx = np.where(valid)[0]
            t_hit = np.einsum('ij,ij->i', locations - origins[valid][index_ray], dirs[valid][index_ray])
            seg_len = lengths[valid][index_ray]
            inside_segment = (t_hit > 1e-4) & (t_hit < seg_len - 1e-4)
            crossing_mask[valid_idx[index_ray[inside_segment]]] = True
    except Exception:
        pass

    return int(crossing_mask.sum()), crossing_mask


def build_local_frame(tangents):
    up = np.tile(np.array([0.0, 0.0, 1.0]), (len(tangents), 1))
    u = np.cross(tangents, up)
    norm_u = np.linalg.norm(u, axis=1, keepdims=True)
    degenerate = (norm_u[:, 0] < 1e-6)
    if np.any(degenerate):
        alt = np.array([1.0, 0.0, 0.0])
        u[degenerate] = np.cross(tangents[degenerate], alt)
        norm_u = np.linalg.norm(u, axis=1, keepdims=True)
    u = u / (norm_u + 1e-8)
    v = np.cross(tangents, u)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return u, v


def scan_virtual_sensors(points, prev_points, next_points, local_pq, radius, n_sensors=SENSOR_COUNT_DEFAULT):
    n = len(points)
    tangents = next_points - prev_points
    tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8)
    u, v = build_local_frame(tangents)

    angles = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)
    directions = [np.cos(ang) * u + np.sin(ang) * v for ang in angles]

    all_sensor_pts = np.concatenate([points + d * radius for d in directions], axis=0)
    _, sensor_dists, _ = local_pq.on_surface(all_sensor_pts)
    sensor_dists = sensor_dists.reshape(n_sensors, n)

    close_mask = sensor_dists < radius
    material_ratio = close_mask.mean(axis=0).astype(np.float32)

    far_mask = (~close_mask).astype(np.float32)
    void_dir_world = np.zeros((n, 3), dtype=np.float32)
    for i, d in enumerate(directions):
        void_dir_world += d * far_mask[i][:, None]

    norm_void = np.linalg.norm(void_dir_world, axis=1, keepdims=True)
    nonzero = norm_void[:, 0] > 1e-6
    void_dir_world[nonzero] = void_dir_world[nonzero] / norm_void[nonzero]

    return material_ratio.astype(np.float32), void_dir_world.astype(np.float32), u, v, tangents


def scan_lookahead_radar(points, tangents, local_pq, base_radius, d_soft, n_probes=3, lookahead_factor=4.0):
    n = len(points)
    lookahead_dists = np.linspace(base_radius, base_radius * lookahead_factor, n_probes)

    close_counts = np.zeros(n, dtype=np.float32)
    for d in lookahead_dists:
        probe_pts = points + tangents * d
        _, probe_dists, _ = local_pq.on_surface(probe_pts)
        close_counts += (probe_dists < d_soft).astype(np.float32)

    ahead_ratio = (close_counts / n_probes).astype(np.float32)
    return ahead_ratio


# =========================================================================
# OUTILS ANTI-COLLISION ET LISSAGE ADAPTATIF
# =========================================================================

def compute_collision_penalties(waypoints, mesh, config, local_pq=None):
    """
    Calcule les pénalités de collisions exactes (segment-triangle)
    et les pénétrations dans le volume du câble.
    """
    n_crossings, crossing_mask = count_segment_face_crossings(waypoints, mesh)

    test_pts = get_segment_test_points(waypoints, steps=config.get("segment_test_steps", 10))

    # Prise en compte de ProximityQuery (3 retours) ou KDTree (2 retours)
    if local_pq is not None:
        _, dists, _ = local_pq.on_surface(test_pts)
    else:
        dists, _ = mesh.kdtree.query(test_pts)  # <-- CORRECTION : 2 valeurs unpacked

    min_clearance = config.get("tube_radius", 6.0) + config.get("min_margin", 10.0)
    tube_violations = np.sum(dists < min_clearance)

    penalty = (n_crossings * config.get("segment_violation_penalty", 200.0)) + (tube_violations * 10.0)
    return penalty, crossing_mask


def apply_spatial_smoothing(action_displacements, smoothing_factor=0.8):
    """
    Applique un filtre laplacien 1D pour lisser les actions
    et éviter l'effet "dents de scie".
    """
    smoothed = np.copy(action_displacements)
    smoothed[1:-1] = (
            smoothing_factor * action_displacements[1:-1] +
            ((1.0 - smoothing_factor) / 2.0) * (action_displacements[:-2] + action_displacements[2:])
    )
    return smoothed


def adaptively_refine_trajectory(waypoints, crossing_mask, collision_streaks, config):
    """
    Insère un point milieu sur les segments bloqués de manière persistante.
    """
    new_waypoints = []
    n_pts = len(waypoints)
    max_pts = config.get("max_points", 150)
    threshold = config.get("adaptive_insert_streak_threshold", 8)

    for i in range(n_pts - 1):
        new_waypoints.append(waypoints[i])

        if i < len(crossing_mask) and crossing_mask[i]:
            collision_streaks[i] = collision_streaks.get(i, 0) + 1
        else:
            collision_streaks[i] = 0

        if collision_streaks.get(i, 0) >= threshold and len(new_waypoints) < max_pts:
            mid_point = 0.5 * (waypoints[i] + waypoints[i + 1])
            new_waypoints.append(mid_point)
            collision_streaks[i] = 0

    new_waypoints.append(waypoints[-1])
    return np.array(new_waypoints, dtype=np.float32)
