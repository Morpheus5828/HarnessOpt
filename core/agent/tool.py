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
            "height": float(bmax[2]), "check_points": check_points}


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


def place_crabes_greedy(waypoints, eligible, min_spacing, is_valid=None):
    seg_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1) if len(waypoints) > 1 else np.array([])
    cum_length = np.insert(np.cumsum(seg_lengths), 0, 0.0) if len(seg_lengths) > 0 else np.zeros(len(waypoints))
    placed_idx = []
    last_len = -min_spacing
    for i in range(len(waypoints)):
        if eligible[i] and (cum_length[i] - last_len) >= min_spacing:
            if is_valid is not None and not is_valid(i):
                continue
            placed_idx.append(i)
            last_len = cum_length[i]
    return placed_idx


def compute_crabes(waypoints, local_pq, mesh, crabe_geometry, normal_cos_threshold, surface_tolerance,
                   straightness_tolerance, min_spacing, max_clearance=None, clash_tolerance=0.5):
    if crabe_geometry is None or len(waypoints) < 2:
        return [], np.zeros(len(waypoints), dtype=bool), {}

    dx, dy = crabe_geometry["dx"], crabe_geometry["dy"]
    eligible, _, x_axis, y_axis, normal, surface_pts, diag = evaluate_crabe_alignment(
        waypoints, local_pq, mesh, dx, dy, normal_cos_threshold, surface_tolerance, straightness_tolerance,
        max_clearance=max_clearance
    )
    clash_rejects = [0]

    def _clash_ok(i):
        ok = is_crabe_clash_free(surface_pts[i], x_axis[i], y_axis[i], normal[i],
                                 crabe_geometry, local_pq, mesh, tolerance=clash_tolerance)
        if not ok:
            clash_rejects[0] += 1
        return ok

    placed_idx = place_crabes_greedy(waypoints, eligible, min_spacing, is_valid=_clash_ok)
    diag["clash_rejets"] = clash_rejects[0]

    crabes = []
    for i in placed_idx:
        crabes.append({
            "position": waypoints[i].astype(np.float32),
            "surface_position": np.asarray(surface_pts[i], dtype=np.float32),
            "x_axis": x_axis[i].astype(np.float32),
            "y_axis": y_axis[i].astype(np.float32),
            "normal": normal[i].astype(np.float32),
        })
    return crabes, eligible, diag



def resample_curve(points, num_points):
    if len(points) < 2:
        return points
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(segment_lengths), 0, 0.0)
    total_dist = cum_dist[-1]
    if total_dist == 0:
        return np.tile(points[0], (num_points, 1))
    new_dist = np.linspace(0, total_dist, num_points)
    new_points = np.zeros((num_points, 3))
    for i in range(3):
        new_points[:, i] = np.interp(new_dist, cum_dist, points[:, i])
    return new_points.astype(np.float32)




def generate_dense_waypoints(start, goal, num_points, mesh=None, intermediate_points=None):
    num_points = int(max(2, num_points))
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

            _, s_idx = mesh.kdtree.query(s_pt)
            _, e_idx = mesh.kdtree.query(e_pt)

            try:
                # On essaie de coller à la surface
                geo_path = pv_mesh.geodesic(s_idx, e_idx)
                segment_pts = geo_path.points
            except Exception :
                # ⚠️ Si la surface est coupée (déconnectée), on fait un saut en ligne droite locale !
                print(f"⚠️ Sceau géodésique au tronçon {i+1} (Régions déconnectées), pont en ligne droite activé.")
                segment_pts = np.linspace(s_pt, e_pt, 10, dtype=np.float32)

            # On évite de dupliquer le point de jonction entre les segments
            if len(all_path_points) > 0:
                all_path_points.extend(segment_pts[1:])
            else:
                all_path_points.extend(segment_pts)

        # 4. Ré-échantillonnage global
        return resample_curve(np.array(all_path_points), num_points)

    except Exception as e:
        print(f"⚠️ Échec global géodésique ({e}), repli sur ligne droite euclidienne A->B.")
        return np.linspace(start, goal, num_points, dtype=np.float32)


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

