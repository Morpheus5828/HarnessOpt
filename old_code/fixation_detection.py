import os

os.environ["OMP_NUM_THREADS"] = "1"

import glob
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore")
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def preprocess_geometry(mesh, voxel_size):
    """Voxelise et calcule les normales du nuage de points."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2.5
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=40))
    pcd_down.orient_normals_towards_camera_location(pcd_down.get_center() + np.array([0, 0, 10000]))

    return pcd_down


def test_single_rotation_fast(rx, ry, rz, clamp_pcd_coarse, center, env_pcd_coarse, distance_threshold):
    """Teste UNE rotation en utilisant uniquement un ICP grossier très rapide."""
    pcd_rot = o3d.geometry.PointCloud()
    pcd_rot.points = o3d.utility.Vector3dVector(np.asarray(clamp_pcd_coarse.points))
    pcd_rot.normals = o3d.utility.Vector3dVector(np.asarray(clamp_pcd_coarse.normals))

    T_init = np.eye(4)

    if rx != 0 or ry != 0 or rz != 0:
        R = pcd_rot.get_rotation_matrix_from_xyz((np.radians(rx), np.radians(ry), np.radians(rz)))
        T_init[:3, :3] = R
        T_init[:3, 3] = center - R @ center
        pcd_rot.transform(T_init)

    try:
        icp_coarse = o3d.pipelines.registration.registration_icp(
            pcd_rot, env_pcd_coarse, distance_threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
        )

        final_transform = icp_coarse.transformation @ T_init
        return icp_coarse.fitness, final_transform

    except RuntimeError:
        return -1.0, np.eye(4)


def check_and_peel_optimized(clamp_path, env_pcd_fine, env_pcd_coarse, voxel_fine, voxel_coarse):
    """Gère le balayage multi-échelle et le peeling."""
    clamp_mesh = o3d.io.read_triangle_mesh(clamp_path)

    if len(clamp_mesh.vertices) == 0:
        return False, 0.0, np.eye(4), env_pcd_fine, env_pcd_coarse, None

    clamp_pcd_fine = preprocess_geometry(clamp_mesh, voxel_fine)
    clamp_pcd_coarse = clamp_pcd_fine.voxel_down_sample(voxel_coarse)

    if len(clamp_pcd_coarse.points) < 5:
        return False, 0.0, np.eye(4), env_pcd_fine, env_pcd_coarse, clamp_mesh.get_center()

    center = clamp_pcd_coarse.get_center()
    distance_threshold_coarse = voxel_coarse * 3.0

    rotations_angles = [
        (0, 0, 0), (90, 0, 0), (180, 0, 0), (270, 0, 0),
        (0, 90, 0), (90, 90, 0), (180, 90, 0), (270, 90, 0),
        (0, 270, 0), (90, 270, 0), (180, 270, 0), (270, 270, 0),
        (0, 0, 90), (90, 0, 90), (180, 0, 90), (270, 0, 90),
        (0, 0, 270), (90, 0, 270), (180, 0, 270), (270, 0, 270),
        (0, 180, 90), (0, 180, 270), (90, 180, 90), (90, 180, 270)
    ]

    best_coarse_fitness = 0.0
    best_coarse_transform = np.eye(4)

    # Lancement du Multi-Threading sécurisé par OMP_NUM_THREADS=1
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as executor:
        futures = [
            executor.submit(
                test_single_rotation_fast, rx, ry, rz, clamp_pcd_coarse, center, env_pcd_coarse,
                distance_threshold_coarse
            ) for rx, ry, rz in rotations_angles
        ]

        for future in futures:
            fit, trans = future.result()
            if fit > best_coarse_fitness:
                best_coarse_fitness = fit
                best_coarse_transform = trans

    clamp_pcd_fine_best = o3d.geometry.PointCloud()
    clamp_pcd_fine_best.points = o3d.utility.Vector3dVector(np.asarray(clamp_pcd_fine.points))
    clamp_pcd_fine_best.normals = o3d.utility.Vector3dVector(np.asarray(clamp_pcd_fine.normals))

    clamp_pcd_fine_best.transform(best_coarse_transform)

    try:
        icp_fine = o3d.pipelines.registration.registration_icp(
            clamp_pcd_fine_best, env_pcd_fine, voxel_fine * 3.0, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        eval_strict = o3d.pipelines.registration.evaluate_registration(
            clamp_pcd_fine_best, env_pcd_fine, 2.0, icp_fine.transformation
        )

        final_strict_transform = icp_fine.transformation @ best_coarse_transform
        best_strict_fitness = eval_strict.fitness

    except RuntimeError:
        best_strict_fitness = 0.0
        final_strict_transform = np.eye(4)

    is_present = best_strict_fitness >= 0.6

    if is_present:
        best_pcd = o3d.geometry.PointCloud()
        best_pcd.points = o3d.utility.Vector3dVector(np.asarray(clamp_pcd_fine.points))
        best_pcd.normals = o3d.utility.Vector3dVector(np.asarray(clamp_pcd_fine.normals))
        best_pcd.transform(final_strict_transform)

        env_pts = np.asarray(env_pcd_fine.points)
        clamp_pts = np.asarray(best_pcd.points)

        scene_tree = cKDTree(env_pts)
        dists, idxs = scene_tree.query(clamp_pts, distance_upper_bound=2.0)

        valid_idxs = idxs[dists != np.inf]
        indices_to_remove = set(valid_idxs)
        keep_indices = list(set(range(len(env_pts))) - indices_to_remove)

        env_pcd_fine = env_pcd_fine.select_by_index(keep_indices)
        env_pcd_coarse = env_pcd_fine.voxel_down_sample(voxel_coarse)

    return is_present, best_strict_fitness, final_strict_transform, env_pcd_fine, env_pcd_coarse, clamp_mesh.get_center()


def subdiviser_peigne_n_butees(mesh_local, transform_globale, n_butees):
    """Calcule les points IN et OUT des passages pour un peigne via OBB."""
    obb = mesh_local.get_oriented_bounding_box()
    center = obb.center
    extent = obb.extent
    R_obb = obb.R

    axes_tries = np.argsort(extent)
    idx_longueur = axes_tries[2]
    idx_decalage = axes_tries[1]
    idx_passage = axes_tries[0]

    vec_longueur = R_obb[:, idx_longueur]
    vec_decalage = R_obb[:, idx_decalage]
    vec_passage = R_obb[:, idx_passage]

    step = extent[idx_longueur] / n_butees
    start_pos = - (extent[idx_longueur] / 2.0) + (step / 2.0)

    R_glob = transform_globale[:3, :3]
    T_glob = transform_globale[:3, 3]

    passages = []
    for i in range(n_butees):
        decalage_l = start_pos + (i * step)
        decalage_d = - (extent[idx_decalage] / 2.0) * 0.85

        current_center_local = center + (decalage_l * vec_longueur) + (decalage_d * vec_decalage)
        p_in_local = current_center_local - (extent[idx_passage] / 2.0) * vec_passage
        p_out_local = current_center_local + (extent[idx_passage] / 2.0) * vec_passage

        p_in_global = (R_glob @ p_in_local) + T_glob
        p_out_global = (R_glob @ p_out_local) + T_glob

        passages.append({'p_in': p_in_global, 'p_out': p_out_global})

    return passages


def run_detection_for_agent(scene_path, clamps_folder, voxel_fine=0.8):
    """Fonction point d'entrée appelée par controller.py."""
    print("\n📡 Initialisation du Radar Open3D (Multi-échelle + Threads Optimisés)...")
    voxel_coarse = 2.0
    env_mesh = o3d.io.read_triangle_mesh(scene_path)

    print(f"🌍 Scan de l'environnement en cours (Fine: {voxel_fine}mm, Coarse: {voxel_coarse}mm)...")
    env_pcd_fine = preprocess_geometry(env_mesh, voxel_fine)
    env_pcd_coarse = env_pcd_fine.voxel_down_sample(voxel_coarse)

    stl_files = glob.glob(os.path.join(clamps_folder, "*.stl")) + glob.glob(os.path.join(clamps_folder, "*.STL"))

    print("\n" + "=" * 70)
    print(f"{'FICHIER CAO':<45} | {'PRÉSENT ?':<15}")
    print("=" * 70)

    detected_positions = []

    for idx, f in enumerate(stl_files):
        name = os.path.basename(f)

        is_present, score, transform, env_pcd_fine, env_pcd_coarse, center_orig = check_and_peel_optimized(
            f, env_pcd_fine, env_pcd_coarse, voxel_fine, voxel_coarse
        )

        if is_present:
            # Calcul du centre transformé pour l'agent (point d'attraction standard)
            center_homo = np.array([center_orig[0], center_orig[1], center_orig[2], 1.0])
            transformed_center = (transform @ center_homo)[:3]

            clamp_data = {
                "file_path": f,
                "name": name,
                "transform": transform.tolist(),
                "position": transformed_center.tolist(),
                "score": score
            }

            # --- Pré-calcul géométrique IN/OUT si c'est un peigne ---
            if name.upper().startswith("X") or "XA453420" in name:
                mesh_local = o3d.io.read_triangle_mesh(f)
                n_points = 13 if "XA453420" in name else 1  # Logique ML/Hardcodée de votre script

                passages = subdiviser_peigne_n_butees(mesh_local, transform, n_points)

                routing_pts = []
                for p in passages:
                    routing_pts.extend([p['p_in'].tolist(), p['p_out'].tolist()])

                # Injection dans le dictionnaire pour que l'AppController le lise directement
                clamp_data["routing_points"] = routing_pts
                print(
                    f"✅ [{idx + 1}/{len(stl_files)}] {name:<35} | OUI  (Match: {score * 100:>2.0f}%, {n_points} passages)")
            else:
                print(f"✅ [{idx + 1}/{len(stl_files)}] {name:<35} | OUI  (Match: {score * 100:>2.0f}%)")

            detected_positions.append(clamp_data)
        else:
            print(f"❌ [{idx + 1}/{len(stl_files)}] {name:<35} | NON  (Max: {score * 100:>2.0f}%)")

    print("=" * 70)
    return detected_positions
