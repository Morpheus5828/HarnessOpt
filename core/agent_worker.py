import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="trimesh")

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from trimesh.proximity import ProximityQuery


import time
import threading
import traceback
import torch
from scipy.spatial import KDTree

from core.agent.agent import *
from core.agent.buffer import *
from core.agent.tool import *


benchmark_algos = {
    "TD3_Geodesique": {"agent": RLAgent(use_td3=True), "buffer": ReplayBuffer(STATE_DIM, ACTION_DIM, use_cer=True),
                       "color": "cyan", "role": "explorer"},
    "SAC_Geodesique": {"agent": SACAgent(), "buffer": ReplayBuffer(STATE_DIM, ACTION_DIM, use_cer=False),
                       "color": "blue", "role": "optimizer"},

    "TD3_ChasseurCrabe_Geodesique": {"agent": RLAgent(use_td3=True),
                                     "buffer": ReplayBuffer(STATE_DIM, ACTION_DIM, use_cer=True),
                                     "color": "gold", "role": "explorer", "crabe_focus": True},

    "TD3_BiGRU_Geodesique": {"agent": RecurrentTD3Agent(),
                             "buffer": SequenceReplayBuffer(STATE_DIM, ACTION_DIM,
                                                            max_len=CONFIG["max_points"]),
                             "color": "black", "role": "explorer"},
    "TD3_BiGRU_ChasseurCrabe": {"agent": RecurrentTD3Agent(),
                                "buffer": SequenceReplayBuffer(STATE_DIM, ACTION_DIM,
                                                               max_len=CONFIG["max_points"]),
                                "color": "sienna", "role": "explorer", "crabe_focus": True},
}

data_lock = threading.Lock()
geom_lock = threading.Lock()

shared_state = {
    "is_playing": False,
    "is_running": True,
    "algos": {},
    "config": CONFIG,
}

# ⚡ Réglage CPU : OMP_NUM_THREADS/MKL_NUM_THREADS/... sont bridés à 1 en haut de ce fichier,
# volontairement, pour éviter que chacun des agents ci-dessus (qui tournent déjà chacun dans
# son propre thread, en parallèle) ne lance EN PLUS son propre pool de threads BLAS/OMP par
# dessus -> sur-souscription (N_agents x N_threads_BLAS threads pour N_coeurs coeurs), qui
# ralentit tout au lieu d'accélérer. Mais les laisser bloqués à 1 pour toujours gâche la
# puissance CPU dispo sur une machine avec plus de coeurs que d'agents. On calcule donc un
# budget de threads par agent (coeurs CPU / nb d'agents) et on relève la limite en
# conséquence : via threadpoolctl (agit sur les pools BLAS déjà chargés, contrairement aux
# variables d'environnement qui ne comptent qu'à l'import) pour numpy/trimesh/scipy, et via
# torch.set_num_threads (réglable à tout moment) pour PyTorch.
_N_AGENTS = max(1, len(benchmark_algos))
_CPU_COUNT = os.cpu_count() or 4
THREADS_PER_AGENT = max(1, _CPU_COUNT // _N_AGENTS)

try:
    torch.set_num_threads(THREADS_PER_AGENT)
except Exception:
    pass

try:
    import threadpoolctl
    threadpoolctl.threadpool_limits(limits=THREADS_PER_AGENT)
    print(f"⚙️ {_N_AGENTS} agents sur {_CPU_COUNT} coeurs CPU détectés -> {THREADS_PER_AGENT} thread(s) "
          f"BLAS/PyTorch par agent (au lieu du plancher de sécurité à 1).")
except ImportError:
    print(f"⚙️ {_N_AGENTS} agents détectés sur {_CPU_COUNT} coeurs CPU, mais 'threadpoolctl' n'est pas "
          f"installé -> les threads BLAS/OMP restent au plancher de sécurité (1). "
          f"`pip install threadpoolctl` pour exploiter jusqu'à {THREADS_PER_AGENT} thread(s)/agent.")


def algo_worker(
        algo_name,
        benchmark_algos,
        initial_waypoints,
        mesh,
        shared_state,
        data_lock,
        geom_lock,
        point_A,
        point_B
):
    try:
        if data_lock is None:
            data_lock = threading.Lock()
        if geom_lock is None:
            geom_lock = threading.Lock()

        agent = benchmark_algos[algo_name]["agent"]
        replay_buffer = benchmark_algos[algo_name]["buffer"]
        role = benchmark_algos[algo_name]["role"]
        crabe_focus = benchmark_algos[algo_name].get("crabe_focus", False)

        local_pq = ProximityQuery(mesh)

        boundary_kdtree = None
        boundary_edges_idx = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
        if len(boundary_edges_idx) > 0:
            boundary_edges = mesh.edges[boundary_edges_idx]
            boundary_vertices = mesh.vertices[np.unique(boundary_edges)]
            boundary_kdtree = KDTree(boundary_vertices)

        with data_lock:
            cfg = shared_state["config"]
            exploration_noise = cfg["exploration_noise_start"]
            local_pts = int(cfg["initial_points"])

        # ⚠️ Le contrôleur a DÉJÀ calculé ce chemin géodésique une seule fois (de façon sûre,
        # sur le thread principal) et l'a passé ici via le paramètre 'initial_waypoints' : on le
        # réutilise tel quel au lieu de relancer un second calcul géodésique. Le recalculer ici
        # faisait tourner ~5 calculs géodésiques concurrents (un par thread d'agent) à l'instant
        # du Play, ce qui pouvait faire échouer silencieusement certains d'entre eux (repli sur
        # une ligne droite) -> c'est pour ça que le tracé géodésique ne s'affichait pas
        # correctement dès le début.
        original_waypoints = np.asarray(initial_waypoints, dtype=np.float32).copy()
        with data_lock:
            shared_state["algos"][algo_name]["waypoints"] = original_waypoints.copy()
            shared_state["algos"][algo_name]["initial_waypoints"] = original_waypoints.copy()

        success_streak = 0
        last_iter = -1
        seg_streak = None
        sharp_streak = None

        best_collisions = None
        best_reward = None
        iters_since_best = 0
        insert_backoff = 1.0
        last_insert_event_iter = None
        best_at_last_insert = None
        best_waypoints = None
        best_original_waypoints = None
        best_local_pts = None
        best_locked_crabe_indices = set()
        best_locked_crabe_data = {}
        regress_streak = 0

        locked_crabe_indices = set()
        locked_crabe_data = {}

        bbox_min, bbox_max = mesh.bounds[0], mesh.bounds[1]

        while shared_state["is_running"]:
            shape_changed_this_iter = False
            with data_lock:
                playing = shared_state["is_playing"]
                # 📸 Copie privée de la config : le verrou n'est tenu que pour CES quelques
                # lectures. Tout le dépouillement ci-dessous (dizaines de cfg.get(...)) se fait
                # ensuite sur cette copie, SANS verrou -> beaucoup moins de contention entre les
                # threads d'agent qui, sinon, se bloquaient tous mutuellement à chaque itération
                # pour une simple lecture de config qui ne change quasiment jamais.
                cfg = dict(shared_state["config"])
                current_iter = shared_state["algos"][algo_name]["iteration"]
                wp_current = shared_state["algos"][algo_name]["waypoints"].copy()
                done = shared_state["algos"][algo_name]["done"]
                prev_disp = shared_state["algos"][algo_name].get("prev_disp", None)

            safe_margin = cfg["min_margin"]
            d_soft = cfg["max_margin"]
            max_iter = cfg["iterations"]
            b_margin = cfg.get("min_border_margin", 35.0)
            max_bend = cfg.get("min_bend_angle", 0.20)

            sensor_count = int(cfg.get("sensor_count", SENSOR_COUNT_DEFAULT))
            sensor_weight = cfg.get("sensor_weight", 20.0)
            sensor_void_penalty = cfg.get("sensor_void_penalty", 80.0)
            sensor_void_ratio_threshold = cfg.get("sensor_void_ratio_threshold", 0.35)
            sensor_push_factor = cfg.get("sensor_push_factor", 1.5)
            hard_min_clearance = safe_margin * cfg.get("hard_min_clearance_factor", 0.5)
            hard_max_clearance = cfg.get("hard_max_clearance", 100.0)

            wide_sensor_radius_factor = cfg.get("wide_sensor_radius_factor", 3.0)
            flatness_weight = cfg.get("flatness_weight", 15.0)
            flatness_penalty = cfg.get("flatness_penalty", 40.0)
            flatness_ratio_threshold = cfg.get("flatness_ratio_threshold", 0.5)

            lookahead_probes = int(cfg.get("lookahead_probes", 3))
            lookahead_factor = cfg.get("lookahead_factor", 4.0)
            lookahead_weight = cfg.get("lookahead_weight", 15.0)
            lookahead_void_penalty = cfg.get("lookahead_void_penalty", 60.0)
            lookahead_void_ratio_threshold = cfg.get("lookahead_void_ratio_threshold", 0.34)
            lookahead_push_factor = cfg.get("lookahead_push_factor", 1.5)

            adaptive_insert_streak_threshold = cfg.get("adaptive_insert_streak_threshold", 8)
            adaptive_insert_max_per_event = int(cfg.get("adaptive_insert_max_per_event", 4))
            segment_violation_penalty = cfg.get("segment_violation_penalty", 15.0)
            segment_test_steps = int(cfg.get("segment_test_steps", 6))
            max_points_cfg = int(cfg.get("max_points", 150))

            regression_tolerance = cfg.get("regression_tolerance", 2)
            regression_patience = cfg.get("regression_patience", 15)
            regression_noise_cut = cfg.get("regression_noise_cut", 0.5)

            band_width = max(d_soft - safe_margin, 1e-3)
            max_shift_band_multiplier = cfg.get("max_shift_band_multiplier", 3.0)
            max_step_band_multiplier = cfg.get("max_step_band_multiplier", 6.0)

            laplace_weight = cfg.get("laplace_weight", 20.0)
            smooth_bonus_weight = cfg.get("smooth_bonus_weight", 60.0)
            smooth_sharp_penalty = cfg.get("smooth_sharp_penalty", 150.0)
            smoothing_iterations = int(cfg.get("smoothing_iterations", 6))
            smoothing_blend = float(np.clip(cfg.get("smoothing_blend", 0.65), 0.0, 0.95))
            disp_spatial_smoothing = float(np.clip(cfg.get("disp_spatial_smoothing", 0.8), 0.0, 1.0))
            success_noise_streak = int(cfg.get("success_noise_streak", 20))
            freeze_success_streak = max(1, int(cfg.get("freeze_success_streak", 30)))
            stagnation_freeze_start = int(cfg.get("stagnation_freeze_start", 80))
            corner_cut_cos_threshold = cfg.get("corner_cut_cos_threshold", 0.87)
            corner_cut_streak_threshold = cfg.get("corner_cut_streak_threshold", 8)
            corner_cut_max_per_event = int(cfg.get("corner_cut_max_per_event", 2))
            crabe_focus_noise_floor = cfg.get("crabe_focus_noise_floor", 0.15)
            patience_mult = int(cfg.get("crabe_focus_patience_mult", 4)) if crabe_focus else 1

            if role == "explorer":
                local_max_shift = cfg["local_max_shift"] * 1.5
                noise_decay = cfg.get("crabe_focus_noise_decay", 0.9985) if crabe_focus \
                    else cfg.get("exploration_decay", 0.95)
                batch_size_gpu = min(cfg.get("batch_size", 256), 256)
                train_steps_gpu = 1
                step_momentum = float(np.clip(cfg.get("step_momentum_explorer", 0.3), 0.0, 0.95))
            else:
                local_max_shift = cfg["local_max_shift"] * 0.7
                noise_decay = 0.95
                batch_size_gpu = cfg.get("batch_size", 256)
                train_steps_gpu = 3
                step_momentum = float(np.clip(cfg.get("step_momentum_optimizer", 0.7), 0.0, 0.95))

            local_max_shift = min(local_max_shift, band_width * max_shift_band_multiplier)
            max_total_step = band_width * max_step_band_multiplier

            crabe_stl_path = cfg.get("crabe_stl_path", "")
            crabe_geometry = get_crabe_geometry(crabe_stl_path)
            crabe_normal_cos_threshold = cfg.get("crabe_normal_cos_threshold", 0.94)
            crabe_surface_tolerance = cfg.get("crabe_surface_tolerance", 3.0)
            crabe_straightness_tolerance = cfg.get("crabe_straightness_tolerance", 2.0)
            crabe_min_spacing = cfg.get("crabe_min_spacing", 250.0)
            crabe_clash_tolerance = cfg.get("crabe_clash_tolerance", 0.5)
            crabe_reward_weight = cfg.get("crabe_reward_weight", 10.0)
            crabe_count_bonus_weight = cfg.get("crabe_count_bonus_weight", 4.0)
            crabe_focus_multiplier = cfg.get("crabe_focus_multiplier", 10.0)

            crabe_max_clearance = cfg.get("crabe_max_clearance", 0.0)
            if not crabe_max_clearance or crabe_max_clearance <= 0:
                clip_height = crabe_geometry["height"] if crabe_geometry else 0.0
                crabe_max_clearance = max(d_soft, clip_height) + crabe_surface_tolerance
            crabe_debug_every = int(cfg.get("crabe_debug_every", 200))

            if current_iter < last_iter:
                exploration_noise = cfg["exploration_noise_start"]
                local_pts = int(cfg["initial_points"])

                # Même logique qu'au démarrage : le contrôleur vient d'écrire le nouveau
                # chemin de départ dans shared_state["waypoints"] juste avant de remettre
                # l'itération à 0 (wp_current, lu juste au-dessus, EST déjà ce chemin) ->
                # on le réutilise au lieu de relancer un calcul géodésique en concurrence
                # avec tous les autres threads d'agent au moment du reset.
                original_waypoints = wp_current.copy()
                success_streak = 0
                wp_current = original_waypoints.copy()
                prev_disp = None
                seg_streak = None
                sharp_streak = None
                shape_changed_this_iter = True
                best_collisions = None
                best_reward = None
                iters_since_best = 0
                insert_backoff = 1.0
                last_insert_event_iter = None
                best_at_last_insert = None
                best_waypoints = None
                best_original_waypoints = None
                best_local_pts = None
                regress_streak = 0
                locked_crabe_indices = set()
                locked_crabe_data = {}
                best_locked_crabe_indices = set()
                best_locked_crabe_data = {}
            last_iter = current_iter

            if not playing or done or current_iter >= max_iter:
                time.sleep(0.01)
                continue

            with geom_lock:
                closest_pts, distances, face_indices = local_pq.on_surface(wp_current)
                face_normals = mesh.face_normals[face_indices]

            new_waypoints = wp_current.copy()

            vectors_to_wp = wp_current - closest_pts
            inside_mask = np.einsum('ij,ij->i', vectors_to_wp, face_normals) < 0

            danger_indices = np.arange(1, len(wp_current) - 1)
            if crabe_focus and locked_crabe_indices:
                danger_indices = np.array(
                    [i for i in danger_indices if i not in locked_crabe_indices], dtype=danger_indices.dtype
                )

            local_reward = 0.0
            r_det = {"Margin": 0.0, "Towards": 0.0, "Smooth": 0.0, "Crash": 0.0, "Sensor": 0.0, "Flat": 0.0,
                     "Ahead": 0.0, "Crabe": 0.0, "Tunnel": 0.0}

            if len(danger_indices) > 0:
                dir_to_point = wp_current[danger_indices] - closest_pts[danger_indices]
                norms = np.linalg.norm(dir_to_point, axis=1, keepdims=True) + 1e-8
                dir_escapes = np.where(inside_mask[danger_indices, None], face_normals[danger_indices],
                                       -dir_to_point / norms)

                dists_raw = distances[danger_indices, None]
                dists_norm = np.clip(
                    np.where(inside_mask[danger_indices, None], -dists_raw / safe_margin, dists_raw / safe_margin),
                    -2.5, 2.5)

                vec_prevs = np.clip((wp_current[danger_indices - 1] - wp_current[danger_indices]) / 100.0, -2.5, 2.5)
                vec_nexts = np.clip((wp_current[danger_indices + 1] - wp_current[danger_indices]) / 100.0, -2.5, 2.5)
                vec_to_origin = np.clip(
                    (original_waypoints[danger_indices] - wp_current[danger_indices]) / (safe_margin * 2), -2.5, 2.5)

                if boundary_kdtree is not None:
                    b_dists, b_indices = boundary_kdtree.query(wp_current[danger_indices])
                    b_pts = boundary_kdtree.data[b_indices]
                    dir_to_border = b_pts - wp_current[danger_indices]
                    dir_to_border_norm = dir_to_border / (np.linalg.norm(dir_to_border, axis=1, keepdims=True) + 1e-8)
                else:
                    b_dists = np.ones(len(danger_indices)) * 100.0
                    dir_to_border_norm = np.zeros((len(danger_indices), 3), dtype=np.float32)

                sensor_radius = max(safe_margin, 1e-3)
                material_ratio, void_dir_world, frame_u, frame_v, frame_t = scan_virtual_sensors(
                    wp_current[danger_indices], wp_current[danger_indices - 1], wp_current[danger_indices + 1],
                    local_pq, sensor_radius, n_sensors=sensor_count
                )
                void_dir_local = np.stack([
                    np.einsum('ij,ij->i', void_dir_world, frame_u),
                    np.einsum('ij,ij->i', void_dir_world, frame_v),
                    np.einsum('ij,ij->i', void_dir_world, frame_t),
                ], axis=1)

                wide_radius = sensor_radius * wide_sensor_radius_factor
                material_ratio_wide, void_dir_world_wide, _, _, _ = scan_virtual_sensors(
                    wp_current[danger_indices], wp_current[danger_indices - 1], wp_current[danger_indices + 1],
                    local_pq, wide_radius, n_sensors=sensor_count
                )

                flatness_score = np.minimum(material_ratio, material_ratio_wide).astype(np.float32)
                mask_narrow_ledge = (material_ratio >= sensor_void_ratio_threshold) & (
                        material_ratio_wide < flatness_ratio_threshold)

                ahead_ratio = scan_lookahead_radar(
                    wp_current[danger_indices], frame_t, local_pq, sensor_radius, d_soft,
                    n_probes=lookahead_probes, lookahead_factor=lookahead_factor
                )
                mask_ahead_void = ahead_ratio < lookahead_void_ratio_threshold

                obs_batch = np.hstack([
                    dists_norm, dir_escapes, vec_prevs, vec_nexts, vec_to_origin,
                    material_ratio[:, None] * 2.0 - 1.0,
                    void_dir_local,
                    material_ratio_wide[:, None] * 2.0 - 1.0,
                    ahead_ratio[:, None] * 2.0 - 1.0,
                    flatness_score[:, None] * 2.0 - 1.0,
                ]).astype(np.float32)

                actions_batch = agent.select_action(obs_batch, noise=exploration_noise)
                movements = actions_batch * local_max_shift

                guiding_forces = np.zeros_like(movements)
                mask_sky_zone = distances[danger_indices] >= d_soft
                guiding_forces[mask_sky_zone] = dir_escapes[mask_sky_zone] * (local_max_shift * 2.0)

                if boundary_kdtree is not None:
                    mask_near_border = (b_dists < b_margin) & (~mask_sky_zone)
                    dir_away_from_border = -dir_to_border_norm
                    guiding_forces[mask_near_border] += dir_away_from_border[mask_near_border] * (
                            local_max_shift * 2.5)

                mask_in_void = material_ratio < sensor_void_ratio_threshold
                if np.any(mask_in_void):
                    guiding_forces[mask_in_void] += (-void_dir_world[mask_in_void]) * (
                            local_max_shift * sensor_push_factor)

                if np.any(mask_narrow_ledge):
                    guiding_forces[mask_narrow_ledge] += (-void_dir_world_wide[mask_narrow_ledge]) * (
                            local_max_shift * sensor_push_factor)

                if np.any(mask_ahead_void):
                    lateral_bias = frame_u[mask_ahead_void] * np.einsum(
                        'ij,ij->i', -void_dir_world_wide[mask_ahead_void], frame_u[mask_ahead_void])[:, None] + \
                                   frame_v[mask_ahead_void] * np.einsum(
                        'ij,ij->i', -void_dir_world_wide[mask_ahead_void], frame_v[mask_ahead_void])[:, None]
                    lateral_norm = np.linalg.norm(lateral_bias, axis=1, keepdims=True)
                    nonzero_lat = lateral_norm[:, 0] > 1e-6
                    lateral_bias[nonzero_lat] = lateral_bias[nonzero_lat] / lateral_norm[nonzero_lat]
                    guiding_forces[mask_ahead_void] += lateral_bias * (local_max_shift * lookahead_push_factor)

                total_disp = movements + guiding_forces
                disp_norm = np.linalg.norm(total_disp, axis=1, keepdims=True)
                scale_down = np.minimum(1.0, max_total_step / (disp_norm + 1e-8))
                total_disp = total_disp * scale_down

                if prev_disp is not None and prev_disp.shape == total_disp.shape:
                    total_disp = step_momentum * prev_disp + (1.0 - step_momentum) * total_disp

                """if disp_spatial_smoothing > 0.0 and len(danger_indices) > 1:
                    full_disp = np.zeros_like(wp_current)
                    full_disp[danger_indices] = total_disp
                    neighbor_mean = (full_disp[:-2] + full_disp[2:]) / 2.0
                    full_disp[1:-1] = (1.0 - disp_spatial_smoothing) * full_disp[1:-1] + \
                                      disp_spatial_smoothing * neighbor_mean
                    total_disp = full_disp[danger_indices]"""

                if disp_spatial_smoothing > 0.0 and len(danger_indices) > 1:
                    total_disp = apply_spatial_smoothing(total_disp, smoothing_factor=disp_spatial_smoothing)

                freeze_ratio = success_streak / float(freeze_success_streak * patience_mult)

                if best_collisions is not None:
                    freeze_ratio = max(freeze_ratio,
                                       (iters_since_best - stagnation_freeze_start)
                                       / float(freeze_success_streak * patience_mult))
                total_disp = total_disp * max(0.0, 1.0 - freeze_ratio)
                next_prev_disp = total_disp.copy()

                proposed_wps = wp_current[danger_indices] + total_disp

                # ==========================================================
                # 🤩 RÉCOMPENSES
                # ==========================================================
                R_margin = np.zeros_like(distances[danger_indices])
                mask_soft = (distances[danger_indices] >= safe_margin) & (distances[danger_indices] < d_soft)
                mask_hard = distances[danger_indices] < safe_margin
                mask_sky = distances[danger_indices] >= d_soft

                R_margin[mask_soft] = 100.0
                R_margin[mask_hard] = -50.0 * (safe_margin - distances[danger_indices][mask_hard]) / safe_margin
                R_margin[mask_sky] = -5.0 * (distances[danger_indices][mask_sky] - d_soft)

                dist_to_goal_current = np.linalg.norm(wp_current - point_B, axis=1)
                progression_error = dist_to_goal_current[danger_indices] - dist_to_goal_current[danger_indices - 1]
                R_sequence = np.where(progression_error > 0, -5.0, 0.0)

                progress = np.einsum('ij,ij->i', total_disp, dir_escapes)
                R_towards = (progress * 1.0) + R_sequence

                v_in = proposed_wps - wp_current[danger_indices - 1]
                v_out = wp_current[danger_indices + 1] - proposed_wps
                norm_in = np.linalg.norm(v_in, axis=1) + 1e-8
                norm_out = np.linalg.norm(v_out, axis=1) + 1e-8
                cos_angle = np.einsum('ij,ij->i', v_in, v_out) / (norm_in * norm_out)

                R_smooth = np.where(cos_angle < max_bend, -smooth_sharp_penalty, smooth_bonus_weight * cos_angle)
                laplacian = wp_current[danger_indices - 1] - 2 * proposed_wps + wp_current[danger_indices + 1]
                R_laplace = -laplace_weight * np.linalg.norm(laplacian, axis=1)

                R_smooth_total = R_smooth + R_laplace

                R_c = np.where(inside_mask[danger_indices], -100.0, 0.0)
                if boundary_kdtree is not None:
                    R_c += np.where(b_dists < b_margin, -150.0 * (1.0 - (b_dists / b_margin)), 0.0)

                R_sensor = sensor_weight * material_ratio
                R_sensor = np.where(mask_in_void, R_sensor - sensor_void_penalty, R_sensor)

                R_flatness = flatness_weight * flatness_score
                R_flatness = np.where(mask_narrow_ledge, R_flatness - flatness_penalty, R_flatness)

                R_lookahead = lookahead_weight * ahead_ratio
                R_lookahead = np.where(mask_ahead_void, R_lookahead - lookahead_void_penalty, R_lookahead)

                candidate_full = wp_current.copy()
                candidate_full[danger_indices] = proposed_wps

                """if segment_violation_penalty > 0 and len(candidate_full) > 1:
                    seg_test_pts = get_segment_test_points(candidate_full, steps=segment_test_steps)
                    with geom_lock:
                        _, seg_test_dist, seg_test_faces = local_pq.on_surface(seg_test_pts)
                        seg_centroids = mesh.vertices[mesh.faces[seg_test_faces]].mean(axis=1)
                        seg_inside = np.einsum('ij,ij->i', seg_test_pts - seg_centroids,
                                               mesh.face_normals[seg_test_faces]) < 0
                    seg_viol_count = (seg_inside | (seg_test_dist < safe_margin)).reshape(
                        len(candidate_full) - 1, -1).sum(axis=1).astype(np.float32)
                    viol_per_point = np.zeros(len(candidate_full), dtype=np.float32)
                    viol_per_point[:-1] += seg_viol_count
                    viol_per_point[1:] += seg_viol_count

                    viol_fraction = viol_per_point[danger_indices] / float(2 * segment_test_steps)
                    R_tunnel = -segment_violation_penalty * viol_fraction
                else:
                    R_tunnel = np.zeros(len(danger_indices), dtype=np.float32)"""

                if cfg.get("segment_violation_penalty", 0) > 0 and len(candidate_full) > 1:
                    with geom_lock:
                        tunnel_penalty, crossing_mask = compute_collision_penalties(
                            candidate_full, mesh, cfg, local_pq=local_pq
                        )

                    R_tunnel = -tunnel_penalty / max(1, len(danger_indices))
                else:
                    R_tunnel = np.zeros(len(danger_indices), dtype=np.float32)

                if crabe_geometry is not None:
                    crabe_eligible_full, crabe_margin_full, _, _, _, _, _ = evaluate_crabe_alignment(
                        candidate_full, local_pq, mesh, crabe_geometry["dx"], crabe_geometry["dy"],
                        crabe_normal_cos_threshold, crabe_surface_tolerance, crabe_straightness_tolerance,
                        max_clearance=crabe_max_clearance
                    )
                    crabe_eligible_next = crabe_eligible_full[danger_indices]
                    crabe_margin_next = crabe_margin_full[danger_indices]

                    focus_mult = crabe_focus_multiplier if crabe_focus else 1.0

                    R_crabe = (crabe_reward_weight * focus_mult) * np.clip(crabe_margin_next, 0.0, 1.0)
                    R_crabe = np.where(crabe_eligible_next, R_crabe + 5.0 * focus_mult, R_crabe)

                    current_crabes = place_crabes_greedy(candidate_full, crabe_eligible_full, crabe_min_spacing)
                    crabe_count_for_reward = len(current_crabes)

                    R_crabe = R_crabe + crabe_count_bonus_weight * crabe_count_for_reward
                else:
                    R_crabe = np.zeros(len(danger_indices), dtype=np.float32)

                # ==========================================================
                # 🧲 GUIDAGE & RÉCOMPENSE VERS LES FIXATIONS EXISTANTES
                # ==========================================================
                existing_crabes = cfg.get("existing_crabes", [])
                R_existing = np.zeros(len(danger_indices), dtype=np.float32)

                if existing_crabes:
                    for crabe in existing_crabes:
                        c_pos = crabe["position"]
                        dists_to_c = np.linalg.norm(wp_current[danger_indices] - c_pos, axis=1)

                        # 1. Force d'attraction (L'aimant)
                        mask_near = dists_to_c < cfg.get("existing_crabe_attraction_radius", 150.0)
                        if np.any(mask_near):
                            dir_to_c = c_pos - wp_current[danger_indices][mask_near]
                            dir_to_c_norm = dir_to_c / (np.linalg.norm(dir_to_c, axis=1, keepdims=True) + 1e-8)
                            guiding_forces[mask_near] += dir_to_c_norm * (
                                        local_max_shift * cfg.get("existing_crabe_attraction_force", 3.0))

                        # 2. Jackpot (Validation)
                        mask_hit = dists_to_c < 20.0  # Rayon de tolérance pour considérer la fixation "capturée"
                        R_existing[mask_hit] += cfg.get("existing_crabe_reward", 500.0)

                rewards_batch = (R_margin + R_towards + R_smooth_total + R_c + R_sensor + R_flatness
                                 + R_lookahead + R_crabe + R_tunnel + R_existing)  # <-- Ajouté ici

                r_det = {
                    "Margin": float(R_margin.mean()),
                    "Towards": float(R_towards.mean()),
                    "Smooth": float(R_smooth_total.mean()),
                    "Crash": float(R_c.mean()),
                    "Sensor": float(R_sensor.mean()),
                    "Flat": float(R_flatness.mean()),
                    "Ahead": float(R_lookahead.mean()),
                    "Crabe": float(R_crabe.mean()),
                    "Tunnel": float(R_tunnel.mean()),
                    "Existant": float(R_existing.mean()),
                }
                local_reward = sum(r_det.values())

                dists_next_raw = dists_raw - progress[:, None]
                dists_next_norm = np.clip(np.where(inside_mask[danger_indices, None], -dists_next_raw / safe_margin,
                                                   dists_next_raw / safe_margin), -2.5, 2.5)
                vec_to_origin_next = np.clip((original_waypoints[danger_indices] - proposed_wps) / (safe_margin * 2),
                                             -2.5, 2.5)

                material_ratio_next, void_dir_world_next, frame_u_n, frame_v_n, frame_t_n = scan_virtual_sensors(
                    proposed_wps, wp_current[danger_indices - 1], wp_current[danger_indices + 1],
                    local_pq, sensor_radius, n_sensors=sensor_count
                )
                void_dir_local_next = np.stack([
                    np.einsum('ij,ij->i', void_dir_world_next, frame_u_n),
                    np.einsum('ij,ij->i', void_dir_world_next, frame_v_n),
                    np.einsum('ij,ij->i', void_dir_world_next, frame_t_n),
                ], axis=1)

                material_ratio_wide_next, _, _, _, _ = scan_virtual_sensors(
                    proposed_wps, wp_current[danger_indices - 1], wp_current[danger_indices + 1],
                    local_pq, wide_radius, n_sensors=sensor_count
                )
                ahead_ratio_next = scan_lookahead_radar(
                    proposed_wps, frame_t_n, local_pq, sensor_radius, d_soft,
                    n_probes=lookahead_probes, lookahead_factor=lookahead_factor
                )

                flatness_score_next = np.minimum(material_ratio_next, material_ratio_wide_next).astype(np.float32)

                next_obs_batch = np.hstack([
                    dists_next_norm, dir_escapes, vec_prevs, vec_nexts, vec_to_origin_next,
                    material_ratio_next[:, None] * 2.0 - 1.0,
                    void_dir_local_next,
                    material_ratio_wide_next[:, None] * 2.0 - 1.0,
                    ahead_ratio_next[:, None] * 2.0 - 1.0,
                    flatness_score_next[:, None] * 2.0 - 1.0,
                ]).astype(np.float32)

                replay_buffer.add(obs_batch, actions_batch, next_obs_batch, rewards_batch)

                if replay_buffer.size > 256 and current_iter % 5 == 0:
                    for _ in range(train_steps_gpu):
                        agent.train(replay_buffer, batch_size=256)

                new_waypoints[danger_indices] = proposed_wps

            new_waypoints = np.clip(new_waypoints, bbox_min, bbox_max)

            # ==========================================================
            # 📏 LISSAGE ET DÉCALAGE DE BORDURE PHYSIQUE
            # ==========================================================
            smoothed_waypoints = new_waypoints.copy()
            for _ in range(smoothing_iterations):
                pts_mid = (smoothed_waypoints[:-2] + smoothed_waypoints[2:]) / 2.0
                candidate_inner = (1.0 - smoothing_blend) * smoothed_waypoints[1:-1] + \
                                  smoothing_blend * pts_mid
                with geom_lock:
                    cand_closest, cand_dist, cand_faces = local_pq.on_surface(candidate_inner)
                cand_inside = np.einsum('ij,ij->i', candidate_inner - cand_closest,
                                        mesh.face_normals[cand_faces]) < 0

                keep = (~cand_inside) & (cand_dist >= safe_margin)
                smoothed_waypoints[1:-1][keep] = candidate_inner[keep]

            if boundary_kdtree is not None:
                b_dists_post, b_idx_post = boundary_kdtree.query(smoothed_waypoints[1:-1])
                mask_too_close_border = (b_dists_post < b_margin) & (distances[1:-1] < d_soft)
                if np.any(mask_too_close_border):
                    vec_away = smoothed_waypoints[1:-1][mask_too_close_border] - boundary_kdtree.data[
                        b_idx_post[mask_too_close_border]]
                    vec_away_norm = vec_away / (np.linalg.norm(vec_away, axis=1, keepdims=True) + 1e-8)
                    push_dist = b_margin - b_dists_post[mask_too_close_border]
                    smoothed_waypoints[1:-1][mask_too_close_border] += vec_away_norm * push_dist[:, None]

            with geom_lock:
                _, distances_post, face_idx_post = local_pq.on_surface(smoothed_waypoints)
            vectors_post = smoothed_waypoints - closest_pts
            inside_post = np.einsum('ij,ij->i', vectors_post, face_normals) < 0

            ideal_target = (safe_margin + d_soft) / 2.0
            for idx in range(1, len(smoothed_waypoints) - 1):
                vec_to_mesh = smoothed_waypoints[idx] - closest_pts[idx]
                norm_vec = np.linalg.norm(vec_to_mesh)
                if inside_post[idx] or distances_post[idx] < safe_margin:
                    dir_push = face_normals[idx] if inside_post[idx] else (vec_to_mesh / (norm_vec + 1e-8))
                    smoothed_waypoints[idx] = closest_pts[idx] + dir_push * ideal_target

            # ==========================================================
            # 🔒 VERROU DUR ANTI-CONTACT
            # ==========================================================
            with geom_lock:
                _, distances_final, face_idx_final = local_pq.on_surface(smoothed_waypoints)
            closest_final = local_pq.on_surface(smoothed_waypoints)[0]
            normals_final = mesh.face_normals[face_idx_final]
            vectors_final = smoothed_waypoints - closest_final
            inside_final = np.einsum('ij,ij->i', vectors_final, normals_final) < 0

            violation = inside_final | (distances_final < hard_min_clearance)
            if np.any(violation):
                vio_idx = np.where(violation)[0]
                vio_idx = vio_idx[(vio_idx > 0) & (vio_idx < len(smoothed_waypoints) - 1)]
                if len(vio_idx) > 0:
                    dir_push_hard = np.where(
                        inside_final[vio_idx, None],
                        normals_final[vio_idx],
                        (vectors_final[vio_idx] / (
                                    np.linalg.norm(vectors_final[vio_idx], axis=1, keepdims=True) + 1e-8))
                    )
                    smoothed_waypoints[vio_idx] = closest_final[vio_idx] + dir_push_hard * hard_min_clearance

            over_violation = (~inside_final) & (distances_final > hard_max_clearance)
            if np.any(over_violation):
                over_idx = np.where(over_violation)[0]
                over_idx = over_idx[(over_idx > 0) & (over_idx < len(smoothed_waypoints) - 1)]
                if len(over_idx) > 0:
                    dir_back = vectors_final[over_idx] / (
                            np.linalg.norm(vectors_final[over_idx], axis=1, keepdims=True) + 1e-8)
                    smoothed_waypoints[over_idx] = closest_final[over_idx] + dir_back * hard_max_clearance

            if crabe_focus and locked_crabe_indices:
                for idx in locked_crabe_indices:
                    if 0 <= idx < len(smoothed_waypoints) and idx in locked_crabe_data:
                        smoothed_waypoints[idx] = locked_crabe_data[idx]["position"]

            test_pts = get_segment_test_points(smoothed_waypoints, steps=10)
            with geom_lock:
                _, distances_test, face_idx_test = local_pq.on_surface(test_pts)
                ins_test = np.einsum('ij,ij->i', test_pts - mesh.vertices[mesh.faces[face_idx_test]].mean(axis=1),
                                     mesh.face_normals[face_idx_test]) < 0

            with geom_lock:
                n_crossings, crossing_mask = count_segment_face_crossings(smoothed_waypoints, mesh)

            current_collisions = int(np.sum(ins_test | (distances_test < safe_margin)))

            # ==========================================================
            # 🧬 RAFFINEMENT ADAPTATIF
            # ==========================================================
            """max_points_reached_msg = ""
            n_segments = len(smoothed_waypoints) - 1
            if n_segments > 0:
                viol_per_test = (ins_test | (distances_test < safe_margin)).reshape(n_segments, -1)
                segment_violation = viol_per_test.any(axis=1)
                deep_per_test = (ins_test | (distances_test < hard_min_clearance)).reshape(n_segments, -1)
                segment_hard_violation = deep_per_test.any(axis=1) | crossing_mask

                if seg_streak is None or len(seg_streak) != n_segments:
                    seg_streak = np.zeros(n_segments, dtype=np.float32)
                seg_streak = np.where(segment_violation | segment_hard_violation,
                                      seg_streak + 1.0, np.maximum(seg_streak - 2, 0))

                violating_idx = np.where(seg_streak > adaptive_insert_streak_threshold * insert_backoff)[0]
                budget = max_points_cfg - len(smoothed_waypoints)

                if len(violating_idx) > 0 and budget > 0 and regress_streak == 0:
                    n_insert = min(budget, adaptive_insert_max_per_event)
                    ranked = violating_idx[np.argsort(-seg_streak[violating_idx])][:n_insert]
                    for seg_idx in np.sort(ranked)[::-1]:
                        mid_point = (smoothed_waypoints[seg_idx] + smoothed_waypoints[seg_idx + 1]) / 2.0
                        with geom_lock:
                            mid_closest, mid_dist, mid_face = local_pq.on_surface(mid_point[None, :])
                        mid_normal = mesh.face_normals[mid_face[0]]
                        mid_vec = mid_point - mid_closest[0]
                        mid_norm = np.linalg.norm(mid_vec)
                        mid_inside = mid_norm < 1e-8 or np.dot(mid_vec, mid_normal) < 0
                        mid_dir = mid_normal if mid_inside else mid_vec / mid_norm
                        shift = (mid_closest[0] + mid_dir * ideal_target) - mid_point
                        shift_norm = np.linalg.norm(shift)

                        if shift_norm > max_total_step:
                            shift = shift * (max_total_step / shift_norm)
                        new_point = (mid_point + shift).astype(smoothed_waypoints.dtype)
                        smoothed_waypoints = np.insert(smoothed_waypoints, seg_idx + 1, new_point, axis=0)
                        new_origin_point = (original_waypoints[seg_idx] + original_waypoints[seg_idx + 1]) / 2.0
                        original_waypoints = np.insert(original_waypoints, seg_idx + 1, new_origin_point, axis=0)
                        if crabe_focus and locked_crabe_indices:
                            shifted_indices = set()
                            shifted_data = {}
                            for lidx in locked_crabe_indices:
                                new_lidx = lidx + 1 if lidx >= seg_idx + 1 else lidx
                                shifted_indices.add(new_lidx)
                                shifted_data[new_lidx] = locked_crabe_data[lidx]
                            locked_crabe_indices = shifted_indices
                            locked_crabe_data = shifted_data
                    seg_streak = None
                    local_pts = len(smoothed_waypoints)
                    prev_disp = None
                    success_streak = 0
                    shape_changed_this_iter = True
                    last_insert_event_iter = current_iter
                    best_at_last_insert = best_collisions
                elif len(violating_idx) > 0 and budget <= 0:
                    max_points_reached_msg = f" ⚠️ max_points({max_points_cfg}) atteint, encore {len(violating_idx)} segment(s) en collision"

                need_budget = (len(violating_idx) > 0 and budget < adaptive_insert_max_per_event) or \
                              (len(smoothed_waypoints) > 0.9 * max_points_cfg)
                if need_budget and not shape_changed_this_iter and len(smoothed_waypoints) > 8:
                    deviation = np.linalg.norm(
                        smoothed_waypoints[1:-1] - (smoothed_waypoints[:-2] + smoothed_waypoints[2:]) / 2.0,
                        axis=1)
                    clean_seg = ~segment_violation
                    removable = clean_seg[:-1] & clean_seg[1:] & (deviation < band_width * 0.2)
                    candidates = np.where(removable)[0] + 1
                    if crabe_focus and locked_crabe_indices:
                        candidates = np.array([i for i in candidates if i not in locked_crabe_indices],
                                              dtype=int)
                    picked = []
                    for i in candidates[np.argsort(deviation[candidates - 1])] if len(candidates) else []:
                        if all(abs(int(i) - j) >= 2 for j in picked):
                            picked.append(int(i))
                        if len(picked) >= adaptive_insert_max_per_event:
                            break
                    for i in sorted(picked, reverse=True):
                        smoothed_waypoints = np.delete(smoothed_waypoints, i, axis=0)
                        original_waypoints = np.delete(original_waypoints, i, axis=0)
                        if crabe_focus and locked_crabe_indices:
                            locked_crabe_indices = {(l - 1 if l > i else l) for l in locked_crabe_indices}
                            locked_crabe_data = {(l - 1 if l > i else l): d
                                                 for l, d in locked_crabe_data.items()}
                    if picked:
                        seg_streak = None
                        local_pts = len(smoothed_waypoints)
                        prev_disp = None
                        shape_changed_this_iter = True"""

            # ==========================================================
            # 🧬 RAFFINEMENT ADAPTATIF (VERSION CORRIGÉE & SÉCURISÉE)
            # ==========================================================
            max_points_reached_msg = ""
            n_segments = len(smoothed_waypoints) - 1

            if n_segments > 0:
                # 1. Analyse des collisions par segment (échantillonnage + traversées exactes)
                viol_per_test = (ins_test | (distances_test < safe_margin)).reshape(n_segments, -1)
                segment_violation = viol_per_test.any(axis=1)

                deep_per_test = (ins_test | (distances_test < hard_min_clearance)).reshape(n_segments, -1)

                # Prise en compte explicite de crossing_mask (traversée exacte de triangle)
                segment_hard_violation = deep_per_test.any(axis=1)
                if crossing_mask is not None and len(crossing_mask) == n_segments:
                    segment_hard_violation |= crossing_mask

                # 2. Accumulation des streaks de collision
                if seg_streak is None or len(seg_streak) != n_segments:
                    seg_streak = np.zeros(n_segments, dtype=np.float32)

                seg_streak = np.where(
                    segment_violation | segment_hard_violation,
                    seg_streak + 1.0,
                    np.maximum(seg_streak - 2.0, 0.0)
                )

                # 3. Identification des segments bloqués nécessitant un ajout de point
                violating_idx = np.where(seg_streak > adaptive_insert_streak_threshold * insert_backoff)[0]
                budget = max_points_cfg - len(smoothed_waypoints)

                # A. CAS 1 : INSERTION DE NOUVEAUX POINTS
                if len(violating_idx) > 0 and budget > 0 and regress_streak == 0:
                    n_insert = min(budget, adaptive_insert_max_per_event)
                    ranked = violating_idx[np.argsort(-seg_streak[violating_idx])][:n_insert]

                    # Insertion en partant de la fin pour ne pas fausser les index pendant la boucle
                    for seg_idx in np.sort(ranked)[::-1]:
                        mid_point = (smoothed_waypoints[seg_idx] + smoothed_waypoints[seg_idx + 1]) / 2.0

                        # Reprojection géométrique du point au-dessus de la surface
                        with geom_lock:
                            mid_closest, mid_dist, mid_face = local_pq.on_surface(mid_point[None, :])

                        mid_normal = mesh.face_normals[mid_face[0]]
                        mid_vec = mid_point - mid_closest[0]
                        mid_norm = np.linalg.norm(mid_vec)

                        mid_inside = (mid_norm < 1e-8) or (np.dot(mid_vec, mid_normal) < 0)
                        mid_dir = mid_normal if mid_inside else (mid_vec / (mid_norm + 1e-8))

                        shift = (mid_closest[0] + mid_dir * ideal_target) - mid_point
                        shift_norm = np.linalg.norm(shift)

                        if shift_norm > max_total_step:
                            shift = shift * (max_total_step / shift_norm)

                        new_point = (mid_point + shift).astype(smoothed_waypoints.dtype)

                        # Insertion dans les trajectoires
                        smoothed_waypoints = np.insert(smoothed_waypoints, seg_idx + 1, new_point, axis=0)

                        new_origin_point = (original_waypoints[seg_idx] + original_waypoints[seg_idx + 1]) / 2.0
                        original_waypoints = np.insert(original_waypoints, seg_idx + 1, new_origin_point, axis=0)

                        # Mise à jour des index des fixations/crabes verrouillés
                        if crabe_focus and locked_crabe_indices:
                            shifted_indices = set()
                            shifted_data = {}
                            for lidx in locked_crabe_indices:
                                new_lidx = lidx + 1 if lidx >= seg_idx + 1 else lidx
                                shifted_indices.add(new_lidx)
                                shifted_data[new_lidx] = locked_crabe_data[lidx]
                            locked_crabe_indices = shifted_indices
                            locked_crabe_data = shifted_data

                    # Reset des états post-insertion
                    seg_streak = None
                    local_pts = len(smoothed_waypoints)
                    prev_disp = None
                    success_streak = 0
                    shape_changed_this_iter = True
                    last_insert_event_iter = current_iter
                    best_at_last_insert = best_collisions

                elif len(violating_idx) > 0 and budget <= 0:
                    max_points_reached_msg = f" ⚠️ max_points({max_points_cfg}) atteint, encore {len(violating_idx)} segment(s) en collision"

                # B. CAS 2 : NETTOYAGE / SUPPRESSION DE POINTS INUTILES SI LE BUDGET EST SERRE
                need_budget = (len(violating_idx) > 0 and budget < adaptive_insert_max_per_event) or \
                              (len(smoothed_waypoints) > 0.9 * max_points_cfg)

                if need_budget and not shape_changed_this_iter and len(smoothed_waypoints) > 8:
                    # Calcul de la déviation géométrique (alignement local)
                    deviation = np.linalg.norm(
                        smoothed_waypoints[1:-1] - (smoothed_waypoints[:-2] + smoothed_waypoints[2:]) / 2.0,
                        axis=1
                    )
                    clean_seg = ~segment_violation
                    removable = clean_seg[:-1] & clean_seg[1:] & (deviation < band_width * 0.2)
                    candidates = np.where(removable)[0] + 1

                    # Protection des fixations verrouillées
                    if crabe_focus and locked_crabe_indices:
                        candidates = np.array([i for i in candidates if i not in locked_crabe_indices], dtype=int)

                    picked = []
                    if len(candidates) > 0:
                        for i in candidates[np.argsort(deviation[candidates - 1])]:
                            if all(abs(int(i) - j) >= 2 for j in picked):
                                picked.append(int(i))
                            if len(picked) >= adaptive_insert_max_per_event:
                                break

                    # Suppression des points inutiles
                    for i in sorted(picked, reverse=True):
                        smoothed_waypoints = np.delete(smoothed_waypoints, i, axis=0)
                        original_waypoints = np.delete(original_waypoints, i, axis=0)

                        if crabe_focus and locked_crabe_indices:
                            locked_crabe_indices = {(l - 1 if l > i else l) for l in locked_crabe_indices}
                            locked_crabe_data = {(l - 1 if l > i else l): d for l, d in locked_crabe_data.items()}

                    if picked:
                        seg_streak = None
                        local_pts = len(smoothed_waypoints)
                        prev_disp = None
                        shape_changed_this_iter = True

            v_in_all = smoothed_waypoints[1:-1] - smoothed_waypoints[:-2]
            v_out_all = smoothed_waypoints[2:] - smoothed_waypoints[1:-1]
            n_in_all = np.linalg.norm(v_in_all, axis=1) + 1e-8
            n_out_all = np.linalg.norm(v_out_all, axis=1) + 1e-8
            cos_angle_all = np.einsum('ij,ij->i', v_in_all, v_out_all) / (n_in_all * n_out_all)

            if current_collisions == 0 and n_crossings == 0 and np.all(cos_angle_all > max_bend):
                success_streak += 1
            else:
                success_streak = 0

            # ==========================================================
            # ✂️ ARRONDI PROGRESSIF DES COINS
            # ==========================================================
            if corner_cut_max_per_event > 0 and len(smoothed_waypoints) >= 4:
                n_interior = len(smoothed_waypoints) - 2
                if sharp_streak is None or len(sharp_streak) != n_interior:
                    sharp_streak = np.zeros(n_interior, dtype=np.float32)
                is_sharp = cos_angle_all < corner_cut_cos_threshold
                sharp_streak = np.where(is_sharp, sharp_streak + 1, np.maximum(sharp_streak - 2, 0))

                corner_budget = max_points_cfg - len(smoothed_waypoints)
                if not shape_changed_this_iter and corner_budget > 0:
                    corner_cand = np.where(sharp_streak > corner_cut_streak_threshold)[0] + 1
                    if crabe_focus and locked_crabe_indices:
                        corner_cand = np.array(
                            [i for i in corner_cand if i not in locked_crabe_indices], dtype=int)
                    picked_corners = []
                    n_cut = min(corner_cut_max_per_event, corner_budget)
                    for i in (corner_cand[np.argsort(cos_angle_all[corner_cand - 1])]
                    if len(corner_cand) else []):
                        if all(abs(int(i) - j) >= 2 for j in picked_corners):
                            picked_corners.append(int(i))
                        if len(picked_corners) >= n_cut:
                            break
                    for i in sorted(picked_corners, reverse=True):
                        p_prev, p, p_next = smoothed_waypoints[i - 1], smoothed_waypoints[i], \
                            smoothed_waypoints[i + 1]
                        cut_pts = np.asarray([0.75 * p + 0.25 * p_prev, 0.75 * p + 0.25 * p_next],
                                             dtype=smoothed_waypoints.dtype)
                        smoothed_waypoints = np.concatenate(
                            [smoothed_waypoints[:i], cut_pts, smoothed_waypoints[i + 1:]], axis=0)
                        o_prev, o, o_next = original_waypoints[i - 1], original_waypoints[i], \
                            original_waypoints[i + 1]
                        cut_orig = np.asarray([0.75 * o + 0.25 * o_prev, 0.75 * o + 0.25 * o_next],
                                              dtype=original_waypoints.dtype)
                        original_waypoints = np.concatenate(
                            [original_waypoints[:i], cut_orig, original_waypoints[i + 1:]], axis=0)
                        if crabe_focus and locked_crabe_indices:
                            locked_crabe_indices = {(l + 1 if l > i else l) for l in locked_crabe_indices}
                            locked_crabe_data = {(l + 1 if l > i else l): d
                                                 for l, d in locked_crabe_data.items()}
                    if picked_corners:
                        sharp_streak = None
                        seg_streak = None
                        local_pts = len(smoothed_waypoints)
                        prev_disp = None
                        shape_changed_this_iter = True

            noise_floor = crabe_focus_noise_floor if crabe_focus else 0.05
            exploration_noise = max(noise_floor, exploration_noise * noise_decay)
            if success_streak > success_noise_streak * patience_mult:
                exploration_noise = 0.0
            if best_collisions is not None and \
                    iters_since_best > stagnation_freeze_start + freeze_success_streak * patience_mult:
                exploration_noise = 0.0
            curr_mean_dist = float(np.mean(distances_post[1:-1])) if len(distances_post) > 2 else 0.0

            # ==========================================================
            # 🏆 MÉMOIRE DE LA MEILLEURE SOLUTION + ROLLBACK ANTI-DIVERGENCE
            # ==========================================================
            is_new_best = (
                    best_collisions is None
                    or current_collisions < best_collisions
                    or (current_collisions == best_collisions and (best_reward is None or local_reward > best_reward))
            )

            if last_insert_event_iter is not None:
                if is_new_best and best_collisions is not None and best_at_last_insert is not None \
                        and current_collisions < best_at_last_insert:
                    insert_backoff = 1.0
                    last_insert_event_iter = None
                elif current_iter - last_insert_event_iter > 40:
                    insert_backoff = min(insert_backoff * 2.0, 32.0)
                    last_insert_event_iter = None

            if is_new_best:
                best_collisions = current_collisions
                best_reward = local_reward
                best_waypoints = smoothed_waypoints.copy()
                best_original_waypoints = original_waypoints.copy()
                best_local_pts = local_pts
                best_locked_crabe_indices = set(locked_crabe_indices)
                best_locked_crabe_data = dict(locked_crabe_data)
                regress_streak = 0
                iters_since_best = 0
            elif best_collisions is not None and current_collisions > best_collisions + regression_tolerance:
                regress_streak += 1
                iters_since_best += 1
            else:
                regress_streak = max(0, regress_streak - 1)
                iters_since_best += 1

            if regress_streak > regression_patience and best_waypoints is not None:
                smoothed_waypoints = best_waypoints.copy()
                original_waypoints = best_original_waypoints.copy()
                local_pts = best_local_pts
                current_collisions = best_collisions
                exploration_noise = max(0.05, exploration_noise * regression_noise_cut)
                prev_disp = None
                seg_streak = None
                sharp_streak = None
                success_streak = 0
                regress_streak = 0
                shape_changed_this_iter = True
                locked_crabe_indices = set(best_locked_crabe_indices)
                locked_crabe_data = dict(best_locked_crabe_data)

                with geom_lock:
                    _, distances_post, _ = local_pq.on_surface(smoothed_waypoints)
                curr_mean_dist = float(np.mean(distances_post[1:-1])) if len(distances_post) > 2 else 0.0

            existing_crabes_list = cfg.get("existing_crabes", [])
            existing_positions_list = []
            for c in existing_crabes_list:
                if "routing_points" in c and c["routing_points"]:
                    existing_positions_list.extend(c["routing_points"])
                else:
                    existing_positions_list.append(c["position"])

            existing_positions_arr = np.array(existing_positions_list,
                                              dtype=np.float32) if existing_positions_list else None

            with geom_lock:
                final_crabes, _, crabe_diag = compute_crabes(
                    smoothed_waypoints, local_pq, mesh, crabe_geometry, crabe_normal_cos_threshold,
                    crabe_surface_tolerance, crabe_straightness_tolerance, crabe_min_spacing,
                    max_clearance=crabe_max_clearance, clash_tolerance=crabe_clash_tolerance,
                    existing_positions=existing_positions_arr
                )

            if crabe_debug_every > 0 and current_iter % crabe_debug_every == 0 and crabe_geometry is None:
                if not crabe_stl_path:
                    print(f"🦀 [{algo_name}] fonctionnalité crabe DÉSACTIVÉE : crabe_stl_path est vide "
                          f"(renseignez CRABE_STL_PATH en tête de fichier ou la variable d'environnement)")
                else:
                    print(f"🦀 [{algo_name}] AUCUN crabe possible : le clip '{crabe_stl_path}' n'a pas pu "
                          f"être chargé par trimesh (voir l'avertissement au démarrage)")
            if crabe_geometry is not None and crabe_diag and crabe_debug_every > 0 \
                    and current_iter % crabe_debug_every == 0:
                print(f"🦀 [{algo_name}] iter {current_iter} : {crabe_diag['points']} pts | "
                      f"hauteur<= {crabe_max_clearance:.0f}mm : {crabe_diag['hauteur_ok']} | "
                      f"plan : {crabe_diag['plan_ok']} | droit : {crabe_diag['droit_ok']} | "
                      f"éligibles : {crabe_diag['eligibles']} | rejets clash : {crabe_diag['clash_rejets']} | "
                      f"posés : {len(final_crabes)}")

            with data_lock:
                shared_state["algos"][algo_name]["waypoints"] = smoothed_waypoints
                shared_state["algos"][algo_name]["iteration"] += 1
                shared_state["algos"][algo_name]["collisions"] = current_collisions

                shared_state["algos"][algo_name]["crossings"] = n_crossings
                shared_state["algos"][algo_name].setdefault("crossing_history", []).append(n_crossings)

                traj_length = float(np.sum(np.linalg.norm(np.diff(smoothed_waypoints, axis=0), axis=1)))
                shared_state["algos"][algo_name]["length"] = traj_length
                shared_state["algos"][algo_name].setdefault("length_history", []).append(traj_length)
                shared_state["algos"][algo_name]["reward"] = local_reward
                shared_state["algos"][algo_name]["reward_history"].append(local_reward)
                shared_state["algos"][algo_name]["reward_details"] = r_det
                shared_state["algos"][algo_name]["density"] = local_pts
                shared_state["algos"][algo_name]["min_dist"] = float(np.min(distances_post[1:-1])) if len(
                    distances_post) > 2 else 0.0
                shared_state["algos"][algo_name]["max_dist"] = float(np.max(distances_post[1:-1])) if len(
                    distances_post) > 2 else 0.0
                shared_state["algos"][algo_name]["mean_dist"] = curr_mean_dist
                shared_state["algos"][algo_name]["dist_history"].append(curr_mean_dist)
                shared_state["algos"][algo_name]["collision_history"].append(current_collisions)
                shared_state["algos"][algo_name]["point_count_history"].append(local_pts)
                shared_state["algos"][algo_name]["crabes"] = final_crabes
                shared_state["algos"][algo_name]["crabe_count"] = len(final_crabes)
                shared_state["algos"][algo_name].setdefault("crabe_count_history", []).append(len(final_crabes))

                shared_state["algos"][algo_name]["prev_disp"] = next_prev_disp if (
                        not shape_changed_this_iter and len(danger_indices) > 0) else None
                shared_state["algos"][algo_name]["best_collisions"] = best_collisions

                shared_state["algos"][algo_name]["stuck_msg"] = (
                    f" | Pt:{local_pts} | Traversées:{n_crossings}{max_points_reached_msg}")

                if success_streak > 120 * patience_mult:
                    shared_state["algos"][algo_name]["done"] = True

    except Exception:
        traceback.print_exc()
        with data_lock:
            shared_state["algos"][algo_name]["stuck_msg"] = " (CRASH RUNTIME)"
            shared_state["algos"][algo_name]["done"] = True
