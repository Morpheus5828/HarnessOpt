import torch

CRABE_STL_PATH = r"C:\Users\a609568\Desktop\STL\Fixations_H160\ECS0792A012A_--D_STD01_CABLE-TIE SUPPORT.1_OFFSET.stl"

_CRABE_GEOMETRY_CACHE = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

CONFIG = {
    # --- Trajectoire / marges ---
    "initial_points": 32,
    "max_points": 150,
    "min_margin": 10.0,
    "max_margin": 15.0,
    "iterations": 1500,

    # --- Exploration RL ---
    "exploration_noise_start": 0.5,
    "exploration_noise_min": 0.05,
    "exploration_decay": 0.95,  # 📉 Modifié : s'effondre plus vite pour éviter de détruire une bonne solution
    "local_max_shift": 3.0,
    "batch_size": 256,
    "train_steps": 3,
    "min_border_margin": 35.0,
    "min_bend_angle": 0.20,

    # --- Capteurs fictifs / anti-vide ---
    "sensor_count": 14,
    "sensor_weight": 20.0,
    "sensor_void_penalty": 80.0,
    "sensor_void_ratio_threshold": 0.35,
    "sensor_push_factor": 1.5,
    "hard_min_clearance_factor": 0.5,

    # --- Garde-fous anti-zigzag / anti-explosion de distance ---
    "max_shift_band_multiplier": 3.0,
    "max_step_band_multiplier": 6.0,
    "hard_max_clearance": 100.0,

    # --- "Route" (grande surface plane) vs "chemin de terre" (rebord étroit) ---
    "wide_sensor_radius_factor": 3.0,
    "flatness_weight": 15.0,
    "flatness_penalty": 40.0,
    "flatness_ratio_threshold": 0.5,

    # --- Radar prédictif vers l'avant (anticipation des vides) ---
    "lookahead_probes": 3,
    "lookahead_factor": 4.0,
    "lookahead_weight": 15.0,
    "lookahead_void_penalty": 60.0,
    "lookahead_void_ratio_threshold": 0.34,
    "lookahead_push_factor": 1.5,

    # --- Raffinement adaptatif (insertion de points là où ça bloque) ---
    "adaptive_insert_streak_threshold": 8,
    "adaptive_insert_max_per_event": 4,

    # --- Anti-tunnel (récompense) ---
    "segment_violation_penalty": 120.0,
    "segment_test_steps": 6,

    # --- Garde-fou anti-divergence (retour à la meilleure solution vue) ---
    "regression_tolerance": 2,
    "regression_patience": 15,  # 📉 Modifié : Rollback plus rapide si dégradation
    "regression_noise_cut": 0.5,

    # --- Lissage ---
    "laplace_weight": 20.0,  # 📈 Modifié : Pénalité très lourde sur les zigzags
    "smooth_bonus_weight": 60.0,
    "smooth_sharp_penalty": 150.0,  # 📈 Modifié : Sanction sévère des angles cassants
    "smoothing_iterations": 6,
    "smoothing_blend": 0.65,
    "step_momentum_explorer": 0.3,
    "step_momentum_optimizer": 0.7,
    "disp_spatial_smoothing": 0.8,  # 📈 Modifié : Force le câble à bouger d'un bloc, sans dents de scie

    # --- Convergence / gel (c'est CE mécanisme qui fige et polit la courbe finale) ---
    "success_noise_streak": 20,
    "freeze_success_streak": 30,
    "stagnation_freeze_start": 80,  # 📉 Modifié : Fige l'apprentissage beaucoup plus tôt si ça stagne

    # --- Arrondi progressif des coins (lissage par AJOUT de points, façon Chaikin) ---
    "corner_cut_cos_threshold": 0.87,
    "corner_cut_streak_threshold": 8,
    "corner_cut_max_per_event": 2,

    # --- Exploration prolongée des agents "chasseurs de crabe" ---
    "crabe_focus_noise_floor": 0.08,
    "crabe_focus_patience_mult": 2,
    "crabe_focus_noise_decay": 0.9985,

    # --- Fixations "crabe" ---
    "crabe_stl_path": CRABE_STL_PATH,
    "crabe_normal_cos_threshold": 0.85,
    "crabe_surface_tolerance": 6.0,
    "crabe_straightness_tolerance": 6.0,
    "crabe_max_clearance": 0.0,
    "crabe_min_spacing": 250.0,
    "crabe_clash_tolerance": 1.5,
    "crabe_reward_weight": 6.0,
    "crabe_count_bonus_weight": 25.0,
    "crabe_focus_multiplier": 3.0,
    "crabe_debug_every": 200,

    "existing_crabe_reward": 500.0,           # Jackpot massif si l'agent passe dedans
    "existing_crabe_attraction_radius": 150.0,# À quelle distance l'aimant commence à tirer (en mm)
    "existing_crabe_attraction_force": 3.0,   # Puissance d'aspiration vers la fixation
}

LR = 3e-4
TAU = 0.005
STATE_DIM = 20
ACTION_DIM = 3
SENSOR_COUNT_DEFAULT = 14
