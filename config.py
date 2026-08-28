from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BASE_CACHE = BASE_DIR / "resources" / "cache"
BASE_CACHE.mkdir(parents=True, exist_ok=True)

MODEL_PATH = BASE_DIR / "resources" / "model" / "best_mesh_classifier.joblib"

INPUT_DIR = BASE_DIR / "resources"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

STL_FOLDER = BASE_CACHE / "stl"
COLOR_PATH = BASE_CACHE / "color" / "dmu_color_parts.xlsx"
FUSION_DIR = BASE_CACHE / "fusion"

COLOR_BG = ("#F0F2F5", "#1A1C1E")

# Sidebar
COLOR_SIDEBAR_BG = ("#FFFFFF", "#0D1117")
COLOR_SIDEBAR_BORDER = ("#E1E4E8", "#3F444C")

# Cartes et Panneaux
COLOR_CARD_BG = ("#FFFFFF", "#23272D")
COLOR_CARD_BORDER = ("#E1E4E8", "#3F444C")

# Couleurs d'accentuation et d'état
COLOR_ACCENT = "#58A6FF"
COLOR_SUCCESS = ("#1A7F37", "#2ECC71")
COLOR_CANCEL = ("#CF222E", "#E63946")

# Textes
COLOR_TEXT_MAIN = ("#24292F", "#C9D1D9")
COLOR_TEXT_SUB = ("#57606A", "#8B949E")

# Widgets spécifiques
COLOR_ENTRY_BG = ("#F6F8FA", "#0D1117")
COLOR_INACTIVE = ("#E1E4E8", "#2D3748")

RANDOM_SEED = 42
DEFAULT_NUM_POINTS = 1000

BOUNDS = {
    "x_min": 0.0,
    "x_max": 200.0,
    "y_min": 0.0,
    "y_max": 200.0,
    "z_min": 0.0,
    "z_max": 200.0,
}

DX = 5.0
DY = 5.0
DZ = 5.0

DEFAULT_POINT_RADIUS = 5

EXPLORATION_DISTANCE = 100.0

MAX_DISTANCE = None
DISTANCE_MIN_WITH_STRUCTURE = 0
DISTANCE_MAX_WITH_STRUCTURE = 40
DISTANCE_WITH_VENTILATION_REFRIGERANT = 10
DISTANCE_HOT_AIR_LINES = 20
DISTANCE_WITH_HIGH_PRESSURE_HYDRAULIC_LINE = 70

HARNESS_DIAMETER = 40

LOG_LEVEL = "INFO"

CURVE_RADIUS = HARNESS_DIAMETER * 6

def as_dict():
    return {k: v for k, v in globals().items() if k.isupper()}
