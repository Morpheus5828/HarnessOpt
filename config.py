"""Constantes historiques du projet.

Les chemins de travail sont désormais définis par :mod:`core.paths`, qui choisit
un emplacement valable sur toutes les plateformes. Ce module les réexporte pour
les scripts existants et, contrairement à sa version précédente, ne crée plus
aucun dossier au moment de l'import : construire un dossier de cache parce
qu'un module importe une constante de couleur est un effet de bord indésirable.

Les valeurs de distance ci-dessous restent la référence des règles
d'intégration ; elles alimentent ``DEFAULT_FAMILY_CLEARANCE`` dans
:mod:`core.routing_rules`.
"""

from pathlib import Path

from core.paths import BASE_CACHE, COLOR_DIR, FUSION_DIR, STL_DIR

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "resources" / "model" / "best_mesh_classifier.joblib"

INPUT_DIR = BASE_DIR / "resources"
OUTPUT_DIR = BASE_DIR / "output"

STL_FOLDER = STL_DIR
COLOR_PATH = COLOR_DIR / "dmu_color_parts.xlsx"

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

# Rayon de cintrage admissible du toron. Cette constante n'était utilisée nulle
# part ; elle est désormais la valeur par défaut de la règle « rayon de cintrage »
# (voir HarnessSpec.bend_radius_factor dans core/routing_rules.py).
CURVE_RADIUS = HARNESS_DIAMETER * 6

def as_dict():
    return {k: v for k, v in globals().items() if k.isupper()}
