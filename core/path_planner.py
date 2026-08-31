"""Recherche du chemin de départ dans l'espace libre.

Pourquoi remplacer la géodésique
--------------------------------
Le chemin initial était calculé par ``PolyData.geodesic``, qui suit la
**surface** du maillage. Trois problèmes :

1. Une maquette DMU est la fusion de centaines de pièces **disjointes**. La
   géodésique exige un chemin d'arêtes entre les deux sommets ; entre deux
   pièces séparées, il n'y en a aucun. Sur données réelles elle échouait donc
   systématiquement, et le code la remplaçait *sans le dire* par une ligne
   droite. Le « chemin géodésique » n'existait pas.
2. Même quand elle aboutit, elle colle à la surface : le chemin de départ est à
   distance nulle de la structure, en violation de toutes les règles de
   distance, et les agents passent leurs premières itérations à l'en décoller.
3. Elle contourne les ouvertures au lieu de les traverser, puisqu'elle ne
   connaît que la surface.

On cherche donc désormais dans l'**espace libre**, sur une grille de voxels où
le maillage n'est qu'un champ d'obstacles. Les pièces disjointes ne posent plus
de problème, et le chemin naît directement dans la bande de distance visée.

A* pondéré, dont la recherche gloutonne est un cas particulier
--------------------------------------------------------------
La recherche se fait avec ``f = g + w · h`` :

* ``w = 1``  : A* classique, chemin de coût minimal ;
* ``w`` grand : le terme ``g`` devient négligeable et l'on retrouve la
  **recherche gloutonne** (*greedy best-first search*), plus rapide mais qui
  ne minimise plus rien.

Un seul paramètre couvre donc les deux stratégies. Attention toutefois : le
coût ``g`` porte ici la préférence pour la bande de distance et la pénalité de
changement de direction. Une recherche purement gloutonne, qui ne classe que
sur ``h``, **ignore ces deux règles** — elle file au plus court vers l'arrivée,
au milieu du vide et en zigzag. C'est rapide, mais c'est précisément ce qu'on
cherche à éviter. Un ``w`` légèrement supérieur à 1 offre le bon compromis.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "PlannerSettings",
    "PlanResult",
    "STRATEGIES",
    "plan_route",
    "build_clearance_field",
]

#: Stratégies proposées dans l'interface. La valeur est le poids heuristique.
STRATEGIES: dict[str, dict] = {
    "optimal": {
        "weight": 1.0,
        "label_fr": "Meilleur chemin (A*)",
        "label_en": "Best path (A*)",
        "help_fr": "Cherche le meilleur compromis longueur / distance / virages. Le plus lent.",
        "help_en": "Finds the best length / clearance / turns trade-off. Slowest.",
    },
    "balanced": {
        "weight": 1.4,
        "label_fr": "Équilibré",
        "label_en": "Balanced",
        "help_fr": "Presque aussi bon, nettement plus rapide. Choix par défaut.",
        "help_en": "Nearly as good, much faster. The default.",
    },
    "greedy": {
        "weight": 12.0,
        "label_fr": "Rapide (glouton)",
        "label_en": "Fast (greedy)",
        "help_fr": (
            "Fonce vers l'arrivée. Très rapide, mais ne tient plus compte "
            "de la bande de distance ni des virages."
        ),
        "help_en": (
            "Heads straight for the goal. Very fast, but stops honouring the "
            "clearance band and the turn penalty."
        ),
    },
}


@dataclass
class PlannerSettings:
    """Réglages de la recherche."""

    #: Taille d'une cellule, en mm. ``None`` = déduite de la distance minimale
    #: exigée : une grille plus grossière que la marge visée ne sait pas
    #: produire un chemin qui longe la structure d'aussi près. Plus fin =
    #: plus précis et plus lent, le coût mémoire croissant au cube.
    voxel_mm: float | None = None
    #: Poids de l'heuristique. 1 = A*, grand = recherche gloutonne.
    heuristic_weight: float = 1.4
    #: Coût d'un changement de direction, exprimé en multiples d'un pas.
    #: C'est ce terme qui produit les longues lignes droites.
    turn_penalty: float = 1.5
    #: Surcoût pour s'éloigner au-delà de la distance maximale : le câble doit
    #: longer la structure, pas traverser le vide.
    band_penalty: float = 3.0
    #: Marge de sécurité ajoutée à la distance minimale, en fraction de
    #: cellule. Le champ de distance est mesuré entre centres de cellules :
    #: la surface réelle peut être plus proche que ce qu'il indique, d'au plus
    #: la demi-diagonale d'une cellule. Sans cette marge, le chemin trouvé
    #: frôle les pièces alors que la grille le croyait dégagé.
    clearance_safety: float = 0.87  # sqrt(3) / 2
    #: Garde-fou mémoire : au-delà, la résolution est automatiquement dégradée.
    max_cells: int = 6_000_000
    #: Nombre maximal de cellules dépilées avant abandon.
    max_expansions: int = 2_000_000
    #: Lissage par raccourcis en ligne de vue après la recherche.
    shortcut: bool = True

    def with_strategy(self, name: str) -> "PlannerSettings":
        """Copie des réglages avec le poids heuristique d'une stratégie."""
        weight = STRATEGIES.get(name, STRATEGIES["balanced"])["weight"]
        return PlannerSettings(**{**self.__dict__, "heuristic_weight": weight})


@dataclass
class PlanResult:
    """Issue d'une recherche."""

    success: bool
    points: np.ndarray | None = None
    message_fr: str = ""
    message_en: str = ""
    stats: dict = field(default_factory=dict)

    def message(self, lang: str = "FR") -> str:
        return self.message_en if str(lang).upper().startswith("EN") else self.message_fr


# ---------------------------------------------------------------------------
# Champ de distance
# ---------------------------------------------------------------------------


def auto_voxel_size(min_clearance_mm: float) -> float:
    """Résolution de grille adaptée à la distance minimale visée.

    Une cellule plus grande que la marge exigée empêche de longer la structure
    d'aussi près : la marge de sécurité liée à la discrétisation devient alors
    supérieure à la marge demandée, et le chemin s'éloigne. On prend donc une
    cellule de l'ordre de la marge, bornée pour rester calculable.
    """
    return float(np.clip(min_clearance_mm, 15.0, 60.0))


def build_clearance_field(mesh, start, goal, settings: PlannerSettings, min_clearance_mm: float = 10.0):
    """Construit la grille et le champ « distance au maillage » en mm.

    La grille couvre le maillage **et** les deux extrémités, avec une marge :
    le câble doit pouvoir contourner par l'extérieur si nécessaire.

    Returns:
        ``(distance, origin, pitch)`` où ``distance`` est un tableau 3D en mm.
    """
    from scipy import ndimage
    from trimesh.voxel import creation as voxel_creation

    pitch = float(settings.voxel_mm or 0.0)
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    if pitch <= 0.0:
        pitch = auto_voxel_size(min_clearance_mm)

    margin = 6.0 * pitch
    low = np.minimum(mesh.bounds[0], np.minimum(start, goal)) - margin
    high = np.maximum(mesh.bounds[1], np.maximum(start, goal)) + margin

    # Résolution dégradée plutôt qu'un plantage mémoire sur une grande cellule.
    while True:
        shape = np.ceil((high - low) / pitch).astype(int) + 1
        if int(np.prod(shape)) <= settings.max_cells or pitch > 500.0:
            break
        pitch *= 1.5

    occupied = np.zeros(tuple(shape), dtype=bool)
    voxels = voxel_creation.voxelize_subdivide(mesh, pitch=pitch)
    indices = np.floor((voxels.points - low) / pitch).astype(int)
    inside = np.all((indices >= 0) & (indices < shape), axis=1)
    indices = indices[inside]
    if len(indices):
        occupied[tuple(indices.T)] = True

    # Distance euclidienne exacte à la cellule occupée la plus proche.
    distance = ndimage.distance_transform_edt(~occupied, sampling=pitch)
    return distance.astype(np.float32), low, pitch


# ---------------------------------------------------------------------------
# Recherche
# ---------------------------------------------------------------------------


def _neighbour_offsets():
    """26 voisins, avec la longueur du pas associée (en cellules)."""
    offsets, lengths = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                offsets.append((dx, dy, dz))
                lengths.append(float(np.sqrt(dx * dx + dy * dy + dz * dz)))
    return np.array(offsets, dtype=np.int32), np.array(lengths, dtype=np.float32)


_OFFSETS, _STEP_LENGTHS = _neighbour_offsets()
#: Direction unitaire de chaque voisin, pour mesurer les changements de cap.
_DIRECTIONS = (_OFFSETS / _STEP_LENGTHS[:, None]).astype(np.float32)


def _snap_to_free(distance, index, min_clearance, radius_cells=6):
    """Ramène un point sur la cellule libre exploitable la plus proche.

    Un point de départ saisi à la main tombe volontiers dans la matière ou trop
    près d'une pièce ; on cherche alors la cellule valide la plus proche plutôt
    que de refuser d'emblée.
    """
    index = np.clip(index, 0, np.array(distance.shape) - 1)
    if distance[tuple(index)] >= min_clearance:
        return tuple(index), 0.0

    best, best_d2 = None, None
    for radius in range(1, radius_cells + 1):
        lo = np.maximum(index - radius, 0)
        hi = np.minimum(index + radius + 1, distance.shape)
        window = distance[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        candidates = np.argwhere(window >= min_clearance)
        if len(candidates) == 0:
            continue
        candidates = candidates + lo
        d2 = np.sum((candidates - index) ** 2, axis=1)
        pick = int(np.argmin(d2))
        if best is None or d2[pick] < best_d2:
            best, best_d2 = candidates[pick], d2[pick]
        break

    if best is None:
        return None, None
    return tuple(int(v) for v in best), float(np.sqrt(best_d2))


def plan_route(
    mesh,
    start,
    goal,
    min_clearance_mm: float,
    max_clearance_mm: float,
    settings: PlannerSettings | None = None,
    num_points: int | None = None,
) -> PlanResult:
    """Cherche un chemin de départ entre deux points, dans l'espace libre.

    Args:
        mesh: maillage de l'environnement (trimesh), obstacles.
        start, goal: extrémités, en mm.
        min_clearance_mm: en deçà, la cellule est interdite.
        max_clearance_mm: au-delà, la cellule est pénalisée — le câble doit
            longer la structure.
        settings: réglages de la recherche.
        num_points: rééchantillonnage final du chemin.

    Returns:
        Un :class:`PlanResult`. **En cas d'échec, aucun repli silencieux** :
        c'est ce comportement qui masquait l'inefficacité de la géodésique.
    """
    settings = settings or PlannerSettings()
    started_at = time.time()

    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    distance, origin, pitch = build_clearance_field(mesh, start, goal, settings, min_clearance_mm)
    shape = np.array(distance.shape)

    # Marge exigée pendant la recherche : la valeur demandée, augmentée de
    # l'incertitude propre à la grille. C'est ce qui évite qu'un chemin jugé
    # dégagé par la grille frôle en réalité une pièce.
    requested_clearance = float(min_clearance_mm)
    min_clearance = max(requested_clearance, 0.0) + settings.clearance_safety * pitch

    start_cell, start_shift = _snap_to_free(
        distance, np.floor((start - origin) / pitch).astype(int), min_clearance
    )
    goal_cell, goal_shift = _snap_to_free(
        distance, np.floor((goal - origin) / pitch).astype(int), min_clearance
    )

    if start_cell is None or goal_cell is None:
        which = "de départ" if start_cell is None else "d'arrivée"
        which_en = "start" if start_cell is None else "end"
        return PlanResult(
            success=False,
            message_fr=(
                f"Le point {which} est dans la matière, ou trop près d'une pièce pour "
                f"respecter la distance minimale de {min_clearance:.0f} mm.\n\n"
                "Déplacez-le, ou réduisez la distance minimale à l'étape 2."
            ),
            message_en=(
                f"The {which_en} point is inside material, or too close to a part to "
                f"honour the {min_clearance:.0f} mm minimum clearance.\n\n"
                "Move it, or lower the minimum clearance at step 2."
            ),
            stats={"voxel_mm": pitch, "grid": tuple(int(v) for v in shape)},
        )

    path_cells, expansions = _search(
        distance, start_cell, goal_cell, min_clearance, max_clearance_mm, pitch, settings
    )

    if path_cells is None:
        return PlanResult(
            success=False,
            message_fr=(
                "Aucun passage trouvé entre les deux points en respectant la distance "
                f"minimale de {min_clearance:.0f} mm.\n\n"
                "Essayez une distance minimale plus faible, ou une résolution plus fine "
                "(étape 3, réglages avancés) : un passage étroit peut échapper à une "
                f"grille de {pitch:.0f} mm."
            ),
            message_en=(
                "No corridor found between the two points while honouring the "
                f"{min_clearance:.0f} mm minimum clearance.\n\n"
                "Try a smaller minimum clearance, or a finer resolution (step 3, "
                f"advanced settings): a narrow passage can slip through a {pitch:.0f} mm grid."
            ),
            stats={
                "voxel_mm": pitch,
                "grid": tuple(int(v) for v in shape),
                "expansions": expansions,
                "seconds": time.time() - started_at,
            },
        )

    points = origin + np.asarray(path_cells, dtype=np.float64) * pitch
    # Les extrémités exactes demandées par l'utilisateur priment sur le centre
    # de la cellule où la recherche les a rattachées.
    points[0] = start
    points[-1] = goal

    if settings.shortcut and len(points) > 2:
        points = _shortcut(points, distance, origin, pitch, min_clearance)

    raw_points = points.copy()
    if num_points and num_points >= 2:
        points = _resample(points, int(num_points))

    length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    direct = float(np.linalg.norm(goal - start))

    # Contrôle contre le maillage réel, pas contre la grille : c'est la seule
    # mesure qui vaille, et elle est rapportée telle quelle plutôt que supposée.
    true_clearance = _measure_clearance(mesh, points)
    violations = int(np.count_nonzero(true_clearance < requested_clearance))

    return PlanResult(
        success=True,
        points=points.astype(np.float32),
        message_fr=(
            f"Chemin trouvé : {length:.0f} mm pour {direct:.0f} mm à vol d'oiseau "
            f"({len(raw_points)} points, grille {pitch:.0f} mm)."
        ),
        message_en=(
            f"Path found: {length:.0f} mm for a {direct:.0f} mm straight line "
            f"({len(raw_points)} points, {pitch:.0f} mm grid)."
        ),
        stats={
            "voxel_mm": pitch,
            "grid": tuple(int(v) for v in shape),
            "expansions": expansions,
            "seconds": time.time() - started_at,
            "length_mm": length,
            "direct_mm": direct,
            "detour_ratio": length / direct if direct > 1e-6 else 1.0,
            "n_points_raw": len(raw_points),
            "start_shift_cells": start_shift,
            "goal_shift_cells": goal_shift,
            "min_clearance_mm": float(true_clearance.min()) if len(true_clearance) else 0.0,
            "clearance_violations": violations,
        },
    )


def _search(distance, start_cell, goal_cell, min_clearance, max_clearance, pitch, settings):
    """A* pondéré sur la grille. Renvoie ``(cellules, nb_expansions)``."""
    shape = distance.shape
    n_cells = int(np.prod(shape))
    strides = np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)

    def flatten(cell):
        return int(cell[0] * strides[0] + cell[1] * strides[1] + cell[2] * strides[2])

    def unflatten(index):
        z = index % shape[2]
        y = (index // shape[2]) % shape[1]
        x = index // (shape[1] * shape[2])
        return x, y, z

    goal_arr = np.array(goal_cell, dtype=np.float64)
    weight = float(settings.heuristic_weight)
    turn_cost = float(settings.turn_penalty)
    band_penalty = float(settings.band_penalty)
    max_clearance = max(float(max_clearance), min_clearance + 1e-3)

    g_score = np.full(n_cells, np.inf, dtype=np.float32)
    came_from = np.full(n_cells, -1, dtype=np.int64)
    #: Direction d'arrivée dans chaque cellule, pour facturer les virages.
    arrival_dir = np.full(n_cells, -1, dtype=np.int8)
    closed = np.zeros(n_cells, dtype=bool)

    start_flat = flatten(start_cell)
    goal_flat = flatten(goal_cell)
    g_score[start_flat] = 0.0

    def heuristic(cell):
        return float(np.linalg.norm(np.array(cell, dtype=np.float64) - goal_arr))

    heap = [(weight * heuristic(start_cell), start_flat)]
    expansions = 0

    while heap:
        _, current = heapq.heappop(heap)
        if closed[current]:
            continue
        closed[current] = True
        expansions += 1

        if current == goal_flat:
            break
        if expansions > settings.max_expansions:
            return None, expansions

        cx, cy, cz = unflatten(current)
        current_dir = arrival_dir[current]
        current_g = g_score[current]

        for k, (dx, dy, dz) in enumerate(_OFFSETS):
            nx, ny, nz = cx + int(dx), cy + int(dy), cz + int(dz)
            if not (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                continue

            clearance = distance[nx, ny, nz]
            if clearance < min_clearance:
                continue  # trop près d'une pièce : cellule interdite

            neighbour = nx * strides[0] + ny * strides[1] + nz * strides[2]
            neighbour = int(neighbour)
            if closed[neighbour]:
                continue

            step = float(_STEP_LENGTHS[k])

            # Rester dans la bande de distance : au-delà du maximum, le câble
            # n'a plus rien à longer et devient impossible à fixer.
            if clearance > max_clearance:
                step *= 1.0 + band_penalty * min(
                    (clearance - max_clearance) / max_clearance, 2.0
                )

            # Changement de cap. Le coût réel dépendrait du chemin complet ;
            # on l'approche à partir de la direction d'arrivée mémorisée pour
            # la cellule courante, ce qui suffit à privilégier les longues
            # lignes droites sans faire exploser l'espace d'états.
            if turn_cost > 0.0 and current_dir >= 0:
                alignment = float(np.dot(_DIRECTIONS[current_dir], _DIRECTIONS[k]))
                step += turn_cost * (1.0 - alignment) * 0.5

            tentative = current_g + step
            if tentative < g_score[neighbour]:
                g_score[neighbour] = tentative
                came_from[neighbour] = current
                arrival_dir[neighbour] = k
                priority = tentative + weight * heuristic((nx, ny, nz))
                heapq.heappush(heap, (priority, neighbour))

    if came_from[goal_flat] == -1 and goal_flat != start_flat:
        return None, expansions

    cells = []
    node = goal_flat
    while node != -1:
        cells.append(unflatten(node))
        if node == start_flat:
            break
        node = int(came_from[node])
    cells.reverse()
    return cells, expansions


def _measure_clearance(mesh, points) -> np.ndarray:
    """Distance réelle de chaque point au maillage, en mm.

    La grille ne donne qu'une approximation ; on mesure ici la vraie distance
    pour pouvoir rendre compte honnêtement de la qualité du chemin.
    """
    try:
        from trimesh.proximity import ProximityQuery

        _, distances, _ = ProximityQuery(mesh).on_surface(np.asarray(points, dtype=np.float64))
        return np.asarray(distances, dtype=np.float64)
    except Exception:
        return np.zeros(0)


def _line_is_clear(a, b, distance, origin, pitch, min_clearance) -> bool:
    """Vrai si le segment ``a-b`` reste partout à la distance exigée."""
    steps = int(np.ceil(np.linalg.norm(b - a) / (pitch * 0.5))) + 1
    samples = a + (b - a) * np.linspace(0.0, 1.0, steps)[:, None]
    cells = np.floor((samples - origin) / pitch).astype(int)
    shape = np.array(distance.shape)
    if np.any(cells < 0) or np.any(cells >= shape):
        return False
    return bool(np.all(distance[cells[:, 0], cells[:, 1], cells[:, 2]] >= min_clearance))


def _shortcut(points, distance, origin, pitch, min_clearance):
    """Supprime les marches d'escalier de la grille par raccourcis directs.

    Une recherche sur grille ne produit que des directions à 45°, ce qui donne
    un tracé en escalier. On relie donc chaque point au point le plus lointain
    qu'il puisse atteindre en ligne droite sans violer la distance minimale.
    Le chemin y gagne de longues lignes droites — exactement ce que la règle de
    rectitude demande — et beaucoup moins de sommets.
    """
    kept = [0]
    i = 0
    n = len(points)
    while i < n - 1:
        best = i + 1
        for j in range(n - 1, i, -1):
            if _line_is_clear(points[i], points[j], distance, origin, pitch, min_clearance):
                best = j
                break
        kept.append(best)
        i = best
    return points[kept]


def _resample(points, num_points):
    """Rééchantillonne le chemin à pas constant."""
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = cumulative[-1]
    if total <= 1e-6:
        return np.tile(points[0], (num_points, 1))

    targets = np.linspace(0.0, total, num_points)
    out = np.empty((num_points, 3), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(targets, cumulative, points[:, axis])
    return out
