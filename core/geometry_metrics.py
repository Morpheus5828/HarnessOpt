"""Mesures géométriques d'une trajectoire de harnais.

Toutes les fonctions travaillent sur un tableau ``(n, 3)`` de points en
millimètres et n'utilisent que numpy : elles sont donc utilisables aussi bien
dans la boucle d'apprentissage (des milliers d'appels par seconde) que dans
l'interface pour afficher un indicateur.

Le point important est le **rayon de courbure**. L'ancienne version du projet
jugeait le lissage à partir du cosinus de l'angle entre deux segments
consécutifs. Ce critère dépend de l'espacement des points : comme le raffinement
adaptatif insère et supprime des points en cours de route, un même cosinus
correspondait à des rayons physiques très différents. On mesure donc ici la
**courbure de Menger**, qui est le rayon du cercle passant par trois points
consécutifs : c'est une grandeur physique en millimètres, comparable
directement au rayon de cintrage admissible du toron (typiquement 6 x diamètre).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "segment_lengths",
    "path_length",
    "arc_lengths",
    "turning_angles",
    "curvature_radii",
    "bend_radii",
    "min_bend_radius",
    "min_curvature_radius",
    "straightness",
    "turn_binormals",
    "zigzag_severity",
    "zigzag_metrics",
    "free_spans",
    "longest_free_span",
    "support_gaps",
    "resample_by_arclength",
    "boundary_points",
    "edge_distances",
]

#: En deçà de cette longueur (mm) un segment est considéré dégénéré et ignoré
#: dans les calculs d'angle : deux points confondus ne définissent pas de
#: direction.
_EPS_LEN = 1e-6


def segment_lengths(points: np.ndarray) -> np.ndarray:
    """Longueur de chaque segment, tableau de taille ``n - 1``."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.zeros(0)
    return np.linalg.norm(np.diff(pts, axis=0), axis=1)


def path_length(points: np.ndarray) -> float:
    """Longueur développée totale de la trajectoire, en mm."""
    return float(segment_lengths(points).sum())


def arc_lengths(points: np.ndarray) -> np.ndarray:
    """Abscisse curviligne de chaque point (0 au départ), taille ``n``."""
    lengths = segment_lengths(points)
    return np.concatenate([[0.0], np.cumsum(lengths)])


def turning_angles(points: np.ndarray) -> np.ndarray:
    """Angle de changement de direction à chaque point intérieur, en radians.

    0 rad = le câble continue tout droit ; pi rad = demi-tour complet.
    Tableau de taille ``n - 2``.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return np.zeros(0)

    v_in = pts[1:-1] - pts[:-2]
    v_out = pts[2:] - pts[1:-1]
    n_in = np.linalg.norm(v_in, axis=1)
    n_out = np.linalg.norm(v_out, axis=1)

    valid = (n_in > _EPS_LEN) & (n_out > _EPS_LEN)
    cos_a = np.ones(len(v_in))
    cos_a[valid] = np.einsum("ij,ij->i", v_in[valid], v_out[valid]) / (n_in[valid] * n_out[valid])
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


def curvature_radii(points: np.ndarray) -> np.ndarray:
    """Rayon de courbure en mm à chaque point intérieur (courbure de Menger).

    Le rayon est celui du cercle circonscrit au triangle formé par trois points
    consécutifs :  ``R = (a * b * c) / (4 * Aire)``. Les portions rectilignes
    donnent ``+inf`` (aire nulle), ce qui est le comportement attendu : une
    ligne droite a un rayon de courbure infini.

    Tableau de taille ``n - 2``.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return np.zeros(0)

    p0, p1, p2 = pts[:-2], pts[1:-1], pts[2:]
    a = np.linalg.norm(p2 - p1, axis=1)
    b = np.linalg.norm(p2 - p0, axis=1)
    c = np.linalg.norm(p1 - p0, axis=1)

    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

    radii = np.full(len(p0), np.inf)
    curved = area > _EPS_LEN
    radii[curved] = (a[curved] * b[curved] * c[curved]) / (4.0 * area[curved])

    # Points confondus : aucun rayon exploitable, on ne les compte pas comme
    # une cassure (ils seront éliminés par le rééchantillonnage).
    degenerate = (a <= _EPS_LEN) | (c <= _EPS_LEN)
    radii[degenerate] = np.inf
    return radii


def min_curvature_radius(points: np.ndarray) -> float:
    """Plus petit rayon de courbure de la trajectoire, en mm.

    Renvoie ``inf`` pour une trajectoire parfaitement droite ou trop courte.
    """
    radii = curvature_radii(points)
    return float(radii.min()) if len(radii) else float("inf")


def bend_radii(points: np.ndarray) -> np.ndarray:
    """Rayon de cintrage réalisable à chaque point intérieur, en mm.

    C'est la grandeur qui décide si un toron réel peut suivre la trajectoire.
    Un sommet n'est pas un angle vif « physique » : il sera arrondi par un
    congé. Pour arrondir un changement de direction ``theta`` avec un rayon
    ``R``, il faut une longueur de tangente ``T = R * tan(theta / 2)`` de part
    et d'autre du sommet. En s'autorisant au plus la moitié du segment adjacent
    le plus court (pour que deux congés voisins ne se chevauchent pas) :

        ``R_realisable = (min(L_entrant, L_sortant) / 2) / tan(theta / 2)``

    Interprétation :

    * portion droite (``theta`` -> 0) : rayon infini, aucune contrainte ;
    * courbe finement échantillonnée : la formule converge vers le rayon de
      courbure réel, donc vers la courbure de Menger ;
    * coude franc entre deux longues lignes droites : on obtient le plus grand
      congé que la place disponible autorise, ce qui est exactement la question
      que se pose l'intégrateur.

    Tableau de taille ``n - 2``.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return np.zeros(0)

    theta = turning_angles(pts)
    seg = segment_lengths(pts)
    available = 0.5 * np.minimum(seg[:-1], seg[1:])

    radii = np.full(len(theta), np.inf)
    bending = theta > 1e-9
    radii[bending] = available[bending] / np.tan(np.clip(theta[bending], 1e-9, np.pi - 1e-9) / 2.0)
    return np.maximum(radii, 0.0)


def min_bend_radius(points: np.ndarray) -> float:
    """Plus petit rayon de cintrage réalisable de la trajectoire, en mm.

    C'est l'indicateur à comparer au rayon de cintrage admissible du toron
    (souvent 6 x le diamètre). ``inf`` = trajectoire parfaitement droite.
    """
    radii = bend_radii(points)
    return float(radii.min()) if len(radii) else float("inf")


def _count_bends(pts: np.ndarray, is_straight: np.ndarray) -> int:
    """Compte les coudes distincts d'une trajectoire.

    Des points courbés qui se suivent forment un seul coude tant qu'ils
    tournent dans le même sens : un arc de cercle régulier est bien un coude
    unique. Dès que le sens de rotation s'inverse (zigzag), on compte un
    nouveau coude — c'est ce qui distingue une belle courbe d'une oscillation.
    """
    if len(pts) < 3:
        return 0

    binormals = np.cross(pts[1:-1] - pts[:-2], pts[2:] - pts[1:-1])
    norms = np.linalg.norm(binormals, axis=1)
    safe = norms > _EPS_LEN
    unit = np.zeros_like(binormals)
    unit[safe] = binormals[safe] / norms[safe, None]

    bends = 0
    previous = None
    for i, straight in enumerate(is_straight):
        if straight:
            previous = None
            continue
        if previous is None or float(np.dot(unit[i], previous)) <= 0.0:
            bends += 1
        previous = unit[i]
    return bends


def straightness(points: np.ndarray, angle_tol_deg: float = 3.0) -> dict:
    """Analyse les portions rectilignes de la trajectoire.

    Un point intérieur est dit « droit » si le câble y tourne de moins de
    ``angle_tol_deg``. On mesure ensuite les suites maximales de points droits.

    Returns:
        dict avec ``straight_ratio`` (part de la longueur parcourue en ligne
        droite, 0..1), ``longest_run_mm`` (plus long tronçon droit),
        ``n_bends`` (nombre de coudes distincts : un arc régulier compte pour
        un seul coude, un zigzag en compte un par alternance) et
        ``total_turning_deg`` (somme des changements de direction, indicateur
        global de tortuosité : plus il est faible, plus la route est propre).
    """
    pts = np.asarray(points, dtype=np.float64)
    total = path_length(pts)
    empty = {
        "straight_ratio": 1.0,
        "longest_run_mm": total,
        "n_bends": 0,
        "total_turning_deg": 0.0,
    }
    if len(pts) < 3 or total <= _EPS_LEN:
        return empty

    angles = turning_angles(pts)
    tol = np.radians(angle_tol_deg)
    is_straight = angles < tol

    seg = segment_lengths(pts)
    # Longueur « portée » par un point intérieur i : la moitié du segment
    # entrant et la moitié du segment sortant.
    carried = 0.5 * (seg[:-1] + seg[1:])
    # Les deux demi-segments d'extrémité ne sont portés par aucun point
    # intérieur : on les rattache à leur voisin pour que la somme des
    # longueurs portées vaille exactement la longueur totale.
    carried = carried.copy()
    carried[0] += 0.5 * seg[0]
    carried[-1] += 0.5 * seg[-1]
    straight_len = float(carried[is_straight].sum())

    # Plus longue suite de points droits, mesurée en longueur développée.
    longest = 0.0
    current = 0.0
    for straight, length in zip(is_straight, carried):
        if straight:
            current += float(length)
            longest = max(longest, current)
        else:
            current = 0.0

    bends = _count_bends(pts, is_straight)

    return {
        "straight_ratio": float(np.clip(straight_len / total, 0.0, 1.0)),
        "longest_run_mm": longest,
        "n_bends": bends,
        "total_turning_deg": float(np.degrees(angles.sum())),
    }


def turn_binormals(points: np.ndarray) -> np.ndarray:
    """Binormale unitaire à chaque point intérieur. Tableau ``(n - 2, 3)``.

    Elle donne le **sens** du virage, là où :func:`turning_angles` n'en donne
    que l'amplitude. Deux virages de même amplitude dont les binormales sont
    opposées tournent en sens contraire : c'est la définition géométrique d'un
    zigzag. Les points dégénérés (segments confondus, virage parfaitement
    droit) reçoivent une binormale nulle.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return np.zeros((0, 3))

    binormals = np.cross(pts[1:-1] - pts[:-2], pts[2:] - pts[1:-1])
    norms = np.linalg.norm(binormals, axis=1)
    unit = np.zeros_like(binormals)
    safe = norms > _EPS_LEN
    unit[safe] = binormals[safe] / norms[safe, None]
    return unit


def zigzag_severity(points: np.ndarray, angle_tol_deg: float = 3.0) -> np.ndarray:
    """Gravité de l'inversion de sens à chaque point intérieur, en radians.

    Une valeur nulle signifie « pas d'inversion ici ». Une valeur positive est
    l'amplitude du **plus petit** des deux virages consécutifs qui s'opposent.

    Prendre le plus petit des deux est délibéré : une oscillation minuscule
    entre deux grands virages reste une petite faute de tracé, alors que deux
    grands virages opposés forment un vrai zigzag. Retenir le plus grand, ou
    leur somme, ferait payer au câble le prix d'un zigzag franc pour un simple
    frémissement numérique.

    Tableau de taille ``n - 2``, aligné sur :func:`turning_angles`.
    """
    pts = np.asarray(points, dtype=np.float64)
    angles = turning_angles(pts)
    severity = np.zeros(len(angles))
    if len(angles) < 2:
        return severity

    unit = turn_binormals(pts)
    tol = np.radians(float(angle_tol_deg))
    significant = angles > tol

    # Produit scalaire entre binormales consécutives : négatif = sens inversé.
    dots = np.einsum("ij,ij->i", unit[:-1], unit[1:])
    reversed_turn = (dots < 0.0) & significant[:-1] & significant[1:]

    # La faute est imputée aux deux virages concernés : l'agent doit pouvoir
    # corriger en agissant sur l'un ou l'autre.
    paired = np.minimum(angles[:-1], angles[1:])
    severity[:-1] = np.where(reversed_turn, paired, severity[:-1])
    severity[1:] = np.maximum(severity[1:], np.where(reversed_turn, paired, 0.0))
    return severity


def zigzag_metrics(points: np.ndarray, angle_tol_deg: float = 3.0) -> dict:
    """Indicateurs d'oscillation d'un tracé.

    * ``n_zigzags`` : nombre d'inversions de sens de virage ;
    * ``zigzag_deg`` : somme des amplitudes concernées, en degrés ;
    * ``worst_zigzag_deg`` : la pire inversion, en degrés.
    """
    pts = np.asarray(points, dtype=np.float64)
    angles = turning_angles(pts)
    if len(angles) < 2:
        return {"n_zigzags": 0, "zigzag_deg": 0.0, "worst_zigzag_deg": 0.0}

    unit = turn_binormals(pts)
    tol = np.radians(float(angle_tol_deg))
    significant = angles > tol
    dots = np.einsum("ij,ij->i", unit[:-1], unit[1:])
    reversed_turn = (dots < 0.0) & significant[:-1] & significant[1:]

    paired = np.minimum(angles[:-1], angles[1:])[reversed_turn]
    return {
        "n_zigzags": int(reversed_turn.sum()),
        "zigzag_deg": float(np.degrees(paired.sum())) if len(paired) else 0.0,
        "worst_zigzag_deg": float(np.degrees(paired.max())) if len(paired) else 0.0,
    }


def free_spans(points: np.ndarray, distances: np.ndarray, max_reach_mm: float) -> list[tuple[int, int, float]]:
    """Repère les traversées « dans le vide ».

    Un point dont la structure la plus proche est à plus de ``max_reach_mm`` ne
    peut pas être soutenu : le câble y traverse un volume libre. C'est très
    exactement le cas « passer à travers un carré vide » qu'il faut éviter.

    Args:
        points: trajectoire ``(n, 3)``.
        distances: distance au maillage de chaque point, taille ``n``.
        max_reach_mm: portée au-delà de laquelle plus rien n'est accrochable.

    Returns:
        Liste de ``(index_debut, index_fin, longueur_mm)``, une entrée par
        traversée à vide, bornes incluses.
    """
    pts = np.asarray(points, dtype=np.float64)
    dist = np.asarray(distances, dtype=np.float64)
    if len(pts) < 2 or len(dist) != len(pts):
        return []

    in_void = dist > max_reach_mm
    if not in_void.any():
        return []

    cum = arc_lengths(pts)
    last = len(pts) - 1

    def span_length(first_idx: int, last_idx: int) -> float:
        """Longueur réellement non soutenue autour des points concernés.

        On étend la mesure jusqu'au milieu des segments voisins : un unique
        point perdu dans le vide correspond bien à une portion de câble sans
        appui, pas à une longueur nulle.
        """
        start_s = cum[first_idx]
        if first_idx > 0:
            start_s = 0.5 * (cum[first_idx - 1] + cum[first_idx])
        end_s = cum[last_idx]
        if last_idx < last:
            end_s = 0.5 * (cum[last_idx] + cum[last_idx + 1])
        return float(end_s - start_s)

    spans: list[tuple[int, int, float]] = []
    start: int | None = None
    for i, void in enumerate(in_void):
        if void and start is None:
            start = i
        elif not void and start is not None:
            spans.append((start, i - 1, span_length(start, i - 1)))
            start = None
    if start is not None:
        spans.append((start, last, span_length(start, last)))
    return spans


def longest_free_span(points: np.ndarray, distances: np.ndarray, max_reach_mm: float) -> float:
    """Longueur de la plus longue traversée à vide, en mm (0 s'il n'y en a pas)."""
    spans = free_spans(points, distances, max_reach_mm)
    return max((length for _, _, length in spans), default=0.0)


def support_gaps(points: np.ndarray, support_indices) -> np.ndarray:
    """Écarts, en mm de longueur développée, entre points de fixation.

    Les deux extrémités de la trajectoire comptent toujours comme tenues : un
    harnais part et arrive sur un équipement.

    Returns:
        Tableau des écarts successifs (taille = nombre de supports + 1 au plus).
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.zeros(0)

    cum = arc_lengths(pts)
    n = len(pts)
    idx = {0, n - 1}
    for raw in np.asarray(list(support_indices), dtype=float).ravel():
        i = int(round(raw))
        if 0 <= i < n:
            idx.add(i)

    ordered = np.array(sorted(idx), dtype=int)
    return np.diff(cum[ordered])


def resample_by_arclength(points: np.ndarray, step_mm: float) -> np.ndarray:
    """Rééchantillonne la trajectoire à pas constant.

    Utile pour comparer deux routes sur la même base, et pour produire une
    polyligne d'export régulière.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2 or step_mm <= 0:
        return pts.astype(np.float32)

    cum = arc_lengths(pts)
    total = cum[-1]
    if total <= _EPS_LEN:
        return pts.astype(np.float32)

    n_out = max(2, int(round(total / step_mm)) + 1)
    targets = np.linspace(0.0, total, n_out)
    out = np.empty((n_out, 3), dtype=np.float32)
    for axis in range(3):
        out[:, axis] = np.interp(targets, cum, pts[:, axis])
    return out


# ----------------------------------------------------------------------
# Bords libres de la structure
# ----------------------------------------------------------------------

def boundary_points(mesh, samples_per_edge: int = 3) -> np.ndarray:
    """Points échantillonnés sur les **arêtes libres** du maillage.

    Une arête libre est une arête portée par une seule face : c'est le bord de
    la tôle, là où la matière s'arrête. Un faisceau qui longe un tel bord frotte
    sur un chant, et surtout ne peut pas y être tenu — il n'y a plus de matière
    pour recevoir une fixation.

    On échantillonne le long de chaque arête plutôt que d'en garder les seuls
    sommets : sur un maillage grossier, deux sommets de bord peuvent être
    distants de plusieurs centimètres, et un tracé passant entre les deux
    paraîtrait à bonne distance alors qu'il rase le bord.
    """
    faces = np.asarray(getattr(mesh, "faces", []), dtype=np.int64)
    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
    if len(faces) == 0 or len(vertices) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    free = unique[counts == 1]
    if len(free) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    a, b = vertices[free[:, 0]], vertices[free[:, 1]]
    steps = np.linspace(0.0, 1.0, max(2, int(samples_per_edge)))
    return np.concatenate([a + t * (b - a) for t in steps], axis=0)


def edge_distances(points, boundary) -> np.ndarray:
    """Distance de chaque point au bord libre le plus proche, en mm.

    Renvoie ``+inf`` partout si le maillage n'a aucun bord libre : une pièce
    fermée n'impose aucune contrainte de bord, et rendre zéro condamnerait
    le tracé entier.
    """
    pts = np.asarray(points, dtype=np.float64)
    ref = np.asarray(boundary, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(ref) == 0:
        return np.full(len(pts), np.inf, dtype=np.float64)

    try:
        from scipy.spatial import cKDTree

        return cKDTree(ref).query(pts)[0]
    except ImportError:
        # Repli sans scipy : exact, simplement plus coûteux.
        return np.array([np.linalg.norm(ref - p, axis=1).min() for p in pts])
