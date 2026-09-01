"""Termes de récompense adossés aux règles d'intégration.

Chaque fonction traduit **une** règle de :mod:`core.routing_rules` en signal
d'apprentissage, sur le même vocabulaire géométrique que le rapport de
conformité affiché à l'utilisateur. C'est volontaire : l'agent est ainsi
récompensé exactement sur ce que l'intégrateur va lire à l'écran.

Convention : toutes les fonctions renvoient un tableau ``(n,)`` aligné sur les
points de la trajectoire (extrémités comprises, valeur nulle pour elles quand
la grandeur n'a pas de sens à cet endroit), afin de pouvoir être combinées par
simple somme pondérée.
"""

from __future__ import annotations

import numpy as np

from core import geometry_metrics as gm

__all__ = [
    "clearance_reward",
    "bend_reward",
    "straightness_reward",
    "zigzag_penalty",
    "free_span_penalty",
    "fixation_coverage_reward",
    "detour_penalty",
    "zone_penalty",
    "combine",
]


def _to_full(interior_values: np.ndarray, n: int) -> np.ndarray:
    """Étale des valeurs définies sur les points intérieurs vers ``(n,)``."""
    out = np.zeros(n, dtype=np.float32)
    if n >= 3 and len(interior_values):
        out[1:-1] = interior_values
    return out


def clearance_reward(
    distances,
    required_min,
    max_mm: float,
    band_bonus: float = 100.0,
    too_close_penalty: float = 50.0,
    too_far_penalty: float = 5.0,
) -> np.ndarray:
    """Récompense la bonne distance au DMU, point par point.

    Trois régimes :

    * **dans la bande** ``[required_min, max_mm]`` : prime pleine, c'est la
      position visée (« longer la structure ») ;
    * **trop près** : pénalité proportionnelle au manque, ce qui donne un
      gradient qui pousse le câble à s'écarter au lieu d'un simple mur ;
    * **trop loin** : pénalité proportionnelle à l'excès, le câble se remet à
      longer la structure au lieu de partir dans le vide.

    ``required_min`` peut être un scalaire ou un tableau : c'est ce qui permet
    d'exiger 70 mm au-dessus d'une hydraulique haute pression et 10 mm ailleurs.
    """
    dist = np.asarray(distances, dtype=np.float32)
    required = np.broadcast_to(np.asarray(required_min, dtype=np.float32), dist.shape)

    reward = np.zeros_like(dist)
    in_band = (dist >= required) & (dist <= max_mm)
    too_close = dist < required
    too_far = dist > max_mm

    reward[in_band] = band_bonus
    safe_required = np.maximum(required, 1e-3)
    reward[too_close] = -too_close_penalty * (
        (required[too_close] - dist[too_close]) / safe_required[too_close]
    )
    reward[too_far] = -too_far_penalty * (dist[too_far] - max_mm)
    return reward


def bend_reward(
    points,
    min_bend_radius_mm: float,
    weight: float = 60.0,
    violation_penalty: float = 150.0,
) -> np.ndarray:
    """Récompense un cintrage large et sanctionne les cassures.

    S'appuie sur le rayon de cintrage **réalisable**
    (:func:`core.geometry_metrics.bend_radii`) et non sur le cosinus entre
    segments : le signal ne dépend donc plus de la densité de points, et une
    insertion de point ne change plus artificiellement la note du tronçon.

    La prime sature à deux fois le rayon admissible : au-delà, élargir encore
    le coude n'apporte rien et l'agent peut consacrer son effort au reste.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return np.zeros(n, dtype=np.float32)

    radii = gm.bend_radii(pts)
    limit = max(float(min_bend_radius_mm), 1e-3)

    ratio = np.clip(radii / limit, 0.0, 2.0)
    values = weight * ratio
    violation = radii < limit
    # Pénalité croissante à mesure que la cassure se resserre : un coude à
    # 0,9 x R_min est presque acceptable, un coude à 0,1 x R_min ne l'est pas.
    values[violation] -= violation_penalty * (1.0 - ratio[violation] / 1.0)
    return _to_full(values.astype(np.float32), n)


def straightness_reward(
    points,
    angle_tol_deg: float = 3.0,
    weight: float = 25.0,
    run_bonus: float = 15.0,
    run_scale_mm: float = 500.0,
) -> np.ndarray:
    """Récompense les longues portions rectilignes.

    Deux composantes :

    * ``weight`` : prime locale dès que le câble va tout droit ;
    * ``run_bonus`` : prime supplémentaire proportionnelle à la longueur de la
      ligne droite à laquelle le point appartient, saturée à ``run_scale_mm``.

    La seconde est décisive : sans elle, l'agent obtient la même note pour dix
    petits segments droits séparés par des coudes que pour une belle ligne
    droite continue. C'est elle qui traduit « rester droit le plus longtemps
    possible ».
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return np.zeros(n, dtype=np.float32)

    angles = gm.turning_angles(pts)
    is_straight = angles < np.radians(angle_tol_deg)

    seg = gm.segment_lengths(pts)
    carried = 0.5 * (seg[:-1] + seg[1:])

    # Longueur de la ligne droite contenant chaque point, calculée en deux
    # balayages (cumul avant puis arrière) plutôt qu'avec une boucle par point.
    forward = np.zeros(len(is_straight))
    running = 0.0
    for i, straight in enumerate(is_straight):
        running = running + carried[i] if straight else 0.0
        forward[i] = running

    backward = np.zeros(len(is_straight))
    running = 0.0
    for i in range(len(is_straight) - 1, -1, -1):
        running = running + carried[i] if is_straight[i] else 0.0
        backward[i] = running

    run_length = np.where(is_straight, forward + backward - carried, 0.0)

    values = np.where(is_straight, weight, 0.0)
    values = values + run_bonus * np.clip(run_length / max(run_scale_mm, 1e-3), 0.0, 1.0)
    return _to_full(values.astype(np.float32), n)


def zigzag_penalty(
    points,
    angle_tol_deg: float = 3.0,
    weight: float = 60.0,
    saturation_deg: float = 45.0,
) -> np.ndarray:
    """Sanctionne les inversions du sens de virage.

    Un zigzag n'est pas une question de courbure totale, et c'est pourquoi il
    lui faut un terme propre. Un arc de cercle régulier accumule beaucoup de
    courbure sans jamais osciller ; à l'inverse, une succession de petits
    virages alternés peut totaliser peu de courbure tout en donnant un câble
    visuellement inacceptable, et impossible à poser proprement. Ni
    :func:`bend_reward`, qui regarde le rayon local, ni
    :func:`straightness_reward`, qui récompense les portions droites, ne
    distinguent ces deux cas.

    La pénalité est proportionnelle à la gravité de l'inversion, saturée à
    ``saturation_deg`` pour qu'un demi-tour isolé n'écrase pas tout le reste du
    signal d'apprentissage.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return np.zeros(n, dtype=np.float32)

    severity = np.degrees(gm.zigzag_severity(pts, angle_tol_deg))
    scale = max(float(saturation_deg), 1e-3)
    penalty = -weight * np.clip(severity / scale, 0.0, 1.0)
    return _to_full(penalty.astype(np.float32), n)


def free_span_penalty(
    points,
    distances,
    max_reach_mm: float,
    span_limit_mm: float,
    weight: float = 80.0,
) -> np.ndarray:
    """Sanctionne les traversées de volumes vides.

    C'est la traduction directe de « ne pas passer à travers un carré vide » :
    la pénalité ne porte pas sur le fait d'être loin d'une pièce (un point isolé
    un peu éloigné n'est pas grave) mais sur la **longueur** de la portion sans
    aucun appui possible. Elle ne mord qu'au-delà de ``span_limit_mm``, la
    distance que le câble peut franchir sans support.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    out = np.zeros(n, dtype=np.float32)
    if n < 2:
        return out

    limit = max(float(span_limit_mm), 1e-3)
    for start, end, length in gm.free_spans(pts, distances, max_reach_mm):
        if length <= limit:
            continue
        severity = min((length - limit) / limit, 3.0)
        out[start:end + 1] -= weight * severity
    return out


def fixation_coverage_reward(
    points,
    clamp_arc_positions,
    pitch_mm: float = 250.0,
    weight: float = 30.0,
    gap_penalty: float = 60.0,
) -> np.ndarray:
    """Récompense une trajectoire correctement tenue par des fixations.

    Chaque point reçoit une note fonction de sa distance au support le plus
    proche : pleine s'il est tenu à moins d'un demi-pas, négative dès que la
    portée dépasse le pas réglementaire. Les deux extrémités comptent toujours
    comme des points tenus (départ et arrivée sur équipement).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return np.zeros(n, dtype=np.float32)

    arc = gm.arc_lengths(pts)
    supports = [0.0, float(arc[-1])]
    if clamp_arc_positions is not None:
        supports.extend(float(s) for s in np.asarray(clamp_arc_positions, dtype=float).ravel())
    supports_arr = np.unique(np.asarray(supports, dtype=np.float64))

    # Distance curviligne au support le plus proche.
    gaps = np.abs(arc[:, None] - supports_arr[None, :]).min(axis=1)

    half_pitch = max(float(pitch_mm) / 2.0, 1e-3)
    normalized = gaps / half_pitch
    values = weight * (1.0 - np.clip(normalized, 0.0, 1.0))
    over = normalized > 1.0
    values[over] -= gap_penalty * np.clip(normalized[over] - 1.0, 0.0, 3.0)
    return values.astype(np.float32)


def detour_penalty(
    points,
    reference_length_mm: float,
    weight: float = 40.0,
    tolerance: float = 1.15,
) -> np.ndarray:
    """Sanctionne les détours par rapport au trajet de référence.

    Sans ce terme, rien ne retient un agent d'allonger indéfiniment la route :
    contourner largement un obstacle satisfait toutes les autres règles, et le
    seul frein était l'attraction du laplacien vers le milieu des voisins.

    La référence est la longueur du chemin géodésique initial. On tolère un
    dépassement (``tolerance``) car s'écarter pour respecter les distances
    rallonge nécessairement le parcours ; au-delà, la pénalité croît avec
    l'excès. La pénalité est répartie uniformément : le détour est une
    propriété de la route entière, pas d'un point en particulier.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2 or reference_length_mm <= 1e-6:
        return np.zeros(n, dtype=np.float32)

    ratio = gm.path_length(pts) / (reference_length_mm * max(tolerance, 1e-3))
    if ratio <= 1.0:
        return np.zeros(n, dtype=np.float32)

    return np.full(n, -weight * min(ratio - 1.0, 3.0), dtype=np.float32)


def zone_penalty(
    points,
    start,
    goal,
    factor: float = 1.25,
    weight: float = 90.0,
    saturation: float = 0.5,
) -> np.ndarray:
    """Sanctionne les points sortis du couloir de cheminement.

    Rien ne retenait un point au-delà de l'arrivée. ``detour_penalty`` juge la
    **longueur totale** et répartit sa sanction uniformément : un point parti
    trois mètres trop loin y contribue à peine plus que ses voisins restés en
    place, et ne reçoit donc aucun signal qui lui dise de revenir. D'où des
    trajectoires qui dépassent la zone de destination et y restent.

    Le couloir est le même que celui qui filtre les peignes : l'ellipsoïde de
    foyers A et B, où le détour imposé — ``|PA| + |PB|`` — ne dépasse pas
    ``factor`` fois la distance directe. La sanction est **par point** : elle
    croît avec le dépassement, sature au-delà de ``saturation`` pour ne pas
    écraser toutes les autres règles, et vaut exactement zéro partout dans le
    couloir — un tracé conforme ne doit rien payer.

    A et B eux-mêmes sont sur l'ellipse de rapport 1 : ils ne sont jamais
    sanctionnés, quel que soit le facteur retenu.
    """
    pts = np.asarray(points, dtype=np.float64)
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(goal, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float32)

    direct = float(np.linalg.norm(b - a))
    if direct < 1e-9:
        # A et B confondus : le couloir n'a aucun sens, et prétendre le
        # contraire sanctionnerait la trajectoire entière.
        return np.zeros(len(pts), dtype=np.float32)

    ratio = (np.linalg.norm(pts - a, axis=1) + np.linalg.norm(pts - b, axis=1)) / direct
    excess = np.maximum(ratio - float(factor), 0.0)
    return (-weight * np.minimum(excess / max(saturation, 1e-6), 1.0)).astype(np.float32)


def combine(**terms) -> tuple[np.ndarray, dict]:
    """Somme des termes et détail par contribution.

    Returns:
        ``(total, details)`` où ``details`` donne la moyenne de chaque terme —
        c'est ce dictionnaire qui alimente la ventilation de la récompense
        affichée dans l'interface.
    """
    arrays = [np.asarray(v, dtype=np.float32) for v in terms.values() if v is not None]
    if not arrays:
        return np.zeros(0, dtype=np.float32), {}

    size = max(len(a) for a in arrays)
    total = np.zeros(size, dtype=np.float32)
    details: dict[str, float] = {}
    for name, value in terms.items():
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        if len(arr) != size:
            padded = np.zeros(size, dtype=np.float32)
            padded[: len(arr)] = arr[:size]
            arr = padded
        total += arr
        details[name] = float(arr.mean())
    return total, details
