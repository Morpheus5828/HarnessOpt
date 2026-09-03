"""Ramener une trajectoire dans le domaine admissible, plutôt que la punir.

Les règles rédhibitoires — pénétration dans la structure, distance minimale,
rayon de cintrage — n'étaient portées que par la récompense. Or une grosse
pénalité n'est pas une garantie : elle rend une trajectoire coûteuse, pas
impossible. Un agent peut parfaitement converger vers un optimum qui viole une
contrainte critique si le gain ailleurs le compense, et rien dans la mécanique
d'apprentissage ne l'en empêche.

Le geste correctif retenu ici est la **projection**, pas le rejet. Rejeter une
trajectoire fait perdre l'itération et renvoie l'agent d'où il vient ;
projeter la ramène dans le domaine admissible, et l'agent continue d'optimiser
depuis un point valide. C'est le geste qui marche déjà ailleurs dans ce code :
``snap_comb_passages`` replace le câble dans les encoches, ``offset_from_surface``
le décolle de la surface. On le généralise.

Deux principes gouvernent tout le module :

* **on ne déplace jamais un point gelé.** Extrémités, encoches de peigne,
  points posés à la main : ce sont des décisions, pas des variables. Une
  projection qui les bougerait défairait ce que l'utilisateur ou la maquette
  imposent ;
* **on mesure ce qu'on a obtenu.** Une contrainte peut rester inatteignable —
  un passage plus étroit que deux fois la distance minimale, par exemple. Le
  rapport dit alors ce qui reste violé, plutôt que d'annoncer un domaine qu'on
  n'a pas atteint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core import geometry_metrics as gm

__all__ = [
    "ProjectionReport",
    "project_clearance",
    "project_bend_radius",
    "project_progress",
    "project_anchors",
    "prune_redundant_points",
    "remove_backtracking",
    "forward_only",
    "arc_positions",
    "project",
    "DEFAULT_CLEARANCE_PASSES",
    "DEFAULT_BEND_PASSES",
    "DEFAULT_PROGRESS_MARGIN_MM",
    "DEFAULT_ANCHOR_TOLERANCE_MM",
]

#: Passes de projection de distance. Déplacer un point lui donne un nouveau
#: plus proche voisin sur le maillage : une passe unique laisse des résidus.
DEFAULT_CLEARANCE_PASSES = 3

#: Passes de relâchement des coudes. Assouplir un sommet change l'angle de ses
#: voisins, donc leur rayon réalisable.
DEFAULT_BEND_PASSES = 12

#: Fraction du déplacement laplacien appliquée à chaque passe. Trop grand, le
#: tracé oscille ; trop petit, il ne converge pas dans le budget de passes.
BEND_RELAXATION = 0.35

#: Rayon d'une contrainte posée à la main, en mm. Le câble doit passer *à
#: portée* du point posé, pas exactement dessus : un point exigé au millimètre
#: fige un sommet que les agents ne peuvent plus lisser, et le tracé se plie
#: autour au lieu de s'améliorer.
DEFAULT_ANCHOR_TOLERANCE_MM = 30.0

#: Avancement minimal, en mm, d'un point au suivant le long du chemin de
#: référence. Strictement positif : deux points au même avancement forment un
#: repli plat, aussi peu posable qu'un vrai retour en arrière.
DEFAULT_PROGRESS_MARGIN_MM = 1.0


@dataclass
class ProjectionReport:
    """Ce que la projection a corrigé, et ce qui résiste."""

    n_clearance_moved: int = 0
    n_bend_moved: int = 0
    n_progress_moved: int = 0
    n_anchor_moved: int = 0
    #: Écart maximal restant entre une ancre posée à la main et le câble, en mm.
    worst_anchor_mm: float = 0.0
    #: Distance minimale au DMU après projection, en mm (``inf`` si non mesurée).
    min_clearance_mm: float = float("inf")
    #: Rayon de cintrage réalisable minimal après projection, en mm.
    min_bend_radius_mm: float = float("inf")
    #: Points restant sous la distance exigée, malgré la projection.
    n_clearance_left: int = 0
    #: Coudes restant plus serrés que le rayon minimal.
    n_bend_left: int = 0
    #: Reculs restants le long du chemin de référence.
    n_progress_left: int = 0

    @property
    def n_moved(self) -> int:
        return (self.n_clearance_moved + self.n_bend_moved
                + self.n_progress_moved + self.n_anchor_moved)

    @property
    def feasible(self) -> bool:
        """La projection a-t-elle abouti à un tracé admissible ?"""
        return (self.n_clearance_left == 0 and self.n_bend_left == 0
                and self.n_progress_left == 0)


def _movable(n: int, frozen) -> np.ndarray:
    """Masque des points que la projection s'autorise à déplacer.

    Les extrémités sont toujours exclues : elles appartiennent aux équipements
    et ne se négocient pas plus que les encoches.
    """
    mask = np.zeros(n, dtype=bool)
    if n > 2:
        mask[1:-1] = True
    for index in (frozen or ()):
        if 0 <= int(index) < n:
            mask[int(index)] = False
    return mask


def project_clearance(points, mesh, required_mm, frozen=(),
                      passes: int = DEFAULT_CLEARANCE_PASSES, query=None):
    """Repousse les points trop proches — ou à l'intérieur — de la structure.

    Chaque point fautif est projeté sur la face la plus proche puis repoussé le
    long de sa normale jusqu'à la distance exigée. Un point à l'intérieur de la
    matière est traité comme les autres : la normale sortante l'en fait sortir,
    ce qui règle la pénétration et la distance d'un même geste.

    Args:
        points: trajectoire ``(n, 3)``, modifiée sur place.
        mesh: maillage trimesh de l'environnement.
        required_mm: distance exigée, scalaire ou tableau par point.
        frozen: indices à ne pas déplacer.
        passes: nombre d'itérations.
        query: ``ProximityQuery`` à réutiliser. Les requêtes de proximité de
            trimesh ne sont pas réentrantes sur un maillage partagé : chaque
            fil doit passer la sienne plutôt que d'en laisser créer une ici.

    Returns:
        ``(n_déplacés, distance signée minimale obtenue)``.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3 or mesh is None:
        return 0, float("inf")

    if query is None:
        from trimesh.proximity import ProximityQuery

        query = ProximityQuery(mesh)
    required = np.broadcast_to(np.asarray(required_mm, dtype=np.float64), (n,)).copy()
    movable = _movable(n, frozen)
    moved = np.zeros(n, dtype=bool)

    for _ in range(max(1, int(passes))):
        closest, distance, faces = query.on_surface(pts)
        normals = mesh.face_normals[faces]
        outward = np.einsum("ij,ij->i", pts - closest, normals) >= 0
        signed = np.where(outward, distance, -distance)

        short = (signed < required) & movable
        if not np.any(short):
            break
        pts[short] = closest[short] + normals[short] * required[short, None]
        moved |= short

    closest, distance, faces = query.on_surface(pts)
    outward = np.einsum("ij,ij->i", pts - closest, mesh.face_normals[faces]) >= 0
    signed = np.where(outward, distance, -distance)

    points[:] = pts.astype(np.asarray(points).dtype, copy=False)
    return int(moved.sum()), float(signed.min())


def project_bend_radius(points, min_radius_mm: float, frozen=(),
                        passes: int = DEFAULT_BEND_PASSES):
    """Assouplit les coudes plus serrés que le rayon de cintrage admissible.

    Le rayon jugé est le **rayon de congé réalisable**, celui que la place
    disponible autorise : ``R = (min(L_entrant, L_sortant) / 2) / tan(θ / 2)``.
    Un sommet fautif est ramené vers le milieu de ses voisins, ce qui ouvre son
    angle donc augmente son rayon. Seuls les sommets fautifs bougent : lisser
    tout le tracé effacerait les longues portions droites, qui sont justement
    ce qu'on cherche à obtenir.

    Returns:
        ``(n_déplacés, rayon réalisable minimal obtenu)``.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3 or not np.isfinite(min_radius_mm) or min_radius_mm <= 0:
        return 0, float(gm.min_bend_radius(points))

    movable = _movable(n, frozen)
    moved = np.zeros(n, dtype=bool)

    for _ in range(max(1, int(passes))):
        radii = gm.bend_radii(pts)                     # taille n - 2
        tight = np.zeros(n, dtype=bool)
        tight[1:-1] = radii < float(min_radius_mm)
        tight &= movable
        if not np.any(tight):
            break
        milieu = 0.5 * (pts[:-2] + pts[2:])
        cible = np.zeros_like(pts)
        cible[1:-1] = milieu
        pts[tight] += BEND_RELAXATION * (cible[tight] - pts[tight])
        moved |= tight

    points[:] = pts.astype(np.asarray(points).dtype, copy=False)
    return int(moved.sum()), float(gm.min_bend_radius(pts))


def arc_positions(points, reference):
    """Avancement de chaque point le long du chemin de référence, en mm.

    On projette chaque point sur la polyligne de référence et l'on rend
    l'abscisse curviligne du projeté. C'est cette grandeur — et non la simple
    projection sur l'axe A→B — qui dit si le câble avance : un cheminement en L
    recule franchement le long de A→B sans pour autant revenir sur ses pas.

    Rend aussi la tangente de la référence au projeté, qui donne la direction
    dans laquelle pousser un point en retard.
    """
    pts = np.asarray(points, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if len(pts) == 0 or len(ref) < 2:
        return np.zeros(len(pts)), np.zeros((len(pts), 3))

    a, b = ref[:-1], ref[1:]
    seg = b - a
    lengths = np.linalg.norm(seg, axis=1)
    lengths = np.where(lengths > 1e-9, lengths, 1e-9)
    tangents = seg / lengths[:, None]
    starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])

    # Projection sur chaque segment, bornée à ses extrémités.
    delta = pts[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("ijk,jk->ij", delta, tangents), 0.0, lengths[None, :])
    closest = a[None, :, :] + t[:, :, None] * tangents[None, :, :]
    distances = np.linalg.norm(pts[:, None, :] - closest, axis=2)

    which = np.argmin(distances, axis=1)
    rows = np.arange(len(pts))
    return starts[which] + t[rows, which], tangents[which]


def project_progress(points, reference, frozen=(),
                     margin_mm: float = DEFAULT_PROGRESS_MARGIN_MM, passes: int = 3):
    """Interdit au tracé de revenir sur ses pas.

    Un faisceau qui recule puis repart forme un repli : deux brins côte à côte
    qu'aucun opérateur ne peut router, et que la longueur totale ne sanctionne
    qu'à peine. La pénalité de zigzag et le terme de progression ne suffisent
    pas — même leçon que le rayon de cintrage : une pénalité rend le repli
    coûteux, pas impossible.

    On impose donc que l'avancement le long du chemin de **référence** soit
    strictement croissant. La référence, et non la droite A→B : un cheminement
    en L recule le long de A→B sans revenir sur ses pas, et l'interdire
    supprimerait des trajets parfaitement valides.

    Un point en retard est poussé vers l'avant le long de la tangente de la
    référence, du strict nécessaire.

    Returns:
        ``(n_déplacés, nombre de reculs restants)``.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3 or reference is None or len(reference) < 2:
        return 0, 0

    movable = _movable(n, frozen)
    moved = np.zeros(n, dtype=bool)
    margin = max(float(margin_mm), 0.0)

    for _ in range(max(1, int(passes))):
        s, tangents = arc_positions(pts, reference)
        retreat = np.zeros(n, dtype=bool)
        for index in range(1, n):
            if s[index] < s[index - 1] + margin:
                if not movable[index]:
                    # Point imposé : on ne le bouge pas, mais son avancement
                    # sert de plancher aux suivants.
                    s[index] = max(s[index], s[index - 1])
                    continue
                pts[index] += tangents[index] * (s[index - 1] + margin - s[index])
                s[index] = s[index - 1] + margin
                retreat[index] = True
        if not np.any(retreat):
            break
        moved |= retreat

    points[:] = pts.astype(np.asarray(points).dtype, copy=False)
    final, _ = arc_positions(points, reference)
    return int(moved.sum()), int(np.count_nonzero(np.diff(final) < 0.0))


def project_anchors(points, anchors, tolerance_mm: float = DEFAULT_ANCHOR_TOLERANCE_MM,
                    frozen=()):
    """Impose au câble de passer **à portée** des points posés à la main.

    La première version épinglait le point exactement sur la position lâchée,
    puis le gelait. Deux conséquences, toutes deux mauvaises : le sommet ainsi
    figé ne pouvait plus être lissé, et le tracé se pliait autour au lieu de
    s'améliorer — l'utilisateur défaisait le travail des agents en croyant le
    guider.

    Ici la contrainte est une **zone**. Le point du câble le plus proche est
    ramené à ``tolerance_mm`` de l'ancre, pas dessus, et il n'est **pas gelé** :
    les agents continuent de le déplacer, sans jamais pouvoir s'éloigner
    davantage. L'utilisateur dit « passe par là » ; il ne dit pas « mets un
    sommet exactement ici ».

    Returns:
        ``(n_déplacés, écart maximal restant à une ancre, en mm)``.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if anchors is None or len(anchors) == 0 or n < 3:
        return 0, 0.0

    movable = _movable(n, frozen)
    if not np.any(movable):
        return 0, 0.0

    tolerance = max(float(tolerance_mm), 0.0)
    candidates = np.where(movable)[0]
    moved = 0
    worst = 0.0
    taken: set = set()

    for anchor in np.asarray(anchors, dtype=np.float64).reshape(-1, 3):
        free = [i for i in candidates if i not in taken]
        if not free:
            break
        distances = np.linalg.norm(pts[free] - anchor, axis=1)
        index = free[int(np.argmin(distances))]
        taken.add(index)

        distance = float(distances.min())
        if distance > tolerance:
            # On ne ramène pas le point *sur* l'ancre : jusqu'au bord de la
            # zone seulement. Ce qui reste de liberté appartient à l'agent.
            direction = (pts[index] - anchor) / max(distance, 1e-9)
            pts[index] = anchor + direction * tolerance
            moved += 1
            distance = tolerance
        worst = max(worst, distance)

    points[:] = pts.astype(np.asarray(points).dtype, copy=False)
    return moved, worst


def prune_redundant_points(points, mesh=None, required_mm=None,
                           min_radius_mm=None, frozen=(), max_removals=None):
    """Retire les points dont la présence n'apporte rien à la trajectoire.

    Le raffinement adaptatif ne savait qu'**ajouter** des points, là où la
    distance à la structure n'était pas tenue. Rien ne les enlevait ensuite :
    une fois le passage trouvé, le tracé restait aussi dense qu'au plus fort de
    la difficulté, et cette densité empêche de tendre la courbe — plus il y a
    de points sur un arc, plus il y a de sommets à aligner pour le redresser.

    Un point est retiré si le segment qui le remplace **reste conforme** :
    distance à la structure tenue, rayon de cintrage tenu chez les deux
    voisins. On ne retire donc jamais un point qui sert à contourner quelque
    chose ; on ne retire que ceux dont le détour n'achète rien.

    Les points imposés — extrémités, encoches, points posés à la main — ne sont
    jamais candidats.

    Returns:
        ``(trajectoire, indices retirés)``. La trajectoire est un nouveau
        tableau : retirer un point change les indices, et le faire sur place
        laisserait des index périmés chez l'appelant.
    """
    pts = np.asarray(points, dtype=np.float64).copy()
    n = len(pts)
    removed: list = []
    if n < 5:
        return np.asarray(points), removed

    protected = {0, n - 1} | {int(i) for i in (frozen or ())}
    budget = n if max_removals is None else int(max_removals)
    query = None
    if mesh is not None and required_mm is not None:
        from trimesh.proximity import ProximityQuery

        query = ProximityQuery(mesh)
    required = (np.broadcast_to(np.asarray(required_mm, dtype=np.float64), (n,)).copy()
                if required_mm is not None else None)

    # On parcourt de la fin vers le début : retirer un point ne décale alors
    # que des indices déjà examinés.
    index = len(pts) - 2
    while index >= 1 and len(removed) < budget:
        if index in protected or len(pts) < 5:
            index -= 1
            continue

        candidate = np.delete(pts, index, axis=0)
        if _keeps_shape(candidate, index, mesh, query, required, min_radius_mm):
            pts = candidate
            removed.append(index)
            protected = {i - 1 if i > index else i for i in protected}
        index -= 1

    return pts, sorted(removed)


def _keeps_shape(candidate, index, mesh, query, required, min_radius_mm) -> bool:
    """Le tracé privé d'un point reste-t-il conforme là où il a changé ?"""
    lo, hi = max(index - 2, 0), min(index + 2, len(candidate) - 1)
    if hi - lo < 2:
        return False

    if min_radius_mm:
        radii = gm.bend_radii(candidate[lo:hi + 1])
        if len(radii) and float(radii.min()) < float(min_radius_mm):
            return False

    if query is None or required is None:
        return True

    # Le segment qui remplace le point doit rester à distance sur toute sa
    # longueur, pas seulement à ses extrémités : c'est là qu'un raccourci
    # traverse la matière.
    a, b = candidate[max(index - 1, 0)], candidate[min(index, len(candidate) - 1)]
    samples = a + np.linspace(0.0, 1.0, 9)[:, None] * (b - a)
    closest, distance, faces = query.on_surface(samples)
    outward = np.einsum("ij,ij->i", samples - closest, mesh.face_normals[faces]) >= 0
    signed = np.where(outward, distance, -distance)
    return bool(signed.min() >= float(np.min(required)))


def remove_backtracking(points, min_cos: float = 0.0, keep=(), max_removals=None):
    """Retire les sommets où le tracé fait demi-tour.

    Un repli n'est pas un coude serré : c'est un aller-retour, deux brins côte
    à côte qu'aucun opérateur ne peut poser. Il se reconnaît au produit
    scalaire des deux directions qui se rejoignent au sommet — négatif, la
    seconde repart d'où venait la première.

    Le sommet fautif est **retiré**, pas déplacé : c'est un point dont le tracé
    n'avait pas besoin, et le supprimer raccorde directement ses deux voisins.
    On recommence tant qu'il en reste, un retrait pouvant en révéler un autre.

    Args:
        points: trajectoire ``(n, 3)``.
        min_cos: seuil sous lequel un raccord est jugé replié. ``0`` retient
            les vrais demi-tours ; une valeur positive resserre.
        keep: indices à conserver quoi qu'il arrive.
        max_removals: borne, pour ne pas raboter un tracé entier.

    Returns:
        ``(trajectoire, indices retirés dans la numérotation d'origine)``.
    """
    pts = np.asarray(points, dtype=np.float64).copy()
    if len(pts) < 3:
        return np.asarray(points), []

    protege = {0, len(pts) - 1} | {int(i) for i in (keep or ())}
    vivants = list(range(len(pts)))
    retires: list = []
    budget = len(pts) if max_removals is None else int(max_removals)

    while len(pts) >= 3 and len(retires) < budget:
        seg = np.diff(pts, axis=0)
        norms = np.linalg.norm(seg, axis=1, keepdims=True)
        unit = seg / np.where(norms > 1e-9, norms, 1.0)
        cos = np.einsum("ij,ij->i", unit[:-1], unit[1:])

        candidats = [i + 1 for i, c in enumerate(cos)
                     if c < float(min_cos) and (i + 1) not in protege]
        if not candidats:
            break

        # Le pli le plus franc d'abord : le corriger détend souvent ses voisins.
        fautif = min(candidats, key=lambda i: cos[i - 1])
        pts = np.delete(pts, fautif, axis=0)
        retires.append(vivants.pop(fautif))
        protege = {i - 1 if i > fautif else i for i in protege}

    return pts, sorted(retires)


def forward_only(points, start=None, goal=None, keep=(), tolerance_mm: float = 0.0):
    """Ne conserve que les points qui **avancent** de la source vers la cible.

    Un repli est un demi-tour franc, repérable au produit scalaire de deux
    directions voisines ; ce n'est pas la même chose qu'un recul. Un tracé peut
    n'avoir aucun demi-tour et pourtant revenir vers son point de départ, en
    décrivant une boucle large dont chaque raccord reste ouvert. Mesuré sur une
    maquette de 2 m : la géodésique brute présentait 55,8 mm de recul cumulé
    répartis sur deux sommets, dont un pas de 28 mm en arrière.

    On mesure donc la progression pour ce qu'elle est : la projection de chaque
    point sur la corde source-cible. Un sommet dont la projection est en deçà
    du maximum déjà atteint est **retiré** — comme pour un repli, c'est un
    point dont le tracé n'a pas besoin, et le supprimer raccorde directement
    ses deux voisins. Un seul passage suffit : la projection d'un point ne
    dépend pas de ses voisins, retirer l'un ne change pas celle des autres.

    Aller *de côté* n'est pas reculer : contourner un obstacle laisse la
    projection stagner, non décroître. La contrainte ne coûte donc rien aux
    détours légitimes.

    Args:
        points: trajectoire ``(n, 3)``.
        start, goal: extrémités. Par défaut les deux bouts du tracé.
        keep: indices à conserver quoi qu'il arrive.
        tolerance_mm: recul toléré avant retrait. Zéro n'en tolère aucun.

    Returns:
        ``(trajectoire, indices retirés dans la numérotation d'origine)``.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return pts.copy(), []

    source = pts[0] if start is None else np.asarray(start, dtype=np.float64).reshape(3)
    cible = pts[-1] if goal is None else np.asarray(goal, dtype=np.float64).reshape(3)
    axe = cible - source
    portee = float(np.linalg.norm(axe))
    if portee < 1e-9:
        return pts.copy(), []
    axe = axe / portee

    avance = (pts - source) @ axe
    protege = {0, len(pts) - 1} | {int(i) for i in (keep or ())}
    tolerance = abs(float(tolerance_mm))

    # Dépasser la cible pour y revenir, c'est reculer aussi : le sommet qui va
    # au-delà de B oblige B lui-même à revenir en arrière. Comme B ne peut pas
    # être retiré, c'est le sommet qui dépasse qui saute.
    plafond = float(avance[-1]) + tolerance

    gardes = [0]
    retires: list = []
    sommet = float(avance[0])
    for i in range(1, len(pts)):
        if i in protege or (avance[i] >= sommet - tolerance and avance[i] <= plafond):
            gardes.append(i)
            sommet = max(sommet, min(float(avance[i]), plafond))
        else:
            retires.append(i)

    return pts[gardes].copy(), retires


def project(points, mesh=None, required_mm=None, min_radius_mm=None,
            frozen=(), rounds: int = 2, query=None, reference=None,
            anchors=None, anchor_tolerance_mm: float = DEFAULT_ANCHOR_TOLERANCE_MM
            ) -> ProjectionReport:
    """Ramène la trajectoire dans le domaine des contraintes rédhibitoires.

    Les deux projections se contrarient : écarter un point de la structure
    creuse un coude, assouplir un coude rapproche de la structure. On les
    alterne sur quelques tours, en finissant par la distance — c'est la
    contrainte dont une violation est un clash, donc celle qu'il vaut mieux
    voir satisfaite en sortie.

    Ne lève jamais : une projection impossible est *rapportée*, pas fatale.
    """
    report = ProjectionReport()
    pts = np.asarray(points)
    n = len(pts)
    if n < 3:
        return report

    for _ in range(max(1, int(rounds))):
        if anchors is not None and len(anchors):
            moved, worst = project_anchors(points, anchors, anchor_tolerance_mm, frozen)
            report.n_anchor_moved += moved
            report.worst_anchor_mm = worst
        if reference is not None:
            moved, left = project_progress(points, reference, frozen)
            report.n_progress_moved += moved
            report.n_progress_left = left
        if min_radius_mm:
            moved, radius = project_bend_radius(points, min_radius_mm, frozen)
            report.n_bend_moved += moved
            report.min_bend_radius_mm = radius
        if mesh is not None and required_mm is not None:
            moved, clearance = project_clearance(points, mesh, required_mm, frozen,
                                                 query=query)
            report.n_clearance_moved += moved
            report.min_clearance_mm = clearance

    if mesh is not None and required_mm is not None:
        required = np.broadcast_to(np.asarray(required_mm, dtype=np.float64), (n,))
        if query is None:
            from trimesh.proximity import ProximityQuery

            query = ProximityQuery(mesh)
        closest, distance, faces = query.on_surface(np.asarray(points, dtype=np.float64))
        outward = np.einsum("ij,ij->i", np.asarray(points, dtype=np.float64) - closest,
                            mesh.face_normals[faces]) >= 0
        signed = np.where(outward, distance, -distance)
        report.min_clearance_mm = float(signed.min())
        # Les extrémités sont imposées : les compter en défaut ferait échouer
        # toute trajectoire dont l'équipement lui-même est proche d'une paroi.
        report.n_clearance_left = int(np.count_nonzero(signed[1:-1] < required[1:-1]))

    if min_radius_mm:
        radii = gm.bend_radii(np.asarray(points, dtype=np.float64))
        report.min_bend_radius_mm = float(radii.min()) if len(radii) else float("inf")
        report.n_bend_left = int(np.count_nonzero(radii < float(min_radius_mm)))

    return report
