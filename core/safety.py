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
    "project",
    "DEFAULT_CLEARANCE_PASSES",
    "DEFAULT_BEND_PASSES",
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


@dataclass
class ProjectionReport:
    """Ce que la projection a corrigé, et ce qui résiste."""

    n_clearance_moved: int = 0
    n_bend_moved: int = 0
    #: Distance minimale au DMU après projection, en mm (``inf`` si non mesurée).
    min_clearance_mm: float = float("inf")
    #: Rayon de cintrage réalisable minimal après projection, en mm.
    min_bend_radius_mm: float = float("inf")
    #: Points restant sous la distance exigée, malgré la projection.
    n_clearance_left: int = 0
    #: Coudes restant plus serrés que le rayon minimal.
    n_bend_left: int = 0

    @property
    def n_moved(self) -> int:
        return self.n_clearance_moved + self.n_bend_moved

    @property
    def feasible(self) -> bool:
        """La projection a-t-elle abouti à un tracé admissible ?"""
        return self.n_clearance_left == 0 and self.n_bend_left == 0


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


def project(points, mesh=None, required_mm=None, min_radius_mm=None,
            frozen=(), rounds: int = 2, query=None) -> ProjectionReport:
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
