"""Chemin suivant la surface de la maquette, y compris entre pièces disjointes.

`pyvista.PolyData.geodesic` cherche un chemin d'**arêtes** entre deux sommets.
Sur une maquette d'hélicoptère — la fusion de centaines de pièces qui ne se
touchent pas — il n'en existe aucun dès que le départ et l'arrivée sont sur
deux pièces différentes. La fonction échouait donc systématiquement, et le
tronçon se réduisait à une corde tendue d'un bout à l'autre. C'est ce que
l'utilisateur voyait : il demandait « le long de la surface » et recevait une
ligne droite.

On construit ici le graphe qui manque :

* les **arêtes du maillage**, pondérées par leur longueur : à l'intérieur d'une
  pièce, le plus court chemin dans ce graphe *est* la géodésique discrète ;
* des **ponts** entre sommets proches appartenant à des pièces différentes,
  pondérés par leur longueur multipliée par une pénalité. La pénalité est ce
  qui fait la différence entre « longer la structure » et « couper au plus
  court » : sauter d'une pièce à l'autre coûte plusieurs fois le prix du même
  déplacement le long d'une surface, donc le chemin ne saute que là où il n'a
  pas le choix, et sur la plus courte distance possible.

Le résultat n'est donc pas une géodésique au sens strict — il n'en existe pas
sur une surface discontinue — mais le chemin le plus proche de la surface que
la maquette autorise, avec le compte et la longueur des sauts nécessaires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["SurfacePathResult", "surface_path", "DEFAULT_BRIDGE_PENALTY"]

#: Coût d'un saut entre deux pièces, en multiples de sa longueur. Assez élevé
#: pour que longer une pièce reste préférable à la couper, assez fini pour que
#: franchir un vide reste possible quand c'est le seul passage.
DEFAULT_BRIDGE_PENALTY = 6.0

#: Longueur maximale d'un saut entre pièces, en mm. Au-delà, ce n'est plus un
#: passage d'une pièce à sa voisine mais une traversée à vide.
DEFAULT_MAX_BRIDGE_MM = 300.0

#: Nombre de ponts créés entre deux pièces voisines. Un seul suffirait à
#: connecter le graphe, mais forcerait tous les trajets par le même point :
#: en offrir plusieurs laisse la recherche choisir où franchir.
DEFAULT_BRIDGES_PER_PAIR = 6

#: Au-delà, les sommets d'une pièce sont sous-échantillonnés pour chercher les
#: ponts. Une pièce de 50 000 sommets n'offre pas 50 000 endroits distincts où
#: sauter vers sa voisine.
MAX_PAIR_SAMPLE = 1500

#: Le graphe est restreint aux sommets tels que ``d(A,v) + d(v,B)`` reste sous
#: ce multiple de la distance A-B. Sur une maquette entière, cela ramène le
#: problème à la zone réellement concernée. La valeur est calibrée : sur une
#: maquette de 128 000 sommets, elle divise par dix-huit le temps d'un trajet
#: court sans changer d'un millimètre le chemin trouvé, là où un corridor plus
#: serré (1,8) le rallongeait de 25 %.
DEFAULT_CORRIDOR_FACTOR = 3.5

#: En deçà, un écart au bord ne vaut plus la peine d'être tenté : on passe
#: directement à « sans contrainte » plutôt que de diviser indéfiniment.
MIN_EDGE_CLEARANCE_MM = 2.0

#: Écart toléré entre le tracé rendu et la surface, en mm. Au-delà, le segment
#: fautif est coupé en deux et son milieu ramené sur la surface.
DEFAULT_SURFACE_TOLERANCE_MM = 1.0

#: Nombre de bissections successives. Sur une surface courbe chaque passe
#: divise la longueur des segments par deux, donc la flèche par quatre : la
#: convergence y est immédiate. Sur une **arête vive** elle n'est que linéaire,
#: d'où la réserve de passes.
MAX_STICK_PASSES = 8

#: Longueur en deçà de laquelle un segment n'est plus coupé. Aucune polyligne
#: ne peut épouser une arête vive : là, la corde coupe le coin quoi qu'on
#: fasse, et bissecter indéfiniment n'ajoute que des points. Sous deux
#: millimètres, l'excursion résiduelle n'a plus de sens pour un toron de vingt.
MIN_STICK_SEGMENT_MM = 2.0

#: Plafond de points ajoutés par le collage, en multiple du tracé d'entrée. Un
#: maillage pathologique ne doit pas faire enfler la trajectoire sans fin.
STICK_GROWTH_LIMIT = 6

NO_MESH = "no_mesh"
NO_PATH = "no_path"
NO_SCIPY = "no_scipy"

_REASON_FR = {
    NO_MESH: "Aucun maillage exploitable pour suivre la surface.",
    NO_PATH: "Aucun chemin de surface entre ces deux points, même en autorisant "
             "des sauts entre pièces.",
    NO_SCIPY: "SciPy est requis pour le chemin de surface.",
}

_REASON_EN = {
    NO_MESH: "No usable mesh to follow the surface.",
    NO_PATH: "No surface path between these two points, even allowing hops "
             "between parts.",
    NO_SCIPY: "SciPy is required for the surface path.",
}


@dataclass
class SurfacePathResult:
    """Chemin obtenu, et ce qu'il a coûté en sauts entre pièces."""

    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    success: bool = False
    reason: str = ""
    #: Nombre de sauts d'une pièce à une autre.
    n_bridges: int = 0
    #: Longueur cumulée de ces sauts, en mm.
    bridge_length_mm: float = 0.0
    #: Longueur parcourue en restant sur une surface, en mm.
    surface_length_mm: float = 0.0
    #: Nombre de sommets retenus dans le graphe (indicateur de coût).
    graph_nodes: int = 0
    #: Distance minimale au DMU après décalage, en mm. Mesurée contre le
    #: maillage réel, pas déduite du décalage demandé.
    min_clearance_mm: float = 0.0
    #: Écart au bord libre réellement obtenu, en mm.
    edge_clearance_mm: float = 0.0
    #: Raccords où le tracé repart d'où il venait. Zéro sur un tracé posable.
    n_folds: int = 0
    #: Sommets retirés parce qu'ils reculaient vers la source au lieu d'avancer
    #: vers la cible. Indicateur, pas un défaut : le tracé rendu n'en a plus.
    n_retreats_removed: int = 0
    #: Décalage effectivement appliqué, en mm (0 = chemin resté sur la surface).
    offset_mm: float = 0.0

    @property
    def total_length_mm(self) -> float:
        return self.surface_length_mm + self.bridge_length_mm

    @property
    def surface_ratio(self) -> float:
        """Part du trajet réellement effectuée le long d'une surface, 0..1."""
        total = self.total_length_mm
        return 1.0 if total <= 1e-9 else self.surface_length_mm / total

    def message(self, lang: str = "FR") -> str:
        english = str(lang).upper().startswith("EN")
        if not self.success:
            table = _REASON_EN if english else _REASON_FR
            return table.get(self.reason, self.reason)

        if english:
            text = f"Surface path: {self.total_length_mm:.0f} mm"
            text += (", entirely along the structure" if not self.n_bridges else
                     f", {self.surface_ratio * 100:.0f} % along the structure, "
                     f"{self.n_bridges} hop(s) between parts "
                     f"({self.bridge_length_mm:.0f} mm)")
            if self.offset_mm:
                text += f", lifted to {self.min_clearance_mm:.0f} mm off the DMU"
            if self.edge_clearance_mm:
                text += f", kept {self.edge_clearance_mm:.0f} mm off free edges"
            if self.n_retreats_removed:
                text += f", {self.n_retreats_removed} backward vertex(es) dropped"
            return text + "."

        text = f"Chemin de surface : {self.total_length_mm:.0f} mm"
        text += (", entièrement le long de la structure" if not self.n_bridges else
                 f", {self.surface_ratio * 100:.0f} % le long de la structure, "
                 f"{self.n_bridges} saut(s) entre pièces "
                 f"({self.bridge_length_mm:.0f} mm)")
        if self.offset_mm:
            text += f", décollé à {self.min_clearance_mm:.0f} mm du DMU"
        if self.edge_clearance_mm:
            text += f", à {self.edge_clearance_mm:.0f} mm des bords libres"
        if self.n_retreats_removed:
            text += f", {self.n_retreats_removed} sommet(s) en recul retiré(s)"
        return text + "."


def _corridor_mask(vertices, start, goal, factor: float) -> np.ndarray:
    """Sommets utiles à un trajet A→B, par une enveloppe ellipsoïdale.

    Un chemin raisonnable ne s'éloigne pas arbitrairement : on ne garde que les
    sommets dont la somme des distances aux deux extrémités reste sous
    ``factor`` fois la distance directe. Sur une maquette complète, cela réduit
    le graphe de plusieurs ordres de grandeur sans écarter aucun trajet
    plausible.
    """
    direct = float(np.linalg.norm(goal - start))
    if direct <= 1e-9 or factor <= 0:
        return np.ones(len(vertices), dtype=bool)

    budget = direct * float(factor)
    d_start = np.linalg.norm(vertices - start, axis=1)
    d_goal = np.linalg.norm(vertices - goal, axis=1)
    mask = (d_start + d_goal) <= budget

    # Le corridor doit au minimum contenir de quoi accrocher les deux
    # extrémités : sinon on rend la main sur un graphe vide.
    if mask.sum() < 4:
        return np.ones(len(vertices), dtype=bool)
    return mask


def _edge_mask(mesh, vertices, clearance_mm: float) -> np.ndarray:
    """Sommets assez loin des bords libres pour porter le tracé.

    Le plus court chemin le long d'une tôle passe volontiers par son chant :
    c'est là que la surface est la plus « directe ». Or un chant ne peut
    recevoir aucune fixation, et use la gaine. On retire donc du graphe les
    sommets trop proches d'un bord, plutôt que de laisser le tracé y aller
    pour l'en déloger ensuite à coups de pénalité.

    Si la contrainte vide le graphe — une pièce entièrement étroite — on la
    lève : mieux vaut un tracé perfectible qu'aucun tracé. Le second membre du
    couple dit si elle a réellement été appliquée, afin de ne pas annoncer un
    écart qu'on n'a pas obtenu.
    """
    entier = np.ones(len(vertices), dtype=bool)
    if clearance_mm <= 0:
        return entier, False
    try:
        from core.geometry_metrics import boundary_points, edge_distances

        boundary = boundary_points(mesh)
    except Exception:
        return entier, False
    if len(boundary) == 0:
        # Pièce fermée : aucun bord libre, la contrainte est sans objet.
        return entier, False

    mask = edge_distances(vertices, boundary) >= clearance_mm
    if mask.sum() < 4:
        return entier, False
    return mask, True


def surface_path(
    mesh,
    start,
    goal,
    bridge_penalty: float = DEFAULT_BRIDGE_PENALTY,
    max_bridge_mm: float = DEFAULT_MAX_BRIDGE_MM,
    bridges_per_pair: int = DEFAULT_BRIDGES_PER_PAIR,
    corridor_factor: float = DEFAULT_CORRIDOR_FACTOR,
    num_points: int | None = None,
    offset_mm: float = 0.0,
    edge_clearance_mm: float = 0.0,
) -> SurfacePathResult:
    """Chemin le plus court le long de la surface, sauts entre pièces compris.

    Args:
        mesh: maillage trimesh de l'environnement.
        start, goal: extrémités, en mm.
        bridge_penalty: coût d'un saut, en multiples de sa longueur.
        max_bridge_mm: longueur maximale d'un saut entre deux pièces.
        bridges_per_pair: nombre de points de passage offerts entre deux
            pièces voisines.
        corridor_factor: enveloppe autour de A-B, en multiples de la distance
            directe. 0 désactive la restriction.
        num_points: rééchantillonnage final.
        offset_mm: distance à laquelle décoller le tracé de la surface. Zéro
            le laisse *sur* la surface, donc en interférence sur toute sa
            longueur — utile pour visualiser la géodésique, inutilisable comme
            point de départ pour les agents.
        edge_clearance_mm: distance en deçà de laquelle un sommet trop proche
            d'un bord libre est retiré du graphe. Le chemin le plus court le
            long d'une tôle passe volontiers par son chant — c'est là que la
            surface est la plus « directe » — alors qu'aucune fixation ne peut
            y être posée. Zéro laisse le graphe entier.

    Returns:
        Un :class:`SurfacePathResult`. Jamais d'exception, jamais de repli
        silencieux sur une ligne droite : c'est précisément ce repli qui
        faisait croire que la géodésique fonctionnait.
    """
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components, dijkstra
        from scipy.spatial import cKDTree
    except ImportError:
        return SurfacePathResult(reason=NO_SCIPY)

    if mesh is None or len(getattr(mesh, "vertices", [])) < 2:
        return SurfacePathResult(reason=NO_MESH)

    start = np.asarray(start, dtype=np.float64).reshape(3)
    goal = np.asarray(goal, dtype=np.float64).reshape(3)

    all_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    keep = _corridor_mask(all_vertices, start, goal, corridor_factor)
    edge_mask, edge_applied = _edge_mask(mesh, all_vertices, float(edge_clearance_mm))
    keep &= edge_mask

    index_map = np.full(len(all_vertices), -1, dtype=np.int64)
    index_map[keep] = np.arange(int(keep.sum()))
    vertices = all_vertices[keep]
    n = len(vertices)
    if n < 2:
        return SurfacePathResult(reason=NO_MESH)

    # --- arêtes du maillage, restreintes au corridor --------------------
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    inside = keep[edges[:, 0]] & keep[edges[:, 1]]
    edges = index_map[edges[inside]]

    if len(edges):
        lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    else:
        lengths = np.zeros(0)

    # --- composantes connexes : une par pièce (ou groupe de pièces) ------
    adjacency = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)
    ).tocsr()
    n_parts, labels = connected_components(adjacency, directed=False)

    # --- ponts entre pièces ----------------------------------------------
    # Chercher les k plus proches voisins de chaque sommet ne marche pas : sur
    # une pièce un tant soit peu dense, les huit voisins les plus proches d'un
    # sommet sont tous sur la même pièce, et aucun pont n'est jamais créé. On
    # raisonne donc **par composante** : pour chaque paire de pièces assez
    # proches, on retient les quelques couples de sommets les plus rapprochés.
    bridges = np.zeros((0, 2), dtype=np.int64)
    bridge_lengths = np.zeros(0)
    if n_parts > 1 and bridges_per_pair > 0 and max_bridge_mm > 0:
        bridges, bridge_lengths = _build_bridges(
            vertices, labels, n_parts,
            max_bridge_mm=float(max_bridge_mm),
            bridges_per_pair=int(bridges_per_pair),
        )

    # --- graphe complet : arêtes + ponts + les deux extrémités -----------
    # Les extrémités sont des nœuds supplémentaires, rattachés au sommet le
    # plus proche : le chemin part donc du point demandé, pas d'un sommet du
    # maillage choisi arbitrairement.
    tree = cKDTree(vertices)
    _, start_anchor = tree.query(start)
    _, goal_anchor = tree.query(goal)
    start_node, goal_node = n, n + 1
    total_nodes = n + 2

    rows = np.concatenate([
        edges[:, 0] if len(edges) else np.zeros(0, dtype=np.int64),
        bridges[:, 0] if len(bridges) else np.zeros(0, dtype=np.int64),
        [start_node, goal_node],
    ])
    cols = np.concatenate([
        edges[:, 1] if len(edges) else np.zeros(0, dtype=np.int64),
        bridges[:, 1] if len(bridges) else np.zeros(0, dtype=np.int64),
        [start_anchor, goal_anchor],
    ])
    weights = np.concatenate([
        lengths,
        bridge_lengths * float(bridge_penalty),
        [np.linalg.norm(vertices[start_anchor] - start),
         np.linalg.norm(vertices[goal_anchor] - goal)],
    ])

    graph = coo_matrix(
        (np.maximum(weights, 1e-9), (rows, cols)), shape=(total_nodes, total_nodes)
    ).tocsr()

    distances, predecessors = dijkstra(
        graph, directed=False, indices=start_node, return_predecessors=True
    )
    if not np.isfinite(distances[goal_node]):
        # Un écart au bord trop ambitieux peut couper la seule voie possible :
        # entre le contour d'un panneau et son ouverture, la bande utile est
        # parfois plus étroite que deux fois l'écart demandé. On relâche alors
        # la contrainte plutôt que de refuser tout tracé — et on dira, plus
        # bas, ce qui a réellement été obtenu.
        if edge_clearance_mm > 0:
            return surface_path(
                mesh, start, goal,
                bridge_penalty=bridge_penalty, max_bridge_mm=max_bridge_mm,
                bridges_per_pair=bridges_per_pair, corridor_factor=corridor_factor,
                num_points=num_points, offset_mm=offset_mm,
                edge_clearance_mm=edge_clearance_mm / 2.0
                if edge_clearance_mm > MIN_EDGE_CLEARANCE_MM else 0.0,
            )
        return SurfacePathResult(reason=NO_PATH, graph_nodes=n)

    # --- remontée du chemin ----------------------------------------------
    chain = []
    node = goal_node
    while node != start_node and node >= 0:
        chain.append(node)
        node = predecessors[node]
    chain.append(start_node)
    chain.reverse()

    points = []
    for node in chain:
        if node == start_node:
            points.append(start)
        elif node == goal_node:
            points.append(goal)
        else:
            points.append(vertices[node])
    points = np.asarray(points, dtype=np.float64)

    # --- mesure : ce qui longe la surface, ce qui saute ------------------
    bridge_set = {(int(a), int(b)) for a, b in bridges}
    bridge_set |= {(b, a) for a, b in bridge_set}

    surface_len = 0.0
    bridge_len = 0.0
    n_bridges = 0
    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        step = float(np.linalg.norm(points[i + 1] - points[i]))
        if (int(a), int(b)) in bridge_set:
            bridge_len += step
            n_bridges += 1
        else:
            surface_len += step

    if num_points and len(points) >= 2:
        points = _resample(points, int(num_points))

    # Le plus court chemin d'un graphe ne repasse jamais par un sommet, mais
    # rien ne l'empêche de revenir vers sa source : contourner une traverse le
    # long des arêtes du maillage fait volontiers reculer le tracé de quelques
    # centimètres avant de repartir. Un faisceau ne recule pas — on retire les
    # sommets fautifs avant toute autre chose.
    from core.safety import forward_only

    points, retreats = forward_only(points, start, goal)
    n_retreats = len(retreats)

    # Décollement : les longueurs mesurées ci-dessus décrivent le trajet le
    # long de la surface, elles ne sont donc pas recalculées ici. Ce qui change
    # est la distance au DMU, mesurée et rapportée séparément.
    min_clearance = 0.0
    if offset_mm and offset_mm > 0:
        try:
            points, min_clearance, _ = unfold(mesh, points, float(offset_mm))
            # Le décalage pousse chaque point le long de sa propre normale : il
            # peut recréer un recul là où il n'y en avait pas.
            points, again = forward_only(points, start, goal)
            n_retreats += len(again)
        except Exception:
            offset_mm = 0.0

    if num_points and len(points) != int(num_points):
        # Le dépliage retire des sommets : on rétablit la densité demandée,
        # sans quoi le tracé rendu serait plus grossier que ce qu'on a promis.
        points = _resample(points, int(num_points))

    return SurfacePathResult(
        points=points.astype(np.float32),
        success=True,
        n_bridges=n_bridges,
        bridge_length_mm=bridge_len,
        surface_length_mm=surface_len,
        graph_nodes=n,
        min_clearance_mm=min_clearance,
        # Mesuré sur le tracé rendu, décalage ou non : un compteur qui ne se
        # remplit que dans un cas annonçait zéro repli sur un tracé qui en
        # comptait deux.
        n_folds=_count_folds(points),
        n_retreats_removed=n_retreats,
        # On n'annonce que ce qu'on a obtenu : une contrainte levée faute de
        # place ne doit pas se lire comme une contrainte respectée.
        edge_clearance_mm=float(edge_clearance_mm) if edge_applied else 0.0,
        offset_mm=float(offset_mm or 0.0),
    )


def _measure_clearance(mesh, points, query=None, samples: int = 9) -> float:
    """Marge minimale du tracé rendu, **segments compris**.

    Mesurer les seuls sommets annonçait 25 mm sur un tracé dont les cordes
    redescendaient à 0,4 mm de la structure.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return 0.0
    if len(pts) == 1:
        return float(signed_distances(mesh, pts, query=query)[0][0])

    ratios = np.linspace(0.0, 1.0, max(3, int(samples)))
    echantillons = np.concatenate([
        pts[:-1] + ratio * (pts[1:] - pts[:-1]) for ratio in ratios
    ])
    mesures, _, _ = signed_distances(mesh, echantillons, query=query)
    return float(mesures.min())


def signed_distances(mesh, points, query=None):
    """Distance à la surface, **négative dans la matière**.

    ``ProximityQuery.on_surface`` rend une distance non signée : un point
    enfoncé de 20 mm dans une tôle et un point posé 20 mm au-dessus lui sont
    indiscernables. C'est précisément la confusion qu'il ne faut pas faire ici.
    """
    from trimesh.proximity import ProximityQuery

    pts = np.asarray(points, dtype=np.float64)
    query = query if query is not None else ProximityQuery(mesh)
    closest, distance, faces = query.on_surface(pts)
    normals = mesh.face_normals[faces]
    outward = np.einsum("ij,ij->i", pts - closest, normals) >= 0
    return np.where(outward, distance, -distance), closest, normals


def surface_gap(mesh, points, query=None, samples: int = 9):
    """De combien les **segments** s'écartent de la distance qu'ils promettent.

    Mesurer les seuls sommets ne dit rien : ils viennent du maillage, ou d'une
    projection, et sont donc à la bonne distance par construction — pendant que
    la corde qui les relie plonge dans la matière. On échantillonne donc le
    long de chaque segment.

    L'écart se compte **par rapport à ce qu'annoncent les extrémités**, pas
    dans l'absolu. Un segment qui joint un connecteur posé à 50 mm de la tôle à
    un point de surface est censé traverser cet intervalle, ce n'est pas un
    défaut ; un segment dont les deux bouts sont à 25 mm est censé y rester.
    Ce qui compte est l'écart *en plus* de cette interpolation, dans un sens
    comme dans l'autre : se rapprocher de la structure met le câble en
    interférence, s'en éloigner veut dire qu'il ne la suit plus.

    Returns:
        ``(écart_max_mm, écarts_par_segment)``.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0, np.zeros(0)

    from trimesh.proximity import ProximityQuery

    query = query if query is not None else ProximityQuery(mesh)
    aux_sommets, _, _ = signed_distances(mesh, pts, query=query)

    ratios = np.linspace(0.0, 1.0, max(3, int(samples)))[1:-1]
    debut, fin = pts[:-1], pts[1:]
    d_debut, d_fin = aux_sommets[:-1], aux_sommets[1:]

    ecarts = np.zeros(len(debut))
    for ratio in ratios:
        echantillons = debut + ratio * (fin - debut)
        mesures, _, _ = signed_distances(mesh, echantillons, query=query)
        attendu = d_debut + ratio * (d_fin - d_debut)
        ecarts = np.maximum(ecarts, np.abs(mesures - attendu))

    return float(ecarts.max()), ecarts


def stick_to_surface(mesh, points, tolerance_mm: float = DEFAULT_SURFACE_TOLERANCE_MM,
                     query=None, keep=(), target_mm: float | None = None):
    """Fait tenir au tracé la distance qu'il annonce, segments compris.

    Le plus court chemin rendu par Dijkstra passe par des **sommets** du
    maillage : ceux-là sont bien sur la surface. Les segments droits qui les
    relient, non. Là où le maillage est grossier — le flanc d'un cylindre n'a
    de sommets qu'à ses deux couronnes — la corde coupe au travers : mesuré
    17,4 mm à l'intérieur de la matière sur un cylindre de 300 mm de diamètre.
    Le rééchantillonnage aggrave le défaut au lieu de le corriger, puisqu'il
    interpole **en ligne droite** sur cette polyligne : à 68 points, six
    sommets se retrouvaient dans la matière, jusqu'à 17,1 mm.

    Le même défaut se reproduit un cran plus haut, sur le tracé **décalé** :
    deux points posés à 25 mm de part et d'autre d'une lisse sont bien à 25 mm,
    et la corde qui les joint redescend à 0,4 mm de la structure.

    On coupe donc en deux tout segment qui s'écarte de la distance annoncée par
    ses extrémités, et l'on repose son milieu à cette distance. Une passe
    divise la longueur par deux, donc la flèche par quatre : le tracé ne se
    densifie que là où la géométrie l'exige — une portion plane n'y gagne aucun
    point.

    Les extrémités ne sont jamais déplacées : ce sont le départ et l'arrivée
    demandés, qui n'ont aucune raison d'être sur une tôle.

    Args:
        mesh: maillage trimesh.
        points: tracé ``(n, 3)``.
        tolerance_mm: écart toléré avant bissection.
        query: ``ProximityQuery`` déjà construit, le cas échéant.
        keep: indices à ne pas déplacer, en plus des extrémités.
        target_mm: distance à tenir. ``None`` conserve celle de chaque point,
            ``0`` ramène le tracé sur la surface.

    Returns:
        ``(tracé, écart_maximal_restant_mm)``.
    """
    from trimesh.proximity import ProximityQuery

    pts = np.asarray(points, dtype=np.float64).copy()
    if len(pts) < 2:
        return pts, 0.0

    query = query if query is not None else ProximityQuery(mesh)
    tolerance = max(1e-6, float(tolerance_mm))
    plafond = max(len(pts) * STICK_GROWTH_LIMIT, len(pts) + 8)

    # Les sommets d'abord : le rééchantillonnage les a posés sur des cordes,
    # pas à la distance visée. Les extrémités et les points imposés restent où
    # ils sont — les déplacer changerait le trajet demandé.
    protege = {0, len(pts) - 1} | {int(i) for i in (keep or ())}
    mobiles = np.array([i for i in range(len(pts)) if i not in protege], dtype=int)
    if len(mobiles) and target_mm is not None:
        _, closest, normals = signed_distances(mesh, pts[mobiles], query=query)
        pts[mobiles] = closest + normals * float(target_mm)

    for _ in range(MAX_STICK_PASSES):
        if len(pts) >= plafond:
            break
        _, ecarts = surface_gap(mesh, pts, query=query)
        longueurs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        fautifs = np.where((ecarts > tolerance)
                           & (longueurs > MIN_STICK_SEGMENT_MM))[0]
        if not len(fautifs):
            break

        fautifs = fautifs[: max(1, plafond - len(pts))]
        milieux = 0.5 * (pts[fautifs] + pts[fautifs + 1])

        # Le milieu inséré est reposé à la distance qu'annonçaient ses voisins,
        # le long de la normale de la face la plus proche. Un segment dont une
        # extrémité est loin de la tôle n'est donc pas rabattu dessus.
        aux, _, _ = signed_distances(mesh, pts, query=query)
        attendu = 0.5 * (aux[fautifs] + aux[fautifs + 1])
        _, closest, normals = signed_distances(mesh, milieux, query=query)
        inseres = closest + normals * attendu[:, None]

        pts = np.insert(pts, fautifs + 1, inseres, axis=0)

    return pts, surface_gap(mesh, pts, query=query)[0]


def offset_from_surface(mesh, points, target_mm, passes=4, query=None):
    """Décolle un tracé de la surface, jusqu'à la distance visée.

    Un chemin de surface est **sur** la surface : sa distance au DMU vaut zéro
    et il est donc en interférence sur toute sa longueur. Les agents passent
    alors leurs premières centaines d'itérations à faire ce qu'un simple
    décalage géométrique fait en une fois.

    Chaque point est repoussé le long de la normale de la face la plus proche.
    L'opération est répétée : après un premier décalage, la face la plus proche
    peut avoir changé — près d'une arête, notamment — et un seul passage
    laisserait des points en deçà de la cible.

    Args:
        mesh: maillage trimesh de l'environnement.
        points: tracé ``(n, 3)``.
        target_mm: distance visée, en mm.
        passes: nombre de reprises.
        query: ``ProximityQuery`` déjà construit. Le construire coûte plusieurs
            secondes sur une maquette de 800 000 triangles : les étapes qui se
            succèdent ici se le partagent au lieu d'en bâtir un chacune.

    Returns:
        ``(points, distance_minimale_mesurée)``. La distance est mesurée contre
        le maillage réel, pas déduite du décalage demandé : c'est la seule qui
        engage.
    """
    from trimesh.proximity import ProximityQuery

    result = np.array(points, dtype=np.float64, copy=True)
    if len(result) == 0 or target_mm <= 0:
        return result, 0.0

    query = query if query is not None else ProximityQuery(mesh)
    for _ in range(max(1, int(passes))):
        closest, distance, faces = query.on_surface(result)
        normals = mesh.face_normals[faces]

        # Un point dans la matière doit ressortir : la normale pointe déjà vers
        # l'extérieur, il suffit de partir de sa projection sur la surface.
        outward = np.einsum("ij,ij->i", result - closest, normals) >= 0
        signed = np.where(outward, distance, -distance)

        short = signed < target_mm
        if not np.any(short):
            break
        result[short] = closest[short] + normals[short] * target_mm

    closest, distance, faces = query.on_surface(result)
    inside = np.einsum("ij,ij->i", result - closest, mesh.face_normals[faces]) < 0
    signed = np.where(inside, -distance, distance)
    return result, float(signed.min())


def unfold(mesh, points, target_mm, rounds: int = 3, query=None):
    """Décolle le tracé **sans le replier sur lui-même**.

    Le décalage seul plie le tracé sur une maquette fusionnée. Chaque point est
    repoussé le long de la normale de **sa** face ; or deux points voisins
    tombent volontiers sur des faces différentes — de part et d'autre d'une
    arête, ou sur deux pièces qui se font face — dont les normales sont presque
    opposées. Poussés chacun de son côté, ils se croisent, et le tracé fait un
    aller-retour. Mesuré sur un DMU de huit pièces : quatorze inversions de
    direction pour un écart de 60 mm.

    Lisser les directions de poussée ne règle rien : la marge obtenue cesse
    d'atteindre la cible sans que les replis disparaissent — essayé, mesuré,
    abandonné. On garde donc la poussée exacte, face par face, et l'on
    **retire les sommets repliés** avant de repousser ce qui reste. Un sommet
    replié est un point dont le tracé n'a pas besoin : le supprimer raccorde
    directement ses deux voisins.

    Returns:
        ``(points, marge_minimale, n_replis_restants)``.
    """
    from core.safety import remove_backtracking

    result = np.array(points, dtype=np.float64, copy=True)
    clearance = 0.0
    for _ in range(max(1, int(rounds))):
        result, clearance = offset_from_surface(mesh, result, target_mm, query=query)
        unfolded, removed = remove_backtracking(result)
        if not removed:
            break
        result = unfolded

    result, clearance = offset_from_surface(mesh, result, target_mm, query=query)
    return result, clearance, _count_folds(result)


def _count_folds(points) -> int:
    """Nombre de raccords où le tracé repart d'où il venait."""
    seg = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    norms = np.linalg.norm(seg, axis=1, keepdims=True)
    if len(seg) < 2:
        return 0
    unit = seg / np.where(norms > 1e-9, norms, 1.0)
    return int(np.count_nonzero(np.einsum("ij,ij->i", unit[:-1], unit[1:]) < 0.0))


def _build_bridges(vertices, labels, n_parts, max_bridge_mm, bridges_per_pair):
    """Couples de sommets reliant deux pièces distinctes.

    Les paires de pièces candidates sont d'abord filtrées sur leurs enveloppes :
    comparer toutes les paires serait quadratique en nombre de pièces, et une
    maquette en compte des centaines.
    """
    from scipy.spatial import cKDTree

    groups = [np.flatnonzero(labels == part) for part in range(n_parts)]
    centres = np.array([vertices[g].mean(axis=0) for g in groups])
    radii = np.array([
        float(np.linalg.norm(vertices[g] - centres[i], axis=1).max()) if len(g) else 0.0
        for i, g in enumerate(groups)
    ])

    centre_tree = cKDTree(centres)
    reach = float(radii.max()) + max_bridge_mm

    rows, cols, dists = [], [], []
    seen: set[tuple[int, int]] = set()

    for i, group_i in enumerate(groups):
        if not len(group_i):
            continue
        for j in centre_tree.query_ball_point(centres[i], radii[i] + reach):
            if j <= i or not len(groups[j]):
                continue
            if (i, j) in seen:
                continue
            seen.add((i, j))

            a = _subsample(group_i)
            b = _subsample(groups[j])
            tree_b = cKDTree(vertices[b])
            d, nearest = tree_b.query(vertices[a], k=1, distance_upper_bound=max_bridge_mm)

            reachable = np.isfinite(d)
            if not reachable.any():
                continue
            order = np.argsort(d[reachable])[:bridges_per_pair]
            src = a[reachable][order]
            dst = b[nearest[reachable][order]]
            rows.extend(src.tolist())
            cols.extend(dst.tolist())
            dists.extend(d[reachable][order].tolist())

    if not rows:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0)
    return (np.stack([np.asarray(rows, dtype=np.int64),
                      np.asarray(cols, dtype=np.int64)], axis=1),
            np.asarray(dists, dtype=np.float64))


def _subsample(indices: np.ndarray) -> np.ndarray:
    """Échantillon régulier d'une pièce, borné par :data:`MAX_PAIR_SAMPLE`."""
    if len(indices) <= MAX_PAIR_SAMPLE:
        return indices
    step = len(indices) / MAX_PAIR_SAMPLE
    picks = (np.arange(MAX_PAIR_SAMPLE) * step).astype(np.int64)
    return indices[picks]


def _resample(points: np.ndarray, num_points: int) -> np.ndarray:
    """Rééchantillonne à pas constant, en conservant les deux extrémités."""
    num_points = max(2, int(num_points))
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = cumulative[-1]
    if total <= 1e-9:
        return np.tile(points[0], (num_points, 1))

    targets = np.linspace(0.0, total, num_points)
    out = np.empty((num_points, 3), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(targets, cumulative, points[:, axis])
    return out
