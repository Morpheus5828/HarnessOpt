"""Quelles encoches emprunter, et dans quel sens les traverser.

Un peigne porte plusieurs encoches **côte à côte**, une par faisceau : les
autres appartiennent aux voisins. Un faisceau en emprunte donc **une**. Le code
enfilait au contraire toutes les encoches détectées, ordonnées par la
projection de leur centre sur A→B. Comme les encoches d'un même peigne se
projettent quasiment au même endroit, elles se suivaient dans la liste : le
câble traversait une encoche, ressortait, et repartait aussitôt dans celle
d'à côté au lieu de rejoindre le peigne suivant.

Restent deux libertés, qui ne se décident pas séparément :

* **quelle** encoche prendre sur chaque peigne ;
* **dans quel sens** la traverser. ``p_in`` et ``p_out`` sont interchangeables ;
  ce qui ne l'est pas, c'est leur solidarité — entrer par l'un oblige à
  ressortir par l'autre, jamais par celui d'une encoche voisine.

Les deux sont tranchées ensemble par programmation dynamique sur les peignes
ordonnés le long du trajet : le résultat est le trajet le plus court parmi
**tous** les choix d'encoche et de sens. Un choix local — « l'encoche la plus
proche », « le côté rencontré en premier » — rejouerait le défaut d'origine
sous une autre forme : c'est justement en raisonnant encoche par encoche qu'on
finit par faire la navette sur un même peigne.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Crossing",
    "choose_crossings",
    "comb_center",
    "comb_name",
    "default_selection",
    "describe",
    "detour_ratio",
    "filter_combs",
    "merge_anchors",
    "in_routing_zone",
    "DEFAULT_ZONE_FACTOR",
]

#: Rallongement relatif toléré pour aller chercher un peigne. Le couloir de
#: cheminement est l'ellipsoïde de foyers A et B : un peigne y est admis si le
#: détour qu'il impose — ``|PA| + |PB|`` — ne dépasse pas ce multiple de la
#: distance directe ``|AB|``. 1,25 laisse un écart latéral d'environ 37 % de
#: la longueur du trajet, ce qui couvre largement un cheminement qui serpente,
#: et écarte les fixations reconnues ailleurs dans la maquette.
DEFAULT_ZONE_FACTOR = 1.25


@dataclass(frozen=True)
class Crossing:
    """Une encoche retenue, dans le sens où le câble la traverse."""

    comb: str
    passage: object
    entry: tuple
    exit: tuple
    #: Vrai si l'on traverse de ``p_out`` vers ``p_in``. Sans conséquence
    #: physique — les deux points sont interchangeables — mais c'est ce qui
    #: permet de relire un trajet et de vérifier qu'on n'a pas inversé
    #: l'entrée et la sortie d'un couple par accident.
    flipped: bool = False

    @property
    def points(self) -> list:
        """Le couple, dans l'ordre de la marche."""
        return [list(self.entry), list(self.exit)]

    @property
    def width_mm(self) -> float:
        return float(np.linalg.norm(np.asarray(self.exit) - np.asarray(self.entry)))


def comb_center(passages) -> np.ndarray:
    """Centre d'un peigne : la moyenne de ses encoches.

    C'est le peigne qui est une étape du trajet, pas l'encoche. Ordonner sur le
    centre d'une encoche prise au hasard ferait dépendre la place du peigne
    dans le trajet de l'encoche qu'on lui aura choisie — alors que le choix de
    l'encoche vient après.
    """
    centers = [np.asarray(p.center, dtype=np.float64) for p in passages]
    return np.mean(centers, axis=0) if centers else np.zeros(3)


def comb_name(passages) -> str:
    """Nom du peigne qui porte ces encoches."""
    for passage in passages:
        name = str(getattr(passage, "comb", "") or "")
        if name:
            return name
    return "?"


def detour_ratio(start, goal, point) -> float:
    """Rallongement relatif imposé par un crochet jusqu'à ``point``.

    Vaut 1 sur le segment A–B, et croît avec l'écart. Renvoie ``0`` quand A et
    B sont confondus : le couloir n'a alors aucun sens, et prétendre le
    contraire écarterait tout.
    """
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(goal, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    direct = float(np.linalg.norm(b - a))
    if direct < 1e-9:
        return 0.0
    return (float(np.linalg.norm(p - a)) + float(np.linalg.norm(p - b))) / direct


def in_routing_zone(start, goal, point, factor: float = DEFAULT_ZONE_FACTOR) -> bool:
    """Le point est-il dans le couloir de cheminement ?"""
    if not factor or factor <= 0:
        return True
    return detour_ratio(start, goal, point) <= float(factor)


def filter_combs(start, goal, combs, factor: float = DEFAULT_ZONE_FACTOR) -> list:
    """Écarte les peignes hors du couloir de cheminement.

    Le détecteur balaie **toute** la maquette : il reconnaît aussi bien les
    peignes qui jalonnent le trajet que ceux montés à l'autre bout de
    l'appareil. Les emprunter tous obligeait le câble à aller les chercher,
    donc à ne plus s'arrêter là où il devait. Un peigne est situé par son
    centre, comme partout ailleurs ici.
    """
    kept = []
    for comb in combs:
        passages = list(comb)
        if passages and in_routing_zone(start, goal, comb_center(passages), factor):
            kept.append(passages)
    return kept


def default_selection(combs, crossings) -> dict:
    """Choix proposé à l'utilisateur : l'encoche que l'application retiendrait.

    Le dictionnaire va du nom du peigne à l'index de l'encoche, ou à ``None``
    pour un peigne qu'on n'emprunte pas. C'est la forme qu'attend
    :func:`choose_crossings`, de sorte que ce que l'utilisateur voit proposé
    est exactement ce qu'il peut modifier.
    """
    retained = {c.comb: c.passage.index for c in crossings}
    return {comb_name(comb): retained.get(comb_name(comb)) for comb in combs if list(comb)}


def _endpoints(passage, flipped: bool):
    p_in = np.asarray(passage.p_in, dtype=np.float64)
    p_out = np.asarray(passage.p_out, dtype=np.float64)
    return (p_out, p_in) if flipped else (p_in, p_out)


def choose_crossings(start, goal, combs, zone_factor: float = DEFAULT_ZONE_FACTOR,
                     selection=None) -> list:
    """Une encoche par peigne, dans l'ordre et le sens les plus courts.

    Args:
        start, goal: extrémités du faisceau.
        combs: un itérable de peignes, chaque peigne étant la liste de ses
            passages. Un peigne vide est ignoré ; un peigne à une seule
            encoche n'offre que le choix du sens.
        zone_factor: largeur du couloir de cheminement (voir
            :data:`DEFAULT_ZONE_FACTOR`). ``0`` ou ``None`` désactive le
            filtrage et reprend tous les peignes détectés.
        selection: choix explicite de l'utilisateur, du nom du peigne vers
            l'index de l'encoche voulue — ou ``None`` pour un peigne à ne pas
            emprunter. Un peigne absent du dictionnaire est laissé au calcul.

    Returns:
        La liste des :class:`Crossing` retenus, dans l'ordre du trajet. Le sens
        de traversée reste calculé même sur une encoche imposée : il n'a aucune
        conséquence physique, et l'utilisateur n'a pas à s'en occuper.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    groups = filter_combs(start, goal, combs, zone_factor)
    if selection is not None:
        wanted = dict(selection)
        kept = []
        for group in groups:
            name = comb_name(group)
            if name not in wanted:
                kept.append(group)
                continue
            index = wanted[name]
            if index is None:
                continue          # peigne écarté par l'utilisateur
            chosen = [p for p in group if p.index == index]
            kept.append(chosen or group)
        groups = kept
    if not groups:
        return []

    direction = goal - start
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    groups.sort(key=lambda g: float(np.dot(comb_center(g) - start, direction)))

    # Un état = une encoche parcourue dans un sens. Deux états par encoche,
    # puisque les deux sens sont permis et qu'on ne sait pas encore lequel sert.
    layers = [
        [(passage, flipped) for passage in group for flipped in (False, True)]
        for group in groups
    ]

    costs: list = []
    backs: list = []
    for depth, layer in enumerate(layers):
        layer_costs, layer_backs = [], []
        for passage, flipped in layer:
            entry, exit_point = _endpoints(passage, flipped)
            traversal = float(np.linalg.norm(exit_point - entry))
            if depth == 0:
                layer_costs.append(float(np.linalg.norm(entry - start)) + traversal)
                layer_backs.append(-1)
                continue
            previous = layers[depth - 1]
            options = [
                costs[depth - 1][j]
                + float(np.linalg.norm(entry - _endpoints(*previous[j])[1]))
                for j in range(len(previous))
            ]
            best = int(np.argmin(options))
            layer_costs.append(options[best] + traversal)
            layer_backs.append(best)
        costs.append(layer_costs)
        backs.append(layer_backs)

    # Le retour vers B fait partie du coût : sans lui, le dernier peigne serait
    # traversé dans le sens qui arrange le peigne précédent, pas le trajet.
    last = layers[-1]
    finals = [
        costs[-1][j] + float(np.linalg.norm(goal - _endpoints(*last[j])[1]))
        for j in range(len(last))
    ]
    index = int(np.argmin(finals))

    chosen = []
    for depth in range(len(layers) - 1, -1, -1):
        passage, flipped = layers[depth][index]
        entry, exit_point = _endpoints(passage, flipped)
        chosen.append(
            Crossing(
                comb=str(getattr(passage, "comb", "") or "?"),
                passage=passage,
                entry=tuple(entry),
                exit=tuple(exit_point),
                flipped=flipped,
            )
        )
        index = backs[depth][index]
    chosen.reverse()
    return chosen


def merge_anchors(start, goal, crossings, points) -> list:
    """Fusionne traversées de peigne et fixations simples, dans l'ordre du trajet.

    Une fixation sans encoche — un clip, un crabe déjà monté — n'impose pas de
    traversée : elle impose un **point**. Elle était purement ignorée, faute
    d'encoche à choisir, et le tracé passait donc à côté de fixations qu'il
    aurait pu emprunter. C'est ce que voit l'utilisateur : un cheminement qui
    coupe au plus court au lieu de suivre la ligne de fixations existante.

    Args:
        start, goal: extrémités du faisceau.
        crossings: traversées retenues, une par peigne.
        points: positions des fixations simples.

    Returns:
        Une liste d'étapes, dans l'ordre : chaque élément est soit un
        :class:`Crossing`, soit un point ``(x, y, z)``.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    direction = goal - start
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])

    def rank(anchor):
        position = (np.asarray(anchor.entry, dtype=np.float64)
                    if isinstance(anchor, Crossing)
                    else np.asarray(anchor, dtype=np.float64))
        return float(np.dot(position - start, direction))

    anchors = list(crossings) + [tuple(float(c) for c in p) for p in (points or [])]
    return sorted(anchors, key=rank)


def describe(crossings, combs=None, lang: str = "FR") -> str:
    """Phrase d'état : combien de peignes empruntés, et sur combien d'encoches."""
    english = str(lang).upper().startswith("EN")
    if not crossings:
        return "No fixation crossed." if english else "Aucune fixation empruntée."

    offered = sum(len(list(comb)) for comb in combs) if combs is not None else 0
    if english:
        text = f"{len(crossings)} comb(s) crossed, one notch each"
        return text + (f" out of {offered} detected." if offered else ".")
    text = f"{len(crossings)} peigne(s) emprunté(s), une encoche chacun"
    return text + (f" sur {offered} détectée(s)." if offered else ".")
