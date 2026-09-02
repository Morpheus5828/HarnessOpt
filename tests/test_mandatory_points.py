"""Passages imposés : le câble doit traverser les encoches, pas les frôler.

L'agent dispose déjà d'une attraction par récompense vers les fixations
existantes. Mesuré sur une vraie boucle, elle ne suffit pas : après deux cents
itérations le câble s'en écarte de 220 à 350 mm. Une encoche de peigne ne se
négocie pas à cette distance, d'où l'épinglage.
"""

import numpy as np
import pytest

from core.agent.tool import snap_comb_passages


def trace(n=11):
    return np.linspace([0, 0, 0], [100, 0, 0], n).astype(np.float32)


# ----------------------------------------------------------------------
# Transmission depuis le contrôleur
# ----------------------------------------------------------------------

class _Traducteur:
    lang = "FR"
    is_english = False

    def __call__(self, key, **kwargs):
        return key


class _Vue:
    def __init__(self):
        self.t = _Traducteur()
        self.messages = []

    def after(self, _delay, callback):
        callback()

    def set_status(self, message, tone="neutral"):
        self.messages.append((tone, message))


def _controleur_avec_scan(n_passages=2):
    from controller.app_controller import AppController
    from core.fixation_scan import summarise

    points = []
    for i in range(n_passages):
        points.append([float(i * 100), -40.0, 0.0])
        points.append([float(i * 100), 40.0, 0.0])

    controller = AppController(_Vue())
    controller.point_a = np.array([-500.0, 0.0, 0.0], dtype=np.float32)
    controller.point_b = np.array([500.0, 0.0, 0.0], dtype=np.float32)
    controller.scan_result = summarise([{
        "name": "peigne.stl", "position": [0.0, 0.0, 0.0], "score": 0.9,
        "routing_points": points,
    }])
    return controller


def test_une_encoche_acceptee_devient_un_couple_impose():
    controller = _controleur_avec_scan(2)
    crossings = controller._crossings_to_use({"use_fixations": True})
    imposes = [point for crossing in crossings for point in crossing.points]
    # Deux encoches détectées sur *un* peigne : une seule est empruntée.
    assert len(crossings) == 1
    assert len(imposes) == 2


def test_les_encoches_detectees_restent_toutes_affichables():
    """Une seule est empruntée, mais l'utilisateur doit voir les autres."""
    controller = _controleur_avec_scan(2)
    assert len(controller._passages_to_use({"use_fixations": True})) == 2


def test_les_passages_refuses_n_imposent_rien():
    controller = _controleur_avec_scan(2)
    assert controller._passages_to_use({"use_fixations": False}) == []
    assert controller._crossings_to_use({"use_fixations": False}) == []


def test_un_refus_laisse_le_trajet_a_deux_etapes():
    controller = _controleur_avec_scan(2)
    nodes, forced = controller._route_nodes({"use_fixations": False})
    assert len(nodes) == 2
    assert forced == set()


# ----------------------------------------------------------------------
# C'est l'agent qui choisit son encoche
# ----------------------------------------------------------------------

def peigne(x=600.0, ys=(0.0, 40.0, 80.0, 120.0, 160.0), epaisseur=20.0):
    """Un peigne : plusieurs encoches côte à côte selon y, traversées selon z."""
    return np.array([[[x, y, 0.0], [x, y, epaisseur]] for y in ys], dtype=np.float32)


def cable(y, n=25):
    return np.linspace([0.0, y, 10.0], [1200.0, y, 10.0], n).astype(np.float32)


def encoche_retenue(waypoints, locked):
    return float(waypoints[min(locked)][1])


def test_l_encoche_retenue_est_la_plus_proche_du_cable():
    """L'agent choisit en déplaçant le câble, pas en obéissant à un calcul."""
    for y, attendu in ((0.0, 0.0), (45.0, 40.0), (90.0, 80.0), (155.0, 160.0)):
        points = cable(y)
        locked = snap_comb_passages(points, [peigne()])
        assert encoche_retenue(points, locked) == attendu


def test_le_choix_suit_le_cable_d_une_iteration_a_l_autre():
    """C'est tout l'intérêt : l'agent peut changer d'encoche en chemin."""
    points = cable(10.0)
    premier = encoche_retenue(points, snap_comb_passages(points, [peigne()]))
    points = cable(150.0)
    second = encoche_retenue(points, snap_comb_passages(points, [peigne()]))
    assert premier != second


def test_un_seul_couple_est_retenu_par_peigne():
    points = cable(45.0)
    locked = snap_comb_passages(points, [peigne()])
    assert len(locked) == 2


def test_le_couple_retenu_reste_entier():
    """Entrer dans une encoche et ressortir par une autre est impossible."""
    points = cable(85.0)
    locked = sorted(snap_comb_passages(points, [peigne()]))
    a, b = tuple(points[locked[0]]), tuple(points[locked[1]])
    couples = {(tuple(c[0]), tuple(c[1])) for c in peigne()}
    assert (a, b) in couples or (b, a) in couples


def test_les_deux_points_restent_consecutifs():
    points = cable(45.0)
    locked = sorted(snap_comb_passages(points, [peigne()]))
    assert locked[1] == locked[0] + 1


def test_le_sens_de_traversee_suit_le_cable():
    """p_in et p_out sont réversibles : il n'y a rien à retourner dans une encoche.

    On juge sur une encoche dont l'axe suit le sens de marche : le câble doit
    entrer par le point qu'il rencontre en premier, quel que soit celui que le
    détecteur a nommé ``p_in``.
    """
    unique = np.array([[[590.0, 0.0, 0.0], [610.0, 0.0, 0.0]]], dtype=np.float32)

    aller = np.linspace([0.0, 0.0, 0.0], [1200.0, 0.0, 0.0], 25).astype(np.float32)
    locked = sorted(snap_comb_passages(aller, [unique]))
    assert aller[locked[0]][0] < aller[locked[1]][0], "on traverse dans le sens de marche"

    retour = np.linspace([1200.0, 0.0, 0.0], [0.0, 0.0, 0.0], 25).astype(np.float32)
    locked = sorted(snap_comb_passages(retour, [unique]))
    assert retour[locked[0]][0] > retour[locked[1]][0], "et dans l'autre sens à rebours"


def test_deux_peignes_donnent_deux_traversees():
    points = np.linspace([0.0, 45.0, 10.0], [2000.0, 45.0, 10.0], 40).astype(np.float32)
    locked = snap_comb_passages(points, [peigne(x=600.0), peigne(x=1400.0)])
    assert len(locked) == 4


def test_les_peignes_restent_dans_l_ordre_du_trajet():
    points = np.linspace([0.0, 45.0, 10.0], [2000.0, 45.0, 10.0], 40).astype(np.float32)
    locked = sorted(snap_comb_passages(points, [peigne(x=600.0), peigne(x=1400.0)]))
    assert points[locked[0]][0] < points[locked[2]][0]


def test_les_extremites_restent_intouchees():
    points = cable(45.0)
    depart, arrivee = points[0].copy(), points[-1].copy()
    snap_comb_passages(points, [peigne()])
    assert np.allclose(points[0], depart) and np.allclose(points[-1], arrivee)


def test_les_indices_reserves_sont_evites():
    points = cable(45.0, n=25)
    locked = snap_comb_passages(points, [peigne()], used={12, 13})
    assert {12, 13} <= locked and len(locked) == 4


def test_l_epinglage_par_peigne_est_idempotent():
    """Rappelé à chaque itération, il ne doit pas dériver."""
    points = cable(45.0)
    premier = snap_comb_passages(points, [peigne()])
    apres = points.copy()
    assert snap_comb_passages(points, [peigne()]) == premier
    assert np.allclose(points, apres)


def test_aucun_peigne_ne_verrouille_rien():
    points = cable(45.0)
    avant = points.copy()
    assert snap_comb_passages(points, None) == set()
    assert snap_comb_passages(points, []) == set()
    assert np.allclose(points, avant)


def test_un_peigne_vide_est_ignore():
    points = cable(45.0)
    assert len(snap_comb_passages(points, [np.zeros((0, 2, 3)), peigne()])) == 2


def test_un_trace_trop_court_ne_leve_pas():
    for n in (0, 1, 2, 3):
        points = np.zeros((n, 3), dtype=np.float32)
        assert snap_comb_passages(points, [peigne()]) == set()


def test_l_agent_recoit_bien_toutes_les_encoches():
    """Un peigne réduit à une encoche au lancement rendrait le choix illusoire."""
    source = open("core/agent_worker.py", encoding="utf-8").read()
    assert 'cfg.get("mandatory_combs")' in source
    assert "snap_comb_passages(wp_current, mandatory_combs)" in source

    controleur = open("controller/app_controller.py", encoding="utf-8").read()
    assert 'config["mandatory_combs"]' in controleur
    assert "for passage in comb" in controleur
