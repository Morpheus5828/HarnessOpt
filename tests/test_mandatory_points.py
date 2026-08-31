"""Passages imposés : le câble doit traverser les encoches, pas les frôler.

L'agent dispose déjà d'une attraction par récompense vers les fixations
existantes. Mesuré sur une vraie boucle, elle ne suffit pas : après deux cents
itérations le câble s'en écarte de 220 à 350 mm. Une encoche de peigne ne se
négocie pas à cette distance, d'où l'épinglage.
"""

import numpy as np
import pytest

from core.agent.tool import snap_mandatory_points


def trace(n=11):
    return np.linspace([0, 0, 0], [100, 0, 0], n).astype(np.float32)


# ----------------------------------------------------------------------
# L'épinglage
# ----------------------------------------------------------------------

def test_le_point_le_plus_proche_est_ramene_sur_la_cible():
    points = trace()
    cible = [52.0, 7.0, 0.0]
    locked = snap_mandatory_points(points, [cible])
    index = next(iter(locked))
    assert np.allclose(points[index], cible)


def test_l_indice_choisi_est_bien_le_plus_proche():
    points = trace()
    locked = snap_mandatory_points(points, [[52.0, 7.0, 0.0]])
    assert locked == {5}          # 50 mm est le point le plus proche de 52


def test_les_extremites_ne_sont_jamais_deplacees():
    """Elles appartiennent aux équipements : elles ne se négocient pas."""
    points = trace()
    depart, arrivee = points[0].copy(), points[-1].copy()
    snap_mandatory_points(points, [[1.0, 0.0, 0.0], [99.0, 0.0, 0.0]])
    assert np.allclose(points[0], depart)
    assert np.allclose(points[-1], arrivee)


def test_deux_passages_ne_partagent_pas_le_meme_point():
    """Sinon l'un des deux serait silencieusement perdu."""
    points = trace()
    locked = snap_mandatory_points(points, [[50.0, 5.0, 0.0], [51.0, -5.0, 0.0]])
    assert len(locked) == 2


def test_chaque_passage_est_atteint_exactement():
    points = trace()
    cibles = [[31.0, -4.0, 0.0], [52.0, 7.0, 0.0], [78.0, 2.0, 0.0]]
    snap_mandatory_points(points, cibles)
    for cible in cibles:
        assert np.linalg.norm(points - np.array(cible), axis=1).min() < 1e-4


def test_une_liste_vide_ne_verrouille_rien():
    points = trace()
    avant = points.copy()
    assert snap_mandatory_points(points, []) == set()
    assert np.allclose(points, avant)


def test_none_est_tolere():
    points = trace()
    assert snap_mandatory_points(points, None) == set()


def test_un_trace_trop_court_ne_leve_pas():
    for n in (0, 1, 2):
        points = np.zeros((n, 3), dtype=np.float32)
        assert snap_mandatory_points(points, [[1.0, 1.0, 1.0]]) == set()


def test_plus_de_passages_que_de_points_libres():
    """Il ne reste que trois points déplaçables pour cinq passages."""
    points = trace(n=5)
    locked = snap_mandatory_points(points, [[float(i), 1.0, 0.0] for i in range(5)])
    assert len(locked) <= 3
    assert 0 not in locked and (len(points) - 1) not in locked


def test_les_indices_deja_reserves_sont_respectes():
    points = trace()
    locked = snap_mandatory_points(points, [[52.0, 7.0, 0.0]], used={5})
    assert 5 in locked
    assert len(locked) == 2       # le point 5 était pris, un autre a été choisi


def test_l_epinglage_est_idempotent():
    """Rappelé à chaque itération, il ne doit pas dériver."""
    points = trace()
    cibles = [[52.0, 7.0, 0.0], [31.0, -4.0, 0.0]]
    premier = snap_mandatory_points(points, cibles)
    apres = points.copy()
    second = snap_mandatory_points(points, cibles)
    assert premier == second
    assert np.allclose(points, apres)


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


def test_les_passages_acceptes_deviennent_des_points_imposes():
    controller = _controleur_avec_scan(2)
    passages = controller._passages_to_use({"use_fixations": True})
    imposes = [list(p) for passage in passages for p in (passage.p_in, passage.p_out)]
    assert len(imposes) == 4


def test_les_passages_refuses_n_imposent_rien():
    controller = _controleur_avec_scan(2)
    assert controller._passages_to_use({"use_fixations": False}) == []


def test_un_refus_laisse_le_trajet_a_deux_etapes():
    controller = _controleur_avec_scan(2)
    nodes, forced = controller._route_nodes({"use_fixations": False})
    assert len(nodes) == 2
    assert forced == set()
