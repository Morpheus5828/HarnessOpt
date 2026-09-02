"""Passer par le maximum de fixations, et pas seulement par les peignes.

Le défaut signalé, image à l'appui : le tracé coupe au plus court alors qu'une
ligne de fixations existantes l'attendait. Une fixation sans encoche — un
clip, un crabe déjà monté — était **purement ignorée**, faute d'encoche à
choisir. Elle n'impose pourtant pas moins un point de passage qu'un peigne.
"""

import numpy as np
import pytest

from core.fixation_scan import Passage, summarise
from core.passage_route import Crossing, choose_crossings, merge_anchors

A = (0.0, 0.0, 0.0)
B = (2000.0, 0.0, 0.0)


def peigne(nom, x, n=3):
    return [Passage(index=i, p_in=(x, i * 40.0, 0.0), p_out=(x, i * 40.0, 20.0),
                    comb=nom)
            for i in range(n)]


# ----------------------------------------------------------------------
# Fusionner peignes et fixations simples
# ----------------------------------------------------------------------

def test_les_deux_familles_se_melangent_dans_l_ordre_du_trajet():
    crossings = choose_crossings(A, B, [peigne("milieu", 1000.0)])
    anchors = merge_anchors(A, B, crossings, [(400.0, 0.0, 0.0), (1600.0, 0.0, 0.0)])
    abscisses = [
        float(a.entry[0]) if isinstance(a, Crossing) else float(a[0]) for a in anchors
    ]
    assert abscisses == sorted(abscisses)
    assert len(anchors) == 3


def test_une_fixation_simple_reste_un_point():
    """Elle impose un passage, pas une traversée : rien à y franchir."""
    anchors = merge_anchors(A, B, [], [(500.0, 0.0, 0.0)])
    assert not isinstance(anchors[0], Crossing)
    assert anchors[0] == (500.0, 0.0, 0.0)


def test_sans_rien_a_emprunter_la_liste_est_vide():
    assert merge_anchors(A, B, [], []) == []
    assert merge_anchors(A, B, [], None) == []


def test_un_depart_confondu_avec_l_arrivee_ne_leve_pas():
    assert len(merge_anchors(A, A, [], [(5.0, 0.0, 0.0)])) == 1


# ----------------------------------------------------------------------
# Ce que le contrôleur en fait
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


def controleur(fixations):
    from controller.app_controller import AppController

    controller = AppController(_Vue())
    controller.point_a = np.array(A, dtype=np.float32)
    controller.point_b = np.array(B, dtype=np.float32)
    controller.scan_result = summarise(fixations)
    return controller


def clip(nom, position):
    return {"name": nom, "position": list(position), "score": 0.85}


def comb_brut(nom, x, n=3):
    points = []
    for i in range(n):
        points += [[x, i * 40.0, 0.0], [x, i * 40.0, 20.0]]
    return {"name": nom, "position": [x, 40.0, 10.0], "score": 0.9,
            "routing_points": points}


def test_un_clip_devient_un_point_de_passage():
    controller = controleur([clip("clip_a.stl", (600.0, 0.0, 0.0))])
    assert controller._fixation_points_to_use({"use_fixations": True}) \
        == [(600.0, 0.0, 0.0)]


def test_un_clip_entre_dans_le_trajet():
    """C'est le défaut corrigé : il n'y entrait pas du tout."""
    controller = controleur([clip("clip_a.stl", (600.0, 0.0, 0.0)),
                             clip("clip_b.stl", (1400.0, 0.0, 0.0))])
    nodes, forced = controller._route_nodes({"use_fixations": True})
    assert len(nodes) == 4, "A, deux clips, B"
    assert forced == set(), "un clip ne se traverse pas"
    assert [float(n[0]) for n in nodes] == [0.0, 600.0, 1400.0, 2000.0]


def test_clips_et_peignes_s_enchainent_dans_l_ordre():
    controller = controleur([
        clip("clip_avant.stl", (400.0, 0.0, 0.0)),
        comb_brut("peigne.stl", 1000.0),
        clip("clip_arriere.stl", (1600.0, 0.0, 0.0)),
    ])
    nodes, forced = controller._route_nodes({"use_fixations": True})
    abscisses = [float(n[0]) for n in nodes]
    assert abscisses == sorted(abscisses)
    assert len(nodes) == 6, "A, clip, entrée, sortie, clip, B"
    assert len(forced) == 1, "seule l'encoche se traverse en ligne droite"


def test_un_clip_hors_zone_est_ecarte():
    """Le détecteur balaie tout le DMU, pas seulement la zone de cheminement."""
    controller = controleur([clip("ailleurs.stl", (1000.0, 9000.0, 0.0))])
    assert controller._fixation_points_to_use({"use_fixations": True}) == []


def test_un_refus_ecarte_aussi_les_clips():
    controller = controleur([clip("clip_a.stl", (600.0, 0.0, 0.0))])
    assert controller._fixation_points_to_use({"use_fixations": False}) == []


def test_un_scan_non_effectue_n_impose_aucun_clip():
    from core.fixation_scan import NO_OPEN3D, ScanResult

    controller = controleur([])
    controller.scan_result = ScanResult(skipped_reason=NO_OPEN3D)
    assert controller._fixation_points_to_use({"use_fixations": True}) == []


def test_les_clips_partent_epingles_vers_les_agents():
    """L'attraction par récompense ne garantit pas plus le passage ici."""
    controller = controleur([clip("clip_a.stl", (600.0, 0.0, 0.0))])
    controller.fixation_points = controller._fixation_points_to_use(
        {"use_fixations": True}
    )
    config = {}
    controller._publish_pinned(config)
    assert config["pinned_points"] == [[600.0, 0.0, 0.0]]


def test_fixations_et_points_manuels_partagent_la_meme_liste():
    """L'agent n'a pas à savoir d'où vient une contrainte pour la respecter."""
    controller = controleur([clip("clip_a.stl", (600.0, 0.0, 0.0))])
    controller.fixation_points = [(600.0, 0.0, 0.0)]
    controller.pinned_points = [[900.0, 50.0, 0.0]]
    config = {}
    controller._publish_pinned(config)
    assert config["pinned_points"] == [[600.0, 0.0, 0.0], [900.0, 50.0, 0.0]]
