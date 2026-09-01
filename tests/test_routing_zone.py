"""Le couloir de cheminement : ce qui n'y est pas n'a rien à y faire.

Deux défauts de même origine. Les peignes reconnus à l'autre bout de la
maquette étaient empruntés, obligeant le câble à aller les chercher. Et rien
ne retenait un point de trajectoire au-delà de l'arrivée : ``detour_penalty``
juge la longueur **totale** et répartit sa sanction uniformément, si bien
qu'un point parti trois mètres trop loin n'y contribue guère plus que ses
voisins restés en place — il ne reçoit donc aucun signal qui lui dise de
revenir.

Une seule notion de zone sert aux deux : l'ellipsoïde de foyers A et B.
"""

import numpy as np
import pytest

from core.passage_route import DEFAULT_ZONE_FACTOR, detour_ratio
from core.reward_terms import zone_penalty

A = (0.0, 0.0, 0.0)
B = (2000.0, 0.0, 0.0)


def penalite(point, **kwargs):
    return float(zone_penalty([point], A, B, **kwargs)[0])


# ----------------------------------------------------------------------
# Un tracé conforme ne paie rien
# ----------------------------------------------------------------------

def test_un_trace_droit_ne_paie_rien():
    droit = np.linspace(A, B, 40)
    assert np.all(zone_penalty(droit, A, B) == 0.0)


def test_les_extremites_ne_sont_jamais_sanctionnees():
    """Elles sont sur l'ellipse de rapport 1, quel que soit le facteur."""
    for facteur in (1.0, 1.25, 3.0):
        assert penalite(A, factor=facteur) == 0.0
        assert penalite(B, factor=facteur) == 0.0


def test_un_ecart_raisonnable_reste_gratuit():
    """S'écarter pour respecter les distances ne doit rien coûter."""
    assert penalite([1000.0, 300.0, 0.0]) == 0.0


# ----------------------------------------------------------------------
# Sortir du couloir coûte
# ----------------------------------------------------------------------

def test_un_point_au_dela_de_l_arrivee_est_sanctionne():
    """Le défaut signalé : la trajectoire dépasse la zone de destination."""
    assert penalite([2500.0, 0.0, 0.0]) < 0.0


def test_la_sanction_croit_avec_le_depassement():
    proche = penalite([2400.0, 0.0, 0.0])
    loin = penalite([2800.0, 0.0, 0.0])
    assert loin < proche < 0.0


def test_la_sanction_sature():
    """Elle ne doit pas écraser toutes les autres règles."""
    assert penalite([9000.0, 0.0, 0.0]) == penalite([90000.0, 0.0, 0.0])
    assert penalite([9000.0, 0.0, 0.0]) == pytest.approx(-90.0)


def test_la_sanction_est_par_point_et_non_diluee():
    """C'est ce qui manquait au détour : un signal local, pas une moyenne."""
    trace = np.linspace(A, B, 20).tolist()
    trace[10] = [3000.0, 0.0, 0.0]
    penalites = zone_penalty(trace, A, B)
    assert penalites[10] < 0.0
    assert np.all(penalites[[i for i in range(20) if i != 10]] == 0.0)


def test_le_poids_commande_l_amplitude():
    assert penalite([9000.0, 0.0, 0.0], weight=10.0) == pytest.approx(-10.0)


def test_un_couloir_plus_large_tolere_plus():
    point = [2500.0, 0.0, 0.0]
    assert penalite(point, factor=1.25) < 0.0
    assert penalite(point, factor=2.0) == 0.0


# ----------------------------------------------------------------------
# La même zone que pour les peignes
# ----------------------------------------------------------------------

def test_la_zone_est_celle_qui_filtre_les_peignes():
    """Deux notions de « zone » finiraient par diverger."""
    for point in ([1000.0, 500.0, 0.0], [2500.0, 0.0, 0.0], [1000.0, 0.0, 0.0]):
        dehors = detour_ratio(A, B, point) > DEFAULT_ZONE_FACTOR
        assert (penalite(point) < 0.0) is dehors


# ----------------------------------------------------------------------
# Cas limites
# ----------------------------------------------------------------------

def test_un_depart_confondu_avec_l_arrivee_ne_sanctionne_rien():
    """Le couloir n'a alors aucun sens : il condamnerait tout le tracé."""
    assert np.all(zone_penalty([[5000.0, 0.0, 0.0]], A, A) == 0.0)


def test_un_trace_vide_ne_leve_pas():
    assert len(zone_penalty([], A, B)) == 0


def test_le_terme_est_branche_sur_l_agent():
    """Un terme non appelé ne corrige rien."""
    source = open("core/agent_worker.py", encoding="utf-8").read()
    assert "rwt.zone_penalty(" in source
    assert "+ R_zone" in source
    assert '"Zone": float(R_zone.mean())' in source
