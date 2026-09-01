"""Choix des encoches : une par peigne, dans le sens le plus court.

Le défaut corrigé : le trajet enfilait **toutes** les encoches détectées,
ordonnées par la projection de leur centre sur A→B. Or les encoches d'un même
peigne sont côte à côte et se projettent au même endroit : elles se suivaient
donc dans la liste, et le câble faisait la navette d'une encoche à sa voisine
au lieu de rejoindre le peigne suivant.
"""

import itertools

import numpy as np
import pytest

from core.fixation_scan import Passage
from core.passage_route import (
    DEFAULT_ZONE_FACTOR,
    Crossing,
    choose_crossings,
    comb_center,
    default_selection,
    describe,
    detour_ratio,
    filter_combs,
    in_routing_zone,
)


def peigne(name, x, n=3, pas=50.0, epaisseur=20.0):
    """Un peigne : ``n`` encoches côte à côte, traversées selon z."""
    return [
        Passage(index=i, p_in=(x, i * pas, 0.0), p_out=(x, i * pas, epaisseur),
                comb=name)
        for i in range(n)
    ]


def longueur(start, goal, crossings):
    points = [np.asarray(start, dtype=float)]
    for crossing in crossings:
        points += [np.asarray(crossing.entry, dtype=float),
                   np.asarray(crossing.exit, dtype=float)]
    points.append(np.asarray(goal, dtype=float))
    return float(sum(np.linalg.norm(points[i + 1] - points[i])
                     for i in range(len(points) - 1)))


# ----------------------------------------------------------------------
# Une encoche par peigne
# ----------------------------------------------------------------------

def test_un_peigne_ne_donne_qu_une_traversee():
    """Les autres encoches sont celles des faisceaux voisins."""
    crossings = choose_crossings((0, 0, 0), (2000, 0, 0), [peigne("A", 1000.0, n=5)])
    assert len(crossings) == 1


def test_chaque_peigne_est_emprunte_une_fois():
    combs = [peigne("A", 100.0), peigne("B", 500.0), peigne("C", 900.0)]
    crossings = choose_crossings((0, 0, 0), (1000, 0, 0), combs)
    assert [c.comb for c in crossings] == ["A", "B", "C"]


def test_le_cable_ne_fait_plus_la_navette_sur_un_meme_peigne():
    """Le symptôme signalé : sortir d'une encoche et repartir dans sa voisine."""
    combs = [peigne("avant", 100.0), peigne("arriere", 500.0)]
    crossings = choose_crossings((0, 0, 0), (600, 0, 0), combs)

    empruntes = [c.comb for c in crossings]
    assert empruntes == ["avant", "arriere"], \
        "on quitte un peigne pour le suivant, pas pour l'encoche d'à côté"
    assert len(set(empruntes)) == len(empruntes), "aucun peigne emprunté deux fois"


def test_l_encoche_retenue_est_celle_qui_sert_le_trajet():
    """Le trajet longe y = 0 : c'est l'encoche du bas qu'il faut prendre."""
    crossings = choose_crossings((0, 0, 0), (2000, 0, 0), [peigne("A", 1000.0, n=5)])
    assert crossings[0].passage.index == 0


def test_un_trajet_decale_change_l_encoche_retenue():
    """Le choix suit le trajet, il n'est pas figé sur la première encoche."""
    crossings = choose_crossings((0, 200, 0), (2000, 200, 0), [peigne("A", 1000.0, n=5)])
    assert crossings[0].passage.index == 4


# ----------------------------------------------------------------------
# Le sens de traversée
# ----------------------------------------------------------------------

def test_les_deux_points_sont_interchangeables():
    """Rien n'impose d'entrer par ``p_in`` : seule leur solidarité compte."""
    passage = Passage(index=0, p_in=(100.0, 0.0, 0.0), p_out=(100.0, 0.0, 20.0),
                      comb="A")
    # On arrive par le haut : entrer par p_out et ressortir par p_in est plus
    # court que de contourner pour entrer par p_in.
    crossings = choose_crossings((0, 0, 40), (200, 0, 0), [[passage]])
    assert crossings[0].flipped
    assert crossings[0].entry == passage.p_out
    assert crossings[0].exit == passage.p_in


def test_le_couple_reste_solidaire():
    """Entrer par un point oblige à ressortir par l'autre, jamais par un tiers."""
    combs = [peigne("A", 100.0, n=4), peigne("B", 400.0, n=4)]
    crossings = choose_crossings((0, 0, 0), (500, 0, 0), combs)
    for crossing in crossings:
        couple = {tuple(crossing.passage.p_in), tuple(crossing.passage.p_out)}
        assert {tuple(crossing.entry), tuple(crossing.exit)} == couple
        assert crossing.entry != crossing.exit


def test_le_sens_du_dernier_peigne_tient_compte_de_l_arrivee():
    """Sinon il serait orienté par le peigne précédent, pas par le trajet."""
    passage = Passage(index=0, p_in=(100.0, 0.0, 0.0), p_out=(100.0, 0.0, 100.0),
                      comb="A")
    vers_le_haut = choose_crossings((0, 0, 0), (200, 0, 500), [[passage]])
    vers_le_bas = choose_crossings((0, 0, 0), (200, 0, -500), [[passage]])
    assert vers_le_haut[0].exit == passage.p_out, "on ressort du côté de B"
    assert vers_le_bas[0].exit == passage.p_in


# ----------------------------------------------------------------------
# Optimalité
# ----------------------------------------------------------------------

def force_brute(start, goal, combs):
    """Tous les choix d'encoche et de sens, peignes pris dans le même ordre."""
    meilleur = None
    for choix in itertools.product(*combs):
        for sens in itertools.product([False, True], repeat=len(combs)):
            faux = [
                Crossing(comb=p.comb, passage=p,
                         entry=p.p_out if f else p.p_in,
                         exit=p.p_in if f else p.p_out, flipped=f)
                for p, f in zip(choix, sens)
            ]
            total = longueur(start, goal, faux)
            if meilleur is None or total < meilleur:
                meilleur = total
    return meilleur


def test_le_choix_est_optimal_et_non_glouton():
    """Un choix encoche par encoche rejouerait le défaut d'origine."""
    rng = np.random.default_rng(7)
    start, goal = (0.0, 0.0, 0.0), (1000.0, 0.0, 0.0)
    for _ in range(25):
        combs = []
        for k in range(3):
            base = 200.0 + k * 300.0
            combs.append([
                Passage(index=i,
                        p_in=tuple(rng.uniform(-120, 120, 3) + [base, 0, 0]),
                        p_out=tuple(rng.uniform(-120, 120, 3) + [base, 0, 0]),
                        comb=f"C{k}")
                for i in range(3)
            ])
        crossings = choose_crossings(start, goal, combs)
        assert longueur(start, goal, crossings) == pytest.approx(
            force_brute(start, goal, combs), rel=1e-9)


def test_l_ordre_des_peignes_suit_le_trajet():
    combs = [peigne("loin", 900.0), peigne("proche", 100.0)]
    crossings = choose_crossings((0, 0, 0), (1000, 0, 0), combs)
    assert [c.comb for c in crossings] == ["proche", "loin"]


def test_le_peigne_est_situe_par_son_centre_pas_par_une_encoche():
    """Sinon sa place dans le trajet dépendrait de l'encoche qu'on lui choisit."""
    centre = comb_center(peigne("A", 100.0, n=3, pas=50.0))
    assert centre == pytest.approx([100.0, 50.0, 10.0])


# ----------------------------------------------------------------------
# Cas limites
# ----------------------------------------------------------------------

def test_aucun_peigne_ne_donne_aucune_traversee():
    assert choose_crossings((0, 0, 0), (100, 0, 0), []) == []


def test_un_peigne_vide_est_ignore():
    crossings = choose_crossings((0, 0, 0), (200, 0, 0), [[], peigne("A", 100.0)])
    assert len(crossings) == 1


def test_un_depart_confondu_avec_l_arrivee_ne_leve_pas():
    crossings = choose_crossings((0, 0, 0), (0, 0, 0), [peigne("A", 100.0)])
    assert len(crossings) == 1


def test_le_resume_annonce_peignes_et_encoches():
    combs = [peigne("A", 100.0, n=3), peigne("B", 500.0, n=3)]
    crossings = choose_crossings((0, 0, 0), (600, 0, 0), combs)
    texte = describe(crossings, combs)
    assert "2 peigne(s)" in texte and "6" in texte


def test_le_resume_est_bilingue():
    combs = [peigne("A", 100.0)]
    crossings = choose_crossings((0, 0, 0), (200, 0, 0), combs)
    assert "comb(s)" in describe(crossings, combs, "EN")
    assert describe([], combs, "EN").strip()
    assert describe([], combs, "FR").strip()


def test_la_largeur_de_traversee_est_celle_de_l_encoche():
    crossings = choose_crossings((0, 0, 0), (200, 0, 0), [peigne("A", 100.0, n=1)])
    assert crossings[0].width_mm == pytest.approx(20.0)


def test_les_points_sortent_dans_l_ordre_de_la_marche():
    crossings = choose_crossings((0, 0, 40), (200, 0, 0),
                                 [[Passage(index=0, p_in=(100.0, 0.0, 0.0),
                                           p_out=(100.0, 0.0, 20.0), comb="A")]])
    assert crossings[0].points == [[100.0, 0.0, 20.0], [100.0, 0.0, 0.0]]


# ----------------------------------------------------------------------
# Le couloir de cheminement
# ----------------------------------------------------------------------

def test_un_point_sur_le_segment_ne_coute_aucun_detour():
    assert detour_ratio((0, 0, 0), (1000, 0, 0), (500, 0, 0)) == pytest.approx(1.0)


def test_le_detour_croit_avec_l_ecart():
    proche = detour_ratio((0, 0, 0), (1000, 0, 0), (500, 100, 0))
    loin = detour_ratio((0, 0, 0), (1000, 0, 0), (500, 400, 0))
    assert 1.0 < proche < loin


def peigne_lointain(name="ailleurs", ecart=6000.0):
    """Un peigne reconnu ailleurs dans la maquette, loin du cheminement."""
    return [Passage(index=i, p_in=(1000.0, ecart + i * 50.0, 0.0),
                    p_out=(1000.0, ecart + i * 50.0, 20.0), comb=name)
            for i in range(3)]


def test_un_peigne_a_l_autre_bout_de_la_maquette_est_ecarte():
    """Le détecteur balaie tout le DMU, pas seulement la zone de cheminement."""
    combs = [peigne("sur_le_trajet", 1000.0), peigne_lointain()]
    kept = filter_combs((0, 0, 0), (2000, 0, 0), combs)
    assert len(kept) == 1
    assert kept[0][0].comb == "sur_le_trajet"


def test_un_peigne_hors_zone_n_est_pas_emprunte():
    """Sinon le câble va le chercher et ne s'arrête plus au bon endroit."""
    loin = [Passage(index=0, p_in=(1000.0, 9000.0, 0.0),
                    p_out=(1000.0, 9000.0, 20.0), comb="loin")]
    crossings = choose_crossings((0, 0, 0), (2000, 0, 0),
                                 [peigne("proche", 1000.0), loin])
    assert [c.comb for c in crossings] == ["proche"]


def test_le_couloir_peut_etre_desactive():
    loin = [Passage(index=0, p_in=(1000.0, 9000.0, 0.0),
                    p_out=(1000.0, 9000.0, 20.0), comb="loin")]
    crossings = choose_crossings((0, 0, 0), (2000, 0, 0), [loin], zone_factor=0)
    assert len(crossings) == 1


def test_un_depart_confondu_avec_l_arrivee_n_ecarte_rien():
    """Le couloir n'a alors aucun sens : prétendre le contraire viderait tout."""
    assert detour_ratio((0, 0, 0), (0, 0, 0), (5000, 0, 0)) == 0.0
    assert in_routing_zone((0, 0, 0), (0, 0, 0), (5000, 0, 0))


def test_la_largeur_du_couloir_est_celle_annoncee():
    """Le facteur est un rallongement relatif, pas une distance."""
    limite = (1000.0, 0.0, 0.0)
    assert in_routing_zone((0, 0, 0), (2000, 0, 0), limite, factor=1.0)
    # Un point qui rallonge le trajet de 30 % sort d'un couloir à 1,25.
    ecart = 2000.0 * 0.5 * (DEFAULT_ZONE_FACTOR ** 2 - 1) ** 0.5
    assert in_routing_zone((0, 0, 0), (2000, 0, 0), (1000.0, ecart * 0.98, 0.0))
    assert not in_routing_zone((0, 0, 0), (2000, 0, 0), (1000.0, ecart * 1.02, 0.0))


# ----------------------------------------------------------------------
# Le choix de l'utilisateur
# ----------------------------------------------------------------------

def test_l_utilisateur_impose_son_encoche():
    combs = [peigne("A", 1000.0, n=5)]
    crossings = choose_crossings((0, 0, 0), (2000, 0, 0), combs, selection={"A": 3})
    assert crossings[0].passage.index == 3


def test_le_sens_reste_calcule_sur_une_encoche_imposee():
    """Il n'a aucune conséquence physique : l'utilisateur n'a pas à s'en occuper."""
    passage = Passage(index=0, p_in=(1000.0, 0.0, 0.0), p_out=(1000.0, 0.0, 100.0),
                      comb="A")
    haut = choose_crossings((0, 0, 0), (2000, 0, 500), [[passage]], selection={"A": 0})
    bas = choose_crossings((0, 0, 0), (2000, 0, -500), [[passage]], selection={"A": 0})
    assert haut[0].exit == passage.p_out
    assert bas[0].exit == passage.p_in


def test_un_peigne_refuse_par_l_utilisateur_est_ignore():
    combs = [peigne("A", 700.0), peigne("B", 1300.0)]
    crossings = choose_crossings((0, 0, 0), (2000, 0, 0), combs, selection={"A": None})
    assert [c.comb for c in crossings] == ["B"]


def test_un_peigne_absent_du_choix_reste_calcule():
    combs = [peigne("A", 700.0), peigne("B", 1300.0)]
    crossings = choose_crossings((0, 0, 0), (2000, 0, 0), combs, selection={"A": 2})
    assert [c.comb for c in crossings] == ["A", "B"]
    assert crossings[0].passage.index == 2


def test_tout_refuser_ne_laisse_aucune_traversee():
    combs = [peigne("A", 700.0), peigne("B", 1300.0)]
    assert choose_crossings((0, 0, 0), (2000, 0, 0), combs,
                            selection={"A": None, "B": None}) == []


def test_une_encoche_inconnue_retombe_sur_le_calcul():
    """Un choix périmé ne doit pas faire disparaître le peigne en silence."""
    combs = [peigne("A", 1000.0, n=3)]
    crossings = choose_crossings((0, 0, 0), (2000, 0, 0), combs, selection={"A": 99})
    assert len(crossings) == 1


def test_le_choix_propose_est_celui_que_l_application_retiendrait():
    combs = [peigne("A", 700.0, n=4), peigne("B", 1300.0, n=4)]
    crossings = choose_crossings((0, 0, 0), (2000, 0, 0), combs)
    propose = default_selection(combs, crossings)
    assert set(propose) == {"A", "B"}
    assert propose == {c.comb: c.passage.index for c in crossings}


def test_le_choix_propose_se_rejoue_a_l_identique():
    """Ce qui est proposé doit être exactement ce qu'on peut modifier."""
    combs = [peigne("A", 700.0, n=4), peigne("B", 1300.0, n=4)]
    crossings = choose_crossings((0, 0, 0), (2000, 0, 0), combs)
    rejoue = choose_crossings((0, 0, 0), (2000, 0, 0), combs,
                              selection=default_selection(combs, crossings))
    assert [(c.comb, c.passage.index) for c in rejoue] \
        == [(c.comb, c.passage.index) for c in crossings]


def test_un_peigne_hors_zone_n_est_pas_proposable():
    loin = [Passage(index=0, p_in=(1000.0, 9000.0, 0.0),
                    p_out=(1000.0, 9000.0, 20.0), comb="loin")]
    combs = filter_combs((0, 0, 0), (2000, 0, 0), [peigne("proche", 1000.0), loin])
    assert set(default_selection(combs, choose_crossings((0, 0, 0), (2000, 0, 0), combs))) \
        == {"proche"}
