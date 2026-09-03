"""Conseils de convergence.

Deux propriétés comptent plus que le contenu des messages :

* on ne conseille rien avant d'avoir vraiment cherché — sinon l'application
  pousse à baisser les exigences au premier obstacle ;
* on ne propose jamais un réglage physiquement intenable, ni de relâcher
  l'interdiction de clash.
"""

import numpy as np

from core import diagnostics as dg
from core.routing_rules import ALL_RULES, ClearanceModel, RoutingRules, evaluate_route


def rules(**kwargs):
    base = dict(clearance=ClearanceModel(default_min_mm=10.0, max_mm=100.0))
    base.update(kwargs)
    return RoutingRules(**base)


def zigzag_route(n=14, amplitude=12.0):
    return np.array([[i * 30.0, amplitude if i % 2 else -amplitude, 0.0] for i in range(n)])


def report_for(route, distances=None, **kwargs):
    if distances is None:
        distances = np.full(len(route), 40.0)
    return evaluate_route(route, rules(**kwargs), distances=distances)


# ----------------------------------------------------------------------
# Quand conseiller
# ----------------------------------------------------------------------

def test_rien_avant_le_minimum_d_iterations():
    route = zigzag_route()
    distances = np.full(len(route), 40.0)
    distances[3] = 3.0
    assert dg.analyse(report_for(route, distances), rules(),
                      iterations=dg.MIN_ITERATIONS - 1, stagnant=True) == []


def test_rien_tant_que_le_score_progresse():
    """Des règles enfreintes mais un score qui progresse : on laisse chercher."""
    route = zigzag_route()
    distances = np.full(len(route), 40.0)
    distances[3] = 3.0
    assert dg.analyse(report_for(route, distances), rules(),
                      iterations=500, stagnant=False) == []


def test_rien_sans_rapport():
    assert dg.analyse(None, rules(), iterations=500, stagnant=True) == []


def test_des_conseils_apparaissent_quand_ca_stagne():
    route = zigzag_route()
    distances = np.full(len(route), 40.0)
    distances[3] = 3.0
    assert dg.analyse(report_for(route, distances), rules(),
                      iterations=500, stagnant=True)


def test_une_route_conforme_et_stagnante_invite_a_conclure():
    droit = np.linspace([0, 0, 0], [900, 0, 0], 12)
    conseils = dg.analyse(
        report_for(droit, fixation_pitch_mm=2000.0, target_straight_ratio=0.1),
        rules(fixation_pitch_mm=2000.0, target_straight_ratio=0.1),
        iterations=500, stagnant=True,
    )
    assert [c.key for c in conseils] == ["converged"]
    assert conseils[0].severity == dg.INFO


# ----------------------------------------------------------------------
# Détection de stagnation
# ----------------------------------------------------------------------

def test_pas_de_stagnation_sans_assez_de_releves():
    assert not dg.is_stagnant([(5,), (4,), (3,)])


def test_un_score_qui_s_ameliore_n_est_pas_stagnant():
    scores = [(n,) for n in range(30, 30 - dg.STAGNATION_WINDOW - 2, -1)]
    assert not dg.is_stagnant(scores)


def test_un_score_immobile_est_stagnant():
    assert dg.is_stagnant([(7,)] * (dg.STAGNATION_WINDOW + 3))


def test_une_amelioration_recente_rompt_la_stagnation():
    scores = [(7,)] * (dg.STAGNATION_WINDOW + 3) + [(2,)]
    assert not dg.is_stagnant(scores)


# ----------------------------------------------------------------------
# Le clash ne se négocie pas
# ----------------------------------------------------------------------

def test_un_clash_ne_propose_aucun_reglage():
    route = np.linspace([0, 0, 0], [900, 0, 0], 12)
    inside = np.zeros(len(route), dtype=bool)
    inside[4] = True
    report = evaluate_route(route, rules(), distances=np.full(len(route), 40.0),
                            inside_mask=inside, n_crossings=2)
    conseils = {c.key: c for c in dg.analyse(report, rules(), iterations=500, stagnant=True)}
    assert "clash" in conseils
    assert conseils["clash"].setting is None
    assert not conseils["clash"].is_applicable


def test_le_conseil_de_clash_est_bloquant_et_premier():
    route = np.linspace([0, 0, 0], [900, 0, 0], 12)
    inside = np.zeros(len(route), dtype=bool)
    inside[4] = True
    report = evaluate_route(route, rules(), distances=np.full(len(route), 40.0),
                            inside_mask=inside, n_crossings=2)
    conseils = dg.analyse(report, rules(), iterations=500, stagnant=True)
    assert conseils[0].severity == dg.BLOCKING


# ----------------------------------------------------------------------
# Les propositions chiffrées
# ----------------------------------------------------------------------

def test_une_distance_mini_intenable_propose_de_la_baisser():
    """Le seul sens qui débloque : l'augmenter durcirait le problème."""
    route = np.linspace([0, 0, 0], [900, 0, 0], 12)
    distances = np.full(len(route), 40.0)
    distances[5] = 6.4
    conseils = {c.key: c for c in dg.analyse(
        report_for(route, distances), rules(), iterations=500, stagnant=True)}
    conseil = conseils["clearance_min"]
    assert conseil.setting == "min_margin"
    assert conseil.value == 6.0
    assert conseil.value < 10.0


def test_une_distance_maxi_depassee_propose_de_l_augmenter():
    route = np.linspace([0, 0, 0], [900, 0, 0], 12)
    distances = np.full(len(route), 40.0)
    distances[5] = 143.0
    conseils = {c.key: c for c in dg.analyse(
        report_for(route, distances), rules(), iterations=500, stagnant=True)}
    conseil = conseils["clearance_max"]
    assert conseil.setting == "max_margin"
    assert conseil.value == 150.0
    assert conseil.value > 100.0


def test_un_frolement_ne_propose_pas_d_accepter_le_contact():
    """Sous le millimètre, on ne règle plus une marge : on accepte un contact."""
    route = np.linspace([0, 0, 0], [900, 0, 0], 12)
    distances = np.full(len(route), 40.0)
    distances[5] = 0.3
    conseils = {c.key: c for c in dg.analyse(
        report_for(route, distances), rules(), iterations=500, stagnant=True)}
    assert not conseils["clearance_min"].is_applicable


def test_un_cintrage_intenable_ne_propose_pas_un_facteur_impossible():
    """Proposer 1 × Ø donnerait une route « conforme » mais impossible à poser."""
    serre = np.array([[0, 0, 0], [300, 0, 0], [310, 5, 0], [320, 60, 0], [330, 300, 0]], float)
    conseils = {c.key: c for c in dg.analyse(
        report_for(serre, np.full(len(serre), 40.0)),
        rules(), iterations=500, stagnant=True)}
    conseil = conseils.get("bend_radius")
    assert conseil is not None
    if conseil.is_applicable:
        assert conseil.value >= dg.MIN_PHYSICAL_BEND_FACTOR


def test_un_pas_de_fixation_intenable_propose_de_l_elargir():
    route = np.linspace([0, 0, 0], [2000, 0, 0], 12)
    conseils = {c.key: c for c in dg.analyse(
        report_for(route, np.full(len(route), 40.0)),
        rules(), iterations=500, stagnant=True)}
    conseil = conseils["fixation_pitch"]
    assert conseil.setting == "fixation_pitch"
    assert conseil.value > 250.0


def test_la_traversee_a_vide_pointe_le_pas_de_fixation():
    """C'est lui qui fixe la limite : proposer un réglage inexistant n'aiderait pas."""
    route = np.linspace([0, 0, 0], [2000, 0, 0], 12)
    distances = np.full(len(route), 400.0)
    conseils = {c.key: c for c in dg.analyse(
        report_for(route, distances), rules(), iterations=500, stagnant=True)}
    if "free_span" in conseils:
        assert conseils["free_span"].setting == "fixation_pitch"


def test_chaque_conseil_applicable_designe_un_reglage_connu():
    """Un conseil qui pointerait un champ inexistant serait inapplicable."""
    connus = {"min_margin", "max_margin", "bend_radius_factor",
              "fixation_pitch", "fixation_parallel_tol"}
    route = zigzag_route()
    distances = np.full(len(route), 40.0)
    distances[3] = 4.0
    distances[7] = 180.0
    for conseil in dg.analyse(report_for(route, distances), rules(),
                              iterations=500, stagnant=True):
        if conseil.is_applicable:
            assert conseil.setting in connus, conseil.key


# ----------------------------------------------------------------------
# Modèle de crabe
# ----------------------------------------------------------------------

def test_un_modele_de_crabe_absent_est_signale_immediatement():
    """Sans attendre la stagnation : la règle ne pourra jamais passer."""
    conseils = dg.analyse(None, rules(), iterations=0, stagnant=False, clamp_model_ok=False)
    assert [c.key for c in conseils] == ["clamp_model"]
    assert conseils[0].severity == dg.BLOCKING


def test_pas_d_alerte_crabe_si_les_regles_de_fixation_sont_decochees():
    sans_fixation = rules(enabled_rules=ALL_RULES - {"fixation_pitch", "fixation_parallel"})
    assert dg.analyse(None, sans_fixation, iterations=0,
                      stagnant=False, clamp_model_ok=False) == []


# ----------------------------------------------------------------------
# Présentation
# ----------------------------------------------------------------------

def test_les_conseils_sont_tries_du_plus_grave_au_moins_grave():
    route = zigzag_route()
    distances = np.full(len(route), 40.0)
    distances[3] = 4.0
    conseils = dg.analyse(report_for(route, distances), rules(),
                          iterations=500, stagnant=True)
    rang = {dg.BLOCKING: 0, dg.MAJOR: 1, dg.INFO: 2}
    rangs = [rang[c.severity] for c in conseils]
    assert rangs == sorted(rangs)


def test_chaque_conseil_est_bilingue():
    route = zigzag_route()
    distances = np.full(len(route), 40.0)
    distances[3] = 4.0
    for conseil in dg.analyse(report_for(route, distances), rules(),
                              iterations=500, stagnant=True):
        for lang in ("FR", "EN"):
            assert conseil.title(lang).strip()
            assert conseil.detail(lang).strip()
            assert conseil.action(lang).strip()
        assert conseil.title("FR") != conseil.title("EN")


def test_le_conseil_de_rectitude_mentionne_les_zigzags():
    conseils = {c.key: c for c in dg.analyse(
        report_for(zigzag_route(), np.full(14, 40.0)),
        rules(), iterations=500, stagnant=True)}
    conseil = conseils.get("straightness")
    assert conseil is not None
    assert "inversion" in conseil.detail("FR")
