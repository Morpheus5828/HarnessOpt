"""Règles activables : décocher une règle doit vraiment la retirer du problème.

Le piège serait qu'une règle décochée disparaisse du rapport tout en
continuant à peser sur le classement ou sur la récompense : l'utilisateur
verrait alors les agents s'acharner sur une contrainte qu'il pense avoir levée.
Ces tests vérifient les trois effets ensemble.
"""

import numpy as np
import pytest

from core.routing_rules import (
    ALL_RULES,
    RULE_CATALOG,
    RULE_IDS,
    ClearanceModel,
    RouteReport,
    RoutingRules,
    Severity,
    evaluate_route,
    rule_info,
)


# ----------------------------------------------------------------------
# Le catalogue
# ----------------------------------------------------------------------

def test_le_catalogue_et_les_identifiants_ne_divergent_pas():
    """C'est la seule liste qui fasse foi : une règle cochable est appliquée."""
    assert set(RULE_IDS) == set(ALL_RULES)
    assert len(RULE_IDS) == len(RULE_CATALOG)


def test_chaque_regle_du_catalogue_est_unique():
    assert len({info.rule_id for info in RULE_CATALOG}) == len(RULE_CATALOG)


def test_chaque_regle_est_documentee_dans_les_deux_langues():
    for info in RULE_CATALOG:
        for lang in ("FR", "EN"):
            assert info.label(lang).strip()
            assert info.help(lang).strip()
        assert info.label("FR") != info.label("EN")


def test_chaque_regle_a_une_gravite_connue():
    valid = {Severity.BLOCKING, Severity.MAJOR, Severity.MINOR}
    assert all(info.severity in valid for info in RULE_CATALOG)


def test_les_identifiants_sont_uniques():
    assert len(RULE_IDS) == len(set(RULE_IDS))


def test_une_regle_inconnue_ne_leve_pas():
    assert rule_info("regle_qui_n_existe_pas") is None


# ----------------------------------------------------------------------
# Le jeu de règles
# ----------------------------------------------------------------------

def test_toutes_les_regles_sont_actives_par_defaut():
    assert RoutingRules().enabled_rules == ALL_RULES


def test_un_identifiant_inconnu_est_ignore():
    """Un réglage enregistré par une version antérieure doit rester lisible."""
    rules = RoutingRules(enabled_rules={"clash", "regle_disparue"})
    assert rules.enabled_rules == frozenset({"clash"})


def test_une_liste_est_acceptee_comme_un_ensemble():
    rules = RoutingRules(enabled_rules=["clash", "bend_radius"])
    assert rules.is_enabled("clash")
    assert not rules.is_enabled("straightness")


def test_with_rules_ne_modifie_pas_l_original():
    base = RoutingRules()
    restreint = base.with_rules({"clash"})
    assert base.enabled_rules == ALL_RULES
    assert restreint.enabled_rules == frozenset({"clash"})


def test_les_autres_reglages_survivent_au_changement_de_regles():
    base = RoutingRules(fixation_pitch_mm=180.0)
    assert base.with_rules({"clash"}).fixation_pitch_mm == 180.0


# ----------------------------------------------------------------------
# L'échelle de récompense
# ----------------------------------------------------------------------

def test_toutes_les_familles_sont_actives_par_defaut():
    scale = RoutingRules().reward_scale()
    assert scale and all(value == 1.0 for value in scale.values())


def test_une_famille_est_neutralisee_quand_toutes_ses_regles_tombent():
    rules = RoutingRules(enabled_rules=ALL_RULES - {"fixation_pitch", "fixation_parallel"})
    assert rules.reward_scale()["fixation"] == 0.0


def test_une_famille_survit_a_la_perte_d_une_seule_de_ses_regles():
    """Décocher la seule distance maximale ne doit pas supprimer toute la
    pression de distance : la distance minimale, elle, reste demandée."""
    rules = RoutingRules(enabled_rules=ALL_RULES - {"clearance_max"})
    assert rules.reward_scale()["clearance"] == 1.0


def test_toutes_les_familles_tombent_si_aucune_regle():
    scale = RoutingRules(enabled_rules=set()).reward_scale()
    assert all(value == 0.0 for value in scale.values())


# ----------------------------------------------------------------------
# L'évaluation
# ----------------------------------------------------------------------

@pytest.fixture
def route():
    """Un tracé volontairement médiocre : coudes serrés, et trop près du DMU."""
    return np.array(
        [[-200, 0, 0], [-100, 0, 0], [-100, 120, 0], [0, 120, 0],
         [0, 0, 0], [100, 0, 0], [100, 90, 0], [200, 90, 0]],
        dtype=float,
    )


@pytest.fixture
def distances(route):
    """Distances au DMU : deux points sous la distance minimale de 10 mm."""
    values = np.full(len(route), 40.0)
    values[2] = 3.0
    values[5] = 6.0
    return values


def evaluate(route, distances, enabled):
    rules = RoutingRules(
        clearance=ClearanceModel(default_min_mm=10.0, max_mm=100.0),
        enabled_rules=enabled,
    )
    return evaluate_route(route, rules, distances=distances)


def test_le_rapport_ne_contient_que_les_regles_cochees(route, distances):
    report = evaluate(route, distances, {"clash", "bend_radius"})
    assert {check.rule_id for check in report.checks} <= {"clash", "bend_radius"}
    assert report.checks, "les règles cochées doivent bien être évaluées"


def test_toutes_les_regles_apparaissent_par_defaut(route, distances):
    report = evaluate(route, distances, ALL_RULES)
    assert len(report.checks) >= 5


def test_les_indicateurs_restent_calcules_meme_decoches(route, distances):
    """L'utilisateur veut souvent continuer à voir la valeur mesurée."""
    report = evaluate(route, distances, {"clash"})
    assert "n_bend_violations" in report.kpis
    assert "length_mm" in report.kpis


def test_decocher_une_regle_enfreinte_rend_la_route_conforme(route, distances):
    complet = evaluate(route, distances, ALL_RULES)
    enfreintes = {check.rule_id for check in complet.failed()}
    assert enfreintes, "le tracé de test doit enfreindre au moins une règle"

    restreint = evaluate(route, distances, ALL_RULES - enfreintes)
    assert restreint.is_compliant


def test_la_distance_minimale_est_bien_enfreinte_puis_levee(route, distances):
    """Vérifie sur une règle précise, pas seulement sur le compte global."""
    avec = evaluate(route, distances, ALL_RULES)
    assert any(not c.passed and c.rule_id == "clearance_min" for c in avec.checks)

    sans = evaluate(route, distances, ALL_RULES - {"clearance_min"})
    assert all(c.rule_id != "clearance_min" for c in sans.checks)


def test_le_rapport_serialise_dit_ce_qui_a_ete_ecarte(route, distances):
    data = evaluate(route, distances, ALL_RULES - {"straightness"}).to_dict()
    assert data["disabled_rules"] == ["straightness"]
    assert "straightness" not in data["enabled_rules"]


def test_la_conformite_ignore_les_regles_decochees(route, distances):
    report = evaluate(route, distances, set())
    assert report.checks == []
    assert report.is_compliant
    assert report.is_deliverable
    assert report.compliance_ratio == 1.0


# ----------------------------------------------------------------------
# Le classement
# ----------------------------------------------------------------------

def make_report(kpis, enabled=None):
    return RouteReport(checks=[], kpis=kpis, enabled_rules=enabled)


def test_une_regle_decochee_ne_pese_plus_sur_le_classement():
    kpis = {"n_clashes": 0, "n_bend_violations": 7, "length_mm": 1000.0}
    avec = make_report(kpis, ALL_RULES).score()
    sans = make_report(kpis, ALL_RULES - {"bend_radius"}).score()
    assert avec != sans
    assert sans < avec


def test_le_classement_complet_est_inchange_sans_precision():
    """``enabled_rules=None`` doit se comporter comme « toutes les règles »."""
    kpis = {"n_clashes": 2, "n_margin_violations": 1, "length_mm": 500.0}
    assert make_report(kpis, None).score() == make_report(kpis, ALL_RULES).score()


def test_la_longueur_departage_toujours():
    """Elle n'est pas une règle d'intégration mais le critère de dernier recours."""
    court = make_report({"length_mm": 100.0}, set()).score()
    long = make_report({"length_mm": 900.0}, set()).score()
    assert court < long


def test_un_clash_reste_prioritaire_sur_le_lissage():
    propre = make_report({"n_clashes": 0, "total_turning_deg": 900.0}, ALL_RULES).score()
    sale = make_report({"n_clashes": 1, "total_turning_deg": 0.0}, ALL_RULES).score()
    assert propre < sale


def test_decocher_le_clash_inverse_bien_cette_priorite():
    actives = ALL_RULES - {"clash"}
    propre = make_report({"n_clashes": 0, "total_turning_deg": 900.0}, actives).score()
    sale = make_report({"n_clashes": 1, "total_turning_deg": 0.0}, actives).score()
    assert sale < propre
