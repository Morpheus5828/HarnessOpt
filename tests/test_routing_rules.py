"""Vérifications du moteur de règles d'intégration."""

import numpy as np
import pytest

from core.routing_rules import (
    ClearanceModel,
    HarnessSpec,
    RoutingRules,
    Severity,
    evaluate_route,
)


def straight(length=1000.0, n=21):
    return np.linspace([0, 0, 0], [length, 0, 0], n)


def route_conforme(rules=None):
    rules = rules or RoutingRules()
    pts = straight(1000.0, 21)
    return evaluate_route(
        pts,
        rules,
        distances=np.full(21, 30.0),
        inside_mask=np.zeros(21, dtype=bool),
        n_crossings=0,
        clamp_arc_positions=[250.0, 500.0, 750.0],
        clamp_tilt_deg=[2.0, 3.0, 1.0],
    )


class TestSpecificationHarnais:
    def test_rayon_admissible_par_defaut(self):
        assert HarnessSpec(diameter_mm=40.0).min_bend_radius_mm == 240.0

    def test_rayon_suit_le_diametre(self):
        assert HarnessSpec(diameter_mm=20.0, bend_radius_factor=6.0).min_bend_radius_mm == 120.0


class TestModeleDeClearance:
    def test_distance_par_defaut_sans_couleur(self):
        model = ClearanceModel(default_min_mm=10.0)
        assert not model.is_differentiated
        assert model.required_min(None, n=5).tolist() == [10.0] * 5

    def test_distance_differenciee_par_famille(self):
        """Une hydraulique haute pression exige davantage qu'une tôle."""
        model = ClearanceModel(default_min_mm=10.0).with_face_families(
            face_family=np.array([0, 0, 1, 1]),
            family_names=["structure", "high_pressure_system"],
        )
        assert model.is_differentiated
        required = model.required_min(np.array([0, 2]))
        assert required[0] == pytest.approx(10.0)
        assert required[1] == pytest.approx(70.0)

    def test_famille_inconnue_retombe_sur_le_defaut(self):
        model = ClearanceModel(default_min_mm=12.0).with_face_families(
            face_family=np.array([0]), family_names=["famille_inedite"]
        )
        assert model.required_min(np.array([0]))[0] == pytest.approx(12.0)

    def test_index_de_face_hors_bornes_ne_plante_pas(self):
        model = ClearanceModel(default_min_mm=10.0).with_face_families(
            face_family=np.array([0, 1]), family_names=["structure", "fuel"]
        )
        assert model.required_min(np.array([99, -1]))[0] == pytest.approx(10.0)

    def test_contrainte_la_plus_severe(self):
        model = ClearanceModel().with_face_families(
            face_family=np.array([0, 1]), family_names=["structure", "high_pressure_system"]
        )
        assert model.strictest_mm == pytest.approx(70.0)


class TestRouteConforme:
    def test_toutes_les_regles_passent(self):
        report = route_conforme()
        assert report.is_compliant
        assert report.is_deliverable
        assert report.compliance_ratio == 1.0
        assert report.failed() == []

    def test_kpis_renseignes(self):
        kpis = route_conforme().kpis
        assert kpis["length_mm"] == pytest.approx(1000.0)
        assert kpis["n_clashes"] == 0
        assert kpis["straight_ratio"] == pytest.approx(1.0)
        assert kpis["min_bend_radius_mm"] == float("inf")


class TestRegleClash:
    def test_traversee_est_bloquante(self):
        rules = RoutingRules()
        report = evaluate_route(
            straight(), rules, distances=np.full(21, 30.0), n_crossings=3,
            clamp_arc_positions=[250.0, 500.0, 750.0],
        )
        assert not report.is_deliverable
        assert "clash" in [c.rule_id for c in report.failed(Severity.BLOCKING)]

    def test_point_dans_la_matiere_est_bloquant(self):
        rules = RoutingRules()
        inside = np.zeros(21, dtype=bool)
        inside[5] = True
        report = evaluate_route(straight(), rules, distances=np.full(21, 30.0), inside_mask=inside)
        assert report.kpis["n_clashes"] == 1
        assert not report.is_deliverable


class TestRegleDistance:
    def test_trop_pres_est_bloquant(self):
        rules = RoutingRules()
        dist = np.full(21, 30.0)
        dist[4] = 3.0
        report = evaluate_route(straight(), rules, distances=dist)
        check = next(c for c in report.checks if c.rule_id == "clearance_min")
        assert not check.passed
        assert check.severity == Severity.BLOCKING

    def test_exigence_differenciee_declenche_la_regle(self):
        """30 mm suffit près d'une tôle, pas près d'une hydraulique HP."""
        rules = RoutingRules()
        dist = np.full(21, 30.0)
        standard = evaluate_route(straight(), rules, distances=dist)
        severe = evaluate_route(
            straight(), rules, distances=dist, required_min=np.full(21, 70.0)
        )
        assert standard.kpis["n_margin_violations"] == 0
        assert severe.kpis["n_margin_violations"] == 21

    def test_cable_qui_flotte_est_signale(self):
        rules = RoutingRules()
        dist = np.full(21, 30.0)
        dist[10] = 500.0
        report = evaluate_route(straight(), rules, distances=dist)
        assert not next(c for c in report.checks if c.rule_id == "clearance_max").passed


class TestRegleCintrage:
    def test_cassure_detectee(self):
        rules = RoutingRules()
        serre = np.array([[0, 0, 0], [50, 0, 0], [50, 50, 0]], dtype=float)
        report = evaluate_route(serre, rules)
        check = next(c for c in report.checks if c.rule_id == "bend_radius")
        assert not check.passed
        assert check.severity == Severity.BLOCKING

    def test_coude_large_accepte(self):
        rules = RoutingRules()
        large = np.array([[0, 0, 0], [800, 0, 0], [800, 800, 0]], dtype=float)
        assert next(c for c in evaluate_route(large, rules).checks if c.rule_id == "bend_radius").passed

    def test_seuil_suit_le_diametre_du_toron(self):
        coude = np.array([[0, 0, 0], [300, 0, 0], [300, 300, 0]], dtype=float)  # congé 150 mm
        gros = RoutingRules()  # toron 40 mm -> 240 mm exigés
        fin = RoutingRules(harness=HarnessSpec(diameter_mm=20.0))  # -> 120 mm exigés
        assert not next(c for c in evaluate_route(coude, gros).checks if c.rule_id == "bend_radius").passed
        assert next(c for c in evaluate_route(coude, fin).checks if c.rule_id == "bend_radius").passed


class TestRegleFixations:
    def test_pas_de_250mm_non_respecte(self):
        rules = RoutingRules()
        report = evaluate_route(straight(2000.0, 21), rules, distances=np.full(21, 30.0))
        check = next(c for c in report.checks if c.rule_id == "fixation_pitch")
        assert not check.passed
        assert check.value == pytest.approx(2000.0)

    def test_nombre_de_crabes_necessaires(self):
        rules = RoutingRules()
        report = evaluate_route(straight(1000.0, 21), rules, distances=np.full(21, 30.0))
        assert report.kpis["clamps_required"] == 3

    def test_crabe_non_parallele_signale(self):
        rules = RoutingRules()
        report = evaluate_route(
            straight(), rules, distances=np.full(21, 30.0),
            clamp_arc_positions=[250.0, 500.0, 750.0], clamp_tilt_deg=[2.0, 40.0, 1.0],
        )
        assert not next(c for c in report.checks if c.rule_id == "fixation_parallel").passed


class TestClassement:
    def test_une_route_propre_bat_une_route_avec_clash(self):
        rules = RoutingRules()
        propre = route_conforme(rules)
        sale = evaluate_route(
            straight(), rules, distances=np.full(21, 30.0), n_crossings=1,
            clamp_arc_positions=[250.0, 500.0, 750.0],
        )
        assert propre.score() < sale.score()

    def test_a_conformite_egale_le_plus_droit_gagne(self):
        rules = RoutingRules()
        droite = evaluate_route(
            straight(), rules, distances=np.full(21, 30.0),
            clamp_arc_positions=[250.0, 500.0, 750.0],
        )
        onduleuse_pts = straight()
        onduleuse_pts[1::2, 1] = 5.0
        onduleuse = evaluate_route(
            onduleuse_pts, rules, distances=np.full(21, 30.0),
            clamp_arc_positions=[250.0, 500.0, 750.0],
        )
        assert droite.score() < onduleuse.score()

    def test_serialisation_du_rapport(self):
        data = route_conforme().to_dict()
        assert data["compliant"] is True
        assert {"id", "label", "passed", "severity"} <= set(data["checks"][0])


class TestRobustesse:
    def test_sans_mesure_de_distance(self):
        """Sans distances mesurées, les règles de distance sont omises, pas supposées bonnes."""
        report = evaluate_route(straight(), RoutingRules())
        assert "clearance_min" not in [c.rule_id for c in report.checks]

    def test_trajectoire_de_deux_points(self):
        report = evaluate_route(np.array([[0, 0, 0], [500, 0, 0]], dtype=float), RoutingRules())
        assert report.kpis["length_mm"] == pytest.approx(500.0)
