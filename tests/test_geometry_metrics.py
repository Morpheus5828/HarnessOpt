"""Vérifications des mesures géométriques d'une trajectoire."""

import numpy as np
import pytest

from core import geometry_metrics as gm


def arc_of_circle(radius=500.0, n=40, sweep=np.pi / 2):
    t = np.linspace(0.0, sweep, n)
    return np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1)


def straight_line(length=1000.0, n=11):
    return np.linspace([0, 0, 0], [length, 0, 0], n)


class TestLongueurs:
    def test_longueur_ligne_droite(self):
        assert gm.path_length(straight_line(1000.0)) == pytest.approx(1000.0)

    def test_abscisse_curviligne_croissante(self):
        arc = gm.arc_lengths(straight_line(1000.0, 11))
        assert arc[0] == 0.0
        assert arc[-1] == pytest.approx(1000.0)
        assert np.all(np.diff(arc) > 0)

    def test_trajectoire_degeneree(self):
        assert gm.path_length(np.zeros((1, 3))) == 0.0
        assert len(gm.turning_angles(np.zeros((2, 3)))) == 0
        assert gm.min_bend_radius(np.zeros((2, 3))) == float("inf")


class TestCourbure:
    def test_ligne_droite_rayon_infini(self):
        assert gm.min_bend_radius(straight_line()) == float("inf")
        assert gm.min_curvature_radius(straight_line()) == float("inf")

    def test_cercle_retrouve_son_rayon(self):
        assert gm.min_curvature_radius(arc_of_circle(500.0, 40)) == pytest.approx(500.0, rel=1e-3)

    def test_rayon_independant_de_lechantillonnage(self):
        """Le point clé : raffiner la trajectoire ne doit pas changer la note."""
        fin = gm.min_bend_radius(arc_of_circle(500.0, 80))
        grossier = gm.min_bend_radius(arc_of_circle(500.0, 15))
        assert fin == pytest.approx(grossier, rel=0.02)
        assert fin == pytest.approx(500.0, rel=0.02)

    def test_coude_franc_avec_de_la_place_est_realisable(self):
        """Un angle droit entre deux longs segments s'arrondit sans difficulté."""
        coude = np.array([[0, 0, 0], [500, 0, 0], [500, 500, 0]], dtype=float)
        assert gm.min_bend_radius(coude) == pytest.approx(250.0)

    def test_coude_serre_est_une_cassure(self):
        serre = np.array([[0, 0, 0], [50, 0, 0], [50, 50, 0]], dtype=float)
        assert gm.min_bend_radius(serre) == pytest.approx(25.0)
        assert gm.min_bend_radius(serre) < 240.0  # sous le rayon admissible d'un toron 40 mm

    def test_angle_de_virage(self):
        coude = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0]], dtype=float)
        assert gm.turning_angles(coude)[0] == pytest.approx(np.pi / 2)


class TestRectitude:
    def test_ligne_droite_totalement_droite(self):
        stats = gm.straightness(straight_line(1000.0, 11))
        assert stats["straight_ratio"] == pytest.approx(1.0)
        assert stats["longest_run_mm"] == pytest.approx(1000.0)
        assert stats["n_bends"] == 0
        assert stats["total_turning_deg"] == pytest.approx(0.0)

    def test_arc_regulier_compte_un_seul_coude(self):
        assert gm.straightness(arc_of_circle(500.0, 30))["n_bends"] == 1

    def test_zigzag_compte_plusieurs_coudes(self):
        zz = np.array(
            [[0, 0, 0], [100, 0, 0], [100, 20, 0], [200, 20, 0], [200, 0, 0], [300, 0, 0]],
            dtype=float,
        )
        stats = gm.straightness(zz)
        assert stats["n_bends"] > 1
        assert stats["total_turning_deg"] > 180.0

    def test_longue_ligne_droite_preferee_a_plusieurs_courtes(self):
        continue_ = straight_line(1000.0, 21)
        hachee = continue_.copy()
        hachee[7, 1] = 60.0
        hachee[14, 1] = -60.0
        assert gm.straightness(continue_)["longest_run_mm"] > gm.straightness(hachee)["longest_run_mm"]


class TestPorteesLibres:
    def test_aucune_portee_libre(self):
        pts = straight_line(1000.0, 11)
        assert gm.free_spans(pts, np.full(11, 5.0), 100.0) == []
        assert gm.longest_free_span(pts, np.full(11, 5.0), 100.0) == 0.0

    def test_point_isole_dans_le_vide_a_une_longueur_non_nulle(self):
        pts = straight_line(1000.0, 11)
        dist = np.full(11, 5.0)
        dist[3] = 400.0
        spans = gm.free_spans(pts, dist, 100.0)
        assert len(spans) == 1
        assert spans[0][2] > 0.0

    def test_longueur_de_la_portee(self):
        pts = straight_line(1000.0, 11)
        dist = np.full(11, 5.0)
        dist[3:7] = 400.0
        assert gm.longest_free_span(pts, dist, 100.0) == pytest.approx(400.0)


class TestSupports:
    def test_extremites_comptent_comme_tenues(self):
        pts = straight_line(1000.0, 11)
        assert gm.support_gaps(pts, []).tolist() == pytest.approx([1000.0])

    def test_ecarts_entre_fixations(self):
        pts = straight_line(1000.0, 11)
        assert gm.support_gaps(pts, [3, 7]).tolist() == pytest.approx([300.0, 400.0, 300.0])


class TestReechantillonnage:
    def test_pas_constant(self):
        out = gm.resample_by_arclength(straight_line(1000.0, 11), 250.0)
        assert len(out) == 5
        assert gm.path_length(out) == pytest.approx(1000.0, rel=1e-4)
