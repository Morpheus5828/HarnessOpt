"""Vérifications des termes de récompense adossés aux règles."""

import numpy as np
import pytest

from core import reward_terms as rt


def straight(length=1000.0, n=21):
    return np.linspace([0, 0, 0], [length, 0, 0], n)


def zigzag(length=1000.0, n=21, amplitude=40.0):
    pts = straight(length, n)
    pts[1::2, 1] = amplitude
    return pts


class TestDistance:
    def test_bande_recompensee(self):
        r = rt.clearance_reward(np.array([30.0]), 10.0, 100.0)
        assert r[0] == pytest.approx(100.0)

    def test_trop_pres_penalise_proportionnellement(self):
        proche = rt.clearance_reward(np.array([5.0]), 10.0, 100.0)[0]
        tres_proche = rt.clearance_reward(np.array([1.0]), 10.0, 100.0)[0]
        assert tres_proche < proche < 0

    def test_trop_loin_penalise(self):
        assert rt.clearance_reward(np.array([300.0]), 10.0, 100.0)[0] < 0

    def test_exigence_par_point(self):
        """Même distance, verdict différent selon la pièce survolée."""
        dist = np.array([30.0, 30.0])
        required = np.array([10.0, 70.0])
        r = rt.clearance_reward(dist, required, 100.0)
        assert r[0] > 0 and r[1] < 0


class TestCintrage:
    def test_ligne_droite_maximise_la_prime(self):
        assert rt.bend_reward(straight(), 240.0).mean() > 0

    def test_cassure_penalisee(self):
        assert rt.bend_reward(zigzag(), 240.0).mean() < 0

    def test_prime_independante_de_lechantillonnage(self):
        """Insérer des points ne doit pas changer artificiellement la note."""
        t = np.linspace(0, np.pi / 2, 30)
        arc_fin = np.stack([500 * np.cos(t), 500 * np.sin(t), np.zeros_like(t)], axis=1)
        t2 = np.linspace(0, np.pi / 2, 60)
        arc_dense = np.stack([500 * np.cos(t2), 500 * np.sin(t2), np.zeros_like(t2)], axis=1)
        moyenne_fin = rt.bend_reward(arc_fin, 240.0)[1:-1].mean()
        moyenne_dense = rt.bend_reward(arc_dense, 240.0)[1:-1].mean()
        assert moyenne_fin == pytest.approx(moyenne_dense, rel=0.05)

    def test_coude_plus_large_mieux_note(self):
        serre = np.array([[0, 0, 0], [300, 0, 0], [300, 300, 0]], dtype=float)
        large = np.array([[0, 0, 0], [900, 0, 0], [900, 900, 0]], dtype=float)
        assert rt.bend_reward(large, 240.0).sum() > rt.bend_reward(serre, 240.0).sum()


class TestRectitude:
    def test_droite_mieux_notee_que_zigzag(self):
        assert rt.straightness_reward(straight()).mean() > rt.straightness_reward(zigzag()).mean()

    def test_longue_ligne_droite_mieux_notee_que_plusieurs_courtes(self):
        """Le cœur de « rester droit le plus longtemps possible »."""
        continue_ = straight(1000.0, 21)
        hachee = continue_.copy()
        hachee[7, 1] = 60.0
        hachee[14, 1] = -60.0
        assert rt.straightness_reward(continue_).sum() > rt.straightness_reward(hachee).sum()

    def test_extremites_neutres(self):
        r = rt.straightness_reward(straight())
        assert r[0] == 0.0 and r[-1] == 0.0


class TestPorteeLibre:
    def test_aucune_penalite_si_tout_est_accrochable(self):
        pts = straight(1000.0, 11)
        assert rt.free_span_penalty(pts, np.full(11, 5.0), 100.0, 250.0).sum() == 0.0

    def test_courte_traversee_toleree(self):
        pts = straight(1000.0, 11)
        dist = np.full(11, 5.0)
        dist[5] = 400.0
        assert rt.free_span_penalty(pts, dist, 100.0, 250.0).sum() == 0.0

    def test_longue_traversee_penalisee(self):
        pts = straight(1000.0, 11)
        dist = np.full(11, 5.0)
        dist[2:8] = 400.0
        assert rt.free_span_penalty(pts, dist, 100.0, 250.0).sum() < 0

    def test_penalite_croit_avec_la_longueur(self):
        pts = straight(2000.0, 21)
        courte = np.full(21, 5.0)
        courte[5:9] = 400.0
        longue = np.full(21, 5.0)
        longue[5:18] = 400.0
        assert rt.free_span_penalty(pts, longue, 100.0, 250.0).sum() < rt.free_span_penalty(
            pts, courte, 100.0, 250.0
        ).sum()


class TestCouvertureFixations:
    def test_sans_fixation_le_milieu_est_penalise(self):
        pts = straight(1000.0, 11)
        r = rt.fixation_coverage_reward(pts, [])
        assert r[5] < 0
        assert r[0] > 0

    def test_fixations_regulieres_suppriment_la_penalite(self):
        pts = straight(1000.0, 11)
        assert rt.fixation_coverage_reward(pts, [250.0, 500.0, 750.0]).min() >= 0

    def test_ajouter_des_fixations_ameliore_la_note(self):
        pts = straight(1000.0, 11)
        assert rt.fixation_coverage_reward(pts, [250.0, 500.0, 750.0]).sum() > rt.fixation_coverage_reward(
            pts, []
        ).sum()


class TestCombinaison:
    def test_somme_et_ventilation(self):
        pts = straight()
        total, details = rt.combine(
            rectitude=rt.straightness_reward(pts),
            cintrage=rt.bend_reward(pts, 240.0),
        )
        assert set(details) == {"rectitude", "cintrage"}
        assert total.sum() == pytest.approx(
            rt.straightness_reward(pts).sum() + rt.bend_reward(pts, 240.0).sum(), rel=1e-5
        )

    def test_termes_absents_ignores(self):
        total, details = rt.combine(a=np.ones(5), b=None)
        assert details == {"a": 1.0}
        assert len(total) == 5

    def test_tailles_heterogenes_tolerees(self):
        total, _ = rt.combine(a=np.ones(5), b=np.ones(3))
        assert len(total) == 5


class TestDetour:
    def test_trajet_direct_non_penalise(self):
        direct = np.linspace([0, 0, 0], [1000, 0, 0], 11)
        assert rt.detour_penalty(direct, 1000.0).sum() == 0.0

    def test_tolerance_avant_penalite(self):
        """S'écarter pour respecter les distances rallonge : c'est admis."""
        legere = np.linspace([0, 0, 0], [1100, 0, 0], 11)
        assert rt.detour_penalty(legere, 1000.0).sum() == 0.0

    def test_detour_franc_penalise(self):
        detour = np.array(
            [[0, 0, 0], [250, 600, 0], [500, 600, 0], [750, 600, 0], [1000, 0, 0]], dtype=float
        )
        assert rt.detour_penalty(detour, 1000.0).sum() < 0

    def test_penalite_croit_avec_le_detour(self):
        court = np.array([[0, 0, 0], [500, 300, 0], [1000, 0, 0]], dtype=float)
        long_ = np.array([[0, 0, 0], [500, 1200, 0], [1000, 0, 0]], dtype=float)
        assert rt.detour_penalty(long_, 1000.0).sum() < rt.detour_penalty(court, 1000.0).sum()

    def test_reference_absente_ne_penalise_pas(self):
        direct = np.linspace([0, 0, 0], [1000, 0, 0], 11)
        assert rt.detour_penalty(direct, 0.0).sum() == 0.0

    def test_penalite_bornee(self):
        """Une route absurde ne doit pas écraser toutes les autres règles."""
        enorme = np.linspace([0, 0, 0], [100_000, 0, 0], 11)
        assert rt.detour_penalty(enorme, 1000.0, weight=40.0)[0] == pytest.approx(-120.0)
