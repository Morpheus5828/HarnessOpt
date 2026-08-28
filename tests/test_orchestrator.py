"""Vérifications de l'équipe d'agents et de l'arbitrage exploration/exploitation."""

import numpy as np
import pytest

from core.orchestrator import (
    ROLES,
    TEAM_PRESETS,
    ExplorationPolicy,
    Orchestrator,
    Phase,
    build_team,
)
from core.routing_rules import RoutingRules, evaluate_route


def report(n_crossings=0, wobble=0.0, distance=30.0, clamps=(250.0, 500.0, 750.0)):
    pts = np.linspace([0, 0, 0], [1000, 0, 0], 21)
    if wobble:
        pts[1::2, 1] = wobble
    return evaluate_route(
        pts,
        RoutingRules(),
        distances=np.full(21, distance),
        n_crossings=n_crossings,
        clamp_arc_positions=list(clamps),
    )


class TestRoles:
    def test_tous_les_roles_sont_complets(self):
        for role in ROLES.values():
            assert role.label_fr and role.label_en
            assert role.description_fr and role.description_en
            assert role.color.startswith("#")
            assert role.algo in {"td3", "sac", "recurrent_td3"}

    def test_aucun_role_ne_debranche_une_regle(self):
        """Un rôle déplace l'effort, il n'autorise jamais à ignorer une règle."""
        for role in ROLES.values():
            assert all(w > 0 for w in role.weights.values()), role.key

    def test_specialisation_effective(self):
        assert ROLES["smoother"].weights["bend"] > ROLES["scout"].weights["bend"]
        assert ROLES["straightener"].weights["straight"] > ROLES["scout"].weights["straight"]
        assert ROLES["fixer"].weights["fixation"] > ROLES["smoother"].weights["fixation"]
        assert ROLES["clearance"].weights["clash"] > ROLES["smoother"].weights["clash"]

    def test_eclaireur_bouge_plus_que_le_lisseur(self):
        assert ROLES["scout"].noise_scale > ROLES["smoother"].noise_scale
        assert ROLES["scout"].shift_scale > ROLES["smoother"].shift_scale


class TestCurseurExploration:
    def test_bornes(self):
        assert ExplorationPolicy(-3.0).temperature == 0.0
        assert ExplorationPolicy(9.0).temperature == 1.0

    def test_monotonie_coherente(self):
        """Tous les réglages doivent bouger dans le même sens que le curseur."""
        froid, chaud = ExplorationPolicy(0.0), ExplorationPolicy(1.0)
        assert chaud.noise_start > froid.noise_start
        assert chaud.noise_floor > froid.noise_floor
        assert chaud.shift_scale > froid.shift_scale
        assert chaud.momentum < froid.momentum            # explorer = moins d'inertie
        assert chaud.migration_interval > froid.migration_interval
        assert chaud.migration_fraction < froid.migration_fraction
        assert chaud.patience_scale > froid.patience_scale

    def test_exploitation_pure_finit_par_se_figer(self):
        assert ExplorationPolicy(0.0).noise_floor == 0.0

    def test_libelles_lisibles(self):
        assert ExplorationPolicy(0.0).label("FR") != ExplorationPolicy(1.0).label("FR")
        assert ExplorationPolicy(0.5).label("EN")


class TestConstitutionEquipe:
    def test_preset_equilibre(self):
        team = build_team("balanced", 0.5)
        assert {s.role for s in team} == set(TEAM_PRESETS["balanced"]["composition"])

    def test_noms_uniques(self):
        team = build_team({"scout": 3, "smoother": 2}, 0.5)
        assert len(team) == 5
        assert len({s.name for s in team}) == 5

    def test_preset_inconnu_retombe_sur_equilibre(self):
        assert len(build_team("n_importe_quoi", 0.5)) == len(build_team("balanced", 0.5))

    def test_composition_vide_retombe_sur_equilibre(self):
        assert len(build_team({}, 0.5)) == len(build_team("balanced", 0.5))

    def test_role_inconnu_ignore(self):
        assert [s.role for s in build_team({"scout": 1, "inexistant": 4}, 0.5)] == ["scout"]

    def test_le_curseur_impacte_toute_lequipe(self):
        froide = build_team("balanced", 0.0)
        chaude = build_team("balanced", 1.0)
        assert sum(s.noise_start for s in chaude) > sum(s.noise_start for s in froide)


class TestCurriculum:
    def test_clash_impose_la_recherche_de_passage(self):
        assert Phase.from_kpis(report(n_crossings=2).kpis) == Phase.FEASIBILITY

    def test_distance_insuffisante_impose_la_mise_aux_distances(self):
        assert Phase.from_kpis(report(distance=2.0).kpis) == Phase.CLEARANCE

    def test_manque_de_fixations_impose_la_pose(self):
        kpis = report(clamps=()).kpis
        kpis["fixation_pitch_mm"] = 250.0
        assert Phase.from_kpis(kpis) == Phase.SUPPORT

    def test_route_saine_passe_au_lissage(self):
        assert Phase.from_kpis(report().kpis) == Phase.POLISH

    def test_kpis_vides_supposent_le_pire(self):
        assert Phase.from_kpis({}) == Phase.FEASIBILITY

    def test_le_role_prioritaire_suit_letape(self):
        orch = Orchestrator("balanced", temperature=0.5, seed=1)
        orch.update_phase(report(n_crossings=3).kpis)
        assert orch.phase_weight_boost("scout") > orch.phase_weight_boost("smoother")
        orch.update_phase(report().kpis)
        assert orch.phase_weight_boost("smoother") > orch.phase_weight_boost("scout")


class TestClassementEtMigration:
    def setup_method(self):
        self.scores = {
            "scout": report(n_crossings=3).score(),
            "clearance": report(wobble=20).score(),
            "smoother": report().score(),
            "straightener": report(n_crossings=1).score(),
            "fixer": report(wobble=40).score(),
        }

    def test_le_meilleur_est_en_tete(self):
        assert Orchestrator().rank(self.scores)[0] == "smoother"

    def test_pas_de_migration_avant_lintervalle(self):
        orch = Orchestrator("balanced", temperature=0.3, seed=1)
        assert orch.plan_migration(self.scores, 5) == []

    def test_migration_apres_lintervalle(self):
        orch = Orchestrator("balanced", temperature=0.3, seed=1)
        orders = orch.plan_migration(self.scores, 500)
        assert orders
        assert all(o.winner == "smoother" for o in orders)

    def test_exploration_maximale_supprime_les_migrations(self):
        """À curseur au maximum, chacun creuse sa piste sans être rappelé."""
        orch = Orchestrator("balanced", temperature=1.0, seed=1)
        assert orch.plan_migration(self.scores, 10_000) == []

    def test_exploitation_rallie_davantage_dagents(self):
        froid = Orchestrator("balanced", temperature=0.0, seed=1)
        tiede = Orchestrator("balanced", temperature=0.6, seed=1)
        assert len(froid.plan_migration(self.scores, 10_000)) >= len(
            tiede.plan_migration(self.scores, 10_000)
        )

    def test_le_meilleur_ne_migre_jamais(self):
        orch = Orchestrator("balanced", temperature=0.0, seed=1)
        assert all(o.loser != "smoother" for o in orch.plan_migration(self.scores, 10_000))

    def test_un_agent_deja_au_niveau_ne_migre_pas(self):
        egaux = {"a": (0, 0), "b": (0, 0), "c": (5, 0)}
        orch = Orchestrator({"scout": 3}, temperature=0.0, seed=1)
        assert all(o.loser != "b" for o in orch.plan_migration(egaux, 10_000))

    def test_les_reglages_sont_perturbes_a_chaque_migration(self):
        orch = Orchestrator("balanced", temperature=0.0, seed=1)
        for order in orch.plan_migration(self.scores, 10_000):
            assert set(order.perturbation) == {"noise_scale", "shift_scale", "momentum_delta"}

    def test_lintervalle_est_respecte_entre_deux_migrations(self):
        orch = Orchestrator("balanced", temperature=0.0, seed=1)
        assert orch.plan_migration(self.scores, 1000)
        assert orch.plan_migration(self.scores, 1005) == []

    def test_equipe_dun_seul_agent(self):
        orch = Orchestrator({"scout": 1}, temperature=0.0, seed=1)
        assert orch.plan_migration({"scout": (0,)}, 10_000) == []


class TestReglageEnCoursDeSession:
    def test_changer_le_curseur_conserve_lequipe(self):
        orch = Orchestrator("balanced", temperature=0.2, seed=1)
        noms = [s.name for s in orch.team]
        orch.set_temperature(0.9)
        assert [s.name for s in orch.team] == noms
        assert orch.policy.temperature == pytest.approx(0.9)

    def test_changer_le_curseur_ne_remet_pas_lexploration_a_zero(self):
        orch = Orchestrator("balanced", temperature=0.2, seed=1)
        avant = {s.name: s.noise_start for s in orch.team}
        orch.set_temperature(0.9)
        assert {s.name: s.noise_start for s in orch.team} == avant

    def test_changer_la_composition(self):
        orch = Orchestrator("balanced", temperature=0.5, seed=1)
        orch.set_composition("finishing")
        assert {s.role for s in orch.team} == set(TEAM_PRESETS["finishing"]["composition"])


class TestResume:
    def test_resume_pret_a_afficher(self):
        orch = Orchestrator("balanced", temperature=0.5, seed=1)
        orch.update_phase(report().kpis)
        summary = orch.summary({"smoother": (0,), "scout": (3,)}, lang="FR")
        assert summary["best"] == "smoother"
        assert summary["phase_label"]
        assert summary["temperature_label"]
        assert all("color" in a and "label" in a for a in summary["team"])

    def test_resume_sans_score(self):
        assert Orchestrator("balanced", 0.5, seed=1).summary()["best"] is not None
