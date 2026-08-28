"""Vérifications du superviseur d'équipe (classement, curriculum, migrations).

Le superviseur est testé sur un état partagé simulé : on n'a pas besoin de
faire tourner de vrais agents pour vérifier qu'il classe correctement et qu'il
transmet bien la trajectoire du meilleur aux retardataires.
"""

import threading

import numpy as np
import pytest

from core.orchestrator import Orchestrator


class FakeSupervisor:
    """Reproduit la logique de TeamSupervisor sans dépendre de PyTorch.

    ``core.agent_team`` importe les réseaux PyTorch au chargement du module ;
    on isole donc ici la seule logique d'arbitrage, qui est celle que ces tests
    doivent couvrir.
    """

    def __init__(self, orchestrator, shared_state, data_lock):
        from core.agent_team import TeamSupervisor

        self._impl = TeamSupervisor.__new__(TeamSupervisor)
        self._impl.orchestrator = orchestrator
        self._impl.shared_state = shared_state
        self._impl.data_lock = data_lock
        self._impl.lang = "FR"

    def step(self):
        return self._impl.step()


def make_state(scores, waypoints=None):
    n = len(scores)
    wps = waypoints or {name: np.linspace([0, 0, 0], [1000, i * 10, 0], 12) for i, name in enumerate(scores)}
    return {
        "is_playing": True,
        "is_running": True,
        "config": {"fixation_pitch_mm": 250.0},
        "algos": {
            name: {
                "score": score,
                "waypoints": wps[name],
                "iteration": 5000,
                "kpis": {"n_clashes": 0, "n_margin_violations": 0, "worst_support_gap_mm": 100.0},
            }
            for name, score in scores.items()
        },
    }


torch = pytest.importorskip("torch", reason="le superviseur importe les réseaux PyTorch")


class TestSuperviseur:
    def setup_method(self):
        self.orch = Orchestrator("balanced", temperature=0.0, seed=5)
        self.names = [s.name for s in self.orch.team]
        self.scores = {name: (i,) for i, name in enumerate(self.names)}
        self.state = make_state(self.scores)
        self.lock = threading.Lock()
        self.sup = FakeSupervisor(self.orch, self.state, self.lock)

    def test_le_resume_est_publie(self):
        summary = self.sup.step()
        assert summary["best"] == self.names[0]
        assert self.state["team"]["best"] == self.names[0]

    def test_la_trajectoire_du_meilleur_est_transmise(self):
        self.sup.step()
        migres = [
            (name, st) for name, st in self.state["algos"].items() if "migrate_from" in st
        ]
        assert migres, "aucun agent n'a reçu d'ordre de migration"
        best_wps = self.state["algos"][self.names[0]]["waypoints"]
        for name, st in migres:
            assert np.allclose(st["migrate_from"]["waypoints"], best_wps)
            assert st["migrate_from"]["from"] == self.names[0]

    def test_la_trajectoire_transmise_est_une_copie(self):
        """L'agent receveur ne doit pas partager le tableau du gagnant."""
        self.sup.step()
        st = next(s for s in self.state["algos"].values() if "migrate_from" in s)
        transmise = st["migrate_from"]["waypoints"]
        self.state["algos"][self.names[0]]["waypoints"][0, 0] = 99999.0
        assert transmise[0, 0] != 99999.0

    def test_le_meilleur_ne_recoit_jamais_dordre(self):
        self.sup.step()
        assert "migrate_from" not in self.state["algos"][self.names[0]]

    def test_reglages_perturbes_transmis(self):
        self.sup.step()
        st = next(s for s in self.state["algos"].values() if "migrate_from" in s)
        assert set(st["migrate_from"]["perturbation"]) == {
            "noise_scale", "shift_scale", "momentum_delta"
        }

    def test_pas_darbitrage_en_pause(self):
        self.state["is_playing"] = False
        assert self.sup.step() == {}
        assert all("migrate_from" not in s for s in self.state["algos"].values())

    def test_sans_score_publie_aucun_arbitrage(self):
        for st in self.state["algos"].values():
            st["score"] = None
        assert self.sup.step() == {}

    def test_letape_suit_les_indicateurs_du_meilleur(self):
        self.state["algos"][self.names[0]]["kpis"]["n_clashes"] = 4
        self.sup.step()
        assert self.orch.phase == "feasibility"

    def test_trajectoire_trop_courte_non_transmise(self):
        self.state["algos"][self.names[0]]["waypoints"] = np.zeros((1, 3))
        self.sup.step()
        assert all("migrate_from" not in s for s in self.state["algos"].values())


class TestFabriqueDequipe:
    def test_chaque_role_recoit_le_bon_couple_reseau_memoire(self):
        from core.agent.agent import RecurrentTD3Agent, RLAgent, SACAgent
        from core.agent.buffer import ReplayBuffer, SequenceReplayBuffer
        from core.agent_team import build_benchmark_algos
        from core.orchestrator import build_team

        algos = build_benchmark_algos(build_team("balanced", 0.5), max_points=64)
        assert isinstance(algos["scout"]["agent"], RLAgent)
        assert isinstance(algos["smoother"]["agent"], SACAgent)
        assert isinstance(algos["straightener"]["agent"], RecurrentTD3Agent)
        assert isinstance(algos["straightener"]["buffer"], SequenceReplayBuffer)
        assert isinstance(algos["scout"]["buffer"], ReplayBuffer)

    def test_le_poseur_est_marque_chasseur_de_crabes(self):
        from core.agent_team import build_benchmark_algos
        from core.orchestrator import build_team

        algos = build_benchmark_algos(build_team("balanced", 0.5), max_points=64)
        assert algos["fixer"]["crabe_focus"] is True
        assert algos["smoother"]["crabe_focus"] is False

    def test_la_fiche_de_role_est_transmise_au_worker(self):
        from core.agent_team import build_benchmark_algos
        from core.orchestrator import build_team

        algos = build_benchmark_algos(build_team("balanced", 0.5), max_points=64)
        assert algos["smoother"]["spec"].weights["bend"] > 1.0
