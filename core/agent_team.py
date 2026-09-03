"""Constitution et supervision d'une équipe d'agents.

Ce module fait le lien entre :mod:`core.orchestrator` (qui décide *qui* fait
*quoi*, sans dépendance lourde) et :mod:`core.agent_worker` (qui fait tourner
un agent sur une trajectoire). C'est le seul endroit où l'on instancie les
réseaux PyTorch.

Deux responsabilités :

* **fabriquer** le dictionnaire d'agents attendu par ``algo_worker`` à partir
  d'une équipe décrite par l'orchestrateur ;
* **superviser** l'équipe en cours de route : classer les agents sur leur score
  DMU, faire avancer le curriculum et transmettre les ordres de migration.

Le superviseur tourne dans son propre fil d'exécution et ne touche jamais aux
trajectoires : il se contente de déposer un ordre dans l'état partagé, que
chaque agent lit à son rythme. Aucune synchronisation lourde n'est nécessaire.
"""

from __future__ import annotations

import threading

from core.agent.agent import RecurrentTD3Agent, RLAgent, SACAgent
from core.agent.buffer import ReplayBuffer, SequenceReplayBuffer
from core.agent.config import ACTION_DIM, CONFIG, STATE_DIM
from core.orchestrator import AgentSpec, Orchestrator

__all__ = ["build_benchmark_algos", "TeamSupervisor"]

#: Correspondance entre l'algorithme demandé par un rôle et la paire
#: (réseau, mémoire de rejeu) qui l'implémente.
_ALGO_FACTORIES = {
    "td3": (
        lambda: RLAgent(use_td3=True),
        lambda max_len: ReplayBuffer(STATE_DIM, ACTION_DIM, use_cer=True),
    ),
    "ddpg": (
        lambda: RLAgent(use_td3=False),
        lambda max_len: ReplayBuffer(STATE_DIM, ACTION_DIM, use_cer=False),
    ),
    "sac": (
        lambda: SACAgent(),
        lambda max_len: ReplayBuffer(STATE_DIM, ACTION_DIM, use_cer=False),
    ),
    "recurrent_td3": (
        lambda: RecurrentTD3Agent(),
        lambda max_len: SequenceReplayBuffer(STATE_DIM, ACTION_DIM, max_len=max_len),
    ),
}


def build_benchmark_algos(team: list[AgentSpec], max_points: int | None = None) -> dict:
    """Instancie les réseaux d'une équipe.

    Args:
        team: fiches produites par :func:`core.orchestrator.build_team`.
        max_points: longueur maximale d'une trajectoire, nécessaire au
            dimensionnement des mémoires séquentielles.

    Returns:
        Le dictionnaire attendu par ``algo_worker`` : ``{nom: {agent, buffer,
        color, role, spec, ...}}``.
    """
    max_len = int(max_points or CONFIG.get("max_points", 150))
    algos: dict[str, dict] = {}

    for spec in team:
        factory = _ALGO_FACTORIES.get(spec.algo, _ALGO_FACTORIES["td3"])
        make_agent, make_buffer = factory
        algos[spec.name] = {
            "agent": make_agent(),
            "buffer": make_buffer(max_len),
            "color": spec.color,
            # « explorer » / « optimizer » : distinction historique conservée
            # pour les réglages de pas. La spécialité métier est dans `spec`.
            "role": "explorer" if spec.shift_scale >= 1.0 else "optimizer",
            "crabe_focus": spec.role == "fixer",
            "spec": spec,
        }
    return algos


class TeamSupervisor(threading.Thread):
    """Fil d'arbitrage : classement, curriculum et migrations.

    Le superviseur relit périodiquement l'état partagé, classe les agents sur
    le score DMU publié par chacun, met à jour l'étape du chantier, et dépose
    les ordres de migration dans la boîte de réception des agents en retard.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        shared_state: dict,
        data_lock: threading.Lock,
        period_s: float = 1.0,
        lang: str = "FR",
    ):
        super().__init__(daemon=True, name="TeamSupervisor")
        self.orchestrator = orchestrator
        self.shared_state = shared_state
        self.data_lock = data_lock
        self.period_s = float(period_s)
        self.lang = lang
        self._stop = threading.Event()

    def stop(self):
        """Demande l'arrêt du superviseur."""
        self._stop.set()

    def run(self):
        while not self._stop.is_set() and self.shared_state.get("is_running", True):
            try:
                self.step()
            except Exception as exc:  # un arbitrage raté ne doit rien interrompre
                print(f"⚠️ Superviseur d'équipe : {exc}")
            self._stop.wait(self.period_s)

    def step(self) -> dict:
        """Un tour d'arbitrage. Renvoie le résumé publié dans l'état partagé."""
        with self.data_lock:
            if not self.shared_state.get("is_playing", False):
                return {}
            algos = self.shared_state.get("algos", {})
            scores = {
                name: state["score"]
                for name, state in algos.items()
                if state.get("score") is not None
            }
            iteration = max((state.get("iteration", 0) for state in algos.values()), default=0)
            best_kpis = {}
            if scores:
                best_name = min(scores, key=lambda n: scores[n])
                best_kpis = dict(algos[best_name].get("kpis", {}) or {})
                best_kpis.setdefault(
                    "fixation_pitch_mm", self.shared_state["config"].get("fixation_pitch_mm", 250.0)
                )

        if not scores:
            return {}

        self.orchestrator.update_phase(best_kpis)
        orders = self.orchestrator.plan_migration(scores, iteration)

        with self.data_lock:
            algos = self.shared_state.get("algos", {})
            for order in orders:
                winner = algos.get(order.winner)
                loser = algos.get(order.loser)
                if winner is None or loser is None:
                    continue
                waypoints = winner.get("waypoints")
                if waypoints is None or len(waypoints) < 2:
                    continue
                loser["migrate_from"] = {
                    "waypoints": waypoints.copy(),
                    "perturbation": order.perturbation,
                    "from": order.winner,
                }

            summary = self.orchestrator.summary(scores, lang=self.lang)
            self.shared_state["team"] = summary

        return summary
