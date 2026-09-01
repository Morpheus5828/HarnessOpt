"""Équipe d'agents spécialisés et arbitrage exploration / exploitation.

Auparavant, cinq agents résolvaient le même problème avec la même récompense,
chacun dans son coin, et la seule distinction (« explorer » / « optimizer »)
se réduisait à un pas plus ou moins grand. Ils ne se parlaient pas : un agent
qui avait trouvé un passage propre n'en faisait profiter personne, et un agent
coincé le restait jusqu'à la fin.

Ce module apporte trois choses :

1. **Des rôles réellement spécialisés.** Chaque rôle pondère différemment les
   mêmes règles d'intégration : l'éclaireur cherche un passage, le lisseur
   soigne les rayons de cintrage, le poseur s'occupe des crabes. Ils ne se
   marchent pas dessus, ils se complètent.

2. **Un curseur exploration / exploitation unique.** Une seule valeur entre 0
   et 1 pilote de façon cohérente le bruit, le pas, l'inertie et la fréquence
   des échanges entre agents. L'utilisateur manipule une notion qu'il
   comprend (« chercher large » ou « peaufiner »), pas six hyperparamètres.

3. **Des échanges entre agents (Population-Based Training).** Régulièrement,
   les routes sont classées avec le score DMU de :mod:`core.routing_rules`.
   Les agents en retard repartent de la meilleure route trouvée, avec des
   réglages perturbés. C'est ce mécanisme qui transforme cinq recherches
   isolées en une vraie recherche coordonnée.

En prime, l'équipe suit un **curriculum** : tant qu'il reste des interférences,
on privilégie ceux qui savent en sortir ; une fois la route saine, l'effort
bascule vers les fixations puis vers le lissage. Inutile de polir une route qui
traverse encore une cloison.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace

__all__ = [
    "Role",
    "ROLES",
    "TEAM_PRESETS",
    "Phase",
    "ExplorationPolicy",
    "AgentSpec",
    "MigrationOrder",
    "Orchestrator",
    "build_team",
]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Rôles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Role:
    """Spécialité d'un agent : ce qu'il cherche et comment il s'y prend.

    ``weights`` multiplie les termes de récompense de :mod:`core.reward_terms`.
    Un rôle ne débranche jamais une règle (tous les poids restent > 0) : il
    déplace seulement l'effort. Un lisseur reste tenu de ne pas percuter une
    cloison, il y consacre simplement moins d'attention qu'un éclaireur.
    """

    key: str
    label_fr: str
    label_en: str
    description_fr: str
    description_en: str
    color: str
    algo: str = "td3"  # "td3", "sac" ou "recurrent_td3"
    weights: dict = field(default_factory=dict)
    noise_scale: float = 1.0
    shift_scale: float = 1.0
    momentum: float = 0.5
    train_steps: int = 1
    #: Un rôle « patient » voit ses seuils de stagnation allongés : chercher un
    #: passage prend plus de temps que peaufiner une courbe.
    patience: float = 1.0

    def label(self, lang: str = "FR") -> str:
        return self.label_en if str(lang).upper().startswith("EN") else self.label_fr

    def description(self, lang: str = "FR") -> str:
        return self.description_en if str(lang).upper().startswith("EN") else self.description_fr


#: Poids de référence, valables pour tous les rôles avant spécialisation.
BASE_WEIGHTS = {
    "clearance": 1.0,   # rester dans la bande de distance au DMU
    "clash": 1.0,       # ne rien traverser
    "bend": 1.0,        # rayon de cintrage
    "straight": 1.0,    # longues lignes droites
    "zigzag": 1.0,      # ne pas osciller (aucun rôle ne descend en dessous)
    "free_span": 1.0,   # ne pas traverser le vide
    "fixation": 1.0,    # crabes tous les 250 mm, posés à plat
    "length": 1.0,      # ne pas rallonger inutilement
}


def _weights(**overrides) -> dict:
    merged = dict(BASE_WEIGHTS)
    merged.update(overrides)
    return merged


ROLES: dict[str, Role] = {
    "scout": Role(
        key="scout",
        label_fr="Éclaireur",
        label_en="Scout",
        description_fr=(
            "Cherche un passage praticable, quitte à être approximatif. "
            "Grands déplacements, forte prise de risque."
        ),
        description_en=(
            "Looks for a viable corridor, even a rough one. "
            "Large moves, high risk taking."
        ),
        color="#00B4D8",
        algo="td3",
        weights=_weights(clash=1.6, free_span=1.5, clearance=1.2, bend=0.4, straight=0.4,
                         fixation=0.3, zigzag=1.0),
        noise_scale=1.6,
        shift_scale=1.5,
        momentum=0.3,
        train_steps=1,
        patience=2.0,
    ),
    "clearance": Role(
        key="clearance",
        label_fr="Contrôleur d'écarts",
        label_en="Clearance keeper",
        description_fr=(
            "Traque les interférences et les distances insuffisantes, "
            "y compris les distances renforcées (air chaud, hydraulique)."
        ),
        description_en=(
            "Hunts interferences and insufficient clearances, including "
            "reinforced ones (hot air, hydraulics)."
        ),
        color="#EF476F",
        algo="td3",
        weights=_weights(clearance=2.0, clash=2.0, free_span=1.2, bend=0.7, straight=0.6,
                         fixation=0.5, zigzag=1.0),
        noise_scale=0.9,
        shift_scale=1.0,
        momentum=0.5,
        train_steps=2,
        patience=1.5,
    ),
    "smoother": Role(
        key="smoother",
        label_fr="Lisseur",
        label_en="Smoother",
        description_fr=(
            "Élargit les rayons de cintrage et supprime les cassures. "
            "Petits déplacements, forte inertie."
        ),
        description_en=(
            "Widens bend radii and removes kinks. Small moves, high inertia."
        ),
        color="#118AB2",
        algo="sac",
        weights=_weights(bend=2.2, straight=1.4, clearance=1.0, clash=1.0, fixation=0.6,
                         length=1.2, zigzag=2.0),
        noise_scale=0.4,
        shift_scale=0.6,
        momentum=0.8,
        train_steps=3,
        patience=1.0,
    ),
    "straightener": Role(
        key="straightener",
        label_fr="Rectifieur",
        label_en="Straightener",
        description_fr=(
            "Allonge les lignes droites et réduit le nombre de coudes : "
            "un cheminement propre est d'abord un cheminement droit."
        ),
        description_en=(
            "Extends straight runs and cuts down the number of bends: "
            "a clean route is first of all a straight one."
        ),
        color="#06D6A0",
        algo="recurrent_td3",
        weights=_weights(straight=2.4, bend=1.6, length=1.4, clearance=1.0, clash=1.0,
                         fixation=0.5, zigzag=2.2),
        noise_scale=0.5,
        shift_scale=0.8,
        momentum=0.75,
        train_steps=2,
        patience=1.2,
    ),
    "fixer": Role(
        key="fixer",
        label_fr="Poseur de crabes",
        label_en="Clamp setter",
        description_fr=(
            "Fait passer le câble là où l'on peut réellement le fixer, "
            "tous les 250 mm, avec des crabes à plat sur la structure."
        ),
        description_en=(
            "Routes the cable where it can actually be clamped, every 250 mm, "
            "with clamps seated flat on the structure."
        ),
        color="#FFD166",
        algo="recurrent_td3",
        weights=_weights(fixation=2.6, clearance=1.3, clash=1.2, free_span=1.4, bend=0.8,
                         straight=0.7, zigzag=1.0),
        noise_scale=1.1,
        shift_scale=1.0,
        momentum=0.5,
        train_steps=2,
        patience=2.0,
    ),
}


#: Compositions d'équipe proposées dans l'interface. Chaque entrée indique
#: combien d'agents de chaque rôle lancer.
TEAM_PRESETS: dict[str, dict] = {
    "discovery": {
        "label_fr": "Découverte",
        "label_en": "Discovery",
        "help_fr": "Beaucoup d'éclaireurs : à utiliser quand aucun passage n'est encore connu.",
        "help_en": "Mostly scouts: use when no corridor is known yet.",
        "composition": {"scout": 3, "clearance": 1, "smoother": 1},
    },
    "balanced": {
        "label_fr": "Équilibrée",
        "label_en": "Balanced",
        "help_fr": "Un agent par spécialité : le choix par défaut, adapté à la plupart des cas.",
        "help_en": "One agent per specialty: the default, suitable for most cases.",
        "composition": {"scout": 1, "clearance": 1, "smoother": 1, "straightener": 1, "fixer": 1},
    },
    "finishing": {
        "label_fr": "Finition",
        "label_en": "Finishing",
        "help_fr": "Lissage et fixations : à utiliser sur une route déjà saine, pour la rendre livrable.",
        "help_en": "Smoothing and clamps: use on an already sound route, to make it deliverable.",
        "composition": {"smoother": 2, "straightener": 2, "fixer": 1},
    },
    "compliance": {
        "label_fr": "Mise en conformité",
        "label_en": "Compliance",
        "help_fr": "Priorité aux interférences et aux distances : pour débloquer une route en clash.",
        "help_en": "Interference and clearance first: to unblock a clashing route.",
        "composition": {"clearance": 2, "scout": 1, "fixer": 1, "smoother": 1},
    },
}


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


class Phase:
    """Étape courante du chantier, déduite de l'état de la meilleure route."""

    FEASIBILITY = "feasibility"  # il reste des interférences
    CLEARANCE = "clearance"      # plus de clash, mais des distances trop justes
    SUPPORT = "support"          # route saine, mais mal tenue
    POLISH = "polish"            # tout est conforme, on peaufine

    #: Rôle mis en avant à chaque étape.
    PRIORITY: dict[str, str] = {
        FEASIBILITY: "scout",
        CLEARANCE: "clearance",
        SUPPORT: "fixer",
        POLISH: "smoother",
    }

    LABELS_FR: dict[str, str] = {
        FEASIBILITY: "Recherche d'un passage",
        CLEARANCE: "Mise aux distances",
        SUPPORT: "Pose des fixations",
        POLISH: "Lissage final",
    }

    LABELS_EN: dict[str, str] = {
        FEASIBILITY: "Finding a corridor",
        CLEARANCE: "Meeting clearances",
        SUPPORT: "Placing fixations",
        POLISH: "Final smoothing",
    }

    @staticmethod
    def label(phase: str, lang: str = "FR") -> str:
        table = Phase.LABELS_EN if str(lang).upper().startswith("EN") else Phase.LABELS_FR
        return table.get(phase, phase)

    @staticmethod
    def from_kpis(kpis: dict) -> str:
        """Déduit l'étape en cours des indicateurs de la meilleure route."""
        if not kpis:
            return Phase.FEASIBILITY
        if int(kpis.get("n_clashes", 0)) > 0:
            return Phase.FEASIBILITY
        if int(kpis.get("n_margin_violations", 0)) > 0:
            return Phase.CLEARANCE
        gap = float(kpis.get("worst_support_gap_mm", 0.0))
        pitch = float(kpis.get("fixation_pitch_mm", 250.0))
        if gap > pitch or float(kpis.get("longest_free_span_mm", 0.0)) > pitch:
            return Phase.SUPPORT
        return Phase.POLISH


#: Trois repères nommés sur le curseur exploration / exploitation. Un réglage
#: continu sans butée ne se compare pas : pour juger de l'effet de
#: l'exploration, il faut pouvoir relancer deux fois exactement au même
#: endroit. Les valeurs sont les deux extrêmes et leur milieu, rien de plus.
EXPLORATION_MODES: dict[str, float] = {
    "explore": 1.0,
    "balanced": 0.5,
    "exploit": 0.0,
}


# ---------------------------------------------------------------------------
# Exploration / exploitation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplorationPolicy:
    """Traduit un curseur unique en réglages cohérents pour toute l'équipe.

    ``temperature`` va de 0 (exploitation pure : on peaufine ce qu'on a) à 1
    (exploration pure : on cherche large, quitte à casser ce qui marchait).

    Le point important est la **cohérence** : monter le bruit sans allonger la
    patience ni espacer les échanges revient à agiter les agents sans leur
    laisser le temps d'aboutir. Ici les six réglages bougent ensemble.
    """

    temperature: float = 0.5

    def __post_init__(self):
        object.__setattr__(self, "temperature", float(min(max(self.temperature, 0.0), 1.0)))

    @property
    def noise_start(self) -> float:
        """Bruit d'exploration au démarrage."""
        return _lerp(0.10, 0.90, self.temperature)

    @property
    def noise_floor(self) -> float:
        """Bruit résiduel : en exploitation pure, l'agent finit par se figer."""
        return _lerp(0.0, 0.25, self.temperature)

    @property
    def noise_decay(self) -> float:
        """Vitesse d'extinction du bruit (proche de 1 = s'éteint lentement)."""
        return _lerp(0.980, 0.9995, self.temperature)

    @property
    def shift_scale(self) -> float:
        """Amplitude des déplacements, relative au réglage de base."""
        return _lerp(0.5, 1.8, self.temperature)

    @property
    def momentum(self) -> float:
        """Inertie : forte en exploitation (mouvements réguliers)."""
        return _lerp(0.85, 0.25, self.temperature)

    @property
    def migration_interval(self) -> int:
        """Nombre d'itérations entre deux échanges entre agents."""
        return int(round(_lerp(25, 250, self.temperature)))

    @property
    def migration_fraction(self) -> float:
        """Part de l'équipe redémarrée depuis la meilleure route trouvée."""
        return _lerp(0.5, 0.0, self.temperature)

    @property
    def patience_scale(self) -> float:
        """Allongement des seuils de stagnation."""
        return _lerp(0.6, 2.5, self.temperature)

    def label(self, lang: str = "FR") -> str:
        """Formulation lisible du réglage, pour l'interface."""
        en = str(lang).upper().startswith("EN")
        if self.temperature < 0.2:
            return "Refine what works" if en else "Peaufiner l'existant"
        if self.temperature < 0.45:
            return "Careful" if en else "Prudent"
        if self.temperature < 0.65:
            return "Balanced" if en else "Équilibré"
        if self.temperature < 0.85:
            return "Adventurous" if en else "Audacieux"
        return "Search wide" if en else "Chercher large"

    def as_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "noise_start": self.noise_start,
            "noise_floor": self.noise_floor,
            "noise_decay": self.noise_decay,
            "shift_scale": self.shift_scale,
            "momentum": self.momentum,
            "migration_interval": self.migration_interval,
            "migration_fraction": self.migration_fraction,
            "patience_scale": self.patience_scale,
        }


# ---------------------------------------------------------------------------
# Constitution de l'équipe
# ---------------------------------------------------------------------------


@dataclass
class AgentSpec:
    """Fiche d'un agent : son rôle, ses réglages, sa couleur d'affichage."""

    name: str
    role: str
    algo: str
    color: str
    weights: dict
    noise_start: float
    noise_floor: float
    noise_decay: float
    shift_scale: float
    momentum: float
    train_steps: int
    patience: float

    def label(self, lang: str = "FR") -> str:
        role = ROLES.get(self.role)
        base = role.label(lang) if role else self.role
        suffix = self.name.rsplit("#", 1)[-1] if "#" in self.name else ""
        return f"{base} {suffix}".strip()

    def as_dict(self) -> dict:
        return asdict(self)


def build_team(
    composition: dict | str = "balanced",
    policy: ExplorationPolicy | float = 0.5,
) -> list[AgentSpec]:
    """Constitue une équipe d'agents à partir d'une composition et d'un curseur.

    Args:
        composition: nom d'un preset de :data:`TEAM_PRESETS`, ou dict
            ``{clé de rôle: nombre d'agents}``.
        policy: :class:`ExplorationPolicy`, ou directement la température.

    Returns:
        Les fiches des agents à lancer, dans l'ordre d'affichage.
    """
    if isinstance(policy, (int, float)):
        policy = ExplorationPolicy(float(policy))

    if isinstance(composition, str):
        preset = TEAM_PRESETS.get(composition, TEAM_PRESETS["balanced"])
        counts = dict(preset["composition"])
    else:
        counts = {k: int(v) for k, v in dict(composition).items() if int(v) > 0}

    if not counts:
        counts = dict(TEAM_PRESETS["balanced"]["composition"])

    team: list[AgentSpec] = []
    for role_key, count in counts.items():
        role = ROLES.get(role_key)
        if role is None:
            continue
        for i in range(count):
            suffix = f"#{i + 1}" if count > 1 else ""
            team.append(
                AgentSpec(
                    name=f"{role_key}{suffix}",
                    role=role_key,
                    algo=role.algo,
                    color=role.color,
                    weights=dict(role.weights),
                    noise_start=policy.noise_start * role.noise_scale,
                    noise_floor=policy.noise_floor * role.noise_scale,
                    noise_decay=policy.noise_decay,
                    shift_scale=policy.shift_scale * role.shift_scale,
                    momentum=role.momentum * (policy.momentum / 0.55),
                    train_steps=role.train_steps,
                    patience=role.patience * policy.patience_scale,
                )
            )
    return team


# ---------------------------------------------------------------------------
# Arbitrage
# ---------------------------------------------------------------------------


@dataclass
class MigrationOrder:
    """Ordre d'échange : ``loser`` repart de la route de ``winner``."""

    loser: str
    winner: str
    reason_fr: str
    reason_en: str
    perturbation: dict

    def reason(self, lang: str = "FR") -> str:
        return self.reason_en if str(lang).upper().startswith("EN") else self.reason_fr


class Orchestrator:
    """Arbitre l'équipe : classement, curriculum et échanges entre agents.

    L'orchestrateur ne calcule aucune trajectoire. Il observe les scores DMU
    publiés par les agents et décide qui repart de quoi. Il est volontairement
    sans dépendance lourde (ni torch, ni trimesh) pour rester testable seul.
    """

    def __init__(
        self,
        composition: dict | str = "balanced",
        temperature: float = 0.5,
        seed: int | None = None,
    ):
        self.policy = ExplorationPolicy(temperature)
        self.composition = composition
        self.team = build_team(composition, self.policy)
        self.phase = Phase.FEASIBILITY
        self.history: list[MigrationOrder] = []
        self._last_migration_iter = 0
        self._rng = random.Random(seed)

    # -- réglages ------------------------------------------------------

    def set_temperature(self, temperature: float) -> list[AgentSpec]:
        """Change le curseur exploration/exploitation et régénère les réglages.

        Les agents en cours ne sont pas relancés : seuls leurs réglages
        évoluent, ce qui permet de bouger le curseur en pleine session.
        """
        self.policy = ExplorationPolicy(temperature)
        refreshed = build_team(self.composition, self.policy)
        by_name = {spec.name: spec for spec in refreshed}
        for i, spec in enumerate(self.team):
            new = by_name.get(spec.name)
            if new is not None:
                self.team[i] = replace(
                    new,
                    # Le bruit courant appartient à l'agent : on ne remet pas
                    # son exploration à zéro parce que l'utilisateur a bougé
                    # le curseur.
                    noise_start=spec.noise_start,
                )
        return self.team

    def set_composition(self, composition: dict | str) -> list[AgentSpec]:
        """Change la composition de l'équipe (nécessite un redémarrage)."""
        self.composition = composition
        self.team = build_team(composition, self.policy)
        return self.team

    # -- lecture de l'état ---------------------------------------------

    @staticmethod
    def rank(scores: dict[str, tuple]) -> list[str]:
        """Classe les agents par score DMU croissant (le meilleur en tête)."""
        return [name for name, _ in sorted(scores.items(), key=lambda kv: kv[1])]

    def update_phase(self, best_kpis: dict) -> str:
        """Met à jour l'étape du chantier et renvoie la nouvelle valeur."""
        self.phase = Phase.from_kpis(best_kpis or {})
        return self.phase

    def phase_weight_boost(self, role_key: str) -> float:
        """Coefficient appliqué au rôle prioritaire de l'étape en cours.

        Le rôle attendu à cette étape travaille à pleine puissance, les autres
        lèvent légèrement le pied : inutile de polir une route qui traverse
        encore une cloison.
        """
        return 1.0 if Phase.PRIORITY.get(self.phase) == role_key else 0.85

    # -- échanges entre agents -----------------------------------------

    def should_migrate(self, iteration: int) -> bool:
        """Indique s'il est temps de procéder à un échange."""
        if self.policy.migration_fraction <= 0.0:
            return False
        return (iteration - self._last_migration_iter) >= self.policy.migration_interval

    def plan_migration(self, scores: dict[str, tuple], iteration: int) -> list[MigrationOrder]:
        """Désigne les agents qui repartent de la meilleure route trouvée.

        C'est l'étape « exploit » du Population-Based Training : on recopie la
        solution du meilleur chez les retardataires, puis on perturbe leurs
        réglages (étape « explore ») pour qu'ils n'aillent pas tous refaire
        exactement la même chose.

        Le dosage est entièrement porté par le curseur : en exploitation, la
        moitié de l'équipe se rallie souvent à la meilleure route ; en
        exploration, plus personne n'est rappelé et chacun creuse sa piste,
        même si elle paraît moins bonne pour l'instant.
        """
        if len(scores) < 2 or not self.should_migrate(iteration):
            return []

        ordered = self.rank(scores)
        winner = ordered[0]
        n_losers = int(len(ordered) * self.policy.migration_fraction)
        if n_losers < 1:
            return []

        orders: list[MigrationOrder] = []
        for loser in ordered[-n_losers:]:
            # Migrer transmet une *route*, pas un rôle : l'agent qui repart de
            # la meilleure solution garde ses propres poids et continue donc à
            # travailler sa spécialité. La spécialisation de l'équipe est
            # préservée sans avoir besoin de protéger qui que ce soit.
            if loser == winner or scores[loser] <= scores[winner]:
                continue
            orders.append(
                MigrationOrder(
                    loser=loser,
                    winner=winner,
                    reason_fr=f"Repart de la route de « {winner} », en retard au classement DMU.",
                    reason_en=f"Restarts from “{winner}”'s route, trailing in the DMU ranking.",
                    perturbation=self._perturb(),
                )
            )

        if orders:
            self._last_migration_iter = iteration
            self.history.extend(orders)
            del self.history[:-50]  # on ne conserve que l'historique récent
        return orders

    def _perturb(self) -> dict:
        """Perturbe les hyperparamètres d'un agent qui vient de migrer.

        Sans cette perturbation, tous les agents finiraient par explorer à
        l'identique autour de la même route : la population perdrait sa
        diversité et n'apporterait plus rien par rapport à un agent unique.
        """
        return {
            "noise_scale": self._rng.uniform(0.7, 1.5),
            "shift_scale": self._rng.uniform(0.8, 1.3),
            "momentum_delta": self._rng.uniform(-0.15, 0.15),
        }

    # -- restitution ---------------------------------------------------

    def summary(self, scores: dict[str, tuple] | None = None, lang: str = "FR") -> dict:
        """État de l'équipe, prêt à afficher dans l'interface."""
        ordered = self.rank(scores) if scores else [spec.name for spec in self.team]
        return {
            "phase": self.phase,
            "phase_label": Phase.label(self.phase, lang),
            "temperature": self.policy.temperature,
            "temperature_label": self.policy.label(lang),
            "ranking": ordered,
            "best": ordered[0] if ordered else None,
            "team": [
                {
                    "name": spec.name,
                    "role": spec.role,
                    "label": spec.label(lang),
                    "color": spec.color,
                    "algo": spec.algo,
                    "rank": ordered.index(spec.name) + 1 if spec.name in ordered else None,
                }
                for spec in self.team
            ],
            "migrations": [
                {"loser": m.loser, "winner": m.winner, "reason": m.reason(lang)}
                for m in self.history[-5:]
            ],
        }
