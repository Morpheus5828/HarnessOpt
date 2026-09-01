"""Règles d'intégration (DMU) applicables à un cheminement de harnais.

Ce module est le **point de vérité unique** du projet : la même définition de
« route conforme » sert à la fonction de récompense des agents et au panneau de
conformité de l'interface. Tant qu'il n'y avait pas de définition partagée, un
agent pouvait converger vers une route que l'intégrateur aurait refusée.

Les règles couvertes :

===========================  ==================================================
Règle                        Signification métier
===========================  ==================================================
``clash``                    Aucune interférence : le câble ne traverse ni ne
                             touche une pièce du DMU.
``clearance_min``            Distance minimale respectée, éventuellement
                             différenciée selon la nature de la pièce (air
                             chaud, hydraulique haute pression, ...).
``clearance_max``            Le câble reste à portée d'une structure : il doit
                             longer le DMU, pas flotter au milieu du vide.
``bend_radius``              Rayon de cintrage réalisable >= rayon admissible
                             du toron : pas de cassure.
``free_span``                Pas de traversée à vide plus longue que le pas de
                             fixation : sinon le câble pend.
``fixation_pitch``           Un point de fixation au moins tous les 250 mm.
``fixation_parallel``        Chaque crabe repose à plat sur la structure.
===========================  ==================================================

La sévérité distingue ce qui interdit la livraison (``BLOCKING``) de ce qui la
dégrade (``MAJOR``) et de ce qui relève de la qualité (``MINOR``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from core import geometry_metrics as gm

__all__ = [
    "Severity",
    "HarnessSpec",
    "ClearanceModel",
    "RoutingRules",
    "RuleCheck",
    "RouteReport",
    "evaluate_route",
    "DEFAULT_FAMILY_CLEARANCE",
    "RULE_CATALOG",
    "RULE_IDS",
    "ALL_RULES",
    "rule_info",
]


class Severity:
    """Gravité d'une règle non respectée."""

    BLOCKING = "blocking"  # rédhibitoire : la route ne peut pas être livrée
    MAJOR = "major"        # à corriger : la route est livrable sous réserve
    MINOR = "minor"        # qualité : la route est acceptable mais perfectible


@dataclass(frozen=True)
class RuleInfo:
    """Fiche d'identité d'une règle, telle que présentée à l'utilisateur.

    Elle sert à construire la page « Règles » sans y recopier de libellés :
    ajouter une règle au catalogue suffit à la faire apparaître à l'écran avec
    sa case à cocher.
    """

    rule_id: str
    label_fr: str
    label_en: str
    help_fr: str
    help_en: str
    severity: str
    #: Famille de récompense à neutraliser si la règle est décochée.
    reward_key: str = ""

    def label(self, lang: str = "FR") -> str:
        return self.label_en if str(lang).upper().startswith("EN") else self.label_fr

    def help(self, lang: str = "FR") -> str:
        return self.help_en if str(lang).upper().startswith("EN") else self.help_fr


#: Toutes les règles connues, dans l'ordre d'affichage. C'est la seule liste
#: qui fasse foi : l'interface, le moteur d'évaluation et la récompense s'y
#: réfèrent tous, ce qui interdit qu'une règle soit cochable sans être
#: réellement appliquée.
RULE_CATALOG: tuple[RuleInfo, ...] = (
    RuleInfo(
        "clash",
        "Aucune interférence avec la structure",
        "No interference with the structure",
        "Le câble ne touche ni ne traverse aucune pièce. Décocher cette règle "
        "revient à accepter des clashs : à ne faire que pour une étude "
        "exploratoire.",
        "The cable neither touches nor crosses any part. Unticking this means "
        "accepting clashes: exploratory studies only.",
        Severity.BLOCKING,
        "clash",
    ),
    RuleInfo(
        "clearance_min",
        "Distance minimale à la structure",
        "Minimum clearance to structure",
        "Le câble reste au moins à la distance minimale de chaque pièce, "
        "renforcée pour les familles qui l'exigent.",
        "The cable stays at least the minimum distance from every part, "
        "increased for the families that require it.",
        Severity.BLOCKING,
        "clearance",
    ),
    RuleInfo(
        "clearance_max",
        "Câble maintenu à portée de la structure",
        "Cable kept within reach of structure",
        "Le câble ne s'éloigne pas au-delà de la distance maximale : au-delà, "
        "plus rien ne permet de le fixer.",
        "The cable never drifts beyond the maximum distance, past which "
        "nothing can hold it.",
        Severity.MAJOR,
        "clearance",
    ),
    RuleInfo(
        "edge_clearance",
        "Distance minimale au bord de tôle",
        "Minimum clearance to sheet edge",
        "Le câble ne longe pas un bord libre de la structure. Un chant de tôle "
        "use la gaine, et surtout ne peut recevoir aucune fixation : longer un "
        "bord, c'est router là où rien ne tiendra le faisceau.",
        "The cable does not run along a free edge of the structure. A sheet "
        "edge chafes the sleeve and, above all, can take no fixation: routing "
        "along an edge means routing where nothing will hold the harness.",
        Severity.MAJOR,
        reward_key="edge",
    ),
    RuleInfo(
        "free_span",
        "Pas de traversée dans le vide",
        "No unsupported crossing",
        "Le câble ne traverse pas un espace vide sur une longueur où il ne "
        "pourrait être tenu.",
        "The cable does not cross empty space over a length where it could "
        "not be held.",
        Severity.MAJOR,
        "free_span",
    ),
    RuleInfo(
        "bend_radius",
        "Rayon de cintrage admissible",
        "Allowable bend radius",
        "Aucun coude plus serré que le rayon de cintrage minimal du toron. "
        "C'est cette règle qui garantit un câble lisse plutôt que plié.",
        "No bend tighter than the harness minimum bend radius. This is the "
        "rule that yields a smooth cable rather than a kinked one.",
        Severity.BLOCKING,
        "bend",
    ),
    RuleInfo(
        "fixation_pitch",
        "Une fixation au moins tous les N mm",
        "A fixation at least every N mm",
        "Un point de fixation existant ou un crabe posé par l'agent au moins "
        "tous les N millimètres.",
        "An existing fixation point, or a clamp placed by the agent, at least "
        "every N millimetres.",
        Severity.MAJOR,
        "fixation",
    ),
    RuleInfo(
        "fixation_parallel",
        "Crabes posés à plat sur la structure",
        "Clamps seated flat on the structure",
        "L'embase de chaque crabe reste parallèle à la structure, à la "
        "tolérance de pose près.",
        "Every clamp base stays parallel to the structure, within the seating "
        "tolerance.",
        Severity.MAJOR,
        "fixation",
    ),
    RuleInfo(
        "straightness",
        "Tracé droit le plus longtemps possible",
        "Straight run kept as long as possible",
        "Le câble privilégie les longues portions rectilignes plutôt qu'une "
        "succession de petits virages.",
        "The cable favours long straight runs over a succession of small "
        "turns.",
        Severity.MINOR,
        "straight",
    ),
)

#: Identifiants des règles, dans l'ordre du catalogue.
RULE_IDS: tuple[str, ...] = tuple(info.rule_id for info in RULE_CATALOG)

#: Jeu complet : toutes les règles appliquées. C'est le comportement par défaut.
ALL_RULES: frozenset = frozenset(RULE_IDS)

_RULE_BY_ID = {info.rule_id: info for info in RULE_CATALOG}


def rule_info(rule_id: str) -> RuleInfo | None:
    """Fiche d'une règle, ou ``None`` si l'identifiant est inconnu."""
    return _RULE_BY_ID.get(rule_id)


#: Distance minimale (mm) entre le câble et chaque famille de pièces du DMU.
#: Les valeurs reprennent celles déjà présentes dans ``config.py`` et
#: constituent le point d'entrée de la demande « certaines couleurs de maillage
#: ont besoin d'une distance plus grande ».
DEFAULT_FAMILY_CLEARANCE: dict[str, float] = {
    "standard": 10.0,
    "structure": 10.0,
    "equipement": 10.0,
    "insonorisation": 10.0,
    "copper_foils": 10.0,
    "mecanical_installation": 10.0,
    "flight_control_system-fcs": 25.0,
    "ecs_air_circuit": 20.0,        # air chaud : DISTANCE_HOT_AIR_LINES
    "ecs_cold_circuit": 10.0,       # ventilation / réfrigérant
    "p3_circuit": 20.0,             # prélèvement moteur, chaud
    "fuel": 25.0,
    "high_pressure_system": 70.0,   # hydraulique haute pression
    "return_system": 25.0,
    "air_azote": 20.0,
    "suction": 20.0,
}


@dataclass
class HarnessSpec:
    """Caractéristiques du toron à cheminer."""

    #: Diamètre extérieur du toron, en mm.
    diameter_mm: float = 40.0
    #: Rayon de cintrage admissible exprimé en multiples du diamètre.
    #: 6 x D est la valeur usuelle pour un faisceau électrique aéronautique.
    bend_radius_factor: float = 6.0

    @property
    def radius_mm(self) -> float:
        """Rayon du toron (demi-diamètre)."""
        return self.diameter_mm / 2.0

    @property
    def min_bend_radius_mm(self) -> float:
        """Rayon de cintrage minimal admissible, en mm."""
        return self.diameter_mm * self.bend_radius_factor


@dataclass
class ClearanceModel:
    """Distances de sécurité, éventuellement différenciées par famille de pièce.

    Sans information de couleur, le modèle se comporte comme un simple couple
    (mini, maxi). Dès que la fusion fournit la table « face -> famille »
    (:mod:`core.mesh_processor`), la distance minimale exigée devient propre à
    la pièce réellement survolée par le câble.
    """

    #: Distance minimale par défaut, en mm.
    default_min_mm: float = 10.0
    #: Distance maximale, en mm : au-delà, le câble n'a plus rien à longer.
    max_mm: float = 100.0
    #: Distance minimale par famille de couleur DMU.
    per_family: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FAMILY_CLEARANCE))
    #: Famille de chaque face du maillage fusionné (taille ``n_cells``).
    face_family: np.ndarray | None = None
    #: Noms des familles, indexés par les codes de ``face_family``.
    family_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._table: np.ndarray | None = None
        self._build_table()

    def _build_table(self):
        """Pré-calcule un tableau « code famille -> distance mini »."""
        if not self.family_names:
            self._table = None
            return
        self._table = np.array(
            [float(self.per_family.get(name, self.default_min_mm)) for name in self.family_names],
            dtype=np.float32,
        )

    def with_face_families(self, face_family: np.ndarray, family_names) -> "ClearanceModel":
        """Renvoie une copie du modèle enrichie de la table des faces."""
        return ClearanceModel(
            default_min_mm=self.default_min_mm,
            max_mm=self.max_mm,
            per_family=dict(self.per_family),
            face_family=np.asarray(face_family, dtype=np.int32),
            family_names=[str(n) for n in family_names],
        )

    @property
    def is_differentiated(self) -> bool:
        """True si le modèle sait distinguer les familles de pièces."""
        return self._table is not None and self.face_family is not None

    @property
    def strictest_mm(self) -> float:
        """Distance minimale la plus contraignante réellement applicable."""
        if self._table is None or self.face_family is None or len(self.face_family) == 0:
            return self.default_min_mm
        present = np.unique(self.face_family)
        present = present[(present >= 0) & (present < len(self._table))]
        if len(present) == 0:
            return self.default_min_mm
        return float(self._table[present].max())

    def required_min(self, face_indices=None, n: int | None = None) -> np.ndarray:
        """Distance minimale exigée pour chaque point d'une trajectoire.

        Args:
            face_indices: index de la face du DMU la plus proche de chaque
                point (fourni par ``ProximityQuery.on_surface``). ``None``
                pour appliquer la distance par défaut partout.
            n: nombre de points, utilisé quand ``face_indices`` vaut ``None``.

        Returns:
            Tableau ``(n,)`` de distances minimales, en mm.
        """
        if face_indices is None:
            size = int(n or 0)
            return np.full(size, self.default_min_mm, dtype=np.float32)

        faces = np.asarray(face_indices, dtype=np.int64)
        if self._table is None or self.face_family is None:
            return np.full(len(faces), self.default_min_mm, dtype=np.float32)

        valid = (faces >= 0) & (faces < len(self.face_family))
        out = np.full(len(faces), self.default_min_mm, dtype=np.float32)
        codes = self.face_family[faces[valid]]
        in_range = (codes >= 0) & (codes < len(self._table))
        chosen = np.full(len(codes), self.default_min_mm, dtype=np.float32)
        chosen[in_range] = self._table[codes[in_range]]
        out[valid] = chosen
        return out


@dataclass
class RoutingRules:
    """Jeu complet de règles appliquées à un cheminement."""

    harness: HarnessSpec = field(default_factory=HarnessSpec)
    clearance: ClearanceModel = field(default_factory=ClearanceModel)

    #: Écart maximal admis entre deux points de fixation, en mm.
    fixation_pitch_mm: float = 250.0
    #: Écart angulaire toléré entre l'embase d'un crabe et la structure, en degrés.
    fixation_parallel_tol_deg: float = 15.0
    #: Distance minimale à un bord libre de la structure, en mm. Un bord libre
    #: est une arête portée par une seule face : le chant de la tôle.
    edge_clearance_mm: float = 25.0
    #: Longueur maximale d'une traversée à vide, en mm. Par défaut, le pas de
    #: fixation : au-delà, plus rien ne tient le câble.
    max_free_span_mm: float | None = None
    #: Angle en deçà duquel le câble est considéré comme rectiligne, en degrés.
    straight_tol_deg: float = 3.0
    #: Part de longueur droite visée (indicateur de qualité, 0..1).
    target_straight_ratio: float = 0.6
    #: Règles réellement appliquées. Par défaut toutes ; l'utilisateur peut en
    #: décocher depuis la page « Règles ». Une règle absente n'est ni évaluée,
    #: ni comptée dans le classement, ni récompensée : la décocher la retire
    #: vraiment du problème, elle ne se contente pas de disparaître du rapport.
    enabled_rules: frozenset = ALL_RULES

    def __post_init__(self):
        # Tolère une liste ou un ensemble, et ignore les identifiants inconnus
        # afin qu'un réglage enregistré par une version antérieure reste lisible.
        self.enabled_rules = frozenset(self.enabled_rules) & ALL_RULES

    def is_enabled(self, rule_id: str) -> bool:
        """La règle est-elle appliquée ?"""
        return rule_id in self.enabled_rules

    def with_rules(self, enabled) -> "RoutingRules":
        """Copie des règles avec un autre jeu de règles actives."""
        return replace(self, enabled_rules=frozenset(enabled))

    def reward_scale(self) -> dict:
        """Multiplicateurs de récompense induits par les règles décochées.

        Une famille de récompense n'est neutralisée que si *toutes* les règles
        qui s'y rattachent sont décochées : décocher la seule distance maximale
        ne doit pas supprimer la pression qui éloigne le câble des pièces.
        """
        scale: dict[str, float] = {}
        for info in RULE_CATALOG:
            if not info.reward_key:
                continue
            scale.setdefault(info.reward_key, 0.0)
            if self.is_enabled(info.rule_id):
                scale[info.reward_key] = 1.0
        return scale

    @property
    def free_span_limit_mm(self) -> float:
        """Limite effective de traversée à vide."""
        return self.max_free_span_mm if self.max_free_span_mm else self.fixation_pitch_mm

    def with_harness_diameter(self, diameter_mm: float) -> "RoutingRules":
        """Copie des règles avec un autre diamètre de toron."""
        return replace(self, harness=replace(self.harness, diameter_mm=float(diameter_mm)))


@dataclass
class RuleCheck:
    """Résultat d'une règle : conforme ou non, avec la valeur mesurée."""

    rule_id: str
    label_fr: str
    label_en: str
    passed: bool
    value: float
    limit: float
    unit: str
    severity: str
    detail_fr: str = ""
    detail_en: str = ""

    def label(self, lang: str = "FR") -> str:
        return self.label_en if str(lang).upper().startswith("EN") else self.label_fr

    def detail(self, lang: str = "FR") -> str:
        return self.detail_en if str(lang).upper().startswith("EN") else self.detail_fr


@dataclass
class RouteReport:
    """Verdict complet sur une trajectoire."""

    checks: list[RuleCheck]
    kpis: dict
    #: Règles appliquées lors de cette évaluation. ``None`` = toutes.
    enabled_rules: frozenset | None = None

    @property
    def is_compliant(self) -> bool:
        """True si aucune règle n'est enfreinte."""
        return all(check.passed for check in self.checks)

    @property
    def is_deliverable(self) -> bool:
        """True si aucune règle rédhibitoire n'est enfreinte."""
        return all(check.passed for check in self.checks if check.severity == Severity.BLOCKING)

    def failed(self, severity: str | None = None) -> list[RuleCheck]:
        """Règles non respectées, filtrables par gravité."""
        return [
            c for c in self.checks
            if not c.passed and (severity is None or c.severity == severity)
        ]

    @property
    def compliance_ratio(self) -> float:
        """Part de règles respectées, 0..1 (pour une barre de progression)."""
        if not self.checks:
            return 1.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)

    def score(self) -> tuple:
        """Clé de tri lexicographique, du plus grave au plus anecdotique.

        Comparer deux routes revient à comparer ce tuple : plus il est petit,
        meilleure est la route. L'orchestrateur s'en sert pour désigner le
        meilleur agent de la population, et l'interface pour afficher le
        classement. L'ordre traduit la hiérarchie métier : on ne troque jamais
        un clash contre un peu de lissage.
        """
        k = self.kpis
        active = self.enabled_rules if self.enabled_rules is not None else ALL_RULES

        def on(rule_id: str, value: float) -> float:
            """Neutralise le critère si la règle correspondante est décochée."""
            return value if rule_id in active else 0.0

        return (
            on("clash", int(k.get("n_clashes", 0))),
            on("clearance_min", int(k.get("n_margin_violations", 0))),
            on("clearance_max", int(k.get("n_floating_points", 0))),
            on("bend_radius", int(k.get("n_bend_violations", 0))),
            on("fixation_pitch", round(float(k.get("worst_support_gap_mm", 0.0)), 1)),
            on("fixation_parallel", int(k.get("n_tilted_clamps", 0))),
            on("free_span", round(float(k.get("longest_free_span_mm", 0.0)), 1)),
            on("straightness", round(-float(k.get("straight_ratio", 0.0)), 4)),
            on("straightness", round(float(k.get("total_turning_deg", 0.0)), 1)),
            # Les oscillations ne dépendent d'aucune case à cocher : un zigzag
            # n'est jamais acceptable, quel que soit le jeu de règles retenu.
            int(k.get("n_zigzags", 0)),
            # La longueur reste toujours départageante : elle n'est pas une
            # règle d'intégration mais le critère de dernier recours entre
            # deux routes également conformes.
            round(float(k.get("length_mm", 0.0)), 1),
        )

    def to_dict(self) -> dict:
        """Forme sérialisable, pour l'export du rapport."""
        active = self.enabled_rules if self.enabled_rules is not None else ALL_RULES
        return {
            "compliant": self.is_compliant,
            "deliverable": self.is_deliverable,
            "enabled_rules": sorted(active),
            "disabled_rules": sorted(ALL_RULES - active),
            "kpis": dict(self.kpis),
            "checks": [
                {
                    "id": c.rule_id,
                    "label": c.label_fr,
                    "passed": c.passed,
                    "value": c.value,
                    "limit": c.limit,
                    "unit": c.unit,
                    "severity": c.severity,
                }
                for c in self.checks
            ],
        }


def _fmt(value: float, unit: str = "mm") -> str:
    if value == float("inf"):
        return "∞"
    return f"{value:.0f} {unit}" if abs(value) >= 10 else f"{value:.1f} {unit}"


def evaluate_route(
    points,
    rules: RoutingRules,
    distances=None,
    required_min=None,
    inside_mask=None,
    n_crossings: int = 0,
    clamp_arc_positions=None,
    clamp_tilt_deg=None,
    edge_distances=None,
) -> RouteReport:
    """Confronte une trajectoire aux règles d'intégration.

    Args:
        points: trajectoire ``(n, 3)`` en mm.
        rules: jeu de règles à appliquer.
        distances: distance au DMU de chaque point (``None`` = non mesuré, les
            règles de distance sont alors ignorées plutôt que supposées bonnes).
        required_min: distance minimale exigée point par point. ``None`` pour
            utiliser la distance par défaut du modèle de clearance.
        inside_mask: booléens indiquant les points situés *dans* la matière.
        n_crossings: nombre de segments traversant réellement une face.
        clamp_arc_positions: abscisses curvilignes (mm) des fixations posées.
        clamp_tilt_deg: écart angulaire de chaque crabe vis-à-vis de la
            structure, en degrés.
        edge_distances: distance de chaque point au bord libre le plus proche
            (``None`` = non mesuré ; la règle de bord est alors ignorée plutôt
            que supposée bonne).

    Returns:
        Un :class:`RouteReport` prêt à afficher et à comparer.
    """
    pts = np.asarray(points, dtype=np.float64)
    checks: list[RuleCheck] = []
    kpis: dict = {}

    total_length = gm.path_length(pts)
    straight = gm.straightness(pts, angle_tol_deg=rules.straight_tol_deg)
    r_min_reachable = gm.min_bend_radius(pts)
    bend_limit = rules.harness.min_bend_radius_mm

    kpis.update(
        {
            "n_points": int(len(pts)),
            "length_mm": total_length,
            "min_bend_radius_mm": r_min_reachable,
            "min_curvature_radius_mm": gm.min_curvature_radius(pts),
            "straight_ratio": straight["straight_ratio"],
            "longest_straight_mm": straight["longest_run_mm"],
            "n_bends": straight["n_bends"],
            "total_turning_deg": straight["total_turning_deg"],
        }
    )

    # ------------------------------------------------------------------
    # 1. Interférences (rédhibitoire)
    # ------------------------------------------------------------------
    n_inside = int(np.count_nonzero(inside_mask)) if inside_mask is not None else 0
    n_clashes = int(n_crossings) + n_inside
    kpis["n_clashes"] = n_clashes
    kpis["n_crossings"] = int(n_crossings)
    kpis["n_points_inside"] = n_inside
    checks.append(
        RuleCheck(
            rule_id="clash",
            label_fr="Aucune interférence avec le DMU",
            label_en="No interference with the DMU",
            passed=n_clashes == 0,
            value=float(n_clashes),
            limit=0.0,
            unit="",
            severity=Severity.BLOCKING,
            detail_fr=(
                "Aucun contact ni traversée détectés."
                if n_clashes == 0
                else f"{n_clashes} point(s) ou segment(s) en contact avec la structure."
            ),
            detail_en=(
                "No contact or crossing detected."
                if n_clashes == 0
                else f"{n_clashes} point(s) or segment(s) touching the structure."
            ),
        )
    )

    # ------------------------------------------------------------------
    # 2. Distances au DMU
    # ------------------------------------------------------------------
    if distances is not None and len(distances) == len(pts) and len(pts) > 0:
        dist = np.asarray(distances, dtype=np.float64)
        if required_min is None:
            required = rules.clearance.required_min(None, n=len(pts)).astype(np.float64)
        else:
            required = np.asarray(required_min, dtype=np.float64)

        below = dist < required
        n_below = int(np.count_nonzero(below))
        worst_deficit = float(np.max(required - dist)) if n_below else 0.0

        kpis["min_distance_mm"] = float(dist.min())
        kpis["mean_distance_mm"] = float(dist.mean())
        kpis["max_distance_mm"] = float(dist.max())
        kpis["n_margin_violations"] = n_below
        kpis["required_min_mm"] = float(required.max())

        checks.append(
            RuleCheck(
                rule_id="clearance_min",
                label_fr="Distance mini à la structure respectée",
                label_en="Minimum clearance to structure met",
                passed=n_below == 0,
                value=float(dist.min()),
                limit=float(required.max()),
                unit="mm",
                severity=Severity.BLOCKING,
                detail_fr=(
                    f"Point le plus proche à {_fmt(float(dist.min()))}."
                    if n_below == 0
                    else f"{n_below} point(s) trop près, manque jusqu'à {_fmt(worst_deficit)}."
                ),
                detail_en=(
                    f"Closest point at {_fmt(float(dist.min()))}."
                    if n_below == 0
                    else f"{n_below} point(s) too close, short by up to {_fmt(worst_deficit)}."
                ),
            )
        )

        n_above = int(np.count_nonzero(dist > rules.clearance.max_mm))
        kpis["n_floating_points"] = n_above
        checks.append(
            RuleCheck(
                rule_id="clearance_max",
                label_fr="Câble maintenu à portée de la structure",
                label_en="Cable kept within reach of structure",
                passed=n_above == 0,
                value=float(dist.max()),
                limit=float(rules.clearance.max_mm),
                unit="mm",
                severity=Severity.MAJOR,
                detail_fr=(
                    "Le câble longe la structure sur tout son parcours."
                    if n_above == 0
                    else f"{n_above} point(s) s'éloignent au-delà de {_fmt(rules.clearance.max_mm)}."
                ),
                detail_en=(
                    "The cable follows the structure along its whole run."
                    if n_above == 0
                    else f"{n_above} point(s) drift beyond {_fmt(rules.clearance.max_mm)}."
                ),
            )
        )

        # 3. Traversées à vide
        span = gm.longest_free_span(pts, dist, rules.clearance.max_mm)
        kpis["longest_free_span_mm"] = span
        checks.append(
            RuleCheck(
                rule_id="free_span",
                label_fr="Pas de traversée dans le vide",
                label_en="No unsupported crossing",
                passed=span <= rules.free_span_limit_mm,
                value=span,
                limit=float(rules.free_span_limit_mm),
                unit="mm",
                severity=Severity.MAJOR,
                detail_fr=(
                    "Le câble reste toujours accrochable."
                    if span <= rules.free_span_limit_mm
                    else f"Traversée à vide de {_fmt(span)} : le câble ne peut pas y être tenu."
                ),
                detail_en=(
                    "The cable can be supported everywhere."
                    if span <= rules.free_span_limit_mm
                    else f"{_fmt(span)} unsupported run: the cable cannot be held there."
                ),
            )
        )
    else:
        kpis.setdefault("n_margin_violations", 0)
        kpis.setdefault("longest_free_span_mm", 0.0)

    # ------------------------------------------------------------------
    # 2 bis. Bords libres de la structure
    # ------------------------------------------------------------------
    # Indépendant de la distance à la structure, et c'est tout le sujet : le
    # long d'un bord, la matière est bien là, juste à côté. La règle de
    # distance est donc satisfaite alors que le câble rase un chant.
    if edge_distances is not None and len(edge_distances):
        edges = np.asarray(edge_distances, dtype=np.float64)
        finite = edges[np.isfinite(edges)]
        closest = float(finite.min()) if len(finite) else float("inf")
        n_edge = int(np.count_nonzero(edges < rules.edge_clearance_mm))
        kpis["min_edge_distance_mm"] = closest
        kpis["n_edge_violations"] = n_edge
        checks.append(
            RuleCheck(
                rule_id="edge_clearance",
                label_fr="Distance minimale au bord de tôle",
                label_en="Minimum clearance to sheet edge",
                passed=n_edge == 0,
                value=closest,
                limit=float(rules.edge_clearance_mm),
                unit="mm",
                severity=Severity.MAJOR,
                detail_fr=(
                    "Le câble reste à l'écart des chants de tôle."
                    if n_edge == 0
                    else f"{n_edge} point(s) à moins de "
                         f"{_fmt(rules.edge_clearance_mm)} d'un bord libre : "
                         "rien ne peut y tenir le faisceau."
                ),
                detail_en=(
                    "The cable stays clear of sheet edges."
                    if n_edge == 0
                    else f"{n_edge} point(s) closer than "
                         f"{_fmt(rules.edge_clearance_mm)} to a free edge: "
                         "nothing can hold the harness there."
                ),
            )
        )
    else:
        kpis.setdefault("n_edge_violations", 0)

    # ------------------------------------------------------------------
    # 4. Rayon de cintrage
    # ------------------------------------------------------------------
    radii = gm.bend_radii(pts)
    n_bend_violations = int(np.count_nonzero(radii < bend_limit)) if len(radii) else 0
    kpis["n_bend_violations"] = n_bend_violations
    checks.append(
        RuleCheck(
            rule_id="bend_radius",
            label_fr="Rayon de cintrage admissible",
            label_en="Allowable bend radius",
            passed=n_bend_violations == 0,
            value=r_min_reachable,
            limit=bend_limit,
            unit="mm",
            severity=Severity.BLOCKING,
            detail_fr=(
                f"Cintrage le plus serré : {_fmt(r_min_reachable)} (mini {_fmt(bend_limit)})."
                if n_bend_violations == 0
                else f"{n_bend_violations} cassure(s) : jusqu'à {_fmt(r_min_reachable)} au lieu de {_fmt(bend_limit)}."
            ),
            detail_en=(
                f"Tightest bend: {_fmt(r_min_reachable)} (min {_fmt(bend_limit)})."
                if n_bend_violations == 0
                else f"{n_bend_violations} kink(s): down to {_fmt(r_min_reachable)} instead of {_fmt(bend_limit)}."
            ),
        )
    )

    # ------------------------------------------------------------------
    # 5. Fixations : pas de 250 mm et parallélisme des crabes
    # ------------------------------------------------------------------
    arc = gm.arc_lengths(pts)
    supports = [0.0, float(arc[-1])] if len(arc) else [0.0]
    if clamp_arc_positions is not None:
        supports.extend(float(s) for s in np.asarray(clamp_arc_positions, dtype=float).ravel())
    ordered = np.array(sorted(set(round(s, 3) for s in supports)), dtype=float)
    gaps = np.diff(ordered) if len(ordered) > 1 else np.zeros(0)
    worst_gap = float(gaps.max()) if len(gaps) else float(total_length)

    kpis["n_clamps"] = int(len(np.atleast_1d(clamp_arc_positions))) if clamp_arc_positions is not None else 0
    kpis["worst_support_gap_mm"] = worst_gap
    kpis["clamps_required"] = int(max(0, np.ceil(total_length / rules.fixation_pitch_mm) - 1))

    checks.append(
        RuleCheck(
            rule_id="fixation_pitch",
            label_fr=f"Une fixation au moins tous les {rules.fixation_pitch_mm:.0f} mm",
            label_en=f"A fixation at least every {rules.fixation_pitch_mm:.0f} mm",
            passed=worst_gap <= rules.fixation_pitch_mm,
            value=worst_gap,
            limit=float(rules.fixation_pitch_mm),
            unit="mm",
            severity=Severity.MAJOR,
            detail_fr=(
                f"Écart maximal entre fixations : {_fmt(worst_gap)}."
                if worst_gap <= rules.fixation_pitch_mm
                else f"Portée de {_fmt(worst_gap)} sans fixation : il manque des crabes."
            ),
            detail_en=(
                f"Largest gap between fixations: {_fmt(worst_gap)}."
                if worst_gap <= rules.fixation_pitch_mm
                else f"{_fmt(worst_gap)} without any fixation: clamps are missing."
            ),
        )
    )

    if clamp_tilt_deg is not None and len(np.atleast_1d(clamp_tilt_deg)):
        tilts = np.asarray(clamp_tilt_deg, dtype=float).ravel()
        worst_tilt = float(tilts.max())
        n_tilted = int(np.count_nonzero(tilts > rules.fixation_parallel_tol_deg))
        kpis["worst_clamp_tilt_deg"] = worst_tilt
        kpis["n_tilted_clamps"] = n_tilted
        checks.append(
            RuleCheck(
                rule_id="fixation_parallel",
                label_fr="Crabes posés à plat sur la structure",
                label_en="Clamps seated flat on the structure",
                passed=n_tilted == 0,
                value=worst_tilt,
                limit=float(rules.fixation_parallel_tol_deg),
                unit="°",
                severity=Severity.MAJOR,
                detail_fr=(
                    f"Défaut de parallélisme maximal : {worst_tilt:.1f}°."
                    if n_tilted == 0
                    else f"{n_tilted} crabe(s) mal posé(s), jusqu'à {worst_tilt:.1f}° d'écart."
                ),
                detail_en=(
                    f"Worst parallelism deviation: {worst_tilt:.1f}°."
                    if n_tilted == 0
                    else f"{n_tilted} clamp(s) badly seated, up to {worst_tilt:.1f}° off."
                ),
            )
        )

    # ------------------------------------------------------------------
    # 6. Qualité du tracé : rester droit le plus longtemps possible
    # ------------------------------------------------------------------
    # Oscillations : mesurées à part de la courbure totale, car un arc régulier
    # accumule beaucoup de courbure sans jamais osciller, et une succession de
    # petits virages alternés fait l'inverse.
    kpis.update(gm.zigzag_metrics(pts, rules.straight_tol_deg))

    checks.append(
        RuleCheck(
            rule_id="straightness",
            label_fr="Tracé rectiligne sur la majeure partie du parcours",
            label_en="Mostly straight run",
            passed=straight["straight_ratio"] >= rules.target_straight_ratio,
            value=straight["straight_ratio"] * 100.0,
            limit=rules.target_straight_ratio * 100.0,
            unit="%",
            severity=Severity.MINOR,
            detail_fr=(
                f"{straight['straight_ratio'] * 100:.0f} % du parcours est rectiligne, "
                f"plus longue ligne droite {_fmt(straight['longest_run_mm'])}, "
                f"{straight['n_bends']} coude(s)."
            ),
            detail_en=(
                f"{straight['straight_ratio'] * 100:.0f}% of the run is straight, "
                f"longest straight {_fmt(straight['longest_run_mm'])}, "
                f"{straight['n_bends']} bend(s)."
            ),
        )
    )

    # Les règles décochées sont retirées ici, une fois pour toutes. Les
    # indicateurs, eux, restent tous calculés : l'utilisateur qui a décoché une
    # règle veut souvent continuer à voir la valeur mesurée, simplement sans
    # qu'elle décide de la conformité ni du classement.
    checks = [check for check in checks if rules.is_enabled(check.rule_id)]
    return RouteReport(checks=checks, kpis=kpis, enabled_rules=rules.enabled_rules)
