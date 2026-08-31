"""Conseils à l'utilisateur quand la convergence bloque.

Un agent qui n'arrive pas à respecter une règle ne le dit pas : il continue,
indéfiniment, et l'utilisateur ne peut pas distinguer « c'est long » de « c'est
impossible ». Ce module fait la différence, et surtout il la formule en termes
actionnables : une valeur mesurée, une valeur proposée, et le réglage exact à
modifier.

Deux principes le gouvernent.

**On ne propose jamais de relâcher une règle qu'on peut encore satisfaire.**
Les conseils n'apparaissent qu'après un nombre minimal d'itérations *et* une
stagnation avérée du meilleur score. Sinon l'application inciterait à baisser
les exigences avant même d'avoir cherché.

**On ne propose jamais de relâcher l'interdiction de clash.** Une route qui
traverse la structure n'est pas une route ; le conseil porte alors sur la
recherche — plus d'exploration, plus de points, points de départ mal placés —
jamais sur la règle. C'est la seule règle traitée ainsi, et c'est délibéré.

Le module est pur : il ne connaît ni l'interface, ni les agents, ni trimesh.
Il prend un rapport de conformité et rend une liste de suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Suggestion",
    "analyse",
    "is_stagnant",
    "MIN_ITERATIONS",
    "STAGNATION_WINDOW",
    "MIN_PHYSICAL_BEND_FACTOR",
    "MIN_PHYSICAL_CLEARANCE_MM",
]

#: En deçà, on laisse les agents travailler sans commenter.
MIN_ITERATIONS = 60

#: Nombre de relevés de score consécutifs identiques valant stagnation.
STAGNATION_WINDOW = 12

#: Facteur de cintrage en deçà duquel aucun toron ne se plie sans dommage.
#: Proposer moins produirait une route « conforme » mais impossible à poser.
MIN_PHYSICAL_BEND_FACTOR = 3.0

#: Distance minimale en deçà de laquelle on ne propose plus rien : à moins d'un
#: millimètre du DMU, on ne règle plus une marge, on accepte un contact.
MIN_PHYSICAL_CLEARANCE_MM = 1.0

#: Gravités, reprises de core.routing_rules pour éviter une dépendance croisée.
BLOCKING = "blocking"
MAJOR = "major"
INFO = "info"


@dataclass(frozen=True)
class Suggestion:
    """Un conseil : ce qui est mesuré, ce qui est proposé, où le régler.

    ``setting`` et ``value`` sont facultatifs. Quand ils sont présents,
    l'interface peut proposer d'appliquer la valeur en un clic ; quand ils sont
    absents, le conseil est purement explicatif — c'est le cas des clashs, pour
    lesquels aucun réglage ne doit être touché.
    """

    key: str
    severity: str
    title_fr: str
    title_en: str
    detail_fr: str
    detail_en: str
    action_fr: str
    action_en: str
    setting: str | None = None
    value: float | None = None

    def title(self, lang: str = "FR") -> str:
        return self.title_en if _english(lang) else self.title_fr

    def detail(self, lang: str = "FR") -> str:
        return self.detail_en if _english(lang) else self.detail_fr

    def action(self, lang: str = "FR") -> str:
        return self.action_en if _english(lang) else self.action_fr

    @property
    def is_applicable(self) -> bool:
        """Vrai si l'interface peut proposer un bouton « Appliquer »."""
        return self.setting is not None and self.value is not None


def _english(lang: str) -> bool:
    return str(lang).upper().startswith("EN")


def _round_down(value: float, step: float = 1.0) -> float:
    """Arrondi vers le bas, pour proposer une valeur qu'on sait atteignable."""
    if step <= 0:
        return float(value)
    return float(int(value / step) * step)


def _round_up(value: float, step: float = 1.0) -> float:
    if step <= 0:
        return float(value)
    return float(-(-value // step) * step)


def is_stagnant(scores, window: int = STAGNATION_WINDOW) -> bool:
    """Le meilleur score a-t-il cessé de progresser ?

    ``scores`` est la suite des scores du meilleur agent, du plus ancien au
    plus récent. On considère qu'il stagne si aucun des ``window`` derniers
    relevés n'améliore celui qui les précède tous.
    """
    values = list(scores)
    if len(values) < window:
        return False
    recent = values[-window:]
    reference = recent[0]
    return all(value >= reference for value in recent[1:])


def analyse(
    report,
    rules,
    iterations: int,
    stagnant: bool = False,
    clamp_model_ok: bool = True,
) -> list[Suggestion]:
    """Conseils tirés du rapport du **meilleur** agent.

    Args:
        report: :class:`core.routing_rules.RouteReport` du meilleur agent, ou
            ``None`` si aucun rapport n'est encore disponible.
        rules: jeu de règles en vigueur.
        iterations: nombre d'itérations du meilleur agent.
        stagnant: le score a cessé de progresser (voir :func:`is_stagnant`).
        clamp_model_ok: le modèle de crabe est chargeable. Faux, aucune règle
            de fixation ne pourra jamais passer, et le dire vaut mieux que de
            laisser l'agent tourner.

    Returns:
        Les suggestions, de la plus grave à la plus anecdotique. Liste vide
        tant qu'il est trop tôt pour conclure.
    """
    suggestions: list[Suggestion] = []

    if not clamp_model_ok and _needs_fixations(rules):
        suggestions.append(_clamp_model_missing())

    if report is None or iterations < MIN_ITERATIONS:
        return suggestions

    failed = {check.rule_id: check for check in report.failed()}

    if not failed:
        if stagnant:
            suggestions.append(_converged(report))
        return suggestions

    if not stagnant:
        # Des règles sont enfreintes mais le score progresse encore : laisser
        # travailler. Conseiller ici reviendrait à baisser les exigences au
        # premier obstacle.
        return suggestions

    builders = {
        "clash": _advise_clash,
        "clearance_min": _advise_clearance_min,
        "clearance_max": _advise_clearance_max,
        "bend_radius": _advise_bend_radius,
        "free_span": _advise_free_span,
        "fixation_pitch": _advise_fixation_pitch,
        "fixation_parallel": _advise_fixation_parallel,
        "straightness": _advise_straightness,
    }

    for rule_id, check in failed.items():
        builder = builders.get(rule_id)
        if builder is None:
            continue
        suggestion = builder(check, rules, report)
        if suggestion is not None:
            suggestions.append(suggestion)

    order = {BLOCKING: 0, MAJOR: 1, INFO: 2}
    suggestions.sort(key=lambda s: order.get(s.severity, 3))
    return suggestions


# ----------------------------------------------------------------------
# Conseils par règle
# ----------------------------------------------------------------------

def _needs_fixations(rules) -> bool:
    return rules.is_enabled("fixation_pitch") or rules.is_enabled("fixation_parallel")


def _clamp_model_missing() -> Suggestion:
    return Suggestion(
        key="clamp_model",
        severity=BLOCKING,
        title_fr="Aucun modèle de crabe chargé",
        title_en="No clamp model loaded",
        detail_fr=(
            "Le fichier STL du crabe n'a pas pu être lu. Les agents ne peuvent "
            "poser aucune fixation, donc la règle du pas entre fixations ne "
            "pourra jamais être respectée, quel que soit le temps de calcul."
        ),
        detail_en=(
            "The clamp STL could not be read. The agents cannot place any "
            "fixation, so the fixation pitch rule can never be met, however "
            "long the computation runs."
        ),
        action_fr="Indiquez un fichier de crabe valide à l'étape « Règles », "
                  "ou décochez les règles de fixation.",
        action_en="Point to a valid clamp file in the Rules step, or untick "
                  "the fixation rules.",
    )


def _converged(report) -> Suggestion:
    return Suggestion(
        key="converged",
        severity=INFO,
        title_fr="Toutes les règles sont respectées",
        title_en="All rules are met",
        detail_fr=(
            f"Le meilleur agent respecte les {len(report.checks)} règles "
            "appliquées et son score ne progresse plus."
        ),
        detail_en=(
            f"The best agent meets all {len(report.checks)} applied rules and "
            "its score has stopped improving."
        ),
        action_fr="Vous pouvez arrêter le calcul et passer au rapport.",
        action_en="You can stop the computation and move on to the report.",
    )


def _advise_clash(check, rules, report) -> Suggestion:
    """Un clash ne se règle jamais en relâchant la règle."""
    count = int(report.kpis.get("n_clashes", 0))
    return Suggestion(
        key="clash",
        severity=BLOCKING,
        title_fr="Le câble traverse encore la structure",
        title_en="The cable still crosses the structure",
        detail_fr=(
            f"{count} interférence(s) subsistent et le score ne progresse plus. "
            "Aucun réglage de règle ne peut résoudre cela : une route qui "
            "traverse une pièce n'est pas une route."
        ),
        detail_en=(
            f"{count} interference(s) remain and the score has stopped "
            "improving. No rule setting can fix this: a route that crosses a "
            "part is not a route."
        ),
        action_fr=(
            "Poussez le curseur vers « Explorer », augmentez le nombre de "
            "points, ou vérifiez que les points de départ et d'arrivée sont "
            "bien dans le vide et non dans la matière."
        ),
        action_en=(
            "Push the slider towards « Explore », raise the number of points, "
            "or check that the start and end points sit in free space rather "
            "than inside material."
        ),
    )


def _advise_clearance_min(check, rules, report) -> Suggestion:
    """La distance exigée est plus grande que ce que la maquette permet.

    On propose de **baisser** l'exigence jusqu'à la valeur effectivement
    atteinte. C'est le seul sens qui débloque : augmenter la distance minimale
    rendrait le problème strictement plus difficile.
    """
    achieved = float(check.value)
    required = float(check.limit)
    proposed = _round_down(achieved, 1.0)

    if proposed < MIN_PHYSICAL_CLEARANCE_MM:
        # Le câble frôle la structure : relâcher la marge reviendrait à
        # accepter un contact. C'est le passage qu'il faut revoir.
        return Suggestion(
            key="clearance_min",
            severity=BLOCKING,
            title_fr="Passage trop encombré pour y faire tenir le toron",
            title_en="Corridor too congested for the harness",
            detail_fr=(
                f"Le meilleur agent frôle la structure à {achieved:.1f} mm, pour "
                f"{required:.0f} mm exigés. Descendre l'exigence à cette valeur "
                "reviendrait à accepter un contact."
            ),
            detail_en=(
                f"The best agent grazes the structure at {achieved:.1f} mm "
                f"against {required:.0f} mm required. Lowering the requirement "
                "that far would amount to accepting contact."
            ),
            action_fr=(
                "Déplacez le point de départ ou d'arrivée, ou ouvrez un autre "
                "passage : aucun réglage de marge ne rendra celui-ci viable."
            ),
            action_en=(
                "Move the start or end point, or open another corridor: no "
                "clearance setting will make this one viable."
            ),
        )

    if proposed >= required:
        return None
    return Suggestion(
        key="clearance_min",
        severity=BLOCKING,
        title_fr="Distance minimale trop exigeante pour cette maquette",
        title_en="Minimum clearance too demanding for this mock-up",
        detail_fr=(
            f"Le meilleur agent ne descend pas en dessous de {achieved:.1f} mm "
            f"alors que la règle en exige {required:.0f} mm, et son score ne "
            "progresse plus. Le passage est trop encombré pour cette exigence."
        ),
        detail_en=(
            f"The best agent cannot do better than {achieved:.1f} mm where the "
            f"rule demands {required:.0f} mm, and its score has stopped "
            "improving. The corridor is too congested for that requirement."
        ),
        action_fr=f"Ramener la distance minimale à {proposed:.0f} mm.",
        action_en=f"Lower the minimum clearance to {proposed:.0f} mm.",
        setting="min_margin",
        value=proposed,
    )


def _advise_clearance_max(check, rules, report) -> Suggestion:
    """Le câble s'éloigne plus que permis : c'est ici qu'on augmente."""
    achieved = float(check.value)
    limit = float(check.limit)
    proposed = _round_up(achieved, 10.0)
    if proposed <= limit:
        return None
    return Suggestion(
        key="clearance_max",
        severity=MAJOR,
        title_fr="Le câble s'éloigne plus que la bande ne l'autorise",
        title_en="The cable drifts further than the band allows",
        detail_fr=(
            f"Le point le plus éloigné est à {achieved:.0f} mm de la structure, "
            f"pour une distance maximale de {limit:.0f} mm. Aucun trajet plus "
            "proche n'a été trouvé : la maquette est probablement creuse à cet "
            "endroit."
        ),
        detail_en=(
            f"The farthest point sits {achieved:.0f} mm from the structure, "
            f"against a {limit:.0f} mm maximum. No closer path was found: the "
            "mock-up is likely hollow there."
        ),
        action_fr=f"Porter la distance maximale à {proposed:.0f} mm.",
        action_en=f"Raise the maximum clearance to {proposed:.0f} mm.",
        setting="max_margin",
        value=proposed,
    )


def _advise_bend_radius(check, rules, report) -> Suggestion:
    """Le rayon exigé n'est pas tenable : on propose un facteur atteignable."""
    achieved = float(check.value)
    limit = float(check.limit)
    diameter = float(rules.harness.diameter_mm)
    if diameter <= 0 or achieved >= limit:
        return None
    proposed_factor = _round_down(achieved / diameter, 0.5)
    if proposed_factor < MIN_PHYSICAL_BEND_FACTOR:
        # En dessous, le toron se plie au-delà de ce qu'il supporte : la route
        # serait déclarée conforme sans être posable.
        return Suggestion(
            key="bend_radius",
            severity=BLOCKING,
            title_fr="Coude plus serré que ce qu'un toron peut supporter",
            title_en="Bend tighter than any harness can take",
            detail_fr=(
                f"Le coude le plus serré fait {achieved:.0f} mm de rayon, soit "
                f"{achieved / diameter:.1f} × Ø. En deçà de "
                f"{MIN_PHYSICAL_BEND_FACTOR:.0f} × Ø, aucun réglage ne rend le "
                "cheminement posable."
            ),
            detail_en=(
                f"The tightest bend has a {achieved:.0f} mm radius, that is "
                f"{achieved / diameter:.1f} × Ø. Below "
                f"{MIN_PHYSICAL_BEND_FACTOR:.0f} × Ø no setting makes the route "
                "installable."
            ),
            action_fr=(
                "Ouvrez le passage, ou réduisez le diamètre du toron en le "
                "scindant en plusieurs faisceaux."
            ),
            action_en=(
                "Open up the corridor, or reduce the harness diameter by "
                "splitting it into several bundles."
            ),
        )
    if proposed_factor >= rules.harness.bend_radius_factor:
        return None
    return Suggestion(
        key="bend_radius",
        severity=BLOCKING,
        title_fr="Rayon de cintrage inatteignable dans ce passage",
        title_en="Bend radius unreachable in this corridor",
        detail_fr=(
            f"Le coude le plus serré fait {achieved:.0f} mm de rayon, pour un "
            f"minimum exigé de {limit:.0f} mm ({rules.harness.bend_radius_factor:.1f} × Ø). "
            "Le couloir disponible impose des virages plus courts."
        ),
        detail_en=(
            f"The tightest bend has a {achieved:.0f} mm radius against a "
            f"{limit:.0f} mm minimum ({rules.harness.bend_radius_factor:.1f} × Ø). "
            "The available corridor forces tighter turns."
        ),
        action_fr=(
            f"Ramener le rayon de cintrage à {proposed_factor:.1f} × Ø "
            f"({proposed_factor * diameter:.0f} mm), si le toron le permet — "
            "sinon c'est le passage qu'il faut revoir."
        ),
        action_en=(
            f"Lower the bend radius to {proposed_factor:.1f} × Ø "
            f"({proposed_factor * diameter:.0f} mm) if the harness allows it — "
            "otherwise the corridor itself needs rethinking."
        ),
        setting="bend_radius_factor",
        value=proposed_factor,
    )


def _advise_free_span(check, rules, report) -> Suggestion:
    achieved = float(check.value)
    limit = float(check.limit)
    proposed = _round_up(achieved, 10.0)
    if proposed <= limit:
        return None
    return Suggestion(
        key="free_span",
        severity=MAJOR,
        title_fr="Traversée à vide plus longue que permis",
        title_en="Unsupported crossing longer than allowed",
        detail_fr=(
            f"La plus longue portion sans appui possible fait {achieved:.0f} mm, "
            f"pour une limite de {limit:.0f} mm. Le câble doit franchir un vide "
            "que la maquette n'entoure d'aucune structure."
        ),
        detail_en=(
            f"The longest unsupported run is {achieved:.0f} mm against a "
            f"{limit:.0f} mm limit. The cable has to cross a void the mock-up "
            "surrounds with no structure."
        ),
        action_fr=(
            f"Porter le pas entre fixations à {proposed:.0f} mm : c'est lui qui "
            "fixe la traversée à vide admissible."
        ),
        action_en=(
            f"Raise the fixation pitch to {proposed:.0f} mm: it is what sets "
            "the allowed unsupported crossing."
        ),
        setting="fixation_pitch",
        value=proposed,
    )


def _advise_fixation_pitch(check, rules, report) -> Suggestion:
    achieved = float(check.value)
    limit = float(check.limit)
    proposed = _round_up(achieved, 10.0)
    if proposed <= limit:
        return None
    return Suggestion(
        key="fixation_pitch",
        severity=MAJOR,
        title_fr="Pas entre fixations trop serré pour ce parcours",
        title_en="Fixation pitch too tight for this route",
        detail_fr=(
            f"Le plus grand écart entre deux fixations atteint {achieved:.0f} mm, "
            f"pour un pas exigé de {limit:.0f} mm. Sur cette portion, la "
            "structure n'offre nulle part où poser un crabe."
        ),
        detail_en=(
            f"The largest gap between fixations reaches {achieved:.0f} mm "
            f"against a {limit:.0f} mm pitch. Along that stretch the structure "
            "offers nowhere to seat a clamp."
        ),
        action_fr=f"Porter le pas entre fixations à {proposed:.0f} mm.",
        action_en=f"Raise the fixation pitch to {proposed:.0f} mm.",
        setting="fixation_pitch",
        value=proposed,
    )


def _advise_fixation_parallel(check, rules, report) -> Suggestion:
    achieved = float(check.value)
    limit = float(check.limit)
    proposed = min(89.0, _round_up(achieved, 5.0))
    if proposed <= limit:
        return None
    return Suggestion(
        key="fixation_parallel",
        severity=MAJOR,
        title_fr="Crabes trop inclinés sur la structure",
        title_en="Clamps too tilted on the structure",
        detail_fr=(
            f"Le crabe le plus incliné accuse {achieved:.0f}° d'écart, pour une "
            f"tolérance de {limit:.0f}°. Les surfaces disponibles ne sont pas "
            "parallèles au câble à cet endroit."
        ),
        detail_en=(
            f"The most tilted clamp is off by {achieved:.0f}° against a "
            f"{limit:.0f}° tolerance. The available surfaces are not parallel "
            "to the cable there."
        ),
        action_fr=f"Porter la tolérance de pose à {proposed:.0f}°.",
        action_en=f"Raise the seating tolerance to {proposed:.0f}°.",
        setting="fixation_parallel_tol",
        value=proposed,
    )


def _advise_straightness(check, rules, report) -> Suggestion:
    """Conseil purement explicatif : la cible de rectitude n'est pas réglable.

    ``value`` et ``limit`` sont ici des **pourcentages**, tels que publiés par
    le rapport, et non des rapports entre 0 et 1.
    """
    achieved_pct = float(check.value)
    target_pct = float(check.limit)
    zigzags = int(report.kpis.get("n_zigzags", 0))

    extra_fr = (
        f" Le tracé compte encore {zigzags} inversion(s) de sens de virage."
        if zigzags else ""
    )
    extra_en = (
        f" The route still shows {zigzags} turn-direction reversal(s)."
        if zigzags else ""
    )
    return Suggestion(
        key="straightness",
        severity=INFO,
        title_fr="Tracé moins rectiligne que visé",
        title_en="Route less straight than targeted",
        detail_fr=(
            f"{achieved_pct:.0f} % du parcours est rectiligne, pour "
            f"{target_pct:.0f} % visés.{extra_fr}"
        ),
        detail_en=(
            f"{achieved_pct:.0f} % of the run is straight against "
            f"{target_pct:.0f} % targeted.{extra_en}"
        ),
        action_fr=(
            "Choisissez l'équipe « Finition », qui consacre ses agents au "
            "lissage, et poussez le curseur vers « Peaufiner »."
        ),
        action_en=(
            "Pick the « Finishing » team, whose agents focus on smoothing, and "
            "push the slider towards « Refine »."
        ),
    )
