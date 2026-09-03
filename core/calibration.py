"""Réglages déduits de la géométrie du faisceau, plutôt que constants.

Les valeurs par défaut de l'application ont été posées une fois pour toutes :
48 points, 25 mm de pas maximal, 10 mm de distance minimale. Elles conviennent
à un faisceau court dans une zone dégagée, et à peu près à rien d'autre. Sur un
harnais de deux mètres, 48 points laissent **42 mm entre deux sommets** — le
tracé ne peut pas suivre un congé, et le rayon de cintrage réalisable est
plafonné par l'échantillonnage bien avant de l'être par la matière.

Ce module ne cherche pas les réglages optimaux : il donne un point de départ
*cohérent*, déduit de trois grandeurs que l'intégrateur connaît — la longueur
du faisceau, son diamètre, et la bande de distance à tenir. Le balayage
(:mod:`tools.sweep`) part de là pour explorer.

Le principe, à chaque fois : exprimer un réglage en **unités physiques** puis
le convertir, au lieu de le poser en absolu.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Calibration", "calibrate", "SPACING_PER_BEND_RADIUS", "MIN_SPACING_MM"]

#: Espacement des points, en fraction du rayon de cintrage minimal. Le rayon
#: réalisable vaut ``(pas / 2) / tan(θ/2)`` : pour qu'un coude à 30° atteigne le
#: rayon visé, il faut un pas d'environ la moitié de ce rayon. On prend le
#: quart, qui laisse de la marge sur les coudes plus francs.
SPACING_PER_BEND_RADIUS = 0.25

#: En deçà, on échantillonne plus fin que ce que la géométrie exige, et chaque
#: itération coûte pour rien.
MIN_SPACING_MM = 8.0

#: Au-delà, le tracé ne peut plus suivre un congé.
MAX_SPACING_MM = 60.0

#: Le pas d'un agent, en fraction de l'espacement. Plus grand, il saute
#: par-dessus ses voisins et le tracé se replie ; plus petit, il n'avance pas.
STEP_PER_SPACING = 0.6

#: Marge de manœuvre au-dessus du nombre de points initial : c'est ce budget
#: que consomme le raffinement adaptatif quand un passage résiste.
REFINEMENT_HEADROOM = 2.5


@dataclass(frozen=True)
class Calibration:
    """Un jeu de réglages cohérent, et de quoi expliquer chaque valeur."""

    initial_points: int
    max_points: int
    max_step_mm: float
    spacing_mm: float
    min_margin_mm: float
    max_margin_mm: float
    fixation_pitch_mm: float
    iterations: int
    notes: list = field(default_factory=list)

    def as_config(self) -> dict:
        """Forme directement injectable dans la configuration des agents."""
        return {
            "initial_points": self.initial_points,
            "max_points": self.max_points,
            "max_step_mm": self.max_step_mm,
            "local_max_shift": self.max_step_mm,
            "min_margin": self.min_margin_mm,
            "max_margin": self.max_margin_mm,
            "fixation_pitch_mm": self.fixation_pitch_mm,
            "crabe_min_spacing": self.fixation_pitch_mm,
            "iterations": self.iterations,
        }

    def report(self, lang: str = "FR") -> str:
        """Récapitulatif lisible, avec la raison de chaque valeur."""
        english = str(lang).upper().startswith("EN")
        head = (f"{self.initial_points} points ({self.spacing_mm:.0f} mm apart), "
                f"step {self.max_step_mm:.0f} mm, margin "
                f"{self.min_margin_mm:.0f}–{self.max_margin_mm:.0f} mm"
                if english else
                f"{self.initial_points} points (un tous les {self.spacing_mm:.0f} mm), "
                f"pas {self.max_step_mm:.0f} mm, marge "
                f"{self.min_margin_mm:.0f}–{self.max_margin_mm:.0f} mm")
        return "\n".join([head] + [f"  • {note}" for note in self.notes])


def calibrate(length_mm: float, diameter_mm: float = 20.0,
              min_margin_mm: float | None = None, max_margin_mm: float | None = None,
              bend_radius_factor: float = 6.0, fixation_pitch_mm: float = 250.0,
              iterations: int = 500) -> Calibration:
    """Réglages cohérents pour un faisceau de cette longueur et de ce diamètre.

    Args:
        length_mm: longueur du faisceau, à vol d'oiseau ou le long du tracé de
            départ — l'ordre de grandeur suffit.
        diameter_mm: diamètre du toron.
        min_margin_mm, max_margin_mm: bande de distance à la structure. Déduites
            du diamètre si elles ne sont pas données.
        bend_radius_factor: rayon de cintrage minimal, en diamètres.
        fixation_pitch_mm: écart maximal entre deux fixations.
        iterations: budget d'itérations souhaité.

    Returns:
        Une :class:`Calibration`, chaque valeur accompagnée de sa raison.
    """
    length = max(float(length_mm), 1.0)
    diameter = max(float(diameter_mm), 1.0)
    bend_radius = diameter * float(bend_radius_factor)
    notes: list = []

    # 1. L'espacement des points découle du rayon de cintrage : c'est lui qui
    #    décide si le tracé peut décrire un congé, ou seulement l'approximer.
    spacing = bend_radius * SPACING_PER_BEND_RADIUS
    spacing = min(max(spacing, MIN_SPACING_MM), MAX_SPACING_MM)
    points = int(round(length / spacing)) + 1
    points = max(points, 8)
    notes.append(
        f"un point tous les {spacing:.0f} mm, soit le quart du rayon de cintrage "
        f"minimal ({bend_radius:.0f} mm) : en deçà le tracé ne peut pas décrire "
        f"le congé qu'on lui demande"
    )

    # 2. Le pas d'un agent doit rester une fraction de l'espacement. Au-delà, un
    #    point saute par-dessus ses voisins et le tracé se replie.
    step = max(2.0, spacing * STEP_PER_SPACING)
    notes.append(
        f"pas de {step:.0f} mm, soit {STEP_PER_SPACING:.0%} de l'espacement : "
        f"au-delà un point dépasse ses voisins et le tracé se replie"
    )

    # 3. La bande de distance suit le diamètre quand elle n'est pas imposée.
    low = float(min_margin_mm) if min_margin_mm is not None else max(diameter, 15.0)
    high = float(max_margin_mm) if max_margin_mm is not None else max(low * 6.0, 120.0)
    if high <= low:
        high = low * 2.0
        notes.append("distance maximale relevée : elle était sous la minimale")
    if min_margin_mm is None:
        notes.append(
            f"distance minimale de {low:.0f} mm, soit un diamètre de toron : "
            f"de quoi passer un doigt et une pince"
        )

    # 4. Le budget de raffinement : ce que le tracé peut gagner en points
    #    lorsqu'un passage résiste.
    max_points = int(points * REFINEMENT_HEADROOM)
    notes.append(
        f"jusqu'à {max_points} points, soit {REFINEMENT_HEADROOM:.1f} fois le "
        f"départ : c'est le budget du raffinement là où un passage résiste"
    )

    if points > 120:
        notes.append(
            f"{points} points, c'est beaucoup : chaque itération coûte en "
            f"proportion. Un faisceau de {length / 1000:.1f} m gagne à être "
            f"découpé en tronçons entre deux fixations franches"
        )

    return Calibration(
        initial_points=points, max_points=max_points, max_step_mm=round(step, 1),
        spacing_mm=round(spacing, 1), min_margin_mm=low, max_margin_mm=high,
        fixation_pitch_mm=float(fixation_pitch_mm), iterations=int(iterations),
        notes=notes,
    )
