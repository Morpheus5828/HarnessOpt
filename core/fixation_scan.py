"""Analyse des fixations déjà présentes dans la maquette.

Le détecteur (:mod:`core.path_managment.fixation_detection`) recale chaque
modèle de fixation sur le DMU par ICP et, pour les peignes, calcule les
**passages** : le segment ``p_in`` → ``p_out`` par lequel le câble doit
traverser chaque encoche. Ces points ne sont pas indicatifs, ce sont des
contraintes de passage : un faisceau qui ne passe pas dans l'encoche n'est pas
posable.

Ce module ne fait pas la détection ; il l'enveloppe. Il lui apporte trois
choses qui manquaient :

* **une dégradation propre.** Open3D est une dépendance lourde, absente de
  beaucoup de postes. Sans lui, le scan ne plante pas : il dit pourquoi il n'a
  pas eu lieu, et le cheminement continue sans fixations préexistantes ;
* **une structure lisible.** Le détecteur rend des dictionnaires imbriqués avec
  une liste de points aplatie en ``[p_in, p_out, p_in, p_out, …]`` ; on en tire
  ici des passages numérotés, exploitables par l'interface ;
* **une frontière testable.** La mise en forme se teste sans Open3D, sans DMU
  et sans écran, ce qui est indispensable pour un composant dont la dépendance
  n'est pas installable partout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = [
    "Passage",
    "DetectedFixation",
    "ScanResult",
    "summarise",
    "scan",
    "NO_FOLDER",
    "NO_OPEN3D",
]

#: Raisons pour lesquelles un scan n'a pas eu lieu.
NO_FOLDER = "no_folder"
NO_OPEN3D = "no_open3d"
NO_SCENE = "no_scene"
FAILED = "failed"

_REASON_FR = {
    NO_FOLDER: "Aucun dossier de fixations indiqué à l'étape « Règles ».",
    NO_OPEN3D: "Open3D n'est pas installé sur ce poste : le scan des fixations "
               "existantes est ignoré.",
    NO_SCENE: "La maquette fusionnée n'a pas pu être écrite pour le scan.",
    FAILED: "Le scan des fixations a échoué.",
}

_REASON_EN = {
    NO_FOLDER: "No fixation folder set in the Rules step.",
    NO_OPEN3D: "Open3D is not installed on this workstation: the existing "
               "fixation scan is skipped.",
    NO_SCENE: "The merged mock-up could not be written out for the scan.",
    FAILED: "The fixation scan failed.",
}


def _english(lang: str) -> bool:
    return str(lang).upper().startswith("EN")


@dataclass(frozen=True)
class Passage:
    """Un passage imposé : le câble doit traverser de ``p_in`` vers ``p_out``."""

    index: int
    p_in: tuple
    p_out: tuple

    @property
    def center(self) -> tuple:
        return tuple((a + b) / 2.0 for a, b in zip(self.p_in, self.p_out))

    @property
    def width_mm(self) -> float:
        """Ouverture du passage, en mm."""
        return sum((a - b) ** 2 for a, b in zip(self.p_in, self.p_out)) ** 0.5

    def format(self, lang: str = "FR") -> str:
        """Ligne lisible : « n° 3 — p_in (12, 40, 8) → p_out (12, 96, 8) »."""
        label = "no." if _english(lang) else "n°"
        return (
            f"{label} {self.index + 1} — p_in ({_coords(self.p_in)}) "
            f"→ p_out ({_coords(self.p_out)})"
        )


def _coords(point) -> str:
    return ", ".join(f"{value:.0f}" for value in point)


@dataclass(frozen=True)
class DetectedFixation:
    """Une fixation reconnue dans le DMU."""

    name: str
    position: tuple
    score: float
    passages: list = field(default_factory=list)

    @property
    def is_comb(self) -> bool:
        """Un peigne se reconnaît à ses passages multiples."""
        return len(self.passages) > 0


@dataclass(frozen=True)
class ScanResult:
    """Ce que le scan a trouvé, ou pourquoi il n'a pas eu lieu."""

    fixations: list = field(default_factory=list)
    skipped_reason: str | None = None
    scanned_files: int = 0

    @property
    def ran(self) -> bool:
        return self.skipped_reason is None

    @property
    def n_fixations(self) -> int:
        return len(self.fixations)

    @property
    def n_passages(self) -> int:
        return sum(len(fixation.passages) for fixation in self.fixations)

    @property
    def passages(self) -> list:
        """Tous les passages, toutes fixations confondues."""
        return [p for fixation in self.fixations for p in fixation.passages]

    def routing_points(self) -> list:
        """Points de passage aplatis, tels que les attend l'agent."""
        points = []
        for passage in self.passages:
            points.extend([list(passage.p_in), list(passage.p_out)])
        return points

    def message(self, lang: str = "FR") -> str:
        """Phrase d'état, affichable telle quelle."""
        if self.skipped_reason is not None:
            table = _REASON_EN if _english(lang) else _REASON_FR
            return table.get(self.skipped_reason, self.skipped_reason)

        if not self.fixations:
            return (
                "No existing fixation recognised in the mock-up."
                if _english(lang) else
                "Aucune fixation existante reconnue dans la maquette."
            )

        if _english(lang):
            text = f"{self.n_fixations} existing fixation(s) recognised"
            if self.n_passages:
                text += f", {self.n_passages} imposed passage(s)"
            return text + "."

        text = f"{self.n_fixations} fixation(s) existante(s) reconnue(s)"
        if self.n_passages:
            text += f", {self.n_passages} passage(s) imposé(s)"
        return text + "."


def summarise(raw_clamps, scanned_files: int = 0) -> ScanResult:
    """Met en forme la sortie brute du détecteur.

    ``raw_clamps`` est la liste rendue par
    :func:`core.path_managment.fixation_detection.run_detection_for_agent` :
    un dictionnaire par fixation, dont ``routing_points`` — s'il existe — est
    une liste **aplatie** de couples entrée/sortie.
    """
    fixations = []
    for clamp in raw_clamps or []:
        points = list(clamp.get("routing_points") or [])
        passages = [
            Passage(index=i, p_in=tuple(points[2 * i]), p_out=tuple(points[2 * i + 1]))
            # Un nombre impair de points signifierait un passage tronqué : on
            # ne garde que les couples complets plutôt que d'inventer un point.
            for i in range(len(points) // 2)
        ]
        fixations.append(
            DetectedFixation(
                name=str(clamp.get("name", "?")),
                position=tuple(clamp.get("position") or ()),
                score=float(clamp.get("score", 0.0)),
                passages=passages,
            )
        )
    return ScanResult(fixations=fixations, scanned_files=scanned_files)


def scan(scene_path: str, clamps_folder: str, on_progress=None) -> ScanResult:
    """Lance la détection, ou explique pourquoi elle n'a pas eu lieu.

    Args:
        scene_path: STL de la maquette fusionnée.
        clamps_folder: dossier contenant les modèles de fixation.
        on_progress: fonction appelée avec un message d'avancement.

    Returns:
        Un :class:`ScanResult`. Aucune exception n'est propagée : un scan
        impossible ne doit jamais empêcher le cheminement de démarrer.
    """
    if not clamps_folder or not os.path.isdir(str(clamps_folder)):
        return ScanResult(skipped_reason=NO_FOLDER)

    if not scene_path or not os.path.exists(str(scene_path)):
        return ScanResult(skipped_reason=NO_SCENE)

    try:
        import open3d  # noqa: F401
    except Exception:
        return ScanResult(skipped_reason=NO_OPEN3D)

    try:
        from core.path_managment.fixation_detection import run_detection_for_agent
    except Exception:
        return ScanResult(skipped_reason=NO_OPEN3D)

    if on_progress is not None:
        on_progress("Scan des fixations existantes…")

    try:
        raw = run_detection_for_agent(str(scene_path), str(clamps_folder))
    except Exception:
        return ScanResult(skipped_reason=FAILED)

    import glob

    scanned = len(
        glob.glob(os.path.join(str(clamps_folder), "*.stl"))
        + glob.glob(os.path.join(str(clamps_folder), "*.STL"))
    )
    return summarise(raw, scanned_files=scanned)
