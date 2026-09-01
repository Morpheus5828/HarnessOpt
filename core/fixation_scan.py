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

import glob
import os
import traceback
from dataclasses import dataclass, field

__all__ = [
    "Passage",
    "DetectedFixation",
    "ScanResult",
    "summarise",
    "scan",
    "NO_FOLDER",
    "NO_MODELS",
    "NO_OPEN3D",
    "NO_SCENE",
    "FAILED",
]

#: Raisons pour lesquelles un scan n'a pas eu lieu.
NO_FOLDER = "no_folder"
NO_MODELS = "no_models"
NO_OPEN3D = "no_open3d"
NO_SCENE = "no_scene"
FAILED = "failed"

_REASON_FR = {
    NO_FOLDER: "Aucun dossier de fixations indiqué à l'étape « Règles ».",
    NO_MODELS: "Le dossier de fixations ne contient aucun fichier STL.",
    NO_OPEN3D: "Open3D n'est pas installé sur ce poste : le scan des fixations "
               "existantes est ignoré.",
    NO_SCENE: "La maquette fusionnée n'a pas pu être écrite pour le scan.",
    FAILED: "Le scan des fixations a échoué.",
}

_REASON_EN = {
    NO_FOLDER: "No fixation folder set in the Rules step.",
    NO_MODELS: "The fixation folder contains no STL file.",
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

    def format(self, lang: str = "FR", number: int | None = None) -> str:
        """Ligne lisible : « n° 3 — p_in (12, 40, 8) → p_out (12, 96, 8) ».

        ``index`` repère l'encoche **au sein de son peigne** : deux peignes à
        une encoche portent donc tous deux l'index 0. Pour une liste qui les
        mélange, passez ``number`` afin de les numéroter de façon continue.
        """
        label = "no." if _english(lang) else "n°"
        shown = self.index + 1 if number is None else number
        return (
            f"{label} {shown} — p_in ({_coords(self.p_in)}) "
            f"→ p_out ({_coords(self.p_out)})"
        )


def _coords(point) -> str:
    return ", ".join(f"{value:.0f}" for value in point)


@dataclass(frozen=True)
class DetectedFixation:
    """Une fixation reconnue dans le DMU.

    ``file_path`` et ``transform`` sont conservés à dessein : ce sont eux qui
    permettent d'afficher la fixation avec **sa géométrie réelle**, recalée là
    où le détecteur l'a trouvée. Sans eux, il ne resterait qu'un point, et un
    repère symbolique ne dit rien de l'encombrement d'une fixation.
    """

    name: str
    position: tuple
    score: float
    passages: list = field(default_factory=list)
    #: Modèle STL d'origine, tel que fourni au détecteur.
    file_path: str = ""
    #: Matrice 4x4 de recalage, du repère du modèle vers celui de la maquette.
    transform: tuple = ()

    @property
    def is_comb(self) -> bool:
        """Un peigne se reconnaît à ses passages multiples."""
        return len(self.passages) > 0

    @property
    def is_drawable(self) -> bool:
        """Peut-on dessiner sa géométrie, ou seulement sa position ?"""
        return bool(self.file_path) and len(self.transform) == 4


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

        scanned = self.scanned_files

        if not self.fixations:
            if _english(lang):
                return (f"{scanned} model(s) scanned, none recognised in the mock-up."
                        if scanned else "No existing fixation recognised in the mock-up.")
            return (f"{scanned} modèle(s) examiné(s), aucun reconnu dans la maquette."
                    if scanned else "Aucune fixation existante reconnue dans la maquette.")

        if _english(lang):
            text = f"{self.n_fixations} of {scanned} model(s) recognised" if scanned \
                else f"{self.n_fixations} existing fixation(s) recognised"
            if self.n_passages:
                text += f", {self.n_passages} imposed passage(s)"
            return text + "."

        text = f"{self.n_fixations} fixation(s) reconnue(s) sur {scanned} modèle(s) examiné(s)" \
            if scanned else f"{self.n_fixations} fixation(s) existante(s) reconnue(s)"
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
        transform = clamp.get("transform") or ()
        fixations.append(
            DetectedFixation(
                name=str(clamp.get("name", "?")),
                position=tuple(clamp.get("position") or ()),
                score=float(clamp.get("score", 0.0)),
                passages=passages,
                file_path=str(clamp.get("file_path", "") or ""),
                transform=tuple(tuple(row) for row in transform),
            )
        )
    return ScanResult(fixations=fixations, scanned_files=scanned_files)


#: Extensions de maillage que le détecteur (Open3D) sait relire. Le ``.vtk``
#: dans lequel l'étape d'extraction range la maquette fusionnée n'en fait
#: **pas** partie : Open3D le refuse en rendant un maillage vide, sans lever.
#: C'est la cause d'un scan qui « ne trouve rien » alors que tout est en place.
DETECTOR_SUFFIXES = (".stl", ".ply", ".obj", ".off", ".gltf", ".glb")

#: Nom du STL réexporté à l'intention du détecteur.
SCENE_EXPORT_NAME = "temp_for_detection.stl"


def _printer(log):
    """Le journal du scan part en console par défaut : c'est là qu'on le lit."""
    return print if log is None else log


def _model_files(folder: str) -> list:
    """Modèles de fixation à rechercher, triés pour une sortie stable."""
    return sorted(
        glob.glob(os.path.join(folder, "*.stl"))
        + glob.glob(os.path.join(folder, "*.STL"))
    )


def _scene_export_path() -> str:
    from core import paths

    target = paths.FUSION_DIR / SCENE_EXPORT_NAME
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        import tempfile

        return os.path.join(tempfile.gettempdir(), SCENE_EXPORT_NAME)
    return str(target)


def _export_scene(mesh, say) -> str | None:
    """Réécrit en STL la maquette telle qu'elle est en mémoire."""
    target = _scene_export_path()
    try:
        mesh.export(target)
    except Exception as exc:
        say(f"⚠️  Réexport de la maquette impossible : {exc}")
        return None
    say(f"📐 Maquette réexportée pour le détecteur : {target}")
    return target


def _convert_scene(path: str, say) -> str | None:
    """Convertit un fichier de maquette vers un format lisible par Open3D."""
    try:
        import pyvista as pv
    except Exception as exc:
        say(f"⚠️  Conversion impossible, PyVista absent : {exc}")
        return None
    try:
        data = pv.read(path)
        # ``extract_surface`` sur un maillage déjà surfacique est inutile, et
        # PyVista prévient qu'il changera de comportement : on ne l'appelle
        # que là où il sert vraiment, sur un volume.
        surface = data if isinstance(data, pv.PolyData) else data.extract_surface()
        target = _scene_export_path()
        surface.triangulate().save(target)
    except Exception as exc:
        say(f"⚠️  Conversion de la maquette échouée : {exc}")
        return None
    say(f"📐 Maquette convertie pour le détecteur : {target}")
    return target


def scene_for_detection(scene_path, mesh=None, log=None) -> str | None:
    """Rend un fichier que le détecteur sait relire, quitte à l'écrire.

    Le détecteur reçoit un chemin, pas un maillage, et Open3D ne lit qu'une
    poignée de formats. Dépendre de l'endroit **et du format** où la fusion a
    été rangée, c'est accepter qu'un scan échoue en silence dès que l'un des
    deux change. On préfère donc réexporter la maquette effectivement chargée :
    ce qui est scanné est alors exactement ce que les agents parcourent.
    """
    say = _printer(log)

    if mesh is not None:
        exported = _export_scene(mesh, say)
        if exported is not None:
            return exported

    path = str(scene_path or "")
    if not path or not os.path.exists(path):
        say(f"❌ Maquette introuvable : {path or '(aucune)'}")
        return None

    suffix = os.path.splitext(path)[1].lower()
    if suffix in DETECTOR_SUFFIXES:
        return path

    say(f"ℹ️  Format « {suffix or 'inconnu'} » illisible par le détecteur, conversion…")
    return _convert_scene(path, say)


def _report(result: "ScanResult", say) -> None:
    """Récapitule en console ce que le scan a reconnu, fixation par fixation."""
    say("-" * 70)
    say(result.message("FR"))
    for number, fixation in enumerate(result.fixations, start=1):
        position = _coords(fixation.position) if fixation.position else "?"
        say(f"  ✅ [{number}] {fixation.name} — recalage "
            f"{fixation.score * 100:.0f} % en ({position})")
        for rank, passage in enumerate(fixation.passages, start=1):
            say(f"        {passage.format('FR', number=rank)}")
        if not fixation.is_drawable:
            say("        ⚠️  géométrie non réaffichable "
                "(modèle ou recalage manquant)")
    say("=" * 70)


def scan(scene_path, clamps_folder, on_progress=None, mesh=None, log=None) -> ScanResult:
    """Lance la détection, ou explique pourquoi elle n'a pas eu lieu.

    Chaque étape est journalisée : un scan qui ne trouve rien doit dire s'il
    n'a pas tourné, sur quelle maquette il a tourné, et combien de modèles il a
    comparés. Sans cette trace, « aucune fixation trouvée » est indiscernable
    de « le scan n'a jamais eu lieu ».

    Args:
        scene_path: fichier de la maquette fusionnée.
        clamps_folder: dossier contenant les modèles de fixation.
        on_progress: fonction appelée avec un message d'avancement.
        mesh: maquette déjà chargée. Fournie, elle prime sur ``scene_path`` :
            elle est réexportée en STL, ce qui garantit que le détecteur voit
            la même géométrie que les agents.
        log: destination du journal ; la console par défaut.

    Returns:
        Un :class:`ScanResult`. Aucune exception n'est propagée : un scan
        impossible ne doit jamais empêcher le cheminement de démarrer.
    """
    say = _printer(log)
    say("=" * 70)
    say("SCAN DES FIXATIONS EXISTANTES")
    say("=" * 70)

    folder = str(clamps_folder or "")
    if not folder or not os.path.isdir(folder):
        say(f"❌ Dossier de fixations absent ou invalide : "
            f"{folder or '(non renseigné)'}")
        return ScanResult(skipped_reason=NO_FOLDER)
    say(f"📁 Dossier des modèles : {folder}")

    scene = scene_for_detection(scene_path, mesh=mesh, log=say)
    if scene is None:
        return ScanResult(skipped_reason=NO_SCENE)
    say(f"🌍 Maquette analysée : {scene}")

    models = _model_files(folder)
    say(f"🔎 {len(models)} modèle(s) STL à rechercher :")
    for model in models:
        say(f"   • {os.path.basename(model)}")
    if not models:
        say("❌ Aucun modèle à rechercher : scan ignoré.")
        return ScanResult(skipped_reason=NO_MODELS)

    try:
        import open3d  # noqa: F401
    except Exception as exc:
        say(f"❌ Open3D indisponible ({exc}) : scan ignoré.")
        return ScanResult(skipped_reason=NO_OPEN3D, scanned_files=len(models))

    try:
        from core.path_managment.fixation_detection import run_detection_for_agent
    except Exception as exc:
        say(f"❌ Détecteur indisponible ({exc}) : scan ignoré.")
        return ScanResult(skipped_reason=NO_OPEN3D, scanned_files=len(models))

    if on_progress is not None:
        on_progress("Scan des fixations existantes…")

    try:
        raw = run_detection_for_agent(scene, folder)
    except Exception as exc:
        say(f"❌ Le scan a échoué : {exc}")
        for line in traceback.format_exc().rstrip().splitlines():
            say(f"   {line}")
        return ScanResult(skipped_reason=FAILED, scanned_files=len(models))

    result = summarise(raw, scanned_files=len(models))
    _report(result, say)
    return result
