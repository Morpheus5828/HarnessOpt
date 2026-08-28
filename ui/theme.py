"""Charte graphique de l'application.

Un seul endroit définit couleurs, polices et espacements, pour que les pages
restent cohérentes et qu'un changement de palette (mode sombre, mode daltonien)
se propage partout sans retoucher chaque écran.

Chaque couleur est un couple ``(clair, sombre)`` : c'est la convention de
customtkinter, qui choisit automatiquement selon le mode d'apparence.
"""

from __future__ import annotations

__all__ = ["Palette", "PALETTES", "Theme", "current", "set_palette", "FONT", "SPACE"]


class Palette:
    """Jeu de couleurs complet."""

    def __init__(self, name: str, accent: str, ok: str, warn: str, danger: str, info: str):
        self.name = name
        self.accent = accent
        self.ok = ok
        self.warn = warn
        self.danger = danger
        self.info = info

    # -- fonds ---------------------------------------------------------
    BG = ("#F4F6F9", "#15181D")
    SURFACE = ("#FFFFFF", "#1E232B")
    SURFACE_ALT = ("#F0F3F7", "#252B34")
    BORDER = ("#DDE3EA", "#333B46")
    SIDEBAR = ("#FFFFFF", "#12151A")

    # -- textes --------------------------------------------------------
    TEXT = ("#1B2430", "#E4E9F0")
    TEXT_SOFT = ("#5A6675", "#98A3B3")
    TEXT_FAINT = ("#8A94A3", "#6C7684")
    TEXT_ON_ACCENT = "#FFFFFF"

    # -- états ---------------------------------------------------------
    DISABLED = ("#E6EAF0", "#2A313A")


#: Palette standard.
_DEFAULT = Palette(
    name="default",
    accent="#2D7FF9",
    ok="#1E9E5A",
    warn="#E08A00",
    danger="#D93A45",
    info="#5B62F4",
)

#: Palette sûre pour les daltonismes courants (deutéranopie, protanopie) :
#: on remplace l'opposition rouge/vert par une opposition bleu/orange, qui
#: reste lisible pour tous.
_COLORBLIND = Palette(
    name="colorblind",
    accent="#0072B2",
    ok="#009E73",
    warn="#E69F00",
    danger="#CC3311",
    info="#56B4E9",
)

PALETTES: dict[str, Palette] = {"default": _DEFAULT, "colorblind": _COLORBLIND}


class Theme:
    """État courant de la charte (palette active)."""

    def __init__(self):
        self.palette: Palette = _DEFAULT

    # Raccourcis pour ne pas écrire ``theme.palette.accent`` partout.
    @property
    def accent(self) -> str:
        return self.palette.accent

    @property
    def ok(self) -> str:
        return self.palette.ok

    @property
    def warn(self) -> str:
        return self.palette.warn

    @property
    def danger(self) -> str:
        return self.palette.danger

    @property
    def info(self) -> str:
        return self.palette.info

    BG = Palette.BG
    SURFACE = Palette.SURFACE
    SURFACE_ALT = Palette.SURFACE_ALT
    BORDER = Palette.BORDER
    SIDEBAR = Palette.SIDEBAR
    TEXT = Palette.TEXT
    TEXT_SOFT = Palette.TEXT_SOFT
    TEXT_FAINT = Palette.TEXT_FAINT
    TEXT_ON_ACCENT = Palette.TEXT_ON_ACCENT
    DISABLED = Palette.DISABLED

    def severity_color(self, severity: str, passed: bool) -> str:
        """Couleur d'une ligne de conformité.

        Une règle respectée est toujours verte ; une règle enfreinte prend la
        couleur de sa gravité, pour que l'œil aille d'abord sur ce qui bloque.
        """
        if passed:
            return self.ok
        return {"blocking": self.danger, "major": self.warn}.get(severity, self.info)


#: Instance partagée par toute l'application.
_CURRENT = Theme()


def current() -> Theme:
    """Charte active."""
    return _CURRENT


def set_palette(name: str) -> Theme:
    """Change la palette active (``"default"`` ou ``"colorblind"``)."""
    _CURRENT.palette = PALETTES.get(name, _DEFAULT)
    return _CURRENT


class FONT:
    """Échelle typographique.

    Les tailles sont volontairement généreuses : l'application est utilisée sur
    des postes de conception, souvent en écran large, par des personnes qui
    lisent des valeurs chiffrées toute la journée.
    """

    FAMILY = "Segoe UI"
    MONO = "Consolas"

    TITLE = (FAMILY, 24, "bold")
    H1 = (FAMILY, 19, "bold")
    H2 = (FAMILY, 15, "bold")
    H3 = (FAMILY, 13, "bold")
    BODY = (FAMILY, 13)
    BODY_BOLD = (FAMILY, 13, "bold")
    SMALL = (FAMILY, 11)
    SMALL_BOLD = (FAMILY, 11, "bold")
    TINY = (FAMILY, 10)

    VALUE = (MONO, 26, "bold")
    VALUE_SM = (MONO, 15, "bold")
    CODE = (MONO, 12)


class SPACE:
    """Échelle d'espacement, en pixels."""

    XS = 4
    SM = 8
    MD = 14
    LG = 22
    XL = 32
    RADIUS = 12
    RADIUS_SM = 8
