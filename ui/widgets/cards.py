"""Blocs d'affichage réutilisables : cartes, titres, indicateurs, pastilles."""

from __future__ import annotations

import customtkinter as ctk

from ui.theme import FONT, SPACE, current

__all__ = ["Card", "SectionTitle", "KpiTile", "KpiRow", "StatusPill", "HelpText", "Divider"]


class Card(ctk.CTkFrame):
    """Panneau encadré regroupant un bloc de contenu.

    Un titre facultatif et une phrase d'explication en dessous : c'est la
    brique de base de tous les écrans, pour que chaque bloc dise ce qu'il est
    sans qu'on ait à le deviner.
    """

    def __init__(self, master, title: str = "", subtitle: str = "", icon: str = "", **kwargs):
        theme = current()
        kwargs.setdefault("fg_color", theme.SURFACE)
        kwargs.setdefault("border_color", theme.BORDER)
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("corner_radius", SPACE.RADIUS)
        super().__init__(master, **kwargs)

        self._title_label = None
        self._subtitle_label = None

        if title:
            header = ctk.CTkFrame(self, fg_color="transparent")
            header.pack(fill="x", padx=SPACE.MD, pady=(SPACE.MD, 0))
            self._title_label = ctk.CTkLabel(
                header,
                text=f"{icon}  {title}".strip(),
                font=FONT.H2,
                text_color=theme.TEXT,
                anchor="w",
            )
            self._title_label.pack(side="left")

        if subtitle:
            self._subtitle_label = ctk.CTkLabel(
                self,
                text=subtitle,
                font=FONT.SMALL,
                text_color=theme.TEXT_SOFT,
                anchor="w",
                justify="left",
                wraplength=760,
            )
            self._subtitle_label.pack(fill="x", padx=SPACE.MD, pady=(2, 0))

        #: Conteneur à remplir par l'appelant.
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=SPACE.MD, pady=SPACE.MD)

    def set_title(self, title: str, icon: str = ""):
        if self._title_label is not None:
            self._title_label.configure(text=f"{icon}  {title}".strip())

    def set_subtitle(self, subtitle: str):
        if self._subtitle_label is not None:
            self._subtitle_label.configure(text=subtitle)


class SectionTitle(ctk.CTkLabel):
    """Intertitre à l'intérieur d'une carte."""

    def __init__(self, master, text: str, **kwargs):
        theme = current()
        kwargs.setdefault("font", FONT.H3)
        kwargs.setdefault("text_color", theme.TEXT_SOFT)
        kwargs.setdefault("anchor", "w")
        super().__init__(master, text=text.upper(), **kwargs)


class HelpText(ctk.CTkLabel):
    """Phrase d'explication en langage courant, sous un champ."""

    def __init__(self, master, text: str, width: int = 520, **kwargs):
        theme = current()
        kwargs.setdefault("font", FONT.SMALL)
        kwargs.setdefault("text_color", theme.TEXT_FAINT)
        kwargs.setdefault("anchor", "w")
        kwargs.setdefault("justify", "left")
        kwargs.setdefault("wraplength", width)
        super().__init__(master, text=text, **kwargs)


class Divider(ctk.CTkFrame):
    """Filet de séparation horizontal."""

    def __init__(self, master, **kwargs):
        kwargs.setdefault("height", 1)
        kwargs.setdefault("fg_color", current().BORDER)
        super().__init__(master, **kwargs)


class KpiTile(ctk.CTkFrame):
    """Grand chiffre avec son intitulé.

    Conçue pour être lue d'un coup d'œil à deux mètres : la valeur domine,
    l'intitulé est discret, la couleur porte le verdict.
    """

    def __init__(self, master, label: str, value: str = "—", unit: str = "", **kwargs):
        theme = current()
        kwargs.setdefault("fg_color", theme.SURFACE_ALT)
        kwargs.setdefault("corner_radius", SPACE.RADIUS_SM)
        super().__init__(master, **kwargs)

        self._theme = theme

        self.lbl_caption = ctk.CTkLabel(
            self, text=label, font=FONT.SMALL, text_color=theme.TEXT_SOFT, anchor="w"
        )
        self.lbl_caption.pack(fill="x", padx=SPACE.MD, pady=(SPACE.SM, 0))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=SPACE.MD, pady=(0, SPACE.SM))

        self.lbl_value = ctk.CTkLabel(
            row, text=value, font=FONT.VALUE, text_color=theme.TEXT, anchor="w"
        )
        self.lbl_value.pack(side="left")

        self.lbl_unit = ctk.CTkLabel(
            row, text=unit, font=FONT.SMALL, text_color=theme.TEXT_FAINT, anchor="w"
        )
        self.lbl_unit.pack(side="left", padx=(4, 0), pady=(8, 0))

    def update_value(self, value, unit: str | None = None, color: str | None = None):
        """Met à jour la valeur affichée, et éventuellement sa couleur."""
        self.lbl_value.configure(text=str(value), text_color=color or self._theme.TEXT)
        if unit is not None:
            self.lbl_unit.configure(text=unit)

    def set_label(self, label: str):
        self.lbl_caption.configure(text=label)


class KpiRow(ctk.CTkFrame):
    """Rangée d'indicateurs répartis à parts égales."""

    def __init__(self, master, definitions: list[tuple[str, str, str]], **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)

        self.tiles: dict[str, KpiTile] = {}
        for column, (key, label, unit) in enumerate(definitions):
            tile = KpiTile(self, label=label, unit=unit)
            tile.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else SPACE.SM, 0))
            self.grid_columnconfigure(column, weight=1, uniform="kpi")
            self.tiles[key] = tile

    def update_values(self, values: dict):
        """``values`` : ``{clé: valeur}`` ou ``{clé: (valeur, couleur)}``."""
        for key, tile in self.tiles.items():
            if key not in values:
                continue
            payload = values[key]
            if isinstance(payload, tuple):
                tile.update_value(payload[0], color=payload[1])
            else:
                tile.update_value(payload)

    def set_labels(self, labels: dict):
        for key, label in labels.items():
            if key in self.tiles:
                self.tiles[key].set_label(label)


class StatusPill(ctk.CTkLabel):
    """Étiquette colorée portant un état (prêt, en cours, conforme, en défaut)."""

    def __init__(self, master, text: str = "", tone: str = "neutral", **kwargs):
        theme = current()
        kwargs.setdefault("font", FONT.SMALL_BOLD)
        kwargs.setdefault("corner_radius", SPACE.RADIUS_SM)
        kwargs.setdefault("padx", SPACE.MD)
        kwargs.setdefault("height", 26)
        super().__init__(master, text=text, **kwargs)
        self._theme = theme
        self.set_tone(tone)

    def set_tone(self, tone: str):
        """``tone`` : ``ok``, ``warn``, ``danger``, ``info`` ou ``neutral``."""
        theme = self._theme
        color = {
            "ok": theme.ok,
            "warn": theme.warn,
            "danger": theme.danger,
            "info": theme.info,
            "accent": theme.accent,
        }.get(tone)

        if color is None:
            self.configure(fg_color=theme.SURFACE_ALT, text_color=theme.TEXT_SOFT)
        else:
            self.configure(fg_color=color, text_color=theme.TEXT_ON_ACCENT)

    def update_status(self, text: str, tone: str = "neutral"):
        self.configure(text=text)
        self.set_tone(tone)
