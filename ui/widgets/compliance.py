"""Tableau de conformité : le verdict, règle par règle.

C'est l'écran que regarde l'intégrateur. Chaque ligne dit, en une phrase,
si une règle est respectée, ce qui a été mesuré, et ce qui était exigé.
Aucune connaissance de l'algorithme n'est nécessaire pour le lire.
"""

from __future__ import annotations

import customtkinter as ctk

from ui.theme import FONT, SPACE, current

__all__ = ["ComplianceTable", "VerdictBanner"]


def _format_value(value: float, unit: str) -> str:
    if value == float("inf"):
        return "∞"
    if unit == "":
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.0f} {unit}"
    return f"{value:.1f} {unit}"


class _ComplianceRow(ctk.CTkFrame):
    """Une règle : pastille d'état, intitulé, détail, valeur mesurée / limite."""

    def __init__(self, master, **kwargs):
        theme = current()
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self._theme = theme

        # `height` explicite : un CTkFrame sans enfant conserve sinon sa hauteur
        # par défaut de 200 px et impose cette hauteur à toute la ligne.
        self.marker = ctk.CTkFrame(self, width=4, height=38, corner_radius=2, fg_color=theme.BORDER)
        self.marker.pack(side="left", fill="y", padx=(0, SPACE.SM))

        self.icon = ctk.CTkLabel(self, text="", font=FONT.BODY_BOLD, width=22)
        self.icon.pack(side="left")

        texts = ctk.CTkFrame(self, fg_color="transparent")
        texts.pack(side="left", fill="x", expand=True, padx=(SPACE.SM, 0))

        self.lbl_rule = ctk.CTkLabel(
            texts, text="", font=FONT.BODY_BOLD, text_color=theme.TEXT, anchor="w"
        )
        self.lbl_rule.pack(fill="x")

        self.lbl_detail = ctk.CTkLabel(
            texts, text="", font=FONT.SMALL, text_color=theme.TEXT_SOFT,
            anchor="w", justify="left", wraplength=520,
        )
        self.lbl_detail.pack(fill="x")

        self.lbl_value = ctk.CTkLabel(
            self, text="", font=FONT.VALUE_SM, text_color=theme.TEXT, width=150, anchor="e"
        )
        self.lbl_value.pack(side="right", padx=(SPACE.SM, 0))

    def update_check(self, check, lang: str = "FR"):
        theme = self._theme
        color = theme.severity_color(check.severity, check.passed)

        self.marker.configure(fg_color=color)
        self.icon.configure(text="✓" if check.passed else "✕", text_color=color)
        self.lbl_rule.configure(text=check.label(lang))
        self.lbl_detail.configure(text=check.detail(lang))

        measured = _format_value(check.value, check.unit)
        self.lbl_value.configure(text=measured, text_color=color)


class ComplianceTable(ctk.CTkFrame):
    """Liste des règles avec leur état, alimentée par un ``RouteReport``."""

    def __init__(self, master, lang: str = "FR", **kwargs):
        theme = current()
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self.lang = lang
        self._rows: list[_ComplianceRow] = []

        self.placeholder = ctk.CTkLabel(
            self,
            text="—",
            font=FONT.BODY,
            text_color=theme.TEXT_FAINT,
            anchor="w",
        )
        self.placeholder.pack(fill="x", pady=SPACE.SM)

    def set_language(self, lang: str):
        self.lang = lang

    def set_placeholder(self, text: str):
        self.placeholder.configure(text=text)

    def update_report(self, report):
        """Affiche un rapport, ou l'invite si ``report`` vaut ``None``.

        Les lignes existantes sont réutilisées plutôt que détruites et
        recréées : ce tableau se rafraîchit plusieurs fois par seconde pendant
        le calcul, et recréer des widgets à ce rythme fait clignoter l'écran.
        """
        if report is None:
            for row in self._rows:
                row.pack_forget()
            self.placeholder.pack(fill="x", pady=SPACE.SM)
            return

        self.placeholder.pack_forget()

        checks = report.checks
        while len(self._rows) < len(checks):
            self._rows.append(_ComplianceRow(self))

        for i, check in enumerate(checks):
            row = self._rows[i]
            row.update_check(check, self.lang)
            if not row.winfo_ismapped():
                row.pack(fill="x", pady=(0, SPACE.SM))

        for row in self._rows[len(checks):]:
            row.pack_forget()


class VerdictBanner(ctk.CTkFrame):
    """Bandeau de synthèse : conforme, livrable avec réserves, ou non conforme."""

    def __init__(self, master, **kwargs):
        theme = current()
        kwargs.setdefault("fg_color", theme.SURFACE_ALT)
        kwargs.setdefault("corner_radius", SPACE.RADIUS_SM)
        super().__init__(master, **kwargs)
        self._theme = theme

        self.marker = ctk.CTkFrame(self, width=6, height=56, corner_radius=3, fg_color=theme.BORDER)
        self.marker.pack(side="left", fill="y", padx=(0, SPACE.MD))

        block = ctk.CTkFrame(self, fg_color="transparent")
        block.pack(side="left", fill="both", expand=True, pady=SPACE.MD)

        self.lbl_title = ctk.CTkLabel(
            block, text="", font=FONT.H1, text_color=theme.TEXT, anchor="w"
        )
        self.lbl_title.pack(fill="x")

        self.lbl_detail = ctk.CTkLabel(
            block, text="", font=FONT.BODY, text_color=theme.TEXT_SOFT,
            anchor="w", justify="left", wraplength=680,
        )
        self.lbl_detail.pack(fill="x")

    def update_verdict(self, report, translator):
        """Met à jour le bandeau à partir d'un rapport et du traducteur courant."""
        theme = self._theme

        if report is None:
            self.marker.configure(fg_color=theme.BORDER)
            self.lbl_title.configure(text=translator("report.verdict.none"), text_color=theme.TEXT_SOFT)
            self.lbl_detail.configure(text="")
            return

        blocking = report.failed("blocking")
        major = report.failed("major")

        if report.is_compliant:
            color, title = theme.ok, translator("report.verdict.ok")
        elif not blocking:
            color, title = theme.warn, translator("report.verdict.deliverable")
        else:
            color, title = theme.danger, translator("report.verdict.ko")

        lang = getattr(translator, "lang", "FR")
        problems = [c.label(lang) for c in blocking + major]
        detail = " · ".join(problems[:3])
        if len(problems) > 3:
            detail += f" (+{len(problems) - 3})"
        if not problems:
            k = report.kpis
            detail = (
                f"{k.get('length_mm', 0):.0f} mm · "
                f"{k.get('straight_ratio', 0) * 100:.0f} % rectiligne · "
                f"{k.get('n_clamps', 0)} fixation(s)"
            )

        self.marker.configure(fg_color=color)
        self.lbl_title.configure(text=title, text_color=color)
        self.lbl_detail.configure(text=detail)
