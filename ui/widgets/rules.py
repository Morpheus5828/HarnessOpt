"""Cases à cocher des règles d'intégration.

Une règle n'est pas une option d'affichage : la décocher change ce que
l'application calcule, ce qu'elle classe et ce qu'elle récompense. La ligne le
dit donc explicitement — libellé, gravité, conséquence — plutôt que de se
réduire à une case anonyme.
"""

from __future__ import annotations

import customtkinter as ctk

from core.routing_rules import RULE_CATALOG, Severity
from ui.theme import FONT, SPACE, current

__all__ = ["RuleToggle", "RuleToggleList"]

#: Libellés de gravité, par langue.
SEVERITY_LABELS = {
    Severity.BLOCKING: ("Bloquant", "Blocking"),
    Severity.MAJOR: ("Majeur", "Major"),
    Severity.MINOR: ("Qualité", "Quality"),
}

#: Hauteur d'une ligne. Fixée explicitement : un CTkFrame sans enfant retombe
#: sinon sur 200 px de haut et disloque la liste.
ROW_HEIGHT = 64


def severity_label(severity: str, lang: str = "FR") -> str:
    fr, en = SEVERITY_LABELS.get(severity, ("Règle", "Rule"))
    return en if str(lang).upper().startswith("EN") else fr


class RuleToggle(ctk.CTkFrame):
    """Une règle : case à cocher, libellé, gravité et explication."""

    def __init__(self, master, info, value: bool = True, on_change=None, lang="FR", **kwargs):
        theme = current()
        kwargs.setdefault("fg_color", "transparent")
        kwargs.setdefault("height", ROW_HEIGHT)
        super().__init__(master, **kwargs)
        self.info = info
        self._on_change = on_change
        self._lang = lang

        self.grid_columnconfigure(1, weight=1)
        self.grid_propagate(False)

        self.var = ctk.BooleanVar(value=bool(value))
        self.box = ctk.CTkCheckBox(
            self, text="", width=24, checkbox_width=20, checkbox_height=20,
            variable=self.var, fg_color=theme.accent,
            hover_color=theme.accent, command=self._changed,
        )
        self.box.grid(row=0, column=0, rowspan=2, sticky="n", padx=(SPACE.SM, SPACE.MD),
                      pady=(SPACE.XS, 0))

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=1, sticky="ew")

        self.lbl_name = ctk.CTkLabel(
            head, text=info.label(lang), font=FONT.BODY_BOLD,
            text_color=theme.TEXT, anchor="w",
        )
        self.lbl_name.pack(side="left")

        colour = theme.severity_color(info.severity, passed=False)
        self.lbl_severity = ctk.CTkLabel(
            head, text=severity_label(info.severity, lang), font=FONT.TINY,
            text_color=colour, fg_color="transparent", anchor="w",
        )
        self.lbl_severity.pack(side="left", padx=(SPACE.SM, 0))

        self.lbl_help = ctk.CTkLabel(
            self, text=info.help(lang), font=FONT.SMALL, text_color=theme.TEXT_SOFT,
            anchor="w", justify="left", wraplength=680,
        )
        self.lbl_help.grid(row=1, column=1, sticky="ew", pady=(2, 0))

        self._apply_state()

    # -- lecture et écriture --------------------------------------------

    def get(self) -> bool:
        return bool(self.var.get())

    def set(self, value: bool):
        self.var.set(bool(value))
        self._apply_state()

    def _changed(self):
        self._apply_state()
        if self._on_change is not None:
            self._on_change(self.info.rule_id, self.get())

    def _apply_state(self):
        """Grise la ligne décochée : l'état doit se lire d'un coup d'œil."""
        theme = current()
        active = self.get()
        self.lbl_name.configure(text_color=theme.TEXT if active else theme.TEXT_FAINT)
        self.lbl_help.configure(text_color=theme.TEXT_SOFT if active else theme.TEXT_FAINT)
        self.lbl_severity.configure(
            text_color=theme.severity_color(self.info.severity, passed=False)
            if active else theme.TEXT_FAINT
        )

    def update_language(self, lang: str):
        self._lang = lang
        self.lbl_name.configure(text=self.info.label(lang))
        self.lbl_help.configure(text=self.info.help(lang))
        self.lbl_severity.configure(text=severity_label(self.info.severity, lang))


class RuleToggleList(ctk.CTkFrame):
    """Toutes les règles du catalogue, avec un résumé et une mise en garde.

    Le résumé n'est pas décoratif : décocher une règle rédhibitoire — « aucune
    interférence », par exemple — produit une route que le bureau d'études ne
    pourra pas livrer. L'application le dit au moment où la case est décochée,
    pas au moment du rapport.
    """

    def __init__(self, master, enabled=None, on_change=None, lang="FR", **kwargs):
        theme = current()
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self._on_change = on_change
        self._lang = lang

        active = set(enabled) if enabled is not None else {i.rule_id for i in RULE_CATALOG}

        self.rows: dict[str, RuleToggle] = {}
        for index, info in enumerate(RULE_CATALOG):
            row = RuleToggle(
                self, info, value=info.rule_id in active,
                on_change=self._row_changed, lang=lang,
            )
            row.pack(fill="x", pady=(0, SPACE.XS))
            self.rows[info.rule_id] = row
            if index < len(RULE_CATALOG) - 1:
                ctk.CTkFrame(self, height=1, fg_color=theme.BORDER).pack(
                    fill="x", pady=(0, SPACE.XS)
                )

        self.lbl_summary = ctk.CTkLabel(
            self, text="", font=FONT.SMALL_BOLD, anchor="w",
            justify="left", wraplength=680,
        )
        self.lbl_summary.pack(fill="x", pady=(SPACE.SM, 0))
        self._refresh_summary()

    # -- lecture ---------------------------------------------------------

    def get(self) -> set:
        """Identifiants des règles cochées."""
        return {rule_id for rule_id, row in self.rows.items() if row.get()}

    def set(self, enabled):
        active = set(enabled)
        for rule_id, row in self.rows.items():
            row.set(rule_id in active)
        self._refresh_summary()

    def select_all(self):
        self.set(set(self.rows))

    # -- réactions --------------------------------------------------------

    def _row_changed(self, rule_id: str, value: bool):
        self._refresh_summary()
        if self._on_change is not None:
            self._on_change(rule_id, value)

    def _refresh_summary(self):
        theme = current()
        english = str(self._lang).upper().startswith("EN")
        active = self.get()
        total = len(self.rows)

        dropped_blocking = [
            row.info for rule_id, row in self.rows.items()
            if rule_id not in active and row.info.severity == Severity.BLOCKING
        ]

        if len(active) == total:
            text = ("All %d rules are applied." % total if english
                    else "Les %d règles sont appliquées." % total)
            colour = theme.ok
        elif dropped_blocking:
            names = ", ".join(info.label(self._lang).lower() for info in dropped_blocking)
            text = (
                f"{len(active)}/{total} rules applied. Warning: a blocking rule is off "
                f"({names}). The route may be undeliverable."
                if english else
                f"{len(active)}/{total} règles appliquées. Attention : une règle "
                f"rédhibitoire est désactivée ({names}). La route obtenue pourra ne "
                "pas être livrable."
            )
            colour = theme.danger
        else:
            text = (f"{len(active)}/{total} rules applied."
                    if english else f"{len(active)}/{total} règles appliquées.")
            colour = theme.warn

        self.lbl_summary.configure(text=text, text_color=colour)

    def update_language(self, lang: str):
        self._lang = lang
        for row in self.rows.values():
            row.update_language(lang)
        self._refresh_summary()
