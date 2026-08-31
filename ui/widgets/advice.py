"""Conseils affichés quand la convergence bloque.

Un conseil sans chiffre ne sert à rien : « rapprochez le câble » n'aide pas.
Chaque carte porte donc la valeur mesurée, la valeur proposée, et — lorsqu'un
réglage existe pour cela — un bouton qui l'applique.

Les conseils qui ne portent aucun bouton ne sont pas des conseils au rabais :
ce sont ceux pour lesquels aucun réglage ne doit être touché, typiquement un
clash ou un coude plus serré que ce qu'un toron supporte.
"""

from __future__ import annotations

import customtkinter as ctk

from ui.theme import FONT, SPACE, current

__all__ = ["AdviceBoard", "AdviceCard"]

#: Correspondance gravité → couleur de la charte.
SEVERITY_COLOR = {
    "blocking": "danger",
    "major": "warn",
    "info": "accent",
}

SEVERITY_ICON = {
    "blocking": "⛔",
    "major": "⚠️",
    "info": "💡",
}


class AdviceCard(ctk.CTkFrame):
    """Une suggestion : titre, constat, action, et bouton s'il y a lieu."""

    def __init__(self, master, suggestion, lang="FR", on_apply=None, **kwargs):
        theme = current()
        kwargs.setdefault("fg_color", theme.SURFACE_ALT)
        kwargs.setdefault("corner_radius", SPACE.RADIUS_SM)
        super().__init__(master, **kwargs)
        self.suggestion = suggestion
        self._on_apply = on_apply
        self._lang = lang

        colour = getattr(theme, SEVERITY_COLOR.get(suggestion.severity, "accent"))

        # Liseré de gravité : la couleur du titre seule se repère mal dans une
        # liste, et se perd complètement pour un œil daltonien.
        bar = ctk.CTkFrame(self, width=4, fg_color=colour, corner_radius=2)
        bar.pack(side="left", fill="y", padx=(0, SPACE.MD))
        bar.pack_propagate(False)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(side="left", fill="both", expand=True, padx=(0, SPACE.MD),
                  pady=SPACE.SM)

        icon = SEVERITY_ICON.get(suggestion.severity, "💡")
        self.lbl_title = ctk.CTkLabel(
            body, text=f"{icon}  {suggestion.title(lang)}", font=FONT.BODY_BOLD,
            text_color=theme.TEXT, anchor="w", justify="left", wraplength=560,
        )
        self.lbl_title.pack(fill="x")

        self.lbl_detail = ctk.CTkLabel(
            body, text=suggestion.detail(lang), font=FONT.SMALL,
            text_color=theme.TEXT_SOFT, anchor="w", justify="left", wraplength=560,
        )
        self.lbl_detail.pack(fill="x", pady=(2, 0))

        self.lbl_action = ctk.CTkLabel(
            body, text=f"→ {suggestion.action(lang)}", font=FONT.SMALL_BOLD,
            text_color=colour, anchor="w", justify="left", wraplength=560,
        )
        self.lbl_action.pack(fill="x", pady=(SPACE.XS, 0))

        self.btn_apply = None
        if suggestion.is_applicable and on_apply is not None:
            self.btn_apply = ctk.CTkButton(
                self, text="Appliquer" if not _english(lang) else "Apply",
                width=110, height=30, font=FONT.SMALL_BOLD,
                fg_color=colour, command=self._apply,
            )
            self.btn_apply.pack(side="right", padx=(0, SPACE.MD), pady=SPACE.SM)

    def _apply(self):
        if self._on_apply is not None:
            self._on_apply(self.suggestion)
        if self.btn_apply is not None:
            theme = current()
            self.btn_apply.configure(
                text="Applied" if _english(self._lang) else "Appliqué",
                state="disabled", fg_color=theme.DISABLED,
            )


def _english(lang: str) -> bool:
    return str(lang).upper().startswith("EN")


class AdviceBoard(ctk.CTkFrame):
    """Liste des conseils en cours, reconstruite à chaque rafraîchissement."""

    def __init__(self, master, lang="FR", on_apply=None, **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self.lang = lang
        self._on_apply = on_apply
        self._suggestions: list = []
        self._keys: tuple = ()

        theme = current()
        self.lbl_empty = ctk.CTkLabel(
            self, text="", font=FONT.BODY, text_color=theme.TEXT_FAINT,
            justify="center", wraplength=520,
        )
        self.cards_box = ctk.CTkFrame(self, fg_color="transparent")
        self._show_placeholder(_default_placeholder(lang))

    # -- contenu ----------------------------------------------------------

    def update_advice(self, suggestions):
        """Met à jour la liste. Ne reconstruit que si elle a changé.

        Reconstruire les cartes à chaque rafraîchissement — quatre fois par
        seconde — ferait clignoter le texte et remettrait à zéro les boutons
        déjà cliqués.
        """
        suggestions = list(suggestions or [])
        keys = tuple((s.key, s.setting, s.value) for s in suggestions)
        if keys == self._keys:
            return
        self._keys = keys
        self._suggestions = suggestions

        for widget in self.cards_box.winfo_children():
            widget.destroy()

        if not suggestions:
            self.cards_box.pack_forget()
            self._show_placeholder(_default_placeholder(self.lang))
            return

        self.lbl_empty.pack_forget()
        self.cards_box.pack(fill="both", expand=True)
        for suggestion in suggestions:
            AdviceCard(
                self.cards_box, suggestion, lang=self.lang, on_apply=self._on_apply,
            ).pack(fill="x", pady=(0, SPACE.SM))

    def count(self) -> int:
        return len(self._suggestions)

    def clear(self):
        self._keys = ()
        self._suggestions = []
        for widget in self.cards_box.winfo_children():
            widget.destroy()
        self.cards_box.pack_forget()
        self._show_placeholder(_default_placeholder(self.lang))

    def _show_placeholder(self, text: str):
        self.lbl_empty.configure(text=text)
        self.lbl_empty.pack(expand=True, padx=SPACE.LG, pady=SPACE.XL)

    def update_language(self, lang: str):
        self.lang = lang
        keys, suggestions = self._keys, self._suggestions
        self._keys = ()
        if suggestions:
            self.update_advice(suggestions)
        else:
            self._show_placeholder(_default_placeholder(lang))
        self._keys = keys


def _default_placeholder(lang: str) -> str:
    if _english(lang):
        return (
            "No advice for now.\n\n"
            "Advice appears once the agents have had a real go at it and their "
            "best score stops improving — never before, so as not to suggest "
            "lowering the requirements at the first obstacle."
        )
    return (
        "Aucun conseil pour l'instant.\n\n"
        "Les conseils apparaissent une fois que les agents ont réellement "
        "cherché et que leur meilleur score cesse de progresser — jamais avant, "
        "pour ne pas inciter à baisser les exigences au premier obstacle."
    )
