"""Les courbes des agents, en grand, dans leur propre fenêtre.

Les courbes vivent dans un onglet de la page *Cheminement*, à côté de la
conformité, des conseils et du tableau des agents. C'est la bonne place pour
jeter un œil ; ce n'en est pas une pour **suivre** une session. Quatre séries
empilées dans un quart d'écran donnent des courbes hautes de deux
centimètres : on y voit qu'une valeur monte ou descend, pas où elle en est.

Cette fenêtre les reprend telles quelles, à la taille de l'écran, à côté de la
vue 3D. Elle ne duplique aucune logique : c'est le même
:class:`~ui.charts.ProgressCharts`, alimenté par le même instantané. Deux jeux
de courbes qui divergeraient seraient pires qu'un seul trop petit.
"""

from __future__ import annotations

import customtkinter as ctk

from ui.charts import ProgressCharts
from ui.theme import FONT, SPACE, current

__all__ = ["ChartsWindow"]

#: Taille de repli, en fraction de l'écran, quand on quitte le plein écran.
#: La fenêtre 3D est à côté : sortir du plein écran doit la redécouvrir, pas
#: la remplacer par une fenêtre presque aussi grande.
SCREEN_FRACTION = 0.75


class ChartsWindow(ctk.CTkToplevel):
    """Fenêtre autonome portant les courbes de progression."""

    def __init__(self, master, lang: str = "FR", on_close=None):
        super().__init__(master)
        theme = current()
        self._on_close = on_close

        english = str(lang).upper().startswith("EN")
        self._english = english
        self.title("Agent performance" if english else "Performances des agents")
        self.configure(fg_color=theme.BG)

        try:
            width = int(self.winfo_screenwidth() * SCREEN_FRACTION)
            height = int(self.winfo_screenheight() * SCREEN_FRACTION)
            self.geometry(f"{width}x{height}")
        except Exception:
            self.geometry("1200x900")

        # Plein écran d'emblée : quatre séries empilées ne se lisent qu'en
        # grand, et c'est ce qu'on est venu chercher en ouvrant cette fenêtre.
        # Une fenêtre plein écran sans porte de sortie visible est un piège :
        # Échap, F11 et le bouton en font trois.
        self.fullscreen = False
        self._set_fullscreen(True)
        self.bind("<Escape>", lambda _event: self._set_fullscreen(False))
        self.bind("<F11>", lambda _event: self._set_fullscreen(not self.fullscreen))

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=SPACE.LG, pady=(SPACE.MD, 0))
        ctk.CTkLabel(
            header,
            text=("Agent performance over the run" if english
                  else "Évolution des performances des agents"),
            font=FONT.H1, text_color=theme.TEXT, anchor="w",
        ).pack(side="left")

        self.btn_fullscreen = ctk.CTkButton(
            header, text="", width=150, height=28, font=FONT.SMALL,
            fg_color="transparent", border_width=1, border_color=theme.BORDER,
            text_color=theme.TEXT, hover_color=theme.SURFACE_ALT,
            command=lambda: self._set_fullscreen(not self.fullscreen),
        )
        self.btn_fullscreen.pack(side="right", padx=(SPACE.MD, 0))

        self.lbl_state = ctk.CTkLabel(
            header, text="", font=FONT.SMALL, text_color=theme.TEXT_SOFT, anchor="e"
        )
        self.lbl_state.pack(side="right")

        body = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=SPACE.RADIUS)
        body.pack(fill="both", expand=True, padx=SPACE.LG, pady=SPACE.MD)

        self.charts = ProgressCharts(body, lang=lang)
        self.available = self.charts.start()

        self.protocol("WM_DELETE_WINDOW", self.close)
        self._label_fullscreen()

    # ------------------------------------------------------------------
    def _set_fullscreen(self, plein: bool):
        """Bascule le plein écran. Sans effet si le gestionnaire le refuse."""
        try:
            self.attributes("-fullscreen", bool(plein))
            self.fullscreen = bool(plein)
        except Exception:
            # Certains gestionnaires de fenêtres ignorent l'attribut : la
            # fenêtre reste à sa taille, et le bouton doit dire la vérité.
            self.fullscreen = False
        self._label_fullscreen()

    def _label_fullscreen(self):
        if getattr(self, "btn_fullscreen", None) is None:
            return
        anglais = self._english
        if self.fullscreen:
            texte = "Exit full screen (Esc)" if anglais else "Quitter le plein écran (Échap)"
        else:
            texte = "Full screen (F11)" if anglais else "Plein écran (F11)"
        try:
            self.btn_fullscreen.configure(text=texte)
        except Exception:
            pass

    def set_limits(self, min_margin: float, max_margin: float, min_bend_radius: float):
        if self.available:
            self.charts.set_limits(min_margin, max_margin, min_bend_radius)

    def update_charts(self, snapshot: dict, colors: dict):
        """Rafraîchit les courbes. Silencieux si la fenêtre est déjà fermée."""
        if not self.available:
            return
        try:
            self.charts.update(snapshot, colors)
            iterations = max((s.get("iteration", 0) for s in snapshot.values()),
                            default=0)
            unite = "iteration(s)" if self._english else "itération(s)"
            self.lbl_state.configure(
                text=f"{len(snapshot)} agent(s) · {iterations} {unite}"
            )
        except Exception:
            # Une fenêtre détruite entre deux rafraîchissements ne doit pas
            # emporter la boucle d'affichage du cheminement.
            self.available = False

    def reset(self):
        if self.available:
            self.charts.reset()

    def close(self):
        self.available = False
        # Détruire une fenêtre restée en plein écran laisse, sur certains
        # gestionnaires, l'écran sans barre de tâches jusqu'au prochain clic.
        try:
            self.attributes("-fullscreen", False)
        except Exception:
            pass
        if self._on_close is not None:
            try:
                self._on_close()
            except Exception:
                pass
        try:
            self.destroy()
        except Exception:
            pass
