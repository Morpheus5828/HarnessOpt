"""Tableau de bord de l'équipe d'agents.

L'utilisateur doit pouvoir voir *qui* travaille et *sur quoi*. Chaque agent a
une carte portant sa spécialité en clair, son rang, et son état courant. La
couleur de la carte est celle de sa trajectoire dans la vue 3D : on relie ainsi
d'un coup d'œil une courbe à l'agent qui l'a produite.
"""

from __future__ import annotations

import customtkinter as ctk

from ui.theme import FONT, SPACE, current

__all__ = ["AgentBoard", "PhaseIndicator"]


class _AgentCard(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        theme = current()
        kwargs.setdefault("fg_color", theme.SURFACE_ALT)
        kwargs.setdefault("corner_radius", SPACE.RADIUS_SM)
        super().__init__(master, **kwargs)
        self._theme = theme

        # `height` explicite : sans enfant, un CTkFrame garde 200 px de haut et
        # étirerait la carte d'autant.
        self.stripe = ctk.CTkFrame(self, width=5, height=46, corner_radius=3, fg_color=theme.BORDER)
        self.stripe.pack(side="left", fill="y", padx=(0, SPACE.SM))

        block = ctk.CTkFrame(self, fg_color="transparent")
        block.pack(side="left", fill="x", expand=True, pady=SPACE.SM)

        head = ctk.CTkFrame(block, fg_color="transparent")
        head.pack(fill="x")

        self.lbl_rank = ctk.CTkLabel(
            head, text="", font=FONT.SMALL_BOLD, text_color=theme.TEXT_FAINT, width=26, anchor="w"
        )
        self.lbl_rank.pack(side="left")

        self.lbl_role = ctk.CTkLabel(
            head, text="", font=FONT.BODY_BOLD, text_color=theme.TEXT, anchor="w"
        )
        self.lbl_role.pack(side="left")

        self.lbl_badge = ctk.CTkLabel(
            head, text="", font=FONT.TINY, text_color=theme.accent, anchor="e"
        )
        self.lbl_badge.pack(side="right")

        self.lbl_state = ctk.CTkLabel(
            block, text="", font=FONT.SMALL, text_color=theme.TEXT_SOFT,
            anchor="w", justify="left", wraplength=420,
        )
        self.lbl_state.pack(fill="x")

    def update_agent(self, info: dict):
        theme = self._theme
        # `or` plutôt que la valeur par défaut de `get` : les appelants passent
        # volontiers une clé présente mais nulle, que customtkinter refuse.
        self.stripe.configure(fg_color=info.get("color") or theme.BORDER)
        rank = info.get("rank")
        self.lbl_rank.configure(text=f"#{rank}" if rank else "—")
        self.lbl_role.configure(text=info.get("label") or info.get("name") or "")
        self.lbl_badge.configure(
            text=info.get("badge") or "",
            text_color=info.get("badge_color") or theme.accent,
        )
        self.lbl_state.configure(text=info.get("state") or "")


class AgentBoard(ctk.CTkFrame):
    """Une carte par agent, triées par rang."""

    def __init__(self, master, **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self._cards: list[_AgentCard] = []

    def update_agents(self, agents: list[dict]):
        """Affiche la liste d'agents, en réutilisant les cartes existantes."""
        while len(self._cards) < len(agents):
            self._cards.append(_AgentCard(self))

        for card, info in zip(self._cards, agents):
            card.update_agent(info)
            if not card.winfo_ismapped():
                card.pack(fill="x", pady=(0, SPACE.SM))

        for card in self._cards[len(agents):]:
            card.pack_forget()


class PhaseIndicator(ctk.CTkFrame):
    """Étape du chantier en cours, avec les étapes déjà franchies.

    Traduit le curriculum de l'orchestrateur en quelque chose de lisible :
    l'utilisateur voit que le système cherche d'abord un passage, puis met aux
    distances, puis pose les fixations, puis lisse.
    """

    ORDER = ("feasibility", "clearance", "support", "polish")

    def __init__(self, master, **kwargs):
        theme = current()
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self._theme = theme
        self._dots: dict[str, ctk.CTkFrame] = {}
        self._labels: dict[str, ctk.CTkLabel] = {}

        for i, phase in enumerate(self.ORDER):
            block = ctk.CTkFrame(self, fg_color="transparent")
            block.pack(side="left", padx=(0 if i == 0 else SPACE.MD, 0))

            dot = ctk.CTkFrame(block, width=10, height=10, corner_radius=5, fg_color=theme.BORDER)
            dot.pack(side="left", pady=(0, 1))
            dot.pack_propagate(False)

            label = ctk.CTkLabel(
                block, text="", font=FONT.SMALL, text_color=theme.TEXT_FAINT
            )
            label.pack(side="left", padx=(SPACE.XS, 0))

            self._dots[phase] = dot
            self._labels[phase] = label

    def update_phase(self, phase: str, labels: dict[str, str]):
        """``labels`` : ``{clé d'étape: libellé traduit}``."""
        theme = self._theme
        try:
            current_index = self.ORDER.index(phase)
        except ValueError:
            current_index = -1

        for i, key in enumerate(self.ORDER):
            self._labels[key].configure(text=labels.get(key, key))
            if i < current_index:
                self._dots[key].configure(fg_color=theme.ok)
                self._labels[key].configure(text_color=theme.TEXT_FAINT)
            elif i == current_index:
                self._dots[key].configure(fg_color=theme.accent)
                self._labels[key].configure(text_color=theme.accent)
            else:
                self._dots[key].configure(fg_color=theme.BORDER)
                self._labels[key].configure(text_color=theme.TEXT_FAINT)
