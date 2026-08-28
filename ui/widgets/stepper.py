"""Barre d'étapes : où l'on en est, et ce qu'il reste à faire.

L'ancienne interface proposait deux entrées de menu latéral sans ordre visible.
On explicite ici le déroulé : charger la maquette, régler les contraintes,
faire cheminer, contrôler. Une étape non atteignable est grisée, avec la raison
affichée au survol : l'utilisateur n'est jamais bloqué sans explication.
"""

from __future__ import annotations

import customtkinter as ctk

from ui.theme import FONT, SPACE, current

__all__ = ["Stepper"]


class _Step(ctk.CTkFrame):
    def __init__(self, master, index: int, on_click, **kwargs):
        theme = current()
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self._theme = theme
        self._index = index
        self._on_click = on_click
        self._enabled = False

        self.bubble = ctk.CTkButton(
            self,
            text=str(index + 1),
            width=34,
            height=34,
            corner_radius=17,
            font=FONT.BODY_BOLD,
            fg_color=theme.SURFACE_ALT,
            hover_color=theme.BORDER,
            text_color=theme.TEXT_FAINT,
            command=self._clicked,
        )
        self.bubble.pack(side="left")

        block = ctk.CTkFrame(self, fg_color="transparent")
        block.pack(side="left", padx=(SPACE.SM, 0))

        self.lbl_title = ctk.CTkLabel(
            block, text="", font=FONT.BODY_BOLD, text_color=theme.TEXT_FAINT, anchor="w"
        )
        self.lbl_title.pack(fill="x")

        self.lbl_sub = ctk.CTkLabel(
            block, text="", font=FONT.TINY, text_color=theme.TEXT_FAINT, anchor="w"
        )
        self.lbl_sub.pack(fill="x")

    def _clicked(self):
        if self._enabled and self._on_click is not None:
            self._on_click(self._index)

    def set_texts(self, title: str, subtitle: str):
        self.lbl_title.configure(text=title)
        self.lbl_sub.configure(text=subtitle)

    def set_state(self, state: str):
        """``state`` : ``current``, ``done``, ``available`` ou ``locked``."""
        theme = self._theme
        self._enabled = state != "locked"

        if state == "current":
            self.bubble.configure(
                fg_color=theme.accent, hover_color=theme.accent,
                text_color=theme.TEXT_ON_ACCENT, text=str(self._index + 1),
            )
            self.lbl_title.configure(text_color=theme.TEXT)
            self.lbl_sub.configure(text_color=theme.TEXT_SOFT)
        elif state == "done":
            self.bubble.configure(
                fg_color=theme.ok, hover_color=theme.ok,
                text_color=theme.TEXT_ON_ACCENT, text="✓",
            )
            self.lbl_title.configure(text_color=theme.TEXT_SOFT)
            self.lbl_sub.configure(text_color=theme.TEXT_FAINT)
        elif state == "available":
            self.bubble.configure(
                fg_color=theme.SURFACE_ALT, hover_color=theme.BORDER,
                text_color=theme.TEXT, text=str(self._index + 1),
            )
            self.lbl_title.configure(text_color=theme.TEXT_SOFT)
            self.lbl_sub.configure(text_color=theme.TEXT_FAINT)
        else:
            self.bubble.configure(
                fg_color=theme.DISABLED, hover_color=theme.DISABLED,
                text_color=theme.TEXT_FAINT, text="🔒",
            )
            self.lbl_title.configure(text_color=theme.TEXT_FAINT)
            self.lbl_sub.configure(text_color=theme.TEXT_FAINT)


class Stepper(ctk.CTkFrame):
    """Suite d'étapes cliquables."""

    def __init__(self, master, n_steps: int, on_select=None, **kwargs):
        theme = current()
        kwargs.setdefault("fg_color", theme.SURFACE)
        kwargs.setdefault("corner_radius", SPACE.RADIUS)
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", theme.BORDER)
        super().__init__(master, **kwargs)

        self._steps: list[_Step] = []
        self._current = 0
        self._unlocked = 0

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="x", padx=SPACE.MD, pady=SPACE.MD)

        for i in range(n_steps):
            if i:
                ctk.CTkFrame(inner, height=2, width=40, fg_color=theme.BORDER).pack(
                    side="left", padx=SPACE.MD, pady=(0, 8)
                )
            step = _Step(inner, i, on_select)
            step.pack(side="left")
            self._steps.append(step)

    def set_texts(self, texts: list[tuple[str, str]]):
        for step, (title, subtitle) in zip(self._steps, texts):
            step.set_texts(title, subtitle)

    def set_current(self, index: int):
        self._current = index
        self._refresh()

    def unlock_up_to(self, index: int):
        """Rend accessibles toutes les étapes jusqu'à ``index`` inclus."""
        self._unlocked = max(self._unlocked, index)
        self._refresh()

    def reset_unlock(self, index: int = 0):
        self._unlocked = index
        self._refresh()

    @property
    def unlocked(self) -> int:
        return self._unlocked

    def _refresh(self):
        for i, step in enumerate(self._steps):
            if i == self._current:
                step.set_state("current")
            elif i < self._current and i <= self._unlocked:
                step.set_state("done")
            elif i <= self._unlocked:
                step.set_state("available")
            else:
                step.set_state("locked")
