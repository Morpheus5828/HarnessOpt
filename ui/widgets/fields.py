"""Champs de saisie en langage métier.

Principe directeur : aucun champ n'apparaît sans un intitulé compréhensible,
son unité physique, et une phrase qui explique à quoi il sert. L'utilisateur
n'a pas à deviner ce que signifie une valeur, ni dans quelle unité la saisir.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk

from ui.theme import FONT, SPACE, current

__all__ = [
    "LabeledEntry",
    "NumberField",
    "SliderField",
    "PathField",
    "ChoiceField",
    "CoordinateField",
    "ToggleField",
]


class _Field(ctk.CTkFrame):
    """Base commune : intitulé au-dessus, aide en dessous."""

    def __init__(self, master, label: str, help_text: str = "", width: int = 520, **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        theme = current()

        self.lbl = ctk.CTkLabel(
            self, text=label, font=FONT.BODY_BOLD, text_color=theme.TEXT, anchor="w"
        )
        self.lbl.pack(fill="x")

        self.row = ctk.CTkFrame(self, fg_color="transparent")
        self.row.pack(fill="x", pady=(SPACE.XS, 0))

        self.help = None
        if help_text:
            self.help = ctk.CTkLabel(
                self,
                text=help_text,
                font=FONT.SMALL,
                text_color=theme.TEXT_FAINT,
                anchor="w",
                justify="left",
                wraplength=width,
            )
            self.help.pack(fill="x", pady=(2, 0))

    def set_label(self, label: str, help_text: str | None = None):
        self.lbl.configure(text=label)
        if help_text is not None and self.help is not None:
            self.help.configure(text=help_text)


class LabeledEntry(_Field):
    """Champ texte libre."""

    def __init__(self, master, label: str, help_text: str = "", placeholder: str = "",
                 value: str = "", entry_width: int = 380, **kwargs):
        super().__init__(master, label, help_text, **kwargs)
        theme = current()
        self.entry = ctk.CTkEntry(
            self.row,
            placeholder_text=placeholder,
            width=entry_width,
            height=34,
            font=FONT.BODY,
            fg_color=theme.SURFACE_ALT,
            border_color=theme.BORDER,
            text_color=theme.TEXT,
        )
        self.entry.pack(side="left", fill="x", expand=True)
        if value:
            self.entry.insert(0, value)

    def get(self) -> str:
        return self.entry.get().strip()

    def set(self, value: str):
        self.entry.delete(0, "end")
        self.entry.insert(0, str(value))

    def set_placeholder(self, text: str):
        self.entry.configure(placeholder_text=text)


class NumberField(_Field):
    """Valeur numérique avec son unité affichée à côté du champ."""

    def __init__(self, master, label: str, help_text: str = "", value: float = 0.0,
                 unit: str = "mm", entry_width: int = 110, on_change=None, **kwargs):
        super().__init__(master, label, help_text, **kwargs)
        theme = current()
        self._on_change = on_change

        self.entry = ctk.CTkEntry(
            self.row,
            width=entry_width,
            height=34,
            font=FONT.VALUE_SM,
            justify="right",
            fg_color=theme.SURFACE_ALT,
            border_color=theme.BORDER,
            text_color=theme.TEXT,
        )
        self.entry.pack(side="left")
        self.entry.insert(0, self._format(value))
        self.entry.bind("<KeyRelease>", self._changed)
        self.entry.bind("<FocusOut>", self._changed)

        self.lbl_unit = ctk.CTkLabel(
            self.row, text=unit, font=FONT.BODY, text_color=theme.TEXT_SOFT
        )
        self.lbl_unit.pack(side="left", padx=(SPACE.SM, 0))

    @staticmethod
    def _format(value: float) -> str:
        return f"{value:g}"

    def _changed(self, _event=None):
        if self._on_change is not None:
            self._on_change(self.get())

    def get(self, default: float = 0.0) -> float:
        """Valeur saisie. Tolère la virgule décimale et renvoie ``default`` si illisible."""
        try:
            return float(self.entry.get().replace(",", ".").strip())
        except (TypeError, ValueError):
            return default

    def set(self, value: float):
        self.entry.delete(0, "end")
        self.entry.insert(0, self._format(value))

    def set_unit(self, unit: str):
        self.lbl_unit.configure(text=unit)


class SliderField(_Field):
    """Curseur avec sa valeur courante et, aux extrémités, ce qu'elles signifient.

    Utilisé notamment pour le réglage Exploration ↔ Exploitation : l'utilisateur
    voit en permanence l'effet du réglage écrit en toutes lettres, plutôt qu'un
    nombre sans signification.
    """

    def __init__(self, master, label: str, help_text: str = "", from_: float = 0.0,
                 to: float = 1.0, value: float = 0.5, steps: int | None = None,
                 left_label: str = "", right_label: str = "",
                 value_formatter=None, on_change=None, **kwargs):
        super().__init__(master, label, help_text, **kwargs)
        theme = current()
        self._on_change = on_change
        self._formatter = value_formatter or (lambda v: f"{v:.2f}")

        self.slider = ctk.CTkSlider(
            self.row,
            from_=from_,
            to=to,
            number_of_steps=steps,
            command=self._changed,
            progress_color=theme.accent,
            button_color=theme.accent,
            button_hover_color=theme.accent,
        )
        self.slider.set(value)
        self.slider.pack(side="left", fill="x", expand=True)

        self.lbl_value = ctk.CTkLabel(
            self.row,
            text=self._formatter(value),
            font=FONT.BODY_BOLD,
            text_color=theme.accent,
            width=150,
            anchor="e",
        )
        self.lbl_value.pack(side="left", padx=(SPACE.SM, 0))

        if left_label or right_label:
            ends = ctk.CTkFrame(self, fg_color="transparent")
            ends.pack(fill="x")
            self.lbl_left = ctk.CTkLabel(
                ends, text=f"◀ {left_label}", font=FONT.TINY, text_color=theme.TEXT_FAINT
            )
            self.lbl_left.pack(side="left")
            self.lbl_right = ctk.CTkLabel(
                ends, text=f"{right_label} ▶", font=FONT.TINY, text_color=theme.TEXT_FAINT
            )
            self.lbl_right.pack(side="right", padx=(0, 158))
        else:
            self.lbl_left = self.lbl_right = None

    def _changed(self, value):
        self.lbl_value.configure(text=self._formatter(float(value)))
        if self._on_change is not None:
            self._on_change(float(value))

    def get(self) -> float:
        return float(self.slider.get())

    def set(self, value: float):
        self.slider.set(value)
        self.lbl_value.configure(text=self._formatter(float(value)))

    def set_end_labels(self, left: str, right: str):
        if self.lbl_left is not None:
            self.lbl_left.configure(text=f"◀ {left}")
            self.lbl_right.configure(text=f"{right} ▶")

    def refresh_value_label(self):
        self.lbl_value.configure(text=self._formatter(self.get()))


class PathField(_Field):
    """Chemin de dossier ou de fichier, avec bouton de sélection."""

    def __init__(self, master, label: str, help_text: str = "", value: str = "",
                 browse_text: str = "Parcourir…", mode: str = "directory",
                 filetypes=None, on_change=None, **kwargs):
        super().__init__(master, label, help_text, **kwargs)
        theme = current()
        self._mode = mode
        self._filetypes = filetypes or [("Tous les fichiers", "*.*")]
        self._on_change = on_change

        self.entry = ctk.CTkEntry(
            self.row,
            height=34,
            font=FONT.BODY,
            fg_color=theme.SURFACE_ALT,
            border_color=theme.BORDER,
            text_color=theme.TEXT,
        )
        self.entry.pack(side="left", fill="x", expand=True)
        if value:
            self.entry.insert(0, value)

        self.btn = ctk.CTkButton(
            self.row,
            text=browse_text,
            width=110,
            height=34,
            font=FONT.SMALL_BOLD,
            fg_color=theme.SURFACE_ALT,
            hover_color=theme.BORDER,
            text_color=theme.TEXT,
            border_width=1,
            border_color=theme.BORDER,
            command=self._browse,
        )
        self.btn.pack(side="left", padx=(SPACE.SM, 0))

    def _browse(self):
        if self._mode == "directory":
            chosen = filedialog.askdirectory(title=self.lbl.cget("text"))
        else:
            chosen = filedialog.askopenfilename(title=self.lbl.cget("text"), filetypes=self._filetypes)
        if chosen:
            self.set(chosen)
            if self._on_change is not None:
                self._on_change(chosen)

    def get(self) -> str:
        return self.entry.get().strip()

    def set(self, value: str):
        self.entry.delete(0, "end")
        self.entry.insert(0, str(value))

    def set_browse_text(self, text: str):
        self.btn.configure(text=text)


class ChoiceField(_Field):
    """Choix parmi quelques options, présentées en boutons côte à côte.

    Préféré à une liste déroulante quand les options sont peu nombreuses :
    l'utilisateur voit d'emblée tout ce qui lui est proposé.
    """

    def __init__(self, master, label: str, help_text: str = "", options=None,
                 value: str = "", on_change=None, **kwargs):
        super().__init__(master, label, help_text, **kwargs)
        theme = current()
        self._on_change = on_change
        self._values: list[str] = []
        self._labels: list[str] = []

        self.segmented = ctk.CTkSegmentedButton(
            self.row,
            values=[],
            command=self._changed,
            font=FONT.BODY,
            selected_color=theme.accent,
            selected_hover_color=theme.accent,
            unselected_color=theme.SURFACE_ALT,
            text_color=theme.TEXT,
        )
        self.segmented.pack(side="left", fill="x", expand=True)

        if options:
            self.set_options(options, value)

    def set_options(self, options, value: str = ""):
        """``options`` : liste de ``(valeur, libellé)`` ou de chaînes."""
        pairs = [(o, o) if isinstance(o, str) else tuple(o) for o in options]
        self._values = [p[0] for p in pairs]
        self._labels = [p[1] for p in pairs]
        self.segmented.configure(values=self._labels)
        target = value if value in self._values else (self._values[0] if self._values else "")
        if target:
            self.segmented.set(self._labels[self._values.index(target)])

    def _changed(self, label):
        if self._on_change is not None and label in self._labels:
            self._on_change(self._values[self._labels.index(label)])

    def get(self) -> str:
        label = self.segmented.get()
        return self._values[self._labels.index(label)] if label in self._labels else ""

    def set(self, value: str):
        if value in self._values:
            self.segmented.set(self._labels[self._values.index(value)])


class CoordinateField(_Field):
    """Point 3D saisi en X, Y, Z séparés, plutôt qu'en une chaîne à virgules.

    Trois champs distincts évitent l'erreur de saisie la plus courante (oublier
    une virgule) et rendent immédiatement visible quel axe on est en train de
    modifier.
    """

    AXES = ("X", "Y", "Z")

    def __init__(self, master, label: str, help_text: str = "",
                 value=(0.0, 0.0, 0.0), unit: str = "mm", on_change=None, **kwargs):
        super().__init__(master, label, help_text, **kwargs)
        theme = current()
        self._on_change = on_change
        self.entries: list[ctk.CTkEntry] = []

        for i, axis in enumerate(self.AXES):
            ctk.CTkLabel(
                self.row, text=axis, font=FONT.SMALL_BOLD, text_color=theme.TEXT_SOFT, width=14
            ).pack(side="left", padx=(0 if i == 0 else SPACE.SM, SPACE.XS))
            entry = ctk.CTkEntry(
                self.row,
                width=92,
                height=34,
                font=FONT.VALUE_SM,
                justify="right",
                fg_color=theme.SURFACE_ALT,
                border_color=theme.BORDER,
                text_color=theme.TEXT,
            )
            entry.insert(0, f"{float(value[i]):g}")
            entry.pack(side="left")
            entry.bind("<FocusOut>", self._changed)
            self.entries.append(entry)

        ctk.CTkLabel(
            self.row, text=unit, font=FONT.BODY, text_color=theme.TEXT_SOFT
        ).pack(side="left", padx=(SPACE.SM, 0))

    def _changed(self, _event=None):
        if self._on_change is not None:
            self._on_change(self.get())

    def get(self, default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
        """Coordonnées saisies ; renvoie ``default`` si l'une est illisible."""
        out = []
        for i, entry in enumerate(self.entries):
            try:
                out.append(float(entry.get().replace(",", ".").strip()))
            except (TypeError, ValueError):
                return tuple(default)
        return tuple(out)

    def set(self, value):
        for entry, component in zip(self.entries, value):
            entry.delete(0, "end")
            entry.insert(0, f"{float(component):g}")

    def is_valid(self) -> bool:
        for entry in self.entries:
            try:
                float(entry.get().replace(",", ".").strip())
            except (TypeError, ValueError):
                return False
        return True


class ToggleField(ctk.CTkFrame):
    """Interrupteur oui/non avec son explication."""

    def __init__(self, master, label: str, help_text: str = "", value: bool = False,
                 on_change=None, width: int = 520, **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        theme = current()
        self._on_change = on_change

        self.var = tk.BooleanVar(value=value)
        self.switch = ctk.CTkSwitch(
            self,
            text=label,
            variable=self.var,
            font=FONT.BODY_BOLD,
            text_color=theme.TEXT,
            progress_color=theme.accent,
            command=self._changed,
        )
        self.switch.pack(fill="x", anchor="w")

        self.help = None
        if help_text:
            self.help = ctk.CTkLabel(
                self,
                text=help_text,
                font=FONT.SMALL,
                text_color=theme.TEXT_FAINT,
                anchor="w",
                justify="left",
                wraplength=width,
            )
            self.help.pack(fill="x", padx=(48, 0), pady=(2, 0))

    def _changed(self):
        if self._on_change is not None:
            self._on_change(self.get())

    def get(self) -> bool:
        return bool(self.var.get())

    def set(self, value: bool):
        self.var.set(bool(value))

    def set_label(self, label: str, help_text: str | None = None):
        self.switch.configure(text=label)
        if help_text is not None and self.help is not None:
            self.help.configure(text=help_text)
