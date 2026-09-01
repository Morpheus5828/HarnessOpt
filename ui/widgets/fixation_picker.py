"""Choix des encoches empruntées, peigne par peigne.

La question posée jusqu'ici était binaire : emprunter les fixations, ou pas.
Elle ne suffit plus. Un peigne porte plusieurs encoches côte à côte, une par
faisceau ; l'application en propose une, mais c'est l'intégrateur qui sait
laquelle est libre, laquelle est réservée, laquelle est accessible à l'outil.
Il doit donc pouvoir trancher lui-même, peigne par peigne.

Une liste déroulante par peigne plutôt qu'une case par encoche : un peigne
peut porter treize encoches, et treize cases à cocher dont une seule peut être
retenue est un formulaire qui ment sur ce qu'il autorise. La liste dit la
règle par sa forme même — un choix, et un seul, « ne pas emprunter » compris.

Le choix se répercute en 3D à chaque changement (``on_change``) : l'utilisateur
voit l'encoche qu'il vient de désigner s'allumer avant de valider, plutôt que
d'arbitrer sur des coordonnées.
"""

from __future__ import annotations

import customtkinter as ctk

from ui.theme import FONT, SPACE, current

__all__ = ["FixationPicker", "NO_CROSSING_FR", "NO_CROSSING_EN",
           "comb_name", "index_for", "label_for", "passage_labels"]

NO_CROSSING_FR = "Ne pas emprunter ce peigne"
NO_CROSSING_EN = "Do not use this comb"

#: Au-delà, la fenêtre défile plutôt que de déborder de l'écran.
SCROLL_THRESHOLD = 4


def comb_name(passages) -> str:
    """Nom du peigne qui porte ces encoches.

    Fonction de module et non méthode : ``_name`` est déjà un attribut de tout
    widget Tk, et le redéfinir sur la fenêtre le remplacerait par une méthode.
    """
    for passage in passages:
        name = str(getattr(passage, "comb", "") or "")
        if name:
            return name
    return "?"


def passage_labels(passages, lang: str = "FR") -> list:
    """Étiquettes proposées pour un peigne : le refus, puis chaque encoche.

    Le refus vient en tête pour être atteignable sans dérouler treize lignes,
    et parce qu'il est le seul choix qui n'a pas besoin d'être lu pour être
    compris.
    """
    english = str(lang).upper().startswith("EN")
    labels = [NO_CROSSING_EN if english else NO_CROSSING_FR]
    labels += [passage.format(lang, number=passage.index + 1) for passage in passages]
    return labels


def label_for(passages, index, lang: str = "FR") -> str:
    """Étiquette correspondant à l'encoche ``index``, ou au refus si ``None``."""
    labels = passage_labels(passages, lang)
    for rank, passage in enumerate(passages, start=1):
        if passage.index == index:
            return labels[rank]
    return labels[0]


def index_for(passages, label, lang: str = "FR"):
    """Index d'encoche correspondant à une étiquette, ``None`` pour le refus."""
    for rank, candidate in enumerate(passage_labels(passages, lang)):
        if candidate == label:
            return None if rank == 0 else passages[rank - 1].index
    return None


class FixationPicker(ctk.CTkToplevel):
    """Fenêtre modale : un peigne par ligne, une encoche à choisir.

    ``result`` vaut ``None`` tant que rien n'est validé, puis le dictionnaire
    ``nom de peigne -> index d'encoche`` (``None`` pour un peigne écarté). Tout
    refuser et fermer la fenêtre reviennent au même : aucun passage imposé.
    """

    def __init__(self, master, combs, selection=None, lang="FR", on_change=None,
                 scan_message=""):
        super().__init__(master)
        theme = current()
        self._english = str(lang).upper().startswith("EN")
        self._lang = lang
        self._combs = {comb_name(comb): list(comb) for comb in combs if list(comb)}
        self._selection = dict(selection or {})
        self._on_change = on_change
        self._menus = {}
        self.result = None

        self.title("Existing fixations detected" if self._english
                   else "Fixations existantes détectées")
        self.configure(fg_color=theme.BG)
        self.resizable(False, False)
        self._build(theme, scan_message)

        # Modale : la réponse commande la suite du lancement, la laisser en
        # arrière-plan reviendrait à faire attendre le calcul sans le dire.
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self._refuse)

    # ------------------------------------------------------------------
    def _build(self, theme, scan_message):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=SPACE.LG, pady=SPACE.LG)

        ctk.CTkLabel(
            frame,
            text=("Which notch should the harness go through?"
                  if self._english else
                  "Par quelle encoche le faisceau doit-il passer ?"),
            font=FONT.H1, text_color=theme.TEXT, anchor="w",
        ).pack(fill="x")

        detail = scan_message or ""
        rule = ("One notch per comb: the others belong to the neighbouring harnesses."
                if self._english else
                "Une encoche par peigne : les autres sont celles des faisceaux voisins.")
        ctk.CTkLabel(
            frame, text=f"{detail}\n{rule}".strip(), font=FONT.SMALL,
            text_color=theme.TEXT_SOFT, anchor="w", justify="left",
        ).pack(fill="x", pady=(SPACE.XS, SPACE.MD))

        holder = (ctk.CTkScrollableFrame(frame, fg_color=theme.SURFACE, height=320)
                  if len(self._combs) > SCROLL_THRESHOLD
                  else ctk.CTkFrame(frame, fg_color=theme.SURFACE))
        holder.pack(fill="both", expand=True)

        for name, passages in self._combs.items():
            self._add_row(holder, theme, name, passages)

        buttons = ctk.CTkFrame(frame, fg_color="transparent")
        buttons.pack(fill="x", pady=(SPACE.MD, 0))
        ctk.CTkButton(
            buttons, text="Ignore all" if self._english else "Tout ignorer",
            command=self._refuse, fg_color=theme.SURFACE_ALT,
            text_color=theme.TEXT, hover_color=theme.BORDER, width=140,
        ).pack(side="left")
        ctk.CTkButton(
            buttons, text="Use these notches" if self._english
            else "Emprunter ces encoches",
            command=self._accept, width=200,
        ).pack(side="right")

    def _add_row(self, master, theme, name, passages):
        row = ctk.CTkFrame(master, fg_color="transparent")
        row.pack(fill="x", padx=SPACE.MD, pady=SPACE.SM)

        ctk.CTkLabel(
            row, text=name, font=FONT.BODY_BOLD, text_color=theme.TEXT, anchor="w",
        ).pack(fill="x")

        count = (f"{len(passages)} notch(es)" if self._english
                 else f"{len(passages)} encoche(s)")
        ctk.CTkLabel(
            row, text=count, font=FONT.TINY, text_color=theme.TEXT_FAINT, anchor="w",
        ).pack(fill="x")

        menu = ctk.CTkOptionMenu(
            row, values=passage_labels(passages, self._lang),
            command=lambda label, key=name: self._changed(key, label),
            font=FONT.CODE, dynamic_resizing=False, width=420,
        )
        menu.set(label_for(passages, self._selection.get(name), self._lang))
        menu.pack(fill="x", pady=(SPACE.XS, 0))
        self._menus[name] = menu

    # ------------------------------------------------------------------
    def _changed(self, name, label):
        self._selection[name] = index_for(self._combs[name], label, self._lang)
        if self._on_change is not None:
            try:
                self._on_change(dict(self._selection))
            except Exception:
                # Un aperçu 3D en échec ne doit pas empêcher de répondre.
                pass

    def _accept(self):
        self.result = dict(self._selection)
        self.destroy()

    def _refuse(self):
        self.result = {name: None for name in self._combs}
        self.destroy()

    # ------------------------------------------------------------------
    def choose(self, name, index):
        """Sélectionne une encoche par programme — l'équivalent d'un clic."""
        if name not in self._combs:
            return
        self._menus[name].set(label_for(self._combs[name], index, self._lang))
        self._changed(name, self._menus[name].get())

    @property
    def selection(self) -> dict:
        return dict(self._selection)

    def ask(self) -> dict:
        """Ouvre la fenêtre et rend le choix, ou ``None`` si rien n'est retenu."""
        self.grab_set()
        self.wait_window()
        if self.result is None:
            return None
        return self.result if any(v is not None for v in self.result.values()) else None
