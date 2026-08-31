"""Fenêtre principale : assistant en quatre étapes.

L'ancienne fenêtre proposait deux entrées de menu latéral sans ordre ni
verrouillage : rien n'indiquait qu'il fallait charger la maquette avant de
lancer un agent, et lancer l'agent sans maquette produisait un cube de 1 mm de
côté sans le moindre message.

Ici le déroulé est explicite. Une étape est accessible ou non selon l'état
réel du projet — maquette chargée, règles cohérentes, cheminement lancé — et
non selon les écrans déjà visités ; l'application dit toujours ce qui manque
pour atteindre celle qu'on lui demande.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from core import paths
from ui.i18n import LANGUAGES, Translator
from ui.pages.project_page import ProjectPage
from ui.pages.report_page import ReportPage
from ui.pages.routing_page import RoutingPage
from ui.pages.rules_page import RulesPage
from ui.theme import FONT, SPACE, current, set_palette
from ui.widgets import Stepper

__all__ = ["AppWindow"]

STEP_PROJECT, STEP_RULES, STEP_ROUTING, STEP_REPORT = range(4)


class AppWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.settings = paths.load_settings()
        self.t = Translator(self.settings.get("language", "FR"))
        set_palette(self.settings.get("palette", "default"))

        ctk.set_appearance_mode(self.settings.get("appearance", "light"))
        ctk.set_default_color_theme("blue")

        self.title(self.t("app.title"))
        self.geometry("1560x960")
        self.minsize(1180, 760)
        self.configure(fg_color=current().BG)

        self.controller = None
        self.current_step = STEP_PROJECT

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._prepare_cache()
        self._build_menu()
        self._build_header()
        self._build_pages()
        self._build_statusbar()

        self.show_step(STEP_PROJECT)

    # -- construction -----------------------------------------------------

    def _prepare_cache(self):
        """Propose de reprendre une session précédente ou de repartir à zéro."""
        if paths.cache_has_content():
            size = paths.human_size(paths.cache_size_bytes())
            keep = messagebox.askyesno(
                self.t("cache.title"),
                f"{self.t('cache.question')}\n\n{self.t('cache.size')} : {size}",
            )
            if not keep:
                paths.clear_cache()
        paths.ensure_cache_folders()

    def _build_header(self):
        theme = current()
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=SPACE.LG, pady=(SPACE.MD, SPACE.SM))
        header.grid_columnconfigure(1, weight=1)

        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="w", padx=(0, SPACE.XL))

        # La marque tient sur une seule ligne : la baseline qui la doublait
        # n'apprenait rien à qui utilise l'outil, et volait de la hauteur à la
        # zone de travail sur chaque écran.
        ctk.CTkLabel(
            brand, text="🚁 HarnessOpt", font=FONT.H1, text_color=theme.accent, anchor="w"
        ).pack(anchor="w", pady=(SPACE.XS, SPACE.XS))

        self.stepper = Stepper(header, 4, on_select=self.show_step)
        self.stepper.grid(row=0, column=1, sticky="ew")
        self._refresh_stepper_texts()

        # Filet de séparation : sans lui, l'en-tête et la page se confondent.
        ctk.CTkFrame(header, height=1, fg_color=theme.BORDER).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(SPACE.MD, 0)
        )

    def _build_pages(self):
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.grid(row=1, column=0, sticky="nsew", padx=SPACE.LG, pady=(0, SPACE.SM))
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.pages = {
            STEP_PROJECT: ProjectPage(self.container, self),
            STEP_RULES: RulesPage(self.container, self),
            STEP_ROUTING: RoutingPage(self.container, self),
            STEP_REPORT: ReportPage(self.container, self),
        }

    def _build_statusbar(self):
        theme = current()
        bar = ctk.CTkFrame(self, height=30, fg_color=current().SURFACE, corner_radius=0)
        bar.grid(row=2, column=0, sticky="ew")

        # Une pastille colorée double la couleur du texte. Un message d'erreur
        # se distinguait jusqu'ici d'un message ordinaire par la seule teinte
        # des caractères, ce qui est illisible pour une part non négligeable
        # des utilisateurs, et invisible du coin de l'œil pour tous.
        self.status_dot = ctk.CTkFrame(
            bar, width=10, height=10, corner_radius=5, fg_color=theme.TEXT_FAINT
        )
        self.status_dot.pack(side="left", padx=(SPACE.LG, SPACE.SM))
        self.status_dot.pack_propagate(False)

        self.lbl_status = ctk.CTkLabel(
            bar, text=self.t("app.ready"), font=FONT.SMALL,
            text_color=theme.TEXT_SOFT, anchor="w",
        )
        self.lbl_status.pack(side="left", pady=SPACE.XS)

        # Rappel de l'étape courante : sur un écran large, le fil d'étapes est
        # loin du regard une fois qu'on travaille en bas de page.
        self.lbl_step = ctk.CTkLabel(
            bar, text="", font=FONT.TINY, text_color=theme.TEXT_FAINT, anchor="e"
        )
        self.lbl_step.pack(side="right", padx=(0, SPACE.MD))

        self.lbl_cache = ctk.CTkLabel(
            bar, text="", font=FONT.TINY, text_color=theme.TEXT_FAINT, anchor="e"
        )
        self.lbl_cache.pack(side="right", padx=SPACE.LG)
        self.refresh_cache_label()

    def _build_menu(self):
        if hasattr(self, "menubar"):
            self.menubar.destroy()

        t = self.t
        self.menubar = tk.Menu(self)

        m_file = tk.Menu(self.menubar, tearoff=0)
        m_file.add_command(label=t("menu.cache.open"), command=self._open_cache_folder)
        m_file.add_separator()
        m_file.add_command(label=t("app.quit"), command=self.on_closing)
        self.menubar.add_cascade(label=t("menu.file"), menu=m_file)

        m_edit = tk.Menu(self.menubar, tearoff=0)
        m_edit.add_command(label=t("menu.cache.clear"), command=self._clear_cache)
        self.menubar.add_cascade(label=t("menu.edit"), menu=m_edit)

        m_view = tk.Menu(self.menubar, tearoff=0)
        m_lang = tk.Menu(m_view, tearoff=0)
        for code, label in LANGUAGES.items():
            m_lang.add_command(label=label, command=lambda c=code: self.set_language(c))
        m_view.add_cascade(label=t("menu.language"), menu=m_lang)
        m_view.add_separator()
        m_view.add_command(label=t("menu.theme.light"), command=lambda: self.set_appearance("light"))
        m_view.add_command(label=t("menu.theme.dark"), command=lambda: self.set_appearance("dark"))
        m_view.add_separator()
        m_view.add_command(label=t("menu.colors.default"), command=lambda: self.set_palette_name("default"))
        m_view.add_command(
            label=t("menu.colors.colorblind"), command=lambda: self.set_palette_name("colorblind")
        )
        self.menubar.add_cascade(label=t("menu.view"), menu=m_view)

        self.config(menu=self.menubar)

    # -- navigation --------------------------------------------------------

    def refresh_steps(self) -> int:
        """Recalcule les étapes accessibles et renvoie la plus avancée.

        L'accès se déduit de ce qui est fait (maquette chargée, règles
        cohérentes, cheminement lancé) plutôt que de s'accumuler au fil des
        visites : une étape ne peut donc plus rester fermée parce que personne
        n'est passé l'ouvrir. Si l'écran affiché n'est plus atteignable — la
        maquette a été vidée, les règles sont devenues incohérentes — on
        revient à la dernière étape valide plutôt que de laisser l'utilisateur
        sur une page qui ne correspond plus à rien.
        """
        if self.controller is None:
            return self.stepper.unlocked

        reachable = self.controller.max_reachable_step()
        self.stepper.reset_unlock(reachable)
        if self.current_step > reachable:
            self._display_step(reachable)
        return reachable

    def show_step(self, index: int):
        """Affiche une étape, si elle est accessible."""
        # Le verrouillage est rafraîchi AVANT de refuser : sans cela, l'étape
        # « Cheminement » ne s'ouvrait jamais, puisque son déverrouillage
        # n'avait lieu qu'à la fin de son propre affichage.
        reachable = self.refresh_steps()

        if index > reachable:
            reason = ""
            if self.controller is not None:
                reason = self.controller.locked_reason(index)
            self.set_status(reason or self.t("step.locked"), "warn")
            return

        self._display_step(index)

    def _display_step(self, index: int):
        """Bascule réellement l'affichage, sans revérifier l'accès."""
        self.current_step = index
        for page in self.pages.values():
            page.grid_forget()
        self.pages[index].grid(row=0, column=0, sticky="nsew")
        self.stepper.set_current(index)
        self._refresh_step_label()

        if self.controller is not None:
            self.controller.on_step_shown(index)

    def set_controller(self, controller):
        self.controller = controller

    # -- préférences --------------------------------------------------------

    def remember(self, **values):
        """Mémorise des réglages entre deux sessions."""
        self.settings.update({k: v for k, v in values.items() if v is not None})
        paths.update_settings(**values)

    def set_language(self, code: str):
        self.t.set_language(code)
        self.remember(language=self.t.lang)
        self.title(self.t("app.title"))
        self._build_menu()
        self._refresh_stepper_texts()
        self._refresh_step_label()
        self.refresh_cache_label()
        for page in self.pages.values():
            if hasattr(page, "update_language"):
                page.update_language()

    def set_appearance(self, mode: str):
        ctk.set_appearance_mode(mode)
        self.remember(appearance=mode)

    def set_palette_name(self, name: str):
        set_palette(name)
        self.remember(palette=name)
        messagebox.showinfo(
            self.t("app.title"),
            "La nouvelle palette sera appliquée au prochain démarrage."
            if not self.t.is_english
            else "The new palette will be applied on next start.",
        )

    def _refresh_stepper_texts(self):
        t = self.t
        self.stepper.set_texts(
            [
                (t("step.project"), t("step.project.sub")),
                (t("step.rules"), t("step.rules.sub")),
                (t("step.routing"), t("step.routing.sub")),
                (t("step.report"), t("step.report.sub")),
            ]
        )

    # -- barre d'état ---------------------------------------------------------

    def ask_use_fixations(self, scan_result, default: bool = True) -> bool:
        """Demande si le câble doit emprunter les fixations reconnues.

        La question arrive une fois le scan terminé et les passages déjà
        dessinés en 3D : l'utilisateur répond en voyant ce dont on parle,
        plutôt que sur une liste de coordonnées.
        """
        lang = self.t.lang
        english = self.t.is_english

        detail = scan_result.message(lang)
        lines = [passage.format(lang, number=n)
                 for n, passage in enumerate(scan_result.passages[:6], start=1)]
        if scan_result.n_passages > 6:
            lines.append("…")

        if english:
            question = (
                f"{detail}\n\n"
                + "\n".join(lines)
                + "\n\nShould the harness go through them?\n\n"
                "Yes — the route crosses every notch and the agents keep it there.\n"
                "No — the agents pick their own way."
            )
            title = "Existing fixations detected"
        else:
            question = (
                f"{detail}\n\n"
                + "\n".join(lines)
                + "\n\nFaut-il faire passer le faisceau par ces fixations ?\n\n"
                "Oui — le tracé traverse chaque encoche et les agents l'y maintiennent.\n"
                "Non — les agents choisissent librement leur passage."
            )
            title = "Fixations existantes détectées"

        try:
            return bool(messagebox.askyesno(title, question, default="yes" if default else "no"))
        except Exception:
            return default

    def set_status(self, message: str, tone: str = "neutral"):
        theme = current()
        color = {
            "ok": theme.ok, "warn": theme.warn, "danger": theme.danger, "info": theme.accent
        }.get(tone, theme.TEXT_SOFT)
        self.lbl_status.configure(text=message, text_color=color)
        self.status_dot.configure(fg_color=color)

    def _refresh_step_label(self):
        """Rappelle l'étape courante dans la barre d'état."""
        keys = ("step.project", "step.rules", "step.routing", "step.report")
        index = max(0, min(len(keys) - 1, self.current_step))
        self.lbl_step.configure(
            text=f"{self.t('stepper.step')} {index + 1}/4 — {self.t(keys[index])}"
        )

    def refresh_cache_label(self):
        self.lbl_cache.configure(
            text=f"{self.t('cache.size')} : {paths.human_size(paths.cache_size_bytes())}"
        )

    # -- cache -----------------------------------------------------------------

    def _open_cache_folder(self):
        folder = str(paths.BASE_CACHE)
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)  # noqa: S606 - ouverture de l'explorateur
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            messagebox.showwarning(self.t("app.title"), f"{folder}\n\n{exc}")

    def _clear_cache(self):
        if messagebox.askyesno(self.t("app.title"), self.t("menu.cache.clear") + " ?"):
            paths.clear_cache()
            self.refresh_cache_label()
            self.set_status(self.t("cache.cleared"), "ok")

    # -- fermeture ---------------------------------------------------------------

    def on_closing(self):
        if not messagebox.askyesno(self.t("app.quit"), self.t("app.quit.confirm")):
            return
        if messagebox.askyesno(self.t("app.quit"), self.t("app.quit.cache")):
            paths.clear_cache()
        if self.controller is not None:
            self.controller.shutdown()
        self.quit()
        self.destroy()
