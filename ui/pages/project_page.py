"""Étape 1 : charger la maquette numérique.

L'utilisateur désigne d'où viennent les pièces, lance le chargement, et voit
le résultat : combien de pièces, quelles familles, quel encombrement. Rien de
plus : à ce stade il n'a aucune raison de voir un réglage d'algorithme.
"""

from __future__ import annotations

import os
import time

import customtkinter as ctk

from ui.theme import FONT, SPACE, current
from ui.widgets import Card, ChoiceField, KpiRow, PathField, StatusPill

__all__ = ["ProjectPage"]


class ProjectPage(ctk.CTkScrollableFrame):
    def __init__(self, master, app, **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self.app = app
        self.t = app.t
        self._start_time: float | None = None
        self._build()

    # -- construction ---------------------------------------------------

    def _build(self):
        theme = current()
        t = self.t

        self.lbl_title = ctk.CTkLabel(
            self, text=t("project.title"), font=FONT.TITLE, text_color=theme.TEXT, anchor="w"
        )
        self.lbl_title.pack(fill="x", pady=(0, SPACE.XS))

        self.lbl_intro = ctk.CTkLabel(
            self, text=t("project.intro"), font=FONT.BODY, text_color=theme.TEXT_SOFT,
            anchor="w", justify="left", wraplength=820,
        )
        self.lbl_intro.pack(fill="x", pady=(0, SPACE.LG))

        # --- source des pièces ---------------------------------------
        self.card_source = Card(self, title=t("project.source"), icon="📂")
        self.card_source.pack(fill="x", pady=(0, SPACE.MD))

        self.field_source = ChoiceField(
            self.card_source.body,
            label="",
            options=[
                ("folder", t("project.source.folder")),
                ("catia", t("project.source.catia")),
            ],
            value="folder",
            on_change=self._on_source_changed,
        )
        self.field_source.lbl.pack_forget()
        self.field_source.pack(fill="x", pady=(0, SPACE.MD))

        self.field_folder = PathField(
            self.card_source.body,
            label=t("project.folder"),
            help_text=t("project.folder.help"),
            value=self.app.settings.get("stl_folder", ""),
            browse_text=t("project.browse"),
            on_change=lambda v: self.app.remember(stl_folder=v),
        )
        self.field_folder.pack(fill="x", pady=(0, SPACE.MD))

        self.field_exclude = PathField(
            self.card_source.body,
            label=t("project.exclude"),
            help_text=t("project.exclude.help"),
            value=self.app.settings.get("exclude_filter", ""),
            browse_text=t("project.browse"),
            on_change=lambda v: self.app.remember(exclude_filter=v),
        )
        self.field_exclude.btn.pack_forget()  # motif texte : pas de sélecteur de fichier

        self.field_clamps = PathField(
            self.card_source.body,
            label=t("project.clamps"),
            help_text=t("project.clamps.help"),
            value=self.app.settings.get("clamps_folder", ""),
            browse_text=t("project.browse"),
            on_change=lambda v: self.app.remember(clamps_folder=v),
        )
        self.field_clamps.pack(fill="x", pady=(0, SPACE.MD))

        actions = ctk.CTkFrame(self.card_source.body, fg_color="transparent")
        actions.pack(fill="x", pady=(SPACE.SM, 0))

        self.btn_run = ctk.CTkButton(
            actions, text=t("project.run"), height=42, font=FONT.H3,
            fg_color=theme.accent, command=self._on_run,
        )
        self.btn_run.pack(side="left")

        self.btn_cancel = ctk.CTkButton(
            actions, text=t("project.cancel"), height=42, width=130, font=FONT.BODY_BOLD,
            fg_color="transparent", border_width=1, border_color=theme.danger,
            text_color=theme.danger, hover_color=theme.SURFACE_ALT,
            state="disabled", command=self._on_cancel,
        )
        self.btn_cancel.pack(side="left", padx=(SPACE.SM, 0))

        self.pill = StatusPill(actions, text=self.t("app.ready"), tone="neutral")
        self.pill.pack(side="right")

        # --- progression ----------------------------------------------
        self.card_progress = Card(self, title=t("project.running"), icon="⏳")

        self.progress = ctk.CTkProgressBar(
            self.card_progress.body, height=12, progress_color=theme.accent
        )
        self.progress.set(0)
        self.progress.pack(fill="x", pady=(0, SPACE.SM))

        row = ctk.CTkFrame(self.card_progress.body, fg_color="transparent")
        row.pack(fill="x")

        self.lbl_step = ctk.CTkLabel(
            row, text="", font=FONT.BODY, text_color=theme.TEXT, anchor="w", justify="left"
        )
        self.lbl_step.pack(side="left", fill="x", expand=True)

        self.lbl_timer = ctk.CTkLabel(
            row, text="", font=FONT.VALUE_SM, text_color=theme.TEXT_SOFT, anchor="e"
        )
        self.lbl_timer.pack(side="right")

        # --- résultat --------------------------------------------------
        self.card_result = Card(self, title=t("project.done"), icon="✅")

        self.kpis = KpiRow(
            self.card_result.body,
            [
                ("parts", t("project.stat.parts"), ""),
                ("faces", t("project.stat.faces"), ""),
                ("families", t("project.stat.families"), ""),
                ("size", t("project.stat.size"), "mm"),
            ],
        )
        self.kpis.pack(fill="x", pady=(0, SPACE.MD))

        self.lbl_bbox_title = ctk.CTkLabel(
            self.card_result.body, text=t("project.bbox"), font=FONT.SMALL_BOLD,
            text_color=theme.TEXT_SOFT, anchor="w",
        )
        self.lbl_bbox_title.pack(fill="x")

        self.lbl_bbox = ctk.CTkLabel(
            self.card_result.body, text="", font=FONT.CODE,
            text_color=theme.TEXT, anchor="w",
        )
        self.lbl_bbox.pack(fill="x", pady=(0, SPACE.MD))

        self.lbl_families_title = ctk.CTkLabel(
            self.card_result.body, text=t("project.families.title"), font=FONT.SMALL_BOLD,
            text_color=theme.TEXT_SOFT, anchor="w",
        )
        self.lbl_families_title.pack(fill="x")

        self.families_box = ctk.CTkFrame(self.card_result.body, fg_color="transparent")
        self.families_box.pack(fill="x", pady=(SPACE.XS, 0))
        self._family_rows: list[ctk.CTkFrame] = []

        self._on_source_changed("folder")

    # -- réactions ------------------------------------------------------

    def _on_source_changed(self, value: str):
        if value == "catia":
            self.field_folder.pack_forget()
            self.field_exclude.pack(fill="x", pady=(0, SPACE.MD),
                                    before=self.field_clamps)
        else:
            self.field_exclude.pack_forget()
            self.field_folder.pack(fill="x", pady=(0, SPACE.MD),
                                   before=self.field_clamps)

    def _on_run(self):
        source = self.field_source.get()
        folder = self.field_folder.get()

        if source == "folder":
            if not folder or not os.path.isdir(folder):
                self.pill.update_status(self.t("project.error.folder"), "danger")
                return
            if not any(f.lower().endswith(".stl") for f in os.listdir(folder)):
                self.pill.update_status(self.t("project.error.empty"), "danger")
                return

        self.app.remember(
            stl_folder=folder,
            clamps_folder=self.field_clamps.get(),
            exclude_filter=self.field_exclude.get(),
        )

        self._start_time = time.time()
        self.card_result.pack_forget()
        self.card_progress.pack(fill="x", pady=(0, SPACE.MD))
        self.progress.set(0)
        self.lbl_step.configure(text="…")
        self.btn_run.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.pill.update_status(self.t("project.running"), "info")

        self.app.controller.start_extraction(
            {
                "use_existing": source == "folder",
                "stl_folder": folder,
                "exclude_filter": self.field_exclude.get(),
                "clamps_folder": self.field_clamps.get(),
            }
        )

    def _on_cancel(self):
        self.app.controller.cancel_extraction()

    # -- API appelée par le contrôleur ----------------------------------

    def update_progress(self, message: str, ratio: float | None = None):
        self.lbl_step.configure(text=message)
        if ratio is None:
            return

        self.progress.set(max(0.0, min(1.0, ratio)))
        if not self._start_time or ratio <= 0.01:
            return

        elapsed = time.time() - self._start_time
        remaining = elapsed / ratio - elapsed

        def clock(seconds: float) -> str:
            minutes, secs = divmod(int(max(0.0, seconds)), 60)
            return f"{minutes:02d}:{secs:02d}"

        self.lbl_timer.configure(
            text=f"{self.t('common.elapsed')} {clock(elapsed)}  ·  "
                 f"{self.t('common.eta')} {clock(remaining)}"
        )

    def show_failure(self, message: str):
        self.card_progress.pack_forget()
        self.btn_run.configure(state="normal", text=self.t("project.run"))
        self.btn_cancel.configure(state="disabled")
        self.pill.update_status(message.split("\n")[0][:70], "danger")
        self.lbl_step.configure(text=message)

    def show_results(self, summary: dict):
        """``summary`` : ``n_parts``, ``n_cells``, ``bounds``, ``families``."""
        self.card_progress.pack_forget()
        self.card_result.pack(fill="x", pady=(0, SPACE.MD))
        self.btn_run.configure(state="normal", text=self.t("project.rerun"))
        self.btn_cancel.configure(state="disabled")
        self.pill.update_status(self.t("project.done"), "ok")

        bounds = summary.get("bounds") or (0, 0, 0, 0, 0, 0)
        extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        families = summary.get("families") or {}

        self.kpis.update_values(
            {
                "parts": f"{summary.get('n_parts', 0):,}".replace(",", " "),
                "faces": f"{summary.get('n_cells', 0):,}".replace(",", " "),
                "families": str(len(families)),
                "size": f"{extent:,.0f}".replace(",", " "),
            }
        )

        self.lbl_bbox.configure(
            text=(
                f"X  {bounds[0]:>10.0f} … {bounds[1]:<10.0f}   "
                f"Y  {bounds[2]:>10.0f} … {bounds[3]:<10.0f}   "
                f"Z  {bounds[4]:>10.0f} … {bounds[5]:<10.0f}"
            )
        )
        self._render_families(families)

    def _render_families(self, families: dict):
        """Barres horizontales : une par famille, proportionnelles au nombre de pièces."""
        theme = current()
        for row in self._family_rows:
            row.destroy()
        self._family_rows.clear()

        if not families:
            row = ctk.CTkFrame(self.families_box, fg_color="transparent")
            row.pack(fill="x")
            ctk.CTkLabel(
                row, text=self.t("rules.families.none"), font=FONT.SMALL,
                text_color=theme.TEXT_FAINT, anchor="w", justify="left", wraplength=700,
            ).pack(fill="x")
            self._family_rows.append(row)
            return

        total = max(1, sum(families.values()))
        for name, count in sorted(families.items(), key=lambda kv: -kv[1]):
            row = ctk.CTkFrame(self.families_box, fg_color="transparent")
            row.pack(fill="x", pady=1)

            ctk.CTkLabel(
                row, text=name, font=FONT.SMALL, text_color=theme.TEXT,
                width=210, anchor="w",
            ).pack(side="left")

            bar = ctk.CTkProgressBar(row, height=8, progress_color=theme.accent)
            bar.set(count / total)
            bar.pack(side="left", fill="x", expand=True, padx=SPACE.SM)

            ctk.CTkLabel(
                row, text=str(count), font=FONT.SMALL, text_color=theme.TEXT_SOFT,
                width=50, anchor="e",
            ).pack(side="right")

            self._family_rows.append(row)

    def update_language(self):
        t = self.t
        self.lbl_title.configure(text=t("project.title"))
        self.lbl_intro.configure(text=t("project.intro"))
        self.card_source.set_title(t("project.source"), "📂")
        self.field_source.set_options(
            [("folder", t("project.source.folder")), ("catia", t("project.source.catia"))],
            self.field_source.get(),
        )
        self.field_folder.set_label(t("project.folder"), t("project.folder.help"))
        self.field_exclude.set_label(t("project.exclude"), t("project.exclude.help"))
        self.field_clamps.set_label(t("project.clamps"), t("project.clamps.help"))
        for field in (self.field_folder, self.field_clamps):
            field.set_browse_text(t("project.browse"))
        self.btn_cancel.configure(text=t("project.cancel"))
        self.card_progress.set_title(t("project.running"), "⏳")
        self.card_result.set_title(t("project.done"), "✅")
        self.lbl_bbox_title.configure(text=t("project.bbox"))
        self.lbl_families_title.configure(text=t("project.families.title"))
        self.kpis.set_labels(
            {
                "parts": t("project.stat.parts"),
                "faces": t("project.stat.faces"),
                "families": t("project.stat.families"),
                "size": t("project.stat.size"),
            }
        )
