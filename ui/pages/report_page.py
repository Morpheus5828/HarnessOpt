"""Étape 4 : contrôler le résultat et l'exporter.

Une page volontairement sobre : le verdict en grand, le détail règle par règle,
et les exports. C'est ce qui sert de preuve d'analyse au dossier
d'industrialisation, donc l'export du rapport reprend exactement ce qui est
affiché à l'écran.
"""

from __future__ import annotations

import customtkinter as ctk

from ui.theme import FONT, SPACE, current
from ui.widgets import Card, ChoiceField, ComplianceTable, KpiRow, StatusPill, VerdictBanner

__all__ = ["ReportPage"]


class ReportPage(ctk.CTkScrollableFrame):
    def __init__(self, master, app, **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self.app = app
        self.t = app.t
        self._reports: dict = {}
        self._build()

    def _build(self):
        theme = current()
        t = self.t

        self.lbl_title = ctk.CTkLabel(
            self, text=t("report.title"), font=FONT.TITLE, text_color=theme.TEXT, anchor="w"
        )
        self.lbl_title.pack(fill="x", pady=(0, SPACE.XS))

        self.lbl_intro = ctk.CTkLabel(
            self, text=t("report.intro"), font=FONT.BODY, text_color=theme.TEXT_SOFT,
            anchor="w", justify="left", wraplength=820,
        )
        self.lbl_intro.pack(fill="x", pady=(0, SPACE.LG))

        self.banner = VerdictBanner(self)
        self.banner.pack(fill="x", pady=(0, SPACE.MD))
        self.banner.update_verdict(None, self.app.t)

        self.card_choice = Card(self, title=t("report.choose_agent"), icon="🤖")
        self.card_choice.pack(fill="x", pady=(0, SPACE.MD))

        self.f_agent = ChoiceField(
            self.card_choice.body, label="", options=[], on_change=self._on_agent_changed
        )
        self.f_agent.lbl.pack_forget()
        self.f_agent.pack(fill="x")

        self.kpis = KpiRow(
            self,
            [
                ("length", t("kpi.length"), "mm"),
                ("bend", t("kpi.bend"), "mm"),
                ("straight", t("kpi.straight"), "%"),
                ("bends", t("kpi.bends"), ""),
                ("clamps", t("kpi.clamps"), ""),
                ("distance", t("kpi.distance"), "mm"),
            ],
        )
        self.kpis.pack(fill="x", pady=(0, SPACE.MD))

        self.card_detail = Card(self, title=t("report.detail"), icon="📋")
        self.card_detail.pack(fill="x", pady=(0, SPACE.MD))

        self.compliance = ComplianceTable(self.card_detail.body, lang=self.app.t.lang)
        self.compliance.set_placeholder(t("report.verdict.none"))
        self.compliance.pack(fill="x")

        self.card_export = Card(self, title=t("report.exports"), icon="💾")
        self.card_export.pack(fill="x", pady=(0, SPACE.LG))

        buttons = ctk.CTkFrame(self.card_export.body, fg_color="transparent")
        buttons.pack(fill="x")

        self.btn_stl = ctk.CTkButton(
            buttons, text=t("report.export.stl"), height=42, font=FONT.BODY_BOLD,
            fg_color=theme.accent, command=lambda: self._export("stl"),
        )
        self.btn_stl.pack(side="left")

        self.btn_csv = ctk.CTkButton(
            buttons, text=t("report.export.csv"), height=42, font=FONT.BODY_BOLD,
            fg_color="transparent", border_width=1, border_color=theme.BORDER,
            text_color=theme.TEXT, hover_color=theme.SURFACE_ALT,
            command=lambda: self._export("csv"),
        )
        self.btn_csv.pack(side="left", padx=(SPACE.SM, 0))

        self.btn_report = ctk.CTkButton(
            buttons, text=t("report.export.report"), height=42, font=FONT.BODY_BOLD,
            fg_color="transparent", border_width=1, border_color=theme.BORDER,
            text_color=theme.TEXT, hover_color=theme.SURFACE_ALT,
            command=lambda: self._export("report"),
        )
        self.btn_report.pack(side="left", padx=(SPACE.SM, 0))

        # Fin naturelle du flux : renvoyer le tracé retenu dans le document
        # CATIA ouvert. Sans effet hors Windows, où le bouton signale
        # simplement que CATIA n'est pas joignable.
        self.btn_catia = ctk.CTkButton(
            buttons, text=t("report.export.catia"), height=42, font=FONT.BODY_BOLD,
            fg_color="transparent", border_width=1, border_color=theme.BORDER,
            text_color=theme.TEXT, hover_color=theme.SURFACE_ALT,
            command=lambda: self._export("catia"),
        )
        self.btn_catia.pack(side="left", padx=(SPACE.SM, 0))

        self.pill = StatusPill(buttons, text="", tone="neutral")
        self.pill.pack(side="right")

        self._set_exports_enabled(False)

    # -- réactions -------------------------------------------------------

    def _set_exports_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for button in (self.btn_stl, self.btn_csv, self.btn_report, self.btn_catia):
            button.configure(state=state)

    def _on_agent_changed(self, name: str):
        report = self._reports.get(name)
        self._show(report)

    def _export(self, kind: str):
        name = self.f_agent.get()
        if not name:
            return
        if kind == "catia":
            self.pill.update_status(self.t("report.catia.sending"), "info")
            self.app.controller.export(name, kind)
            return

        result = self.app.controller.export(name, kind)
        if result:
            self.pill.update_status(f"{self.t('report.exported')} · {result}", "ok")
        else:
            self.pill.update_status("", "neutral")

    # -- API appelée par le contrôleur -------------------------------------

    def update_reports(self, reports: dict, ranking: list[str], labels: dict):
        """``reports`` : ``{nom d'agent: RouteReport}``, classés par ``ranking``."""
        self._reports = dict(reports)
        if not reports:
            self._set_exports_enabled(False)
            self._show(None)
            return

        ordered = [n for n in ranking if n in reports] or list(reports)
        current_choice = self.f_agent.get()
        self.f_agent.set_options(
            [(n, labels.get(n, n)) for n in ordered],
            current_choice if current_choice in reports else ordered[0],
        )
        self._set_exports_enabled(True)
        self._show(self._reports.get(self.f_agent.get()))

    def _show(self, report):
        self.banner.update_verdict(report, self.app.t)
        self.compliance.update_report(report)

        if report is None:
            # Sans rapport, les indicateurs doivent repartir à vide : les
            # laisser afficher les valeurs du calcul précédent ferait croire à
            # un résultat qui n'existe plus.
            self.kpis.update_values({key: self.t("common.none") for key in self.kpis.tiles})
            return

        k = report.kpis
        radius = k.get("min_bend_radius_mm", float("inf"))
        self.kpis.update_values(
            {
                "length": f"{k.get('length_mm', 0):.0f}",
                "bend": "∞" if radius == float("inf") else f"{radius:.0f}",
                "straight": f"{k.get('straight_ratio', 0) * 100:.0f}",
                "bends": str(k.get("n_bends", 0)),
                "clamps": str(k.get("n_clamps", 0)),
                "distance": f"{k.get('mean_distance_mm', 0):.0f}",
            }
        )

    def report_catia_result(self, ok: bool, message: str):
        """Rend compte de l'envoi vers CATIA, qui se fait en arrière-plan."""
        if ok:
            self.pill.update_status(self.t("report.catia.done"), "ok")
        else:
            self.pill.update_status(message.split("\n")[0][:80], "danger")

    def update_language(self):
        t = self.t
        self.lbl_title.configure(text=t("report.title"))
        self.lbl_intro.configure(text=t("report.intro"))
        self.card_choice.set_title(t("report.choose_agent"), "🤖")
        self.card_detail.set_title(t("report.detail"), "📋")
        self.card_export.set_title(t("report.exports"), "💾")
        self.btn_stl.configure(text=t("report.export.stl"))
        self.btn_csv.configure(text=t("report.export.csv"))
        self.btn_report.configure(text=t("report.export.report"))
        self.btn_catia.configure(text=t("report.export.catia"))
        self.kpis.set_labels(
            {
                "length": t("kpi.length"), "bend": t("kpi.bend"),
                "straight": t("kpi.straight"), "bends": t("kpi.bends"),
                "clamps": t("kpi.clamps"), "distance": t("kpi.distance"),
            }
        )
        self.compliance.set_language(self.app.t.lang)
        self.compliance.set_placeholder(t("report.verdict.none"))
        self.banner.update_verdict(self._reports.get(self.f_agent.get()), self.app.t)
