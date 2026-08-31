"""Étape 3 : faire cheminer le harnais.

Écran de pilotage. À gauche, ce que l'utilisateur décide : les deux extrémités,
la composition de l'équipe, et le curseur Exploration ↔ Exploitation. À droite,
ce qu'il observe : la vue 3D, la conformité du meilleur tracé, et les agents au
travail.

Deux partis pris :

* **aucun hyperparamètre visible par défaut.** Le curseur et le choix d'équipe
  suffisent ; les réglages fins sont repliés dans « Réglages avancés » pour qui
  veut vraiment y toucher.
* **le curseur est actif pendant le calcul.** C'est ce qui rend la notion
  d'exploration/exploitation concrète : on la déplace et on voit, en quelques
  secondes, les agents changer de comportement.
"""

from __future__ import annotations

import customtkinter as ctk

from core.orchestrator import ROLES, TEAM_PRESETS, ExplorationPolicy, Phase
from core.path_planner import STRATEGIES
from ui.theme import FONT, SPACE, current
from ui.widgets import (
    AdviceBoard,
    AgentBoard,
    Card,
    ChoiceField,
    ComplianceTable,
    CoordinateField,
    KpiRow,
    NumberField,
    PhaseIndicator,
    SliderField,
    StatusPill,
    ToggleField,
)

__all__ = ["RoutingPage"]

#: Au-delà, la liste des passages mangerait tout l'écran.
MAX_LISTED_PASSAGES = 8


class RoutingPage(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self.app = app
        self.t = app.t
        self._advanced_open = False
        self._build()

    # -- construction ---------------------------------------------------

    def _build(self):
        theme = current()
        t = self.t

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, SPACE.MD))

        titles = ctk.CTkFrame(header, fg_color="transparent")
        titles.pack(side="left", fill="x", expand=True)

        self.lbl_title = ctk.CTkLabel(
            titles, text=t("routing.title"), font=FONT.TITLE, text_color=theme.TEXT, anchor="w"
        )
        self.lbl_title.pack(fill="x")

        self.lbl_intro = ctk.CTkLabel(
            titles, text=t("routing.intro"), font=FONT.BODY, text_color=theme.TEXT_SOFT,
            anchor="w", justify="left", wraplength=700,
        )
        self.lbl_intro.pack(fill="x")

        self.pill = StatusPill(header, text=t("app.ready"), tone="neutral")
        self.pill.pack(side="right", pady=(SPACE.SM, 0))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=5, uniform="routing")
        body.grid_columnconfigure(1, weight=6, uniform="routing")
        body.grid_rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_right(body)

    # -- colonne de gauche : ce que l'utilisateur décide -----------------

    def _build_left(self, parent):
        theme = current()
        t = self.t

        # Les commandes tiennent désormais sans défilement : elles se règlent
        # une fois, il n'y a aucune raison d'aller les chercher plus bas.
        left = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE.SM))

        # --- extrémités ------------------------------------------------
        self.card_ends = Card(left, title=t("routing.endpoints"), icon="📍")
        self.card_ends.pack(fill="x", pady=(0, SPACE.MD))

        saved = self.app.settings
        self.f_source = CoordinateField(
            self.card_ends.body, label=t("routing.source"),
            help_text=t("routing.coords.help"),
            value=tuple(saved.get("point_a", (0.0, 0.0, 0.0))),
        )
        self.f_source.pack(fill="x", pady=(0, SPACE.MD))

        self.f_target = CoordinateField(
            self.card_ends.body, label=t("routing.target"),
            help_text=t("routing.coords.help"),
            value=tuple(saved.get("point_b", (0.0, 0.0, 0.0))),
        )
        self.f_target.pack(fill="x")

        # --- stratégie --------------------------------------------------
        self.card_strategy = Card(left, title=t("routing.strategy"), icon="🎯")
        self.card_strategy.pack(fill="x", pady=(0, SPACE.MD))

        self.f_team = ChoiceField(
            self.card_strategy.body,
            label=t("routing.team"),
            help_text=t("routing.team.help"),
            options=[(k, v["label_fr"]) for k, v in TEAM_PRESETS.items()],
            value=saved.get("team_preset", "balanced"),
            on_change=self._on_team_changed,
        )
        self.f_team.pack(fill="x", pady=(0, SPACE.SM))

        self.lbl_team_help = ctk.CTkLabel(
            self.card_strategy.body, text="", font=FONT.SMALL,
            text_color=theme.TEXT_FAINT, anchor="w", justify="left", wraplength=440,
        )
        self.lbl_team_help.pack(fill="x", pady=(0, SPACE.MD))

        # Chemin de départ : c'est un vrai arbitrage vitesse / qualité, on le
        # laisse donc à l'utilisateur plutôt que de le figer.
        self.f_start_path = ChoiceField(
            self.card_strategy.body,
            label=t("routing.start_path"),
            help_text=t("routing.start_path.help"),
            options=self._start_path_options(),
            value=saved.get("start_path", "balanced"),
            on_change=self._on_start_path_changed,
        )
        self.f_start_path.pack(fill="x", pady=(0, SPACE.SM))

        self.lbl_start_path_help = ctk.CTkLabel(
            self.card_strategy.body, text="", font=FONT.SMALL,
            text_color=theme.TEXT_FAINT, anchor="w", justify="left", wraplength=440,
        )
        self.lbl_start_path_help.pack(fill="x", pady=(0, SPACE.MD))

        # Emprunter les fixations existantes est un choix d'intégration, pas un
        # détail : le câble qui les traverse est contraint de passer là, ce qui
        # peut rallonger le trajet. L'utilisateur doit pouvoir le refuser.
        self.f_use_fixations = ToggleField(
            self.card_strategy.body,
            label=t("routing.use_fixations"),
            help_text=t("routing.use_fixations.help"),
            value=bool(saved.get("use_fixations", True)),
            on_change=lambda v: self.app.remember(use_fixations=bool(v)),
        )
        self.f_use_fixations.pack(fill="x", pady=(0, SPACE.MD))

        self.f_explore = SliderField(
            self.card_strategy.body,
            label=t("routing.explore"),
            help_text=t("routing.explore.help"),
            from_=0.0, to=1.0, steps=20,
            value=float(saved.get("temperature", 0.45)),
            left_label=t("routing.explore.left"),
            right_label=t("routing.explore.right"),
            value_formatter=self._format_temperature,
            on_change=self._on_temperature_changed,
        )
        self.f_explore.pack(fill="x")

        # --- commandes ---------------------------------------------------
        controls = ctk.CTkFrame(left, fg_color="transparent")
        controls.pack(fill="x", pady=(0, SPACE.MD))

        self.btn_play = ctk.CTkButton(
            controls, text=t("routing.start"), height=48, font=FONT.H2,
            fg_color=theme.accent, command=self._on_play,
        )
        self.btn_play.pack(side="left", fill="x", expand=True)

        self.btn_reset = ctk.CTkButton(
            controls, text=t("routing.reset"), height=48, width=150, font=FONT.BODY_BOLD,
            fg_color="transparent", border_width=1, border_color=theme.BORDER,
            text_color=theme.TEXT, hover_color=theme.SURFACE_ALT,
            command=self._on_reset,
        )
        self.btn_reset.pack(side="left", padx=(SPACE.SM, 0))

        # --- réglages avancés ------------------------------------------------
        self.btn_advanced = ctk.CTkButton(
            left, text=f"▸ {t('routing.advanced')}", height=32, anchor="w",
            fg_color="transparent", text_color=theme.TEXT_SOFT,
            hover_color=theme.SURFACE_ALT, font=FONT.SMALL_BOLD,
            command=self._toggle_advanced,
        )
        self.btn_advanced.pack(fill="x")

        self.card_advanced = Card(left, subtitle=t("routing.advanced.warn"))
        self.f_points = NumberField(
            self.card_advanced.body, label="Points de départ sur la trajectoire",
            help_text="Plus il y en a, plus le tracé est fin — et plus le calcul est long.",
            value=float(saved.get("initial_points", 48)), unit="pts", entry_width=100,
        )
        self.f_points.pack(fill="x", pady=(0, SPACE.MD))

        self.f_max_points = NumberField(
            self.card_advanced.body, label="Nombre maximal de points",
            help_text="Plafond au-delà duquel l'agent ne peut plus raffiner le tracé.",
            value=float(saved.get("max_points", 150)), unit="pts", entry_width=100,
        )
        self.f_max_points.pack(fill="x", pady=(0, SPACE.MD))

        self.f_step = NumberField(
            self.card_advanced.body, label="Déplacement maximal par itération",
            help_text="De combien un point du câble peut bouger d'un coup.",
            value=float(saved.get("max_step_mm", 25.0)), unit="mm", entry_width=100,
        )
        self.f_step.pack(fill="x", pady=(0, SPACE.MD))

        self.f_voxel = NumberField(
            self.card_advanced.body, label="Résolution de la recherche",
            help_text=("Taille des cellules explorées pour trouver le chemin de départ. "
                       "0 = déduite de la distance minimale. Plus fin = plus précis, plus lent."),
            value=float(saved.get("voxel_mm", 0.0)), unit="mm", entry_width=100,
        )
        self.f_voxel.pack(fill="x", pady=(0, SPACE.MD))

        self.f_iterations = NumberField(
            self.card_advanced.body, label="Nombre d'itérations",
            help_text="Le calcul s'arrête de lui-même s'il converge avant.",
            value=float(saved.get("iterations", 500)), unit="", entry_width=100,
        )
        self.f_iterations.pack(fill="x")

        self._on_team_changed(self.f_team.get())
        self._on_start_path_changed(self.f_start_path.get())
        self.f_explore.refresh_value_label()

    # -- colonne de droite : ce que l'utilisateur observe -----------------

    def _build_right(self, parent):
        theme = current()
        t = self.t

        right = ctk.CTkFrame(parent, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(SPACE.SM, 0))
        right.grid_rowconfigure(3, weight=4)
        right.grid_rowconfigure(4, weight=5)
        right.grid_columnconfigure(0, weight=1)

        # --- avancement ---------------------------------------------------
        progress_card = Card(right)
        progress_card.grid(row=0, column=0, sticky="ew", pady=(0, SPACE.SM))

        phase_row = ctk.CTkFrame(progress_card.body, fg_color="transparent")
        phase_row.pack(fill="x", pady=(0, SPACE.SM))

        self.lbl_phase_caption = ctk.CTkLabel(
            phase_row, text=t("routing.phase"), font=FONT.SMALL_BOLD,
            text_color=theme.TEXT_SOFT,
        )
        self.lbl_phase_caption.pack(side="left", padx=(0, SPACE.MD))

        self.phase_indicator = PhaseIndicator(phase_row)
        self.phase_indicator.pack(side="left")

        self.lbl_run_info = ctk.CTkLabel(
            phase_row, text="", font=FONT.SMALL, text_color=theme.TEXT_FAINT, anchor="e"
        )
        self.lbl_run_info.pack(side="right")

        self.kpis = KpiRow(
            progress_card.body,
            [
                ("length", t("kpi.length"), "mm"),
                ("clashes", t("kpi.clashes"), ""),
                ("bend", t("kpi.bend"), "mm"),
                ("straight", t("kpi.straight"), "%"),
                ("clamps", t("kpi.clamps"), ""),
            ],
        )
        self.kpis.pack(fill="x")

        # --- fixations existantes ---------------------------------------------
        # Le résultat du scan était jusqu'ici invisible : l'utilisateur ne
        # pouvait pas savoir si l'application avait reconnu les fixations déjà
        # montées, ni quels passages elle en avait déduits.
        self.scan_box = ctk.CTkFrame(
            right, fg_color=theme.SURFACE, border_width=1,
            border_color=theme.BORDER, corner_radius=SPACE.RADIUS,
        )

        self.lbl_scan = ctk.CTkLabel(
            self.scan_box, text="", font=FONT.SMALL_BOLD, text_color=theme.TEXT,
            anchor="w", justify="left", wraplength=760,
        )
        self.lbl_scan.pack(fill="x", padx=SPACE.MD, pady=(SPACE.SM, 0))

        self.scan_passages = ctk.CTkFrame(self.scan_box, fg_color="transparent")
        self.scan_passages.pack(fill="x", padx=SPACE.MD, pady=(SPACE.XS, SPACE.SM))

        # --- options d'affichage 3D ------------------------------------------
        view_bar = ctk.CTkFrame(right, fg_color="transparent")
        view_bar.grid(row=2, column=0, sticky="ew", pady=(0, SPACE.XS))

        self.lbl_view = ctk.CTkLabel(
            view_bar, text=t("routing.view"), font=FONT.SMALL_BOLD, text_color=theme.TEXT_SOFT
        )
        self.lbl_view.pack(side="left", padx=(0, SPACE.MD))

        self.view_toggles: dict[str, ctk.CTkCheckBox] = {}
        for key, label, default in (
            ("mesh", t("routing.view.mesh"), True),
            ("edges", t("routing.view.edges"), False),
            ("bbox", t("routing.view.bbox"), False),
            ("clamps", t("routing.view.clamps"), True),
        ):
            box = ctk.CTkCheckBox(
                view_bar, text=label, font=FONT.SMALL, checkbox_width=17, checkbox_height=17,
                fg_color=theme.accent, command=self._on_view_changed,
            )
            box.select() if default else box.deselect()
            box.pack(side="left", padx=(0, SPACE.MD))
            self.view_toggles[key] = box

        self.btn_detach = ctk.CTkButton(
            view_bar, text=t("routing.view.detach"), width=130, height=26,
            font=FONT.SMALL_BOLD, fg_color="transparent", border_width=1,
            border_color=theme.BORDER, text_color=theme.TEXT,
            hover_color=theme.SURFACE_ALT, command=self._on_detach,
        )
        self.btn_detach.pack(side="right")

        # --- vue 3D ------------------------------------------------------------
        view_frame = ctk.CTkFrame(
            right, fg_color=theme.SURFACE, border_width=1,
            border_color=theme.BORDER, corner_radius=SPACE.RADIUS,
        )
        view_frame.grid(row=3, column=0, sticky="nsew", pady=(0, SPACE.SM))

        self.viewer_container = ctk.CTkFrame(view_frame, fg_color=("#EDEFF3", "#0E1116"),
                                             corner_radius=SPACE.RADIUS_SM)
        self.viewer_container.pack(fill="both", expand=True, padx=2, pady=2)

        # --- sorties vivantes, en onglets -----------------------------------------
        # Conformité, agents et courbes se disputaient la place avec la vue 3D.
        # En onglets, chacun dispose de toute la largeur, et la conformité — la
        # seule information dont l'utilisateur a réellement besoin — est celle
        # qui s'affiche par défaut.
        self.tabs = ctk.CTkTabview(
            right, fg_color=theme.SURFACE, segmented_button_selected_color=theme.accent,
            segmented_button_selected_hover_color=theme.accent, corner_radius=SPACE.RADIUS,
        )
        self.tabs.grid(row=4, column=0, sticky="nsew")

        self._tab_names = {
            "compliance": t("routing.compliance"),
            "advice": t("routing.advice"),
            "agents": t("routing.agents"),
            "charts": t("routing.charts"),
        }
        for name in self._tab_names.values():
            self.tabs.add(name)

        compliance_tab = ctk.CTkScrollableFrame(
            self.tabs.tab(self._tab_names["compliance"]), fg_color="transparent"
        )
        compliance_tab.pack(fill="both", expand=True)
        self.compliance = ComplianceTable(compliance_tab, lang=self.app.t.lang)
        self.compliance.set_placeholder(t("report.verdict.none"))
        self.compliance.pack(fill="both", expand=True)

        advice_tab = ctk.CTkScrollableFrame(
            self.tabs.tab(self._tab_names["advice"]), fg_color="transparent"
        )
        advice_tab.pack(fill="both", expand=True)
        self.advice_board = AdviceBoard(
            advice_tab, lang=self.app.t.lang, on_apply=self._on_apply_advice
        )
        self.advice_board.pack(fill="both", expand=True)

        agents_tab = ctk.CTkScrollableFrame(
            self.tabs.tab(self._tab_names["agents"]), fg_color="transparent"
        )
        agents_tab.pack(fill="both", expand=True)
        self.agent_board = AgentBoard(agents_tab)
        self.agent_board.pack(fill="both", expand=True)

        self.charts_container = ctk.CTkFrame(
            self.tabs.tab(self._tab_names["charts"]), fg_color="transparent"
        )
        self.charts_container.pack(fill="both", expand=True)

    # -- réactions -------------------------------------------------------

    def _format_temperature(self, value: float) -> str:
        return ExplorationPolicy(value).label(self.app.t.lang)

    def _start_path_options(self):
        key = "label_en" if self.app.t.is_english else "label_fr"
        options = [(name, spec[key]) for name, spec in STRATEGIES.items()]
        options.append(("geodesic", self.t("routing.start_path.geodesic")))
        return options

    def _on_start_path_changed(self, key: str):
        if key == "geodesic":
            text = self.t("routing.start_path.geodesic.help")
        else:
            spec = STRATEGIES.get(key, {})
            text = spec.get("help_en" if self.app.t.is_english else "help_fr", "")
        self.lbl_start_path_help.configure(text=text)
        self.app.remember(start_path=key)

    def _on_team_changed(self, key: str):
        preset = TEAM_PRESETS.get(key, {})
        help_key = "help_en" if self.app.t.is_english else "help_fr"
        composition = preset.get("composition", {})
        roles = ", ".join(
            f"{count} × {ROLES[role].label(self.app.t.lang)}"
            for role, count in composition.items()
            if role in ROLES
        )
        self.lbl_team_help.configure(text=f"{preset.get(help_key, '')}\n{roles}")
        self.app.remember(team_preset=key)

    def _on_temperature_changed(self, value: float):
        self.app.remember(temperature=value)
        self.app.controller.set_temperature(value)

    def _on_play(self):
        self.app.controller.toggle_routing()

    def _on_reset(self):
        self.app.controller.reset_routing()

    def _on_view_changed(self):
        self.app.controller.update_3d_visibility(
            {key: bool(box.get()) for key, box in self.view_toggles.items()}
        )

    def _on_apply_advice(self, suggestion):
        """Applique un conseil au réglage correspondant, à l'étape « Règles ».

        Le changement n'est pas appliqué à chaud : les agents ont recopié les
        règles au démarrage, et n'en modifier qu'une partie en cours de route
        produirait un mélange incohérent entre la récompense et le rapport. On
        écrit donc dans la page « Règles » et on invite à relancer.
        """
        self.app.controller.apply_suggestion(suggestion)

    def _on_detach(self):
        self.app.controller.detach_3d()

    def _toggle_advanced(self):
        self._advanced_open = not self._advanced_open
        if self._advanced_open:
            self.btn_advanced.configure(text=f"▾ {self.t('routing.advanced')}")
            self.card_advanced.pack(fill="x", pady=(SPACE.XS, SPACE.MD))
        else:
            self.btn_advanced.configure(text=f"▸ {self.t('routing.advanced')}")
            self.card_advanced.pack_forget()

    # -- lecture ---------------------------------------------------------

    def collect(self) -> dict:
        return {
            "point_a": self.f_source.get(),
            "point_b": self.f_target.get(),
            "team_preset": self.f_team.get(),
            "start_path": self.f_start_path.get(),
            "use_fixations": self.f_use_fixations.get(),
            "temperature": self.f_explore.get(),
            "initial_points": int(self.f_points.get(48)),
            "max_points": int(self.f_max_points.get(150)),
            "max_step_mm": self.f_step.get(25.0),
            "voxel_mm": self.f_voxel.get(0.0),
            "iterations": int(self.f_iterations.get(500)),
        }

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.f_source.is_valid() or not self.f_target.is_valid():
            problems.append("Les coordonnées de départ et d'arrivée doivent être des nombres.")
            return problems

        a, b = self.f_source.get(), self.f_target.get()
        if a == b:
            problems.append("Le point de départ et le point d'arrivée sont identiques.")
        if self.f_max_points.get(150) < self.f_points.get(48):
            problems.append(
                "Le nombre maximal de points est inférieur au nombre de points de départ."
            )
        return problems

    # -- API appelée par le contrôleur -------------------------------------

    def set_running_state(self, state: str):
        """``state`` : ``idle``, ``scanning``, ``running`` ou ``paused``."""
        theme = current()
        t = self.t
        if state == "scanning":
            self.btn_play.configure(text=t("routing.scanning"), state="disabled")
            self.pill.update_status(t("routing.scanning"), "info")
        elif state == "running":
            self.btn_play.configure(
                text=t("routing.pause"), state="normal", fg_color=theme.warn
            )
            self.pill.update_status(t("routing.pause").upper(), "accent")
        elif state == "paused":
            self.btn_play.configure(
                text=t("routing.resume"), state="normal", fg_color=theme.accent
            )
            self.pill.update_status(t("routing.resume"), "warn")
        else:
            self.btn_play.configure(
                text=t("routing.start"), state="normal", fg_color=theme.accent
            )
            self.pill.update_status(t("app.ready"), "neutral")

    def update_live(self, snapshot: dict):
        """Rafraîchit les indicateurs, la conformité et les agents."""
        theme = current()

        report = snapshot.get("report")
        self.compliance.update_report(report)
        self.advice_board.update_advice(snapshot.get("advice"))
        self._refresh_advice_tab()

        if report is not None:
            k = report.kpis
            radius = k.get("min_bend_radius_mm", float("inf"))
            limit = k.get("bend_limit_mm", 0.0)
            clashes = int(k.get("n_clashes", 0))
            self.kpis.update_values(
                {
                    "length": f"{k.get('length_mm', 0):.0f}",
                    "clashes": (str(clashes), theme.ok if clashes == 0 else theme.danger),
                    "bend": (
                        "∞" if radius == float("inf") else f"{radius:.0f}",
                        theme.ok if radius >= limit else theme.danger,
                    ),
                    "straight": f"{k.get('straight_ratio', 0) * 100:.0f}",
                    "clamps": str(k.get("n_clamps", 0)),
                }
            )

        team = snapshot.get("team") or {}
        if team:
            labels = {
                key: Phase.label(key, self.app.t.lang) for key in PhaseIndicator.ORDER
            }
            self.phase_indicator.update_phase(team.get("phase", ""), labels)
            self.agent_board.update_agents(snapshot.get("agents", []))

        info = []
        if snapshot.get("iteration") is not None:
            info.append(f"{self.t('routing.iteration')} {snapshot['iteration']}")
        if team.get("best"):
            info.append(f"{self.t('routing.best')} : {team['best']}")
        self.lbl_run_info.configure(text="   ·   ".join(info))

    def show_fixation_scan(self, result):
        """Affiche le résultat du scan des fixations existantes.

        Les passages ``p_in`` / ``p_out`` sont listés explicitement : ce sont
        des contraintes de passage, pas des indications. Un faisceau qui ne
        traverse pas l'encoche n'est pas posable, et l'utilisateur doit pouvoir
        vérifier que l'application a bien reconnu les bonnes.
        """
        theme = current()
        for widget in self.scan_passages.winfo_children():
            widget.destroy()

        if result is None:
            self.scan_box.grid_remove()
            return

        lang = self.app.t.lang
        self.scan_box.grid(row=1, column=0, sticky="ew", pady=(0, SPACE.SM))

        if not result.ran:
            self.lbl_scan.configure(
                text=f"🔎  {result.message(lang)}", text_color=theme.TEXT_FAINT
            )
            return

        colour = theme.ok if result.n_fixations else theme.TEXT_SOFT
        self.lbl_scan.configure(text=f"🔎  {result.message(lang)}", text_color=colour)

        # Numérotation continue : l'index d'un passage est propre à son peigne,
        # deux peignes à une encoche porteraient donc tous deux le n° 1.
        for number, passage in enumerate(result.passages[:MAX_LISTED_PASSAGES], start=1):
            ctk.CTkLabel(
                self.scan_passages, text=passage.format(lang, number=number),
                font=FONT.CODE, text_color=theme.TEXT_SOFT, anchor="w",
            ).pack(fill="x")

        remaining = result.n_passages - MAX_LISTED_PASSAGES
        if remaining > 0:
            more = (f"… and {remaining} more passage(s)"
                    if self.app.t.is_english else
                    f"… et {remaining} passage(s) de plus")
            ctk.CTkLabel(
                self.scan_passages, text=more, font=FONT.TINY,
                text_color=theme.TEXT_FAINT, anchor="w",
            ).pack(fill="x")

    def set_use_fixations(self, value: bool):
        """Reflète la réponse donnée à la fenêtre de confirmation.

        L'utilisateur retrouve son choix là où il l'aurait coché lui-même, et
        il est mémorisé pour la session suivante.
        """
        self.f_use_fixations.var.set(bool(value))
        self.app.remember(use_fixations=bool(value))

    def _refresh_advice_tab(self):
        """Affiche le nombre de conseils sur l'onglet.

        Sans cela, un conseil déposé dans un onglet non affiché n'est jamais vu.
        """
        count = self.advice_board.count()
        label = self.t("routing.advice")
        wanted = f"{label} ({count})" if count else label
        current_name = self._tab_names["advice"]
        if wanted == current_name:
            return
        try:
            self.tabs.rename(current_name, wanted)
        except Exception:
            return
        self._tab_names["advice"] = wanted

    def set_detached(self, detached: bool):
        self.btn_detach.configure(
            text=self.t("routing.view.attach") if detached else self.t("routing.view.detach")
        )

    def update_language(self):
        t = self.t
        self.lbl_title.configure(text=t("routing.title"))
        self.lbl_intro.configure(text=t("routing.intro"))
        self.card_ends.set_title(t("routing.endpoints"), "📍")
        self.card_strategy.set_title(t("routing.strategy"), "🎯")
        self.card_advanced.set_subtitle(t("routing.advanced.warn"))
        self.f_source.set_label(t("routing.source"), t("routing.coords.help"))
        self.f_target.set_label(t("routing.target"), t("routing.coords.help"))
        self.f_team.set_label(t("routing.team"), t("routing.team.help"))
        label_key = "label_en" if self.app.t.is_english else "label_fr"
        self.f_team.set_options(
            [(k, v[label_key]) for k, v in TEAM_PRESETS.items()], self.f_team.get()
        )
        self.f_start_path.set_label(t("routing.start_path"), t("routing.start_path.help"))
        self.f_use_fixations.set_label(t("routing.use_fixations"), t("routing.use_fixations.help"))
        self.f_start_path.set_options(self._start_path_options(), self.f_start_path.get())
        self._on_start_path_changed(self.f_start_path.get())
        self.f_explore.set_label(t("routing.explore"), t("routing.explore.help"))
        self.f_explore.set_end_labels(t("routing.explore.left"), t("routing.explore.right"))
        self.f_explore.refresh_value_label()
        self.btn_reset.configure(text=t("routing.reset"))
        self.btn_advanced.configure(
            text=f"{'▾' if self._advanced_open else '▸'} {t('routing.advanced')}"
        )
        self.lbl_phase_caption.configure(text=t("routing.phase"))
        self.lbl_view.configure(text=t("routing.view"))
        for key, label_key in (
            ("mesh", "routing.view.mesh"), ("edges", "routing.view.edges"),
            ("bbox", "routing.view.bbox"), ("clamps", "routing.view.clamps"),
        ):
            self.view_toggles[key].configure(text=t(label_key))
        self.kpis.set_labels(
            {
                "length": t("kpi.length"), "clashes": t("kpi.clashes"),
                "bend": t("kpi.bend"), "straight": t("kpi.straight"),
                "clamps": t("kpi.clamps"),
            }
        )
        self.compliance.set_language(self.app.t.lang)
        self.compliance.set_placeholder(t("report.verdict.none"))
        self.advice_board.update_language(self.app.t.lang)
        self._on_team_changed(self.f_team.get())
