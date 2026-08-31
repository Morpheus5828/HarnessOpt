"""Étape 2 : régler les contraintes d'intégration.

Tout est exprimé en unités physiques et en vocabulaire métier : diamètre du
toron, rayon de cintrage, distance minimale, pas entre fixations. Aucune de ces
valeurs n'est un hyperparamètre d'algorithme ; ce sont les règles que le bureau
d'études applique déjà.

Le rayon de cintrage minimal se recalcule sous les yeux de l'utilisateur dès
qu'il change le diamètre : il voit immédiatement la conséquence de sa saisie.
"""

from __future__ import annotations

import customtkinter as ctk

from core.routing_rules import ALL_RULES, DEFAULT_FAMILY_CLEARANCE
from ui.theme import FONT, SPACE, current
from ui.widgets import Card, NumberField, PathField, RuleToggleList

__all__ = ["RulesPage"]


class RulesPage(ctk.CTkScrollableFrame):
    def __init__(self, master, app, **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self.app = app
        self.t = app.t
        self._family_fields: dict[str, NumberField] = {}
        self._build()

    # -- construction ---------------------------------------------------

    def _build(self):
        theme = current()
        t = self.t
        saved = self.app.settings

        self.lbl_title = ctk.CTkLabel(
            self, text=t("rules.title"), font=FONT.TITLE, text_color=theme.TEXT, anchor="w"
        )
        self.lbl_title.pack(fill="x", pady=(0, SPACE.XS))

        self.lbl_intro = ctk.CTkLabel(
            self, text=t("rules.intro"), font=FONT.BODY, text_color=theme.TEXT_SOFT,
            anchor="w", justify="left", wraplength=820,
        )
        self.lbl_intro.pack(fill="x", pady=(0, SPACE.LG))

        # --- règles appliquées -------------------------------------------
        # Placée en premier : décider *quelles* règles s'appliquent commande
        # tout le reste. Régler une distance minimale qu'on a par ailleurs
        # désactivée n'aurait aucun sens, et l'ordre de l'écran doit refléter
        # cette dépendance.
        self.card_active = Card(
            self, title=t("rules.active.title"),
            subtitle=t("rules.active.help"), icon="☑️",
        )
        self.card_active.pack(fill="x", pady=(0, SPACE.MD))

        self.rule_list = RuleToggleList(
            self.card_active.body,
            enabled=set(saved.get("enabled_rules", sorted(ALL_RULES))),
            on_change=self._on_rule_toggled,
            lang=self.app.t.lang,
        )
        self.rule_list.pack(fill="x")

        # --- le harnais ------------------------------------------------
        self.card_harness = Card(self, title=t("rules.harness"), icon="🔌")
        self.card_harness.pack(fill="x", pady=(0, SPACE.MD))

        self.f_diameter = NumberField(
            self.card_harness.body,
            label=t("rules.diameter"),
            help_text=t("rules.diameter.help"),
            value=float(saved.get("harness_diameter", 40.0)),
            unit="mm",
            on_change=lambda _v: self._refresh_bend(),
        )
        self.f_diameter.pack(fill="x", pady=(0, SPACE.MD))

        self.f_bend_factor = NumberField(
            self.card_harness.body,
            label=t("rules.bend_factor"),
            help_text=t("rules.bend_factor.help"),
            value=float(saved.get("bend_radius_factor", 6.0)),
            unit="× Ø",
            on_change=lambda _v: self._refresh_bend(),
        )
        self.f_bend_factor.pack(fill="x", pady=(0, SPACE.SM))

        self.lbl_bend = ctk.CTkLabel(
            self.card_harness.body, text="", font=FONT.BODY_BOLD,
            text_color=theme.accent, anchor="w",
        )
        self.lbl_bend.pack(fill="x")

        # --- distances --------------------------------------------------
        self.card_clearance = Card(self, title=t("rules.clearance"), icon="📏")
        self.card_clearance.pack(fill="x", pady=(0, SPACE.MD))

        self.f_min = NumberField(
            self.card_clearance.body,
            label=t("rules.clearance.min"),
            help_text=t("rules.clearance.min.help"),
            value=float(saved.get("min_margin", 10.0)),
            unit="mm",
        )
        self.f_min.pack(fill="x", pady=(0, SPACE.MD))

        self.f_max = NumberField(
            self.card_clearance.body,
            label=t("rules.clearance.max"),
            help_text=t("rules.clearance.max.help"),
            value=float(saved.get("max_margin", 100.0)),
            unit="mm",
        )
        self.f_max.pack(fill="x")

        # --- distances renforcées par famille ---------------------------
        self.card_families = Card(
            self, title=t("rules.families.title"),
            subtitle=t("rules.families.help"), icon="🎨",
        )
        self.card_families.pack(fill="x", pady=(0, SPACE.MD))

        self.lbl_families_state = ctk.CTkLabel(
            self.card_families.body, text=t("rules.families.none"), font=FONT.SMALL,
            text_color=theme.TEXT_FAINT, anchor="w", justify="left", wraplength=760,
        )
        self.lbl_families_state.pack(fill="x", pady=(0, SPACE.SM))

        self.families_box = ctk.CTkFrame(self.card_families.body, fg_color="transparent")
        self.families_box.pack(fill="x")

        # --- fixations ---------------------------------------------------
        self.card_fix = Card(self, title=t("rules.fixations"), icon="🦀")
        self.card_fix.pack(fill="x", pady=(0, SPACE.MD))

        self.f_pitch = NumberField(
            self.card_fix.body,
            label=t("rules.pitch"),
            help_text=t("rules.pitch.help"),
            value=float(saved.get("fixation_pitch", 250.0)),
            unit="mm",
        )
        self.f_pitch.pack(fill="x", pady=(0, SPACE.MD))

        self.f_parallel = NumberField(
            self.card_fix.body,
            label=t("rules.parallel"),
            help_text=t("rules.parallel.help"),
            value=float(saved.get("fixation_parallel_tol", 15.0)),
            unit="°",
        )
        self.f_parallel.pack(fill="x", pady=(0, SPACE.MD))

        self.f_clamp_model = PathField(
            self.card_fix.body,
            label=t("rules.clamp_model"),
            help_text=t("rules.clamp_model.help"),
            value=saved.get("crabe_stl_path", ""),
            browse_text=t("project.browse"),
            mode="file",
            filetypes=[("Fichiers STL", "*.stl"), ("Tous les fichiers", "*.*")],
        )
        self.f_clamp_model.pack(fill="x")

        # --- actions ------------------------------------------------------
        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", pady=(0, SPACE.LG))

        self.btn_reset = ctk.CTkButton(
            actions, text=t("rules.reset"), height=42, width=220, font=FONT.BODY_BOLD,
            fg_color="transparent", border_width=1, border_color=theme.BORDER,
            text_color=theme.TEXT, hover_color=theme.SURFACE_ALT,
            command=self.reset_defaults,
        )
        self.btn_reset.pack(side="left")

        # Un assistant doit dire où aller ensuite. Sans ce bouton, il fallait
        # deviner qu'on avance en cliquant le numéro d'étape en haut de page.
        self.btn_continue = ctk.CTkButton(
            actions, text=t("rules.continue"), height=42, font=FONT.H3,
            fg_color=theme.accent, command=self._on_continue,
        )
        self.btn_continue.pack(side="right")

        self._refresh_bend()

    # -- réactions ------------------------------------------------------

    def _on_continue(self):
        """Valide les règles et passe au cheminement, ou dit ce qui cloche."""
        problems = self.validate()
        if problems:
            self.app.set_status(problems[0], "warn")
            return

        self.app.remember(**{k: v for k, v in self.collect().items() if k != "family_clearance"})
        self.app.remember(family_clearance=self.collect()["family_clearance"])
        self.app.set_status(self.t("rules.ready"), "ok")
        self.app.show_step(2)

    def _on_rule_toggled(self, _rule_id: str, _value: bool):
        """Mémorise le jeu de règles dès qu'il change.

        Sans cela, décocher une règle puis revenir à l'étape précédente
        perdait le réglage sans prévenir.
        """
        self.app.remember(enabled_rules=sorted(self.rule_list.get()))

    def _refresh_bend(self):
        diameter = self.f_diameter.get(40.0)
        factor = self.f_bend_factor.get(6.0)
        self.lbl_bend.configure(
            text=f"{self.t('rules.bend_result')}  {diameter * factor:.0f} mm"
        )

    def reset_defaults(self):
        self.rule_list.select_all()
        self.f_diameter.set(40.0)
        self.f_bend_factor.set(6.0)
        self.f_min.set(10.0)
        self.f_max.set(100.0)
        self.f_pitch.set(250.0)
        self.f_parallel.set(15.0)
        for name, field in self._family_fields.items():
            field.set(DEFAULT_FAMILY_CLEARANCE.get(name, 10.0))
        self._refresh_bend()

    def show_families(self, families: list[str]):
        """Crée un champ de distance par famille présente dans la maquette.

        Tant qu'aucune maquette classée par couleur n'est chargée, la section
        reste vide et explique pourquoi : mieux vaut une absence expliquée
        qu'une liste de familles inventées.
        """
        for widget in self.families_box.winfo_children():
            widget.destroy()
        self._family_fields.clear()

        if not families:
            self.lbl_families_state.configure(text=self.t("rules.families.none"))
            self.lbl_families_state.pack(fill="x", pady=(0, SPACE.SM))
            return

        self.lbl_families_state.configure(
            text=f"{self.t('rules.families.detected')} : {len(families)}"
        )

        saved = self.app.settings.get("family_clearance", {}) or {}
        for i, name in enumerate(sorted(families)):
            value = float(saved.get(name, DEFAULT_FAMILY_CLEARANCE.get(name, 10.0)))
            field = NumberField(
                self.families_box, label=name.replace("_", " "), value=value, unit="mm",
                entry_width=100,
            )
            field.grid(row=i // 2, column=i % 2, sticky="ew",
                       padx=(0, SPACE.LG), pady=(0, SPACE.SM))
            self.families_box.grid_columnconfigure(i % 2, weight=1)
            self._family_fields[name] = field

    # -- lecture --------------------------------------------------------

    def collect(self) -> dict:
        """Valeurs saisies, prêtes à construire un ``RoutingRules``."""
        return {
            "harness_diameter": self.f_diameter.get(40.0),
            "bend_radius_factor": self.f_bend_factor.get(6.0),
            "min_margin": self.f_min.get(10.0),
            "max_margin": self.f_max.get(100.0),
            "fixation_pitch": self.f_pitch.get(250.0),
            "fixation_parallel_tol": self.f_parallel.get(15.0),
            "crabe_stl_path": self.f_clamp_model.get(),
            "enabled_rules": sorted(self.rule_list.get()),
            "family_clearance": {
                name: field.get(10.0) for name, field in self._family_fields.items()
            },
        }

    def validate(self) -> list[str]:
        """Incohérences détectées, formulées en langage clair."""
        problems: list[str] = []
        values = self.collect()

        if not values["enabled_rules"]:
            problems.append(
                "Aucune règle n'est appliquée : les agents n'auraient rien à respecter."
            )
            return problems

        if values["harness_diameter"] <= 0:
            problems.append("Le diamètre du toron doit être supérieur à zéro.")
        if values["bend_radius_factor"] <= 0:
            problems.append("Le rayon de cintrage doit être un multiple positif du diamètre.")
        if values["min_margin"] < 0:
            problems.append("La distance minimale ne peut pas être négative.")
        if values["max_margin"] <= values["min_margin"]:
            problems.append(
                "La distance maximale doit être supérieure à la distance minimale : "
                "sinon aucune position n'est acceptable."
            )
        if values["fixation_pitch"] <= 0:
            problems.append("Le pas entre fixations doit être supérieur à zéro.")
        if not 0 < values["fixation_parallel_tol"] < 90:
            problems.append("L'écart de pose toléré doit être compris entre 0 et 90 degrés.")

        strictest = max(values["family_clearance"].values(), default=values["min_margin"])
        if strictest >= values["max_margin"]:
            problems.append(
                f"La distance renforcée la plus élevée ({strictest:.0f} mm) atteint la distance "
                f"maximale ({values['max_margin']:.0f} mm) : aucun passage ne serait possible "
                "le long de ces pièces."
            )
        return problems

    def update_language(self):
        t = self.t
        self.lbl_title.configure(text=t("rules.title"))
        self.lbl_intro.configure(text=t("rules.intro"))
        self.card_active.set_title(t("rules.active.title"), "☑️")
        self.card_active.set_subtitle(t("rules.active.help"))
        self.rule_list.update_language(self.app.t.lang)
        self.card_harness.set_title(t("rules.harness"), "🔌")
        self.card_clearance.set_title(t("rules.clearance"), "📏")
        self.card_families.set_title(t("rules.families.title"), "🎨")
        self.card_families.set_subtitle(t("rules.families.help"))
        self.card_fix.set_title(t("rules.fixations"), "🦀")
        self.f_diameter.set_label(t("rules.diameter"), t("rules.diameter.help"))
        self.f_bend_factor.set_label(t("rules.bend_factor"), t("rules.bend_factor.help"))
        self.f_min.set_label(t("rules.clearance.min"), t("rules.clearance.min.help"))
        self.f_max.set_label(t("rules.clearance.max"), t("rules.clearance.max.help"))
        self.f_pitch.set_label(t("rules.pitch"), t("rules.pitch.help"))
        self.f_parallel.set_label(t("rules.parallel"), t("rules.parallel.help"))
        self.f_clamp_model.set_label(t("rules.clamp_model"), t("rules.clamp_model.help"))
        self.f_clamp_model.set_browse_text(t("project.browse"))
        self.btn_reset.configure(text=t("rules.reset"))
        self.btn_continue.configure(text=t("rules.continue"))
        self._refresh_bend()
