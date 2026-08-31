"""Vérifications de l'interface, exécutées sans écran.

Ces tests demandent tkinter, customtkinter et un serveur X (``xvfb-run`` sous
Linux). Ils sont ignorés si l'un manque, pour ne pas bloquer la suite de tests
sur une machine de calcul sans environnement graphique.

Ils couvrent ce qu'une relecture ne voit pas : les valeurs par défaut réellement
présentes dans les champs, la validation des saisies, et le fait que chaque
composant survit à un rafraîchissement avec des données partielles — c'est là
que se logent les pannes d'interface.
"""

import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("tkinter", reason="tkinter absent de cet interpréteur")
ctk = pytest.importorskip("customtkinter", reason="customtkinter non installé")

os.environ.setdefault("HARNESSOPT_CACHE", tempfile.mkdtemp(prefix="ho_test_cache_"))
os.environ.setdefault("HARNESSOPT_CONFIG", tempfile.mkdtemp(prefix="ho_test_conf_"))


@pytest.fixture(scope="module")
def root():
    try:
        app = ctk.CTk()
    except Exception as exc:  # pas d'affichage disponible
        pytest.skip(f"aucun serveur graphique disponible : {exc}")
    app.geometry("1400x900")
    yield app
    app.destroy()


def sample_report(**overrides):
    from core.routing_rules import RoutingRules, evaluate_route

    pts = np.linspace([0, 0, 0], [1000, 0, 0], 21)
    kwargs = dict(
        distances=np.full(21, 30.0),
        inside_mask=np.zeros(21, dtype=bool),
        n_crossings=0,
        clamp_arc_positions=[250.0, 500.0, 750.0],
    )
    kwargs.update(overrides)
    return evaluate_route(pts, RoutingRules(), **kwargs)


class TestChamps:
    def test_valeur_numerique_tolere_la_virgule(self, root):
        from ui.widgets import NumberField

        field = NumberField(root, "Diamètre", value=40.0)
        field.entry.delete(0, "end")
        field.entry.insert(0, "12,5")
        assert field.get() == pytest.approx(12.5)

    def test_valeur_numerique_illisible_renvoie_le_defaut(self, root):
        from ui.widgets import NumberField

        field = NumberField(root, "Diamètre", value=40.0)
        field.entry.delete(0, "end")
        field.entry.insert(0, "abc")
        assert field.get(default=7.0) == 7.0

    def test_coordonnees_lues_et_ecrites(self, root):
        from ui.widgets import CoordinateField

        field = CoordinateField(root, "Départ", value=(1.0, 2.0, 3.0))
        assert field.get() == pytest.approx((1.0, 2.0, 3.0))
        field.set((10.5, -4.0, 0.0))
        assert field.get() == pytest.approx((10.5, -4.0, 0.0))
        assert field.is_valid()

    def test_coordonnees_invalides_signalees(self, root):
        from ui.widgets import CoordinateField

        field = CoordinateField(root, "Départ", value=(0, 0, 0))
        field.entries[1].delete(0, "end")
        field.entries[1].insert(0, "n'importe quoi")
        assert not field.is_valid()

    def test_choix_renvoie_la_valeur_pas_le_libelle(self, root):
        from ui.widgets import ChoiceField

        field = ChoiceField(root, "Équipe", options=[("balanced", "Équilibrée"), ("discovery", "Découverte")])
        assert field.get() == "balanced"
        field.set("discovery")
        assert field.get() == "discovery"

    def test_curseur_affiche_un_libelle_lisible(self, root):
        from ui.widgets import SliderField

        field = SliderField(root, "Exploration", from_=0, to=1, value=0.2,
                            value_formatter=lambda v: "Prudent" if v < 0.5 else "Large")
        assert field.lbl_value.cget("text") == "Prudent"
        field.set(0.9)
        assert field.lbl_value.cget("text") == "Large"


class TestConformite:
    def test_les_lignes_restent_compactes(self, root):
        """Une ligne trop haute rend le tableau illisible : on le vérifie."""
        from ui.widgets import ComplianceTable

        table = ComplianceTable(root)
        table.pack(fill="x")
        table.update_report(sample_report())
        root.update()
        assert all(row.winfo_height() < 90 for row in table._rows)
        table.destroy()

    def test_une_ligne_par_regle(self, root):
        from ui.widgets import ComplianceTable

        report = sample_report()
        table = ComplianceTable(root)
        table.update_report(report)
        assert len([r for r in table._rows if r.winfo_ismapped() or True]) >= len(report.checks)

    def test_les_lignes_sont_reutilisees_entre_deux_rafraichissements(self, root):
        """Recréer les widgets à chaque image ferait clignoter l'écran."""
        from ui.widgets import ComplianceTable

        table = ComplianceTable(root)
        table.pack(fill="x")
        table.update_report(sample_report())
        root.update()
        first = list(table._rows)
        table.update_report(sample_report(n_crossings=2))
        assert table._rows == first
        table.destroy()

    def test_rapport_absent_affiche_linvite(self, root):
        from ui.widgets import ComplianceTable

        table = ComplianceTable(root)
        table.pack(fill="x")
        table.update_report(sample_report())
        table.update_report(None)
        root.update()
        assert table.placeholder.winfo_ismapped()
        table.destroy()

    def test_verdict_conforme(self, root):
        from ui.i18n import Translator
        from ui.widgets import VerdictBanner

        banner = VerdictBanner(root)
        banner.update_verdict(sample_report(), Translator("FR"))
        assert "conforme" in banner.lbl_title.cget("text").lower()

    def test_verdict_non_conforme(self, root):
        from ui.i18n import Translator
        from ui.widgets import VerdictBanner

        banner = VerdictBanner(root)
        banner.update_verdict(sample_report(n_crossings=3), Translator("FR"))
        assert "non conforme" in banner.lbl_title.cget("text").lower()


class TestAgents:
    def test_carte_agent_tolere_les_champs_absents(self, root):
        """Le contrôleur envoie volontiers des clés présentes mais nulles."""
        from ui.widgets import AgentBoard

        board = AgentBoard(root)
        board.update_agents([{"name": "scout", "badge_color": None, "color": None, "label": None}])
        root.update()
        assert board._cards

    def test_cartes_compactes(self, root):
        from ui.widgets import AgentBoard

        board = AgentBoard(root)
        board.pack(fill="x")
        board.update_agents(
            [{"name": "a", "label": "Lisseur", "color": "#118AB2", "rank": 1, "state": "ok"}]
        )
        root.update()
        assert board._cards[0].winfo_height() < 110
        board.destroy()

    def test_etapes_du_curriculum(self, root):
        from core.orchestrator import Phase
        from ui.widgets import PhaseIndicator

        indicator = PhaseIndicator(root)
        labels = {k: Phase.label(k, "FR") for k in PhaseIndicator.ORDER}
        for phase in PhaseIndicator.ORDER:
            indicator.update_phase(phase, labels)
        assert indicator._labels["polish"].cget("text") == Phase.label("polish", "FR")


@pytest.fixture(scope="module")
def app():
    """Fenêtre complète, contrôleur branché, partagée par les tests d'écran."""
    from controller.app_controller import AppController
    from ui.app_window import AppWindow

    try:
        window = AppWindow()
    except Exception as exc:
        pytest.skip(f"fenêtre indisponible : {exc}")
    window.set_controller(AppController(window))
    window.update()
    yield window
    window.destroy()


class TestNavigation:
    """Accès aux étapes de l'assistant.

    L'accès se déduit de l'état réel du projet. Une première version le faisait
    dépendre des écrans déjà visités : l'étape « Cheminement » n'était
    déverrouillée qu'à la fin de son propre affichage, lequel commençait par
    refuser d'entrer tant qu'elle n'était pas déverrouillée. Elle était donc
    inatteignable.
    """

    @staticmethod
    def _maquette_chargee(app):
        app.controller.extraction_summary = {
            "n_parts": 12,
            "bounds": (0, 100, 0, 100, 0, 100),
            "families": {},
        }
        app.refresh_steps()

    @staticmethod
    def _aucune_maquette(app):
        app.controller.extraction_summary = {}
        app.refresh_steps()

    def test_sans_maquette_seule_la_premiere_etape_est_ouverte(self, app):
        self._aucune_maquette(app)
        assert app.controller.max_reachable_step() == 0
        app.show_step(2)
        assert app.current_step == 0

    def test_le_motif_du_verrouillage_est_explicite(self, app):
        self._aucune_maquette(app)
        app.show_step(2)
        assert "maquette" in app.lbl_status.cget("text").lower()

    def test_le_cheminement_souvre_des_que_la_maquette_est_chargee(self, app):
        """Régression : c'est exactement le blocage signalé."""
        self._maquette_chargee(app)
        app.show_step(1)
        assert app.pages[1].validate() == []
        app.show_step(2)
        assert app.current_step == 2

    def test_des_regles_incoherentes_referment_le_cheminement(self, app):
        self._maquette_chargee(app)
        app.show_step(1)
        app.pages[1].f_max.set(5.0)  # maxi < mini
        app.show_step(2)
        assert app.current_step == 1
        assert "maximale" in app.lbl_status.cget("text")
        app.pages[1].f_max.set(100.0)

    def test_le_bouton_continuer_fait_avancer(self, app):
        self._maquette_chargee(app)
        app.show_step(1)
        app.pages[1]._on_continue()
        assert app.current_step == 2

    def test_le_bouton_continuer_refuse_des_regles_incoherentes(self, app):
        self._maquette_chargee(app)
        app.show_step(1)
        app.pages[1].f_diameter.set(0.0)
        app.pages[1]._on_continue()
        assert app.current_step == 1
        app.pages[1].f_diameter.set(40.0)

    def test_le_rapport_reste_ferme_sans_cheminement(self, app):
        self._maquette_chargee(app)
        assert app.controller.max_reachable_step() == 2
        app.show_step(3)
        assert app.current_step != 3
        assert "cheminement" in app.lbl_status.cget("text").lower()

    def test_le_verrouillage_se_referme_si_letat_regresse(self, app):
        """Une étape ne doit pas rester ouverte une fois sa condition perdue."""
        self._maquette_chargee(app)
        app.show_step(2)
        assert app.current_step == 2
        self._aucune_maquette(app)
        app.show_step(2)
        assert app.current_step != 2


class TestApplication:
    def test_les_quatre_etapes_existent(self, app):
        assert len(app.pages) == 4

    def test_changement_de_langue_traduit_toutes_les_pages(self, app):
        app.set_language("EN")
        assert "Route the harness" in app.pages[2].lbl_title.cget("text")
        app.set_language("FR")
        assert "cheminer" in app.pages[2].lbl_title.cget("text")

    def test_regles_par_defaut_coherentes(self, app):
        values = app.pages[1].collect()
        assert values["harness_diameter"] == 40.0
        assert values["min_margin"] < values["max_margin"]
        assert values["fixation_pitch"] == 250.0
        assert app.pages[1].validate() == []

    def test_rayon_affiche_suit_le_diametre(self, app):
        page = app.pages[1]
        page.f_diameter.set(20.0)
        page._refresh_bend()
        assert "120" in page.lbl_bend.cget("text")
        page.f_diameter.set(40.0)
        page._refresh_bend()

    def test_regles_incoherentes_signalees_en_clair(self, app):
        page = app.pages[1]
        page.f_max.set(5.0)
        problems = page.validate()
        assert problems and "maximale" in problems[0]
        page.f_max.set(100.0)

    def test_extremites_identiques_refusees(self, app):
        page = app.pages[2]
        page.f_source.set((10, 20, 30))
        page.f_target.set((10, 20, 30))
        assert page.validate()

    def test_extremites_distinctes_acceptees(self, app):
        page = app.pages[2]
        page.f_source.set((0, 0, 0))
        page.f_target.set((1000, 0, 0))
        assert page.validate() == []

    def test_familles_de_couleurs_affichees(self, app):
        page = app.pages[1]
        page.show_families(["structure", "high_pressure_system"])
        values = page.collect()["family_clearance"]
        assert values["high_pressure_system"] == pytest.approx(70.0)
        assert values["structure"] == pytest.approx(10.0)

    def test_sans_famille_le_message_est_explicite(self, app):
        page = app.pages[1]
        page.show_families([])
        assert page.collect()["family_clearance"] == {}
        assert page.lbl_families_state.winfo_ismapped()

    def test_rafraichissement_avec_donnees_partielles(self, app):
        """Le contrôleur envoie un état incomplet tant qu'aucun agent n'a publié."""
        app.pages[2].update_live({"report": None, "team": {}, "agents": []})
        app.update()

    def test_rafraichissement_complet(self, app):
        report = sample_report()
        report.kpis["bend_limit_mm"] = 240.0
        app.pages[2].update_live(
            {
                "report": report,
                "iteration": 42,
                "team": {"phase": "polish", "best": "smoother", "ranking": ["smoother"]},
                "agents": [{"name": "smoother", "label": "Lisseur", "color": "#118AB2",
                            "rank": 1, "state": "0 interférence"}],
            }
        )
        app.update()
        assert "42" in app.pages[2].lbl_run_info.cget("text")

    def test_page_de_rapport_vide_puis_remplie(self, app):
        page = app.pages[3]
        page.update_reports({}, [], {})
        assert page.btn_stl.cget("state") == "disabled"
        page.update_reports({"smoother": sample_report()}, ["smoother"], {"smoother": "Lisseur"})
        assert page.btn_stl.cget("state") == "normal"
        assert "conforme" in page.banner.lbl_title.cget("text").lower()

    def test_indicateurs_remis_a_zero_quand_le_rapport_disparait(self, app):
        page = app.pages[3]
        page.update_reports({"smoother": sample_report()}, ["smoother"], {"smoother": "Lisseur"})
        assert page.kpis.tiles["length"].lbl_value.cget("text") != "—"
        page.update_reports({}, [], {})
        assert page.kpis.tiles["length"].lbl_value.cget("text") == "—"


class TestReglesActivables:
    """Cases à cocher de la page « Règles ».

    Ce qui compte ici n'est pas l'apparence de la case mais le fait qu'elle
    parvienne jusqu'au moteur : une case décochée doit produire un jeu de
    règles réellement amputé.
    """

    def test_toutes_les_regles_sont_cochees_au_depart(self, app):
        from core.routing_rules import ALL_RULES

        assert app.pages[1].rule_list.get() == set(ALL_RULES)

    def test_chaque_regle_du_catalogue_a_sa_ligne(self, app):
        from core.routing_rules import RULE_IDS

        assert set(app.pages[1].rule_list.rows) == set(RULE_IDS)

    def test_decocher_une_regle_la_retire_de_la_collecte(self, app):
        page = app.pages[1]
        page.rule_list.rows["straightness"].set(False)
        try:
            assert "straightness" not in page.collect()["enabled_rules"]
            assert "clash" in page.collect()["enabled_rules"]
        finally:
            page.rule_list.select_all()

    def test_le_controleur_construit_des_regles_amputees(self, app):
        page = app.pages[1]
        page.rule_list.rows["straightness"].set(False)
        page.rule_list.rows["free_span"].set(False)
        try:
            rules = app.controller.build_rules()
            assert not rules.is_enabled("straightness")
            assert not rules.is_enabled("free_span")
            assert rules.is_enabled("clash")
        finally:
            page.rule_list.select_all()

    def test_la_recompense_est_neutralisee_pour_une_famille_entiere(self, app):
        page = app.pages[1]
        page.rule_list.rows["fixation_pitch"].set(False)
        page.rule_list.rows["fixation_parallel"].set(False)
        try:
            assert app.controller.build_rules().reward_scale()["fixation"] == 0.0
        finally:
            page.rule_list.select_all()

    def test_tout_decocher_est_refuse_par_la_validation(self, app):
        page = app.pages[1]
        for row in page.rule_list.rows.values():
            row.set(False)
        try:
            problems = page.validate()
            assert problems and "règle" in problems[0].lower()
        finally:
            page.rule_list.select_all()

    def test_la_remise_a_zero_recoche_tout(self, app):
        page = app.pages[1]
        page.rule_list.rows["clash"].set(False)
        page.reset_defaults()
        assert page.rule_list.get() == set(page.rule_list.rows)

    def test_le_resume_alerte_quand_une_regle_bloquante_tombe(self, app):
        page = app.pages[1]
        page.rule_list.rows["clash"].set(False)
        page.rule_list._refresh_summary()
        try:
            text = page.rule_list.lbl_summary.cget("text").lower()
            assert "attention" in text
        finally:
            page.rule_list.select_all()

    def test_le_resume_est_serein_quand_tout_est_coche(self, app):
        page = app.pages[1]
        page.rule_list.select_all()
        text = page.rule_list.lbl_summary.cget("text").lower()
        assert "attention" not in text

    def test_une_ligne_decochee_est_grisee(self, app):
        from ui.theme import current

        page = app.pages[1]
        row = page.rule_list.rows["straightness"]
        row.set(False)
        try:
            assert row.lbl_name.cget("text_color") == current().TEXT_FAINT
        finally:
            page.rule_list.select_all()

    def test_les_lignes_survivent_au_changement_de_langue(self, app):
        page = app.pages[1]
        page.rule_list.update_language("EN")
        try:
            assert "rule" in page.rule_list.lbl_summary.cget("text").lower()
        finally:
            page.rule_list.update_language("FR")


class TestEnTeteEtBarreDEtat:
    """En-tête allégé et barre d'état lisible."""

    def test_la_baseline_a_disparu(self, app):
        """Le sous-titre sous « HarnessOpt » ne doit plus exister du tout."""
        assert not hasattr(app, "lbl_tagline")

    def test_la_cle_de_baseline_a_ete_retiree_du_catalogue(self):
        from ui.i18n import EN, FR

        assert "app.tagline" not in FR
        assert "app.tagline" not in EN

    def test_la_pastille_suit_la_gravite_du_message(self, app):
        from ui.theme import current

        theme = current()
        app.set_status("échec", "danger")
        assert app.status_dot.cget("fg_color") == theme.danger
        app.set_status("terminé", "ok")
        assert app.status_dot.cget("fg_color") == theme.ok

    def test_la_pastille_et_le_texte_disent_la_meme_chose(self, app):
        app.set_status("attention", "warn")
        assert app.status_dot.cget("fg_color") == app.lbl_status.cget("text_color")

    def test_un_ton_inconnu_reste_neutre(self, app):
        from ui.theme import current

        app.set_status("message", "ton_inexistant")
        assert app.status_dot.cget("fg_color") == current().TEXT_SOFT

    def test_l_etape_courante_est_rappelee(self, app):
        app.show_step(0)
        text = app.lbl_step.cget("text")
        assert "1/4" in text
        assert "projet" in text.lower()

    def test_le_rappel_d_etape_suit_l_ecran_reellement_affiche(self, app):
        """Il doit suivre l'écran affiché, pas celui qu'on a demandé.

        Demander une étape verrouillée ne change pas d'écran : le rappel doit
        alors rester sur l'étape courante, sinon il annoncerait une étape que
        l'utilisateur n'a pas sous les yeux.
        """
        app._display_step(1)
        assert "2/4" in app.lbl_step.cget("text")
        app._display_step(0)
        assert "1/4" in app.lbl_step.cget("text")

    def test_une_etape_verrouillee_ne_change_pas_le_rappel(self, app):
        app.show_step(0)
        app.show_step(2)  # cheminement : inaccessible sans maquette
        assert "1/4" in app.lbl_step.cget("text")

    def test_le_rappel_d_etape_se_traduit(self, app):
        app.set_language("EN")
        try:
            assert "Step" in app.lbl_step.cget("text")
        finally:
            app.set_language("FR")


class TestConseils:
    """Onglet « Conseils » de la page Cheminement."""

    def test_l_onglet_existe(self, app):
        assert "advice" in app.pages[2]._tab_names

    def test_aucun_conseil_affiche_une_explication(self, app):
        board = app.pages[2].advice_board
        board.clear()
        assert board.count() == 0
        assert "conseil" in board.lbl_empty.cget("text").lower()

    def test_un_conseil_produit_une_carte(self, app):
        from core.diagnostics import Suggestion

        board = app.pages[2].advice_board
        board.update_advice([Suggestion(
            key="k", severity="major",
            title_fr="Titre", title_en="Title",
            detail_fr="Constat", detail_en="Detail",
            action_fr="Action", action_en="Action",
            setting="min_margin", value=6.0,
        )])
        try:
            assert board.count() == 1
            assert len(board.cards_box.winfo_children()) == 1
        finally:
            board.clear()

    def test_un_conseil_sans_reglage_n_a_pas_de_bouton(self, app):
        from core.diagnostics import Suggestion
        from ui.widgets import AdviceCard

        page = app.pages[2]
        card = AdviceCard(page, Suggestion(
            key="clash", severity="blocking",
            title_fr="T", title_en="T", detail_fr="D", detail_en="D",
            action_fr="A", action_en="A",
        ), on_apply=lambda _s: None)
        try:
            assert card.btn_apply is None
        finally:
            card.destroy()

    def test_un_conseil_applicable_a_un_bouton(self, app):
        from core.diagnostics import Suggestion
        from ui.widgets import AdviceCard

        page = app.pages[2]
        card = AdviceCard(page, Suggestion(
            key="clearance_min", severity="blocking",
            title_fr="T", title_en="T", detail_fr="D", detail_en="D",
            action_fr="A", action_en="A", setting="min_margin", value=6.0,
        ), on_apply=lambda _s: None)
        try:
            assert card.btn_apply is not None
        finally:
            card.destroy()

    def test_appliquer_un_conseil_ecrit_dans_la_page_des_regles(self, app):
        from core.diagnostics import Suggestion

        rules_page = app.pages[1]
        before = rules_page.f_min.get(10.0)
        try:
            applied = app.controller.apply_suggestion(Suggestion(
                key="clearance_min", severity="blocking",
                title_fr="T", title_en="T", detail_fr="D", detail_en="D",
                action_fr="A", action_en="A", setting="min_margin", value=6.0,
            ))
            assert applied is True
            assert rules_page.f_min.get(0.0) == pytest.approx(6.0)
        finally:
            rules_page.f_min.set(before)

    def test_un_conseil_sans_reglage_n_est_pas_applique(self, app):
        from core.diagnostics import Suggestion

        assert app.controller.apply_suggestion(Suggestion(
            key="clash", severity="blocking",
            title_fr="T", title_en="T", detail_fr="D", detail_en="D",
            action_fr="A", action_en="A",
        )) is False

    def test_l_onglet_annonce_le_nombre_de_conseils(self, app):
        from core.diagnostics import Suggestion

        page = app.pages[2]
        page.advice_board.update_advice([
            Suggestion(key=f"k{i}", severity="info", title_fr="T", title_en="T",
                       detail_fr="D", detail_en="D", action_fr="A", action_en="A")
            for i in range(2)
        ])
        page._refresh_advice_tab()
        try:
            assert "(2)" in page._tab_names["advice"]
        finally:
            page.advice_board.clear()
            page._refresh_advice_tab()


class TestScanDeFixations:
    """Restitution du scan des fixations existantes."""

    @staticmethod
    def _result(n_passages=3):
        from core.fixation_scan import summarise

        points = []
        for i in range(n_passages):
            points.append([float(i * 20), 0.0, 0.0])
            points.append([float(i * 20), 50.0, 0.0])
        return summarise([{
            "name": "peigne.stl", "position": [0, 0, 0], "score": 0.9,
            "routing_points": points,
        }])

    def test_le_bandeau_est_cache_sans_scan(self, app):
        """On teste la présence dans la grille, pas l'affichage à l'écran.

        ``winfo_ismapped`` vaut toujours 0 sur une page non affichée : il
        rendrait le test vert sans rien vérifier.
        """
        page = app.pages[2]
        page.show_fixation_scan(self._result(1))
        assert page.scan_box.grid_info()
        page.show_fixation_scan(None)
        assert not page.scan_box.grid_info()

    def test_un_scan_reussi_annonce_le_compte(self, app):
        page = app.pages[2]
        page.show_fixation_scan(self._result(3))
        app.update()
        try:
            text = page.lbl_scan.cget("text")
            assert "1" in text and "3" in text
        finally:
            page.show_fixation_scan(None)

    def test_les_passages_sont_listes_avec_p_in_et_p_out(self, app):
        page = app.pages[2]
        page.show_fixation_scan(self._result(3))
        app.update()
        try:
            lignes = [w.cget("text") for w in page.scan_passages.winfo_children()]
            assert len(lignes) == 3
            assert all("p_in" in ligne and "p_out" in ligne for ligne in lignes)
        finally:
            page.show_fixation_scan(None)

    def test_une_longue_liste_est_tronquee_avec_mention(self, app):
        from ui.pages.routing_page import MAX_LISTED_PASSAGES

        page = app.pages[2]
        page.show_fixation_scan(self._result(MAX_LISTED_PASSAGES + 5))
        app.update()
        try:
            lignes = page.scan_passages.winfo_children()
            assert len(lignes) == MAX_LISTED_PASSAGES + 1
            assert "5" in lignes[-1].cget("text")
        finally:
            page.show_fixation_scan(None)

    def test_un_scan_impossible_est_explique(self, app):
        from core.fixation_scan import NO_OPEN3D, ScanResult

        page = app.pages[2]
        page.show_fixation_scan(ScanResult(skipped_reason=NO_OPEN3D))
        app.update()
        try:
            assert "Open3D" in page.lbl_scan.cget("text")
            assert page.scan_box.grid_info(), "le bandeau doit rester visible pour expliquer"
        finally:
            page.show_fixation_scan(None)

    def test_le_dossier_de_fixations_est_collecte(self, app):
        assert "clamps_folder" in app.pages[1].collect()


class TestPassageParLesFixations:
    """Interrupteur « emprunter les fixations existantes »."""

    def test_l_option_existe_et_est_active_par_defaut(self, app):
        assert app.pages[2].f_use_fixations.get() is True

    def test_l_option_est_collectee(self, app):
        assert "use_fixations" in app.pages[2].collect()

    def test_decocher_se_repercute_sur_la_collecte(self, app):
        page = app.pages[2]
        page.f_use_fixations.var.set(False)
        try:
            assert page.collect()["use_fixations"] is False
        finally:
            page.f_use_fixations.var.set(True)

    def test_le_controleur_lit_l_option(self, app):
        from core.fixation_scan import summarise

        page = app.pages[2]
        app.controller.scan_result = summarise([{
            "name": "p.stl", "position": [0, 0, 0], "score": 0.9,
            "routing_points": [[10, 0, 0], [10, 50, 0]],
        }])
        try:
            page.f_use_fixations.var.set(False)
            assert app.controller._passages_to_use(page.collect()) == []
            page.f_use_fixations.var.set(True)
            assert len(app.controller._passages_to_use(page.collect())) == 1
        finally:
            page.f_use_fixations.var.set(True)
            app.controller.scan_result = None


class TestAffichage3DDesFixations:
    """Les fixations et les crabes doivent être visibles, pas seulement comptés."""

    class _ViewerFactice:
        """Enregistre les acteurs, sans contexte 3D."""

        def __init__(self):
            self.actors = {}
            self.is_available = True
            self.renders = 0

        def show_sphere(self, center, name, **kwargs):
            self.actors[name] = ("sphere", kwargs.get("color"))

        def show_path(self, points, name, **kwargs):
            self.actors[name] = ("path", kwargs.get("color"))

        def remove_prefix(self, prefix):
            for name in [n for n in self.actors if n.startswith(prefix)]:
                del self.actors[name]

        def render(self):
            self.renders += 1

    @staticmethod
    def _scan(n_passages=2):
        from core.fixation_scan import summarise

        points = []
        for i in range(n_passages):
            points.append([float(i * 20), 0.0, 0.0])
            points.append([float(i * 20), 50.0, 0.0])
        return summarise([{"name": "peigne.stl", "position": [5.0, 5.0, 5.0],
                           "score": 0.9, "routing_points": points}])

    def test_les_fixations_scannees_sont_dessinees(self, app):
        viewer = self._ViewerFactice()
        app.controller.viewer = viewer
        try:
            app.controller._draw_fixations(self._scan(2))
            noms = set(viewer.actors)
            assert "fixation_body_0" in noms
            assert "fixation_in_0" in noms and "fixation_out_0" in noms
            assert "fixation_slot_0" in noms
            assert "fixation_in_1" in noms
        finally:
            app.controller.viewer = None

    def test_l_entree_et_la_sortie_ont_des_couleurs_distinctes(self, app):
        viewer = self._ViewerFactice()
        app.controller.viewer = viewer
        try:
            app.controller._draw_fixations(self._scan(1))
            assert viewer.actors["fixation_in_0"][1] != viewer.actors["fixation_out_0"][1]
        finally:
            app.controller.viewer = None

    def test_un_scan_non_effectue_efface_les_fixations(self, app):
        from core.fixation_scan import NO_OPEN3D, ScanResult

        viewer = self._ViewerFactice()
        app.controller.viewer = viewer
        try:
            app.controller._draw_fixations(self._scan(1))
            assert viewer.actors
            app.controller._draw_fixations(ScanResult(skipped_reason=NO_OPEN3D))
            assert not any(n.startswith("fixation_") for n in viewer.actors)
        finally:
            app.controller.viewer = None

    def test_les_crabes_poses_sont_dessines(self, app):
        viewer = self._ViewerFactice()
        app.controller.viewer = viewer
        app.controller._clamp_signature = ()
        try:
            app.controller._draw_clamps([
                {"arc_mm": 100.0, "tilt_deg": 2.0,
                 "position": np.array([0.0, 0.0, 0.0]),
                 "surface_position": np.array([0.0, 0.0, -20.0])},
            ])
            assert "clamp_0" in viewer.actors
            assert "clamp_leg_0" in viewer.actors
        finally:
            app.controller.viewer = None
            app.controller._clamp_signature = ()

    def test_un_crabe_sans_position_ne_casse_pas(self, app):
        viewer = self._ViewerFactice()
        app.controller.viewer = viewer
        app.controller._clamp_signature = ()
        try:
            app.controller._draw_clamps([{"arc_mm": 1.0, "tilt_deg": 0.0}])
            assert not any(n.startswith("clamp_") for n in viewer.actors)
        finally:
            app.controller.viewer = None
            app.controller._clamp_signature = ()

    def test_les_crabes_identiques_ne_sont_pas_redessines(self, app):
        """Redessiner quatre fois par seconde ferait clignoter la vue."""
        viewer = self._ViewerFactice()
        app.controller.viewer = viewer
        app.controller._clamp_signature = ()
        crabes = [{"arc_mm": 100.0, "tilt_deg": 2.0,
                   "position": np.array([0.0, 0.0, 0.0])}]
        try:
            app.controller._draw_clamps(crabes)
            avant = viewer.renders
            app.controller._draw_clamps(crabes)
            assert viewer.renders == avant
        finally:
            app.controller.viewer = None
            app.controller._clamp_signature = ()

    def test_le_bandeau_annonce_les_modeles_examines(self, app):
        from core.fixation_scan import summarise

        page = app.pages[2]
        result = summarise([{"name": "p.stl", "position": [0, 0, 0], "score": 0.9}],
                           scanned_files=17)
        page.show_fixation_scan(result)
        app.update()
        try:
            assert "17" in page.lbl_scan.cget("text")
        finally:
            page.show_fixation_scan(None)


class TestFenetreDeConfirmationDesFixations:
    """La question posée à l'utilisateur avant d'emprunter les fixations."""

    @staticmethod
    def _scan(n_passages=2, scanned=9):
        from core.fixation_scan import summarise

        points = []
        for i in range(n_passages):
            points.append([float(i * 100), -40.0, 0.0])
            points.append([float(i * 100), 40.0, 0.0])
        return summarise([{"name": "peigne.stl", "position": [0.0, 0.0, 0.0],
                           "score": 0.9, "routing_points": points}],
                         scanned_files=scanned)

    def test_la_question_montre_le_compte_et_les_passages(self, app, monkeypatch):
        vu = {}

        def faux_askyesno(title, message, **kwargs):
            vu["titre"] = title
            vu["message"] = message
            return True

        monkeypatch.setattr("ui.app_window.messagebox.askyesno", faux_askyesno)
        assert app.ask_use_fixations(self._scan(2, scanned=9)) is True
        assert "9" in vu["message"]
        assert "p_in" in vu["message"] and "p_out" in vu["message"]
        assert "n° 1" in vu["message"] and "n° 2" in vu["message"]

    def test_un_refus_est_rendu_tel_quel(self, app, monkeypatch):
        monkeypatch.setattr("ui.app_window.messagebox.askyesno",
                            lambda *a, **k: False)
        assert app.ask_use_fixations(self._scan(1)) is False

    def test_une_fenetre_en_echec_retombe_sur_le_defaut(self, app, monkeypatch):
        def explose(*_a, **_k):
            raise RuntimeError("pas d'affichage")

        monkeypatch.setattr("ui.app_window.messagebox.askyesno", explose)
        assert app.ask_use_fixations(self._scan(1), default=True) is True
        assert app.ask_use_fixations(self._scan(1), default=False) is False

    def test_la_question_n_est_pas_posee_sans_passage(self, app, monkeypatch):
        """Interrompre l'utilisateur pour rien serait gratuit."""
        from core.fixation_scan import ScanResult

        appels = []
        monkeypatch.setattr("ui.app_window.messagebox.askyesno",
                            lambda *a, **k: appels.append(1) or True)
        assert app.controller._ask_use_fixations(ScanResult(), default=False) is False
        assert app.controller._ask_use_fixations(None, default=True) is True
        assert appels == []

    def test_la_reponse_est_reportee_sur_la_page(self, app, monkeypatch):
        monkeypatch.setattr("ui.app_window.messagebox.askyesno",
                            lambda *a, **k: False)
        page = app.pages[2]
        page.f_use_fixations.var.set(True)
        try:
            reponse = app.controller._ask_use_fixations(self._scan(1), default=True)
            app.update()
            assert reponse is False
            assert page.f_use_fixations.get() is False
        finally:
            page.f_use_fixations.var.set(True)

    def test_la_page_expose_un_moyen_de_refleter_la_reponse(self, app):
        page = app.pages[2]
        try:
            page.set_use_fixations(False)
            assert page.f_use_fixations.get() is False
            page.set_use_fixations(True)
            assert page.f_use_fixations.get() is True
        finally:
            page.f_use_fixations.var.set(True)
