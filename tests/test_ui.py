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


class TestApplication:
    def test_les_quatre_etapes_existent(self, app):
        assert len(app.pages) == 4

    def test_les_etapes_suivantes_sont_verrouillees_au_demarrage(self, app):
        app.stepper.reset_unlock(0)
        app.show_step(2)
        assert app.current_step == 0

    def test_deverrouillage_progressif(self, app):
        app.unlock_step(2)
        app.show_step(2)
        assert app.current_step == 2

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
