"""Vérifications de la recherche du chemin de départ.

Le point central : le chemin initial doit naître **dans** la bande de distance
autorisée. L'ancienne géodésique le posait sur la surface, donc en violation de
toutes les règles de distance dès la première itération.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh requis pour la planification")

from core.geometry_metrics import path_length, straightness
from core.path_planner import (
    STRATEGIES,
    PlannerSettings,
    auto_voxel_size,
    plan_route,
)


def clearances(mesh, points):
    from trimesh.proximity import ProximityQuery

    _, distances, _ = ProximityQuery(mesh).on_surface(np.asarray(points, dtype=np.float64))
    return distances


@pytest.fixture(scope="module")
def dmu_disjoint():
    """Maquette faite de pièces séparées : le cas qui met la géodésique en échec.

    Aucune pièce ne relie les deux extrémités — c'est la situation normale d'un
    DMU fusionné, où chaque STL reste une coque indépendante.
    """
    parts = []
    for i in range(4):
        frame = trimesh.creation.box(extents=[40, 800, 600])
        frame.apply_translation([250 + i * 400, 400, 300])
        parts.append(frame)
        # Segments de plancher distincts, séparés par un jeu : rien ne traverse
        # la maquette d'un bout à l'autre.
        slab = trimesh.creation.box(extents=[240, 800, 30])
        slab.apply_translation([250 + i * 400, 400, 15])
        parts.append(slab)

    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    mesh.fix_normals()
    _ = mesh.face_normals
    _ = mesh.kdtree
    return mesh


@pytest.fixture(scope="module")
def endpoints():
    return np.array([120.0, 400.0, 250.0]), np.array([1700.0, 400.0, 250.0])


class TestResolutionAutomatique:
    def test_la_grille_suit_la_marge_visee(self):
        assert auto_voxel_size(20.0) == 20.0
        assert auto_voxel_size(30.0) == 30.0

    def test_la_grille_reste_calculable(self):
        assert auto_voxel_size(2.0) == 15.0     # plancher
        assert auto_voxel_size(500.0) == 60.0   # plafond


class TestRechercheReussie:
    def test_un_chemin_est_trouve_malgre_les_pieces_disjointes(self, dmu_disjoint, endpoints):
        """Régression : la géodésique échouait toujours sur ce cas."""
        start, goal = endpoints
        result = plan_route(dmu_disjoint, start, goal, 20.0, 150.0)
        assert result.success, result.message_fr
        assert len(result.points) >= 2

    def test_les_extremites_sont_respectees(self, dmu_disjoint, endpoints):
        start, goal = endpoints
        result = plan_route(dmu_disjoint, start, goal, 20.0, 150.0, num_points=40)
        assert np.allclose(result.points[0], start, atol=1e-3)
        assert np.allclose(result.points[-1], goal, atol=1e-3)

    def test_le_nombre_de_points_demande_est_respecte(self, dmu_disjoint, endpoints):
        start, goal = endpoints
        assert len(plan_route(dmu_disjoint, start, goal, 20.0, 150.0, num_points=48).points) == 48

    def test_le_chemin_nait_a_bonne_distance(self, dmu_disjoint, endpoints):
        """Le cœur du sujet : plus aucun point en contact avec la structure."""
        start, goal = endpoints
        result = plan_route(dmu_disjoint, start, goal, 20.0, 200.0, num_points=60)
        assert clearances(dmu_disjoint, result.points).min() >= 20.0

    def test_la_distance_reelle_est_mesuree_et_rapportee(self, dmu_disjoint, endpoints):
        """La grille n'est qu'une approximation : le rapport donne la vraie valeur."""
        start, goal = endpoints
        result = plan_route(dmu_disjoint, start, goal, 20.0, 200.0, num_points=40)
        measured = clearances(dmu_disjoint, result.points).min()
        assert result.stats["min_clearance_mm"] == pytest.approx(measured, abs=0.5)
        assert result.stats["clearance_violations"] == 0

    def test_le_detour_reste_raisonnable(self, dmu_disjoint, endpoints):
        start, goal = endpoints
        result = plan_route(dmu_disjoint, start, goal, 20.0, 200.0, num_points=60)
        assert result.stats["detour_ratio"] < 2.5


class TestStrategies:
    def test_les_trois_strategies_aboutissent(self, dmu_disjoint, endpoints):
        start, goal = endpoints
        for name in STRATEGIES:
            settings = PlannerSettings().with_strategy(name)
            assert plan_route(dmu_disjoint, start, goal, 20.0, 200.0, settings).success, name

    def test_le_glouton_explore_beaucoup_moins(self, dmu_disjoint, endpoints):
        start, goal = endpoints
        optimal = plan_route(
            dmu_disjoint, start, goal, 20.0, 200.0, PlannerSettings().with_strategy("optimal")
        )
        greedy = plan_route(
            dmu_disjoint, start, goal, 20.0, 200.0, PlannerSettings().with_strategy("greedy")
        )
        assert greedy.stats["expansions"] < optimal.stats["expansions"]

    def test_le_poids_croit_avec_la_gourmandise(self):
        assert (
            STRATEGIES["optimal"]["weight"]
            < STRATEGIES["balanced"]["weight"]
            < STRATEGIES["greedy"]["weight"]
        )

    def test_toutes_les_strategies_respectent_la_marge(self, dmu_disjoint, endpoints):
        start, goal = endpoints
        for name in STRATEGIES:
            settings = PlannerSettings().with_strategy(name)
            result = plan_route(dmu_disjoint, start, goal, 20.0, 200.0, settings, num_points=50)
            assert clearances(dmu_disjoint, result.points).min() >= 20.0, name


class TestPenaliteDeVirage:
    """La pénalité de virage vit dans le coût cumulé ``g``.

    Une recherche gloutonne ne classe que sur l'heuristique ``h`` : elle ne
    peut donc pas en tenir compte. C'est la raison de fond pour laquelle le
    réglage par défaut n'est pas glouton.
    """

    @staticmethod
    def _tortuosite(strategy, turn_penalty):
        floor = trimesh.creation.box(extents=[2600, 1400, 40])
        floor.apply_translation([1300, 700, 60])
        floor.merge_vertices()
        floor.fix_normals()
        _ = floor.face_normals
        _ = floor.kdtree

        settings = PlannerSettings(
            voxel_mm=40.0, turn_penalty=turn_penalty, band_penalty=0.0, shortcut=False
        ).with_strategy(strategy)
        result = plan_route(
            floor, np.array([150.0, 150.0, 200.0]), np.array([2350.0, 1200.0, 200.0]),
            20.0, 200.0, settings,
        )
        assert result.success
        return straightness(result.points)["total_turning_deg"]

    def test_la_penalite_redresse_la_recherche_ponderee(self):
        assert self._tortuosite("balanced", 6.0) < self._tortuosite("balanced", 0.0) / 2

    def test_le_glouton_reste_insensible_a_la_penalite(self):
        """Constat mesuré, pas une supposition : le glouton ignore ``g``."""
        assert self._tortuosite("greedy", 6.0) == pytest.approx(
            self._tortuosite("greedy", 0.0), rel=0.02
        )


class TestEchecsExplicites:
    def test_un_point_dans_la_matiere_est_rattrape(self, dmu_disjoint):
        """Une extrémité saisie dans la matière est ramenée au plus proche point valide.

        Refuser serait pénible : l'utilisateur saisit des coordonnées à la main
        et tombe volontiers de quelques millimètres dans une pièce. Le
        rattrapage est en revanche rapporté dans les statistiques.
        """
        inside = np.array([250.0, 400.0, 300.0])  # au cœur d'un cadre
        result = plan_route(dmu_disjoint, inside, np.array([1700.0, 400.0, 250.0]), 20.0, 200.0)
        assert result.success
        assert result.stats["start_shift_cells"] > 0

    def test_un_point_irrecuperable_est_refuse(self, dmu_disjoint):
        """Aucun repli silencieux : l'utilisateur doit savoir ce qui bloque."""
        inside = np.array([250.0, 400.0, 300.0])
        result = plan_route(
            dmu_disjoint, inside, np.array([1700.0, 400.0, 250.0]), 400.0, 900.0
        )
        assert not result.success
        assert result.points is None
        assert "matière" in result.message_fr or "trop près" in result.message_fr

    def test_une_marge_irrealisable_est_signalee(self, dmu_disjoint, endpoints):
        start, goal = endpoints
        result = plan_route(dmu_disjoint, start, goal, 5000.0, 6000.0)
        assert not result.success
        assert result.message_fr

    def test_les_messages_existent_dans_les_deux_langues(self, dmu_disjoint, endpoints):
        start, goal = endpoints
        result = plan_route(dmu_disjoint, start, goal, 20.0, 200.0, num_points=20)
        assert result.message("FR") and result.message("EN")
        assert result.message("FR") != result.message("EN")


class TestRaccourcis:
    def test_les_raccourcis_reduisent_le_nombre_de_sommets(self, dmu_disjoint, endpoints):
        start, goal = endpoints
        avec = plan_route(
            dmu_disjoint, start, goal, 20.0, 200.0, PlannerSettings(shortcut=True)
        )
        sans = plan_route(
            dmu_disjoint, start, goal, 20.0, 200.0, PlannerSettings(shortcut=False)
        )
        assert avec.stats["n_points_raw"] < sans.stats["n_points_raw"]

    def test_les_raccourcis_preservent_la_marge(self, dmu_disjoint, endpoints):
        start, goal = endpoints
        result = plan_route(
            dmu_disjoint, start, goal, 20.0, 200.0, PlannerSettings(shortcut=True), num_points=60
        )
        assert clearances(dmu_disjoint, result.points).min() >= 20.0


class TestComparaisonAvecLaGeodesique:
    def test_la_geodesique_previent_de_son_echec(self, dmu_disjoint, endpoints):
        """Le repli sur une ligne droite doit être annoncé, plus subi."""
        from core.agent.tool import generate_dense_waypoints

        start, goal = endpoints
        warnings: list[str] = []
        generate_dense_waypoints(start, goal, 40, mesh=dmu_disjoint, on_warning=warnings.append)
        assert warnings, "l'échec géodésique est resté silencieux"
        assert "disjointes" in warnings[0]

    def test_la_recherche_dans_lespace_libre_fait_mieux(self, dmu_disjoint, endpoints):
        """Comparaison chiffrée, sur le seul critère qui compte au départ."""
        from core.agent.tool import generate_dense_waypoints

        start, goal = endpoints
        geodesic = generate_dense_waypoints(start, goal, 60, mesh=dmu_disjoint)
        planned = plan_route(dmu_disjoint, start, goal, 20.0, 200.0, num_points=60).points

        violations_geo = int(np.count_nonzero(clearances(dmu_disjoint, geodesic) < 20.0))
        violations_plan = int(np.count_nonzero(clearances(dmu_disjoint, planned) < 20.0))
        assert violations_plan < violations_geo
        assert violations_plan == 0


class TestIntegrationControleur:
    """Le contrôleur applique-t-il la stratégie choisie, et rend-il compte ?"""

    class _Traducteur:
        lang = "FR"

        def __call__(self, key, **kwargs):
            return key

    class _Vue:
        def __init__(self):
            self.t = TestIntegrationControleur._Traducteur()
            self.messages = []

        def after(self, _delay, callback):
            callback()

        def set_status(self, message, tone="neutral"):
            self.messages.append((tone, message))

    @staticmethod
    def _controleur():
        from controller.app_controller import AppController

        view = TestIntegrationControleur._Vue()
        controller = AppController(view)
        controller.point_a = np.array([120.0, 400.0, 250.0], dtype=np.float32)
        controller.point_b = np.array([1700.0, 400.0, 250.0], dtype=np.float32)
        return controller, view

    @staticmethod
    def _regles():
        from core.routing_rules import ClearanceModel, HarnessSpec, RoutingRules

        return RoutingRules(
            harness=HarnessSpec(diameter_mm=20.0),
            clearance=ClearanceModel(default_min_mm=20.0, max_mm=200.0),
        )

    @pytest.mark.parametrize("strategy", ["optimal", "balanced", "greedy"])
    def test_chaque_strategie_produit_un_chemin_conforme(self, dmu_disjoint, strategy):
        controller, view = self._controleur()
        points = controller._build_initial_path(
            dmu_disjoint, self._regles(), {"start_path": strategy, "initial_points": 48}
        )
        assert points is not None
        assert len(points) == 48
        assert clearances(dmu_disjoint, points).min() >= 20.0
        assert view.messages[-1][0] == "ok"

    def test_la_geodesique_suit_la_surface_malgre_les_pieces_disjointes(self, dmu_disjoint):
        """Le cas qui échouait : ``pv.geodesic`` rendait une corde tendue.

        Elle exige un chemin d'arêtes, dont une maquette faite de pièces
        séparées ne dispose jamais. Le chemin de surface franchit les
        intervalles par des sauts courts et longe la structure le reste du
        temps : sa forme épouse le DMU, là où une corde l'ignore.
        """
        controller, view = self._controleur()
        points = controller._build_initial_path(
            dmu_disjoint, self._regles(), {"start_path": "geodesic", "initial_points": 48}
        )
        assert points is not None
        assert view.messages[-1][0] == "ok"

        # Il n'est pas une ligne droite : c'est bien la surface qu'il suit.
        droite = np.linspace(points[0], points[-1], len(points))
        assert np.abs(points - droite).max() > 1.0

    def test_la_geodesique_longe_la_structure_sans_la_toucher(self, dmu_disjoint):
        """Suivre la surface n'est pas la raser.

        Un tracé posé *sur* le maillage est en interférence sur toute sa
        longueur : les agents démarrent alors en clash et passent leurs
        premières itérations à s'en extraire. Le chemin de surface est donc
        décollé de la structure, tout en gardant sa forme.
        """
        controller, view = self._controleur()
        regles = self._regles()
        points = controller._build_initial_path(
            dmu_disjoint, regles, {"start_path": "geodesic", "initial_points": 48}
        )
        assert clearances(dmu_disjoint, points).min() >= regles.clearance.default_min_mm

    def test_la_geodesique_respecte_la_distance_la_ou_une_corde_la_viole(self, dmu_disjoint):
        """Critère décisif : la corde traverse la structure, pas le tracé."""
        controller, view = self._controleur()
        regles = self._regles()
        points = controller._build_initial_path(
            dmu_disjoint, regles, {"start_path": "geodesic", "initial_points": 48}
        )
        corde = np.linspace(points[0], points[-1], len(points))
        mini = regles.clearance.default_min_mm
        assert clearances(dmu_disjoint, corde).min() < mini, \
            "la corde tendue doit bien passer trop près : sinon le test ne prouve rien"
        assert clearances(dmu_disjoint, points).min() >= mini

    def test_un_echec_interrompt_le_lancement(self, dmu_disjoint):
        """Le contrôleur ne doit pas démarrer les agents sur un chemin absent."""
        from core.routing_rules import ClearanceModel, RoutingRules

        controller, view = self._controleur()
        impossible = RoutingRules(clearance=ClearanceModel(default_min_mm=5000.0, max_mm=6000.0))
        points = controller._build_initial_path(
            dmu_disjoint, impossible, {"start_path": "balanced", "initial_points": 48}
        )
        assert points is None
        assert view.messages[-1][0] == "danger"

    def test_le_controleur_simporte_sans_environnement_graphique(self):
        """La logique doit rester vérifiable sans écran : tkinter est importé tardivement."""
        import controller.app_controller as module

        assert "filedialog" not in dir(module)


class TestPassageParLesFixations:
    """Le chemin de départ doit réellement traverser les encoches détectées.

    « Passer près d'une fixation » ne suffit pas : un peigne impose au câble
    d'entrer par une face et de ressortir par l'autre. Le trajet est donc
    découpé en tronçons par ces couples entrée/sortie, et la traversée de
    l'encoche elle-même est une ligne droite — y lancer une recherche de chemin
    ferait contourner le peigne au lieu de passer dedans.
    """

    @staticmethod
    def _controleur_avec_passages(passages, name="peigne.stl"):
        """Un seul peigne, portant toutes ces encoches."""
        return TestPassageParLesFixations._controleur_avec_peignes(
            {name: passages}
        )

    @staticmethod
    def _controleur_avec_peignes(peignes):
        """Plusieurs peignes, chacun avec ses propres encoches.

        La distinction est le cœur du sujet : deux encoches d'un même peigne
        sont côte à côte et destinées à des faisceaux différents ; deux peignes
        sont deux étapes du trajet.
        """
        from core.fixation_scan import summarise

        controller, view = TestIntegrationControleur._controleur()
        controller.scan_result = summarise([
            {"name": name, "position": list(np.mean(np.asarray(points), axis=0)),
             "score": 0.9, "routing_points": points}
            for name, points in peignes.items()
        ])
        return controller, view

    @staticmethod
    def _valeurs(**overrides):
        values = {"start_path": "balanced", "initial_points": 60, "use_fixations": True}
        values.update(overrides)
        return values

    def test_sans_fixations_le_trajet_a_deux_etapes(self):
        controller, _ = TestIntegrationControleur._controleur()
        nodes, forced = controller._route_nodes(self._valeurs())
        assert len(nodes) == 2
        assert forced == set()

    def test_une_encoche_ajoute_son_entree_et_sa_sortie(self):
        controller, _ = self._controleur_avec_passages(
            [[900.0, 350.0, 250.0], [900.0, 450.0, 250.0]]
        )
        nodes, forced = controller._route_nodes(self._valeurs())
        assert len(nodes) == 4          # A, p_in, p_out, B
        assert forced == {1}            # le tronçon p_in -> p_out reste droit

    def test_les_peignes_sont_ordonnes_le_long_du_trajet(self):
        """Deux peignes pris à l'envers feraient revenir le câble en arrière."""
        controller, _ = self._controleur_avec_peignes({
            "loin.stl": [[1400.0, 350.0, 250.0], [1400.0, 450.0, 250.0]],
            "proche.stl": [[600.0, 350.0, 250.0], [600.0, 450.0, 250.0]],
        })
        nodes, _ = controller._route_nodes(self._valeurs())
        abscisses = [float(n[0]) for n in nodes]
        assert abscisses == sorted(abscisses)

    def test_un_seul_couple_est_retenu_par_peigne(self):
        """Le défaut signalé : le câble faisait la navette sur un même peigne.

        Un peigne porte une encoche par faisceau ; celui-ci n'en emprunte
        qu'une. Enfilées à la file, les encoches d'un même peigne se suivaient
        dans le trajet — sortir d'une encoche pour entrer aussitôt dans sa
        voisine, au lieu de rejoindre le peigne suivant.
        """
        controller, _ = self._controleur_avec_passages([
            [900.0, 300.0, 250.0], [900.0, 340.0, 250.0],
            [900.0, 380.0, 250.0], [900.0, 420.0, 250.0],
            [900.0, 460.0, 250.0], [900.0, 500.0, 250.0],
        ])
        nodes, forced = controller._route_nodes(self._valeurs())
        assert len(nodes) == 4, "A, une entrée, une sortie, B"
        assert forced == {1}

    def test_chaque_peigne_apporte_exactement_une_traversee(self):
        controller, _ = self._controleur_avec_peignes({
            "avant.stl": [[600.0, 300.0, 250.0], [600.0, 340.0, 250.0],
                          [600.0, 380.0, 250.0], [600.0, 420.0, 250.0]],
            "arriere.stl": [[1400.0, 300.0, 250.0], [1400.0, 340.0, 250.0],
                            [1400.0, 380.0, 250.0], [1400.0, 420.0, 250.0]],
        })
        nodes, forced = controller._route_nodes(self._valeurs())
        assert len(nodes) == 6, "A, deux couples, B"
        assert forced == {1, 3}

    def test_le_sens_de_traversee_suit_le_trajet(self):
        """p_in et p_out sont interchangeables : seul leur couple est imposé."""
        # Encoche décrite à rebours du sens de marche (A vers +x).
        controller, _ = self._controleur_avec_passages(
            [[950.0, 400.0, 250.0], [850.0, 400.0, 250.0]]
        )
        nodes, _ = controller._route_nodes(self._valeurs())
        assert float(nodes[1][0]) < float(nodes[2][0])

    def test_l_entree_et_la_sortie_viennent_de_la_meme_encoche(self):
        """Entrer dans une encoche et ressortir par une autre est impossible."""
        controller, _ = self._controleur_avec_passages([
            [900.0, 300.0, 250.0], [900.0, 340.0, 250.0],
            [900.0, 460.0, 250.0], [900.0, 500.0, 250.0],
        ])
        crossings = controller._crossings_to_use(self._valeurs())
        assert len(crossings) == 1
        couple = {tuple(crossings[0].passage.p_in), tuple(crossings[0].passage.p_out)}
        assert {tuple(crossings[0].entry), tuple(crossings[0].exit)} == couple

    def test_la_traversee_ne_consomme_que_deux_points_du_cable(self, dmu_disjoint):
        """Une encoche de 5 cm n'a pas à recevoir la part de budget d'un mètre."""
        controller, _ = self._controleur_avec_passages(
            [[900.0, 380.0, 250.0], [900.0, 420.0, 250.0]]
        )
        points = controller._build_initial_path(
            dmu_disjoint, TestIntegrationControleur._regles(),
            self._valeurs(start_path="balanced", initial_points=48),
        )
        assert points is not None
        proches = [
            index for index, point in enumerate(points)
            if 370.0 <= float(point[1]) <= 430.0 and abs(float(point[0]) - 900.0) < 1.0
        ]
        assert len(proches) == 2, "l'entrée et la sortie, et rien entre les deux"
        assert proches[1] == proches[0] + 1

    def test_decocher_l_option_ignore_les_fixations(self):
        controller, _ = self._controleur_avec_passages(
            [[900.0, 350.0, 250.0], [900.0, 450.0, 250.0]]
        )
        nodes, forced = controller._route_nodes(self._valeurs(use_fixations=False))
        assert len(nodes) == 2
        assert forced == set()

    def test_un_scan_non_effectue_n_impose_rien(self):
        from core.fixation_scan import NO_OPEN3D, ScanResult

        controller, _ = TestIntegrationControleur._controleur()
        controller.scan_result = ScanResult(skipped_reason=NO_OPEN3D)
        nodes, _ = controller._route_nodes(self._valeurs())
        assert len(nodes) == 2

    def test_le_chemin_passe_reellement_par_l_encoche(self, dmu_disjoint):
        """Le critère qui compte : les deux points de l'encoche sont sur le tracé."""
        p_in = [900.0, 380.0, 250.0]
        p_out = [900.0, 420.0, 250.0]
        controller, view = self._controleur_avec_passages([p_in, p_out])

        points = controller._build_initial_path(
            dmu_disjoint, TestIntegrationControleur._regles(), self._valeurs()
        )
        assert points is not None
        for cible in (p_in, p_out):
            ecart = np.linalg.norm(points - np.array(cible), axis=1).min()
            assert ecart < 1.0, f"le tracé ne passe pas par {cible} (écart {ecart:.1f} mm)"

    def test_la_traversee_de_l_encoche_est_droite(self, dmu_disjoint):
        p_in = np.array([900.0, 380.0, 250.0])
        p_out = np.array([900.0, 420.0, 250.0])
        controller, _ = self._controleur_avec_passages([p_in.tolist(), p_out.tolist()])

        points = controller._build_initial_path(
            dmu_disjoint, TestIntegrationControleur._regles(), self._valeurs()
        )
        i_in = int(np.linalg.norm(points - p_in, axis=1).argmin())
        i_out = int(np.linalg.norm(points - p_out, axis=1).argmin())
        traversee = points[min(i_in, i_out):max(i_in, i_out) + 1]

        droite = np.linspace(traversee[0], traversee[-1], len(traversee))
        assert np.abs(traversee - droite).max() < 1e-3

    def test_le_chemin_reste_utilisable_avec_la_geodesique(self, dmu_disjoint):
        controller, _ = self._controleur_avec_passages(
            [[900.0, 380.0, 250.0], [900.0, 420.0, 250.0]]
        )
        points = controller._build_initial_path(
            dmu_disjoint, TestIntegrationControleur._regles(),
            self._valeurs(start_path="geodesic"),
        )
        assert points is not None
        assert len(points) >= 10
