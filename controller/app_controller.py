"""Contrôleur : relie l'interface au moteur de cheminement.

Responsabilités :

* piloter l'extraction du DMU (processus séparé) ;
* construire le jeu de règles à partir de ce que l'utilisateur a saisi ;
* lancer, mettre en pause et réinitialiser l'équipe d'agents ;
* rapatrier périodiquement l'état partagé vers l'interface ;
* exporter le résultat.

L'interface n'importe jamais ni PyTorch ni trimesh : ces dépendances lourdes
sont chargées ici, à la demande, ce qui garde le démarrage de l'application
rapide et permet d'afficher un message clair si l'une d'elles manque. Pour la
même raison, tkinter n'est importé que dans les fonctions d'export qui ouvrent
une boîte de dialogue : la logique du contrôleur reste ainsi vérifiable sans
environnement graphique.
"""

from __future__ import annotations

import json
import os
import threading
import time
from multiprocessing import Process, Queue

import numpy as np

from core import paths
from core import diagnostics, fixation_scan
from core.orchestrator import ROLES, Orchestrator
from core.routing_rules import ALL_RULES, ClearanceModel, HarnessSpec, RoutingRules

__all__ = ["AppController"]

#: Au-delà, dessiner chaque crabe avec sa géométrie complète coûterait plus
#: cher que ce que la lecture y gagne.
MAX_DRAWN_CLAMPS = 60

#: Attente maximale d'une réponse de l'utilisateur, en secondes. Passé ce
#: délai, le lancement reprend avec le réglage déjà présent sur la page plutôt
#: que de rester bloqué sur une fenêtre que personne ne regarde.
ASK_TIMEOUT_S = 300

#: Cadence de rafraîchissement de l'interface pendant le calcul, en ms.
REFRESH_RUNNING_MS = 250
REFRESH_IDLE_MS = 700


class AppController:
    def __init__(self, view):
        self.view = view
        self.t = view.t

        self.extraction_process: Process | None = None
        self.extraction_summary: dict = {}

        self.mesh = None
        self.viewer = None
        self.charts = None
        self.rules: RoutingRules | None = None
        self.orchestrator: Orchestrator | None = None
        self.supervisor = None

        self.shared_state: dict | None = None
        self.data_lock = threading.Lock()
        self.benchmark_algos: dict = {}
        self.point_a = None
        self.point_b = None

        self.is_scanning = False
        self.threads_started = False
        self._refresh_job = None
        self._detached = False
        #: Dernier état dessiné par agent, pour ne pas redessiner l'identique.
        self._path_signatures: dict[str, tuple] = {}
        #: Dernier jeu de crabes dessiné, pour ne pas redessiner l'identique.
        self._clamp_signature: tuple = ()
        #: Modèle STL du crabe, pour le dessiner tel qu'il est réellement posé.
        self._crabe_stl_path: str = ""
        #: Scores successifs du meilleur agent, pour détecter la stagnation.
        self._best_scores: list = []
        #: Le modèle de crabe a-t-il pu être chargé ? Renseigné au lancement.
        self._clamp_model_ok = True
        #: Résultat du dernier scan de fixations existantes.
        self.scan_result = None

    # ==================================================================
    # Étape 1 : extraction
    # ==================================================================

    def start_extraction(self, params: dict):
        """Lance l'extraction dans un processus séparé."""
        self.view.set_status(self.t("project.running"), "info")

        def run():
            queue = Queue()
            from core.mesh_processor import extraction_worker

            self.extraction_process = Process(
                target=extraction_worker,
                args=(
                    params["stl_folder"],
                    not params["use_existing"],
                    params["exclude_filter"],
                    queue,
                ),
            )
            self.extraction_process.start()

            while True:
                try:
                    message = queue.get(timeout=0.2)
                except Exception:
                    if self.extraction_process and not self.extraction_process.is_alive():
                        self._post(
                            self.view.pages[0].show_failure,
                            "Le chargement s'est interrompu sans résultat.",
                        )
                        break
                    continue

                kind = message[0]
                if kind == "UPDATE":
                    _, text, current, total, pct = message
                    detail = f"{text}\n{current}/{total}" if total else text
                    self._post(self.view.pages[0].update_progress, detail, pct)
                elif kind == "SUCCESS":
                    _, df, n_points, n_parts, bounds, fusion_path = message
                    self._on_extraction_done(df, n_points, n_parts, bounds, fusion_path, params)
                    break
                elif kind == "ERROR":
                    self._post(self.view.pages[0].show_failure, message[1])
                    self._post(self.view.set_status, message[1].split("\n")[0], "danger")
                    break

            if self.extraction_process:
                self.extraction_process.join(timeout=5)
            self.extraction_process = None

        threading.Thread(target=run, daemon=True).start()

    def _on_extraction_done(self, df, n_points, n_parts, bounds, fusion_path, params):
        families: dict[str, int] = {}
        if df is not None and "Color" in df:
            families = df["Color"].fillna("standard").astype(str).value_counts().to_dict()

        self.extraction_summary = {
            "n_parts": n_parts,
            "n_points": n_points,
            "n_cells": self._count_cells(fusion_path),
            "bounds": bounds,
            "families": families,
            "fusion_path": fusion_path,
            "clamps_folder": params.get("clamps_folder", ""),
        }

        self._post(self.view.pages[0].show_results, self.extraction_summary)
        self._post(self.view.pages[1].show_families, sorted(families))
        self._post(self.view.refresh_steps)
        self._post(self.view.refresh_cache_label)
        self._post(
            self.view.set_status,
            f"{self.t('project.done')} — {n_parts} pièces",
            "ok",
        )
        self._post(self._suggest_endpoints, bounds)

    @staticmethod
    def _count_cells(fusion_path) -> int:
        try:
            import pyvista as pv

            return int(pv.read(str(fusion_path)).n_cells)
        except Exception:
            return 0

    def _suggest_endpoints(self, bounds):
        """Pré-remplit des extrémités plausibles si l'utilisateur n'a rien saisi.

        Deux points aux extrémités de la plus grande dimension de la maquette :
        cela ne remplace pas un choix éclairé, mais évite de démarrer sur des
        coordonnées nulles qui n'ont aucun sens dans le repère avion.
        """
        page = self.view.pages[2]
        if any(page.f_source.get()) or any(page.f_target.get()):
            return

        x0, x1, y0, y1, z0, z1 = bounds
        center = ((y0 + y1) / 2, (z0 + z1) / 2)
        margin = (x1 - x0) * 0.1
        page.f_source.set((x0 + margin, center[0], center[1]))
        page.f_target.set((x1 - margin, center[0], center[1]))

    def cancel_extraction(self):
        if self.extraction_process and self.extraction_process.is_alive():
            self.extraction_process.terminate()
            self.view.pages[0].show_failure("Chargement interrompu.")
            self.view.set_status("Chargement interrompu.", "warn")

    # ==================================================================
    # Étape 2 : règles
    # ==================================================================

    def max_reachable_step(self) -> int:
        """Étape la plus avancée actuellement atteignable.

        Cette fonction est la seule autorité sur le verrouillage : l'interface
        s'y réfère au lieu de mémoriser ce qu'elle a déjà ouvert.
        """
        if not self.extraction_summary:
            return 0
        if self.view.pages[1].validate():
            return 1
        if self.shared_state is None:
            return 2
        return 3

    def locked_reason(self, index: int) -> str:
        """Ce qui manque pour atteindre une étape, dit en une phrase."""
        if index >= 1 and not self.extraction_summary:
            return self.t("step.locked.project")
        if index >= 2:
            problems = self.view.pages[1].validate()
            if problems:
                return problems[0]
        if index >= 3 and self.shared_state is None:
            return self.t("step.locked.routing")
        return ""

    def on_step_shown(self, index: int):
        """Appelé à chaque changement d'étape."""
        if index == 1 and self.extraction_summary:
            self.view.pages[1].show_families(sorted(self.extraction_summary.get("families", {})))
        elif index == 3:
            self._publish_reports()

    def build_rules(self) -> RoutingRules:
        """Construit le jeu de règles à partir de l'étape 2."""
        values = self.view.pages[1].collect()

        clearance = ClearanceModel(
            default_min_mm=values["min_margin"],
            max_mm=values["max_margin"],
        )
        clearance.per_family.update(values.get("family_clearance", {}))

        face_family, family_names = self._load_face_families()
        if face_family is not None:
            clearance = clearance.with_face_families(face_family, family_names)
            clearance.per_family.update(values.get("family_clearance", {}))
            clearance._build_table()

        self.rules = RoutingRules(
            harness=HarnessSpec(
                diameter_mm=values["harness_diameter"],
                bend_radius_factor=values["bend_radius_factor"],
            ),
            clearance=clearance,
            fixation_pitch_mm=values["fixation_pitch"],
            fixation_parallel_tol_deg=values["fixation_parallel_tol"],
            enabled_rules=frozenset(values.get("enabled_rules", ALL_RULES)),
        )
        return self.rules

    @staticmethod
    def _load_face_families():
        """Relit la table « face -> famille » produite par la fusion."""
        try:
            if not paths.FACE_FAMILY_PATH.exists():
                return None, []
            data = np.load(str(paths.FACE_FAMILY_PATH), allow_pickle=True)
            return data["face_family"], [str(n) for n in data["family_names"]]
        except Exception as exc:
            print(f"⚠️ Table des familles illisible ({exc}) : distance uniforme appliquée.")
            return None, []

    # ==================================================================
    # Étape 3 : cheminement
    # ==================================================================

    def toggle_routing(self):
        """Démarre, met en pause ou reprend le cheminement."""
        page = self.view.pages[2]

        if self.is_scanning:
            self.view.set_status(self.t("routing.scanning"), "info")
            return

        problems = page.validate()
        if problems:
            self.view.set_status(problems[0], "warn")
            return

        if not self.threads_started:
            self._start_routing()
            return

        with self.data_lock:
            self.shared_state["is_playing"] = not self.shared_state["is_playing"]
            running = self.shared_state["is_playing"]
        page.set_running_state("running" if running else "paused")

    def _start_routing(self):
        page = self.view.pages[2]
        values = page.collect()

        self.view.remember(
            point_a=list(values["point_a"]),
            point_b=list(values["point_b"]),
            temperature=values["temperature"],
            team_preset=values["team_preset"],
            start_path=values["start_path"],
            initial_points=values["initial_points"],
            max_points=values["max_points"],
            max_step_mm=values["max_step_mm"],
            voxel_mm=values["voxel_mm"],
            iterations=values["iterations"],
        )

        page.set_running_state("scanning")
        self.is_scanning = True
        self.view.set_status("Préparation de l'environnement…", "info")
        threading.Thread(target=self._prepare_and_launch, args=(values,), daemon=True).start()

    def _prepare_and_launch(self, values: dict):
        """Charge la maquette, prépare l'équipe puis lance les agents."""
        try:
            from core.agent.config import CONFIG
            from core.agent_team import TeamSupervisor, build_benchmark_algos
            from core.agent_worker import algo_worker

            rules = self.build_rules()
            rule_values = self.view.pages[1].collect()

            mesh = self._load_mesh()
            if mesh is None:
                self._post(
                    self.view.set_status,
                    "Aucune maquette chargée : revenez à l'étape 1.",
                    "danger",
                )
                self.is_scanning = False
                self._post(self.view.pages[2].set_running_state, "idle")
                return

            self.mesh = mesh
            self.point_a = np.asarray(values["point_a"], dtype=np.float32)
            self.point_b = np.asarray(values["point_b"], dtype=np.float32)

            config = dict(CONFIG)
            config.update(
                {
                    "initial_points": values["initial_points"],
                    "max_points": values["max_points"],
                    "max_step_mm": values["max_step_mm"],
                    "iterations": values["iterations"],
                    "min_margin": rules.clearance.default_min_mm,
                    "max_margin": rules.clearance.max_mm,
                    "harness_diameter": rules.harness.diameter_mm,
                    "bend_radius_factor": rules.harness.bend_radius_factor,
                    "min_bend_radius": rules.harness.min_bend_radius_mm,
                    "fixation_pitch_mm": rules.fixation_pitch_mm,
                    "crabe_min_spacing": rules.fixation_pitch_mm,
                    "crabe_stl_path": rule_values.get("crabe_stl_path", ""),
                    "tube_radius": rules.harness.radius_mm,
                    "existing_crabes": [],
                }
            )

            # Le modèle de crabe est vérifié une fois, ici : s'il est
            # illisible, aucune règle de fixation ne pourra jamais passer, et
            # mieux vaut le dire tout de suite que laisser tourner les agents.
            self._crabe_stl_path = config.get("crabe_stl_path", "")
            self._clamp_model_ok = self._check_clamp_model(self._crabe_stl_path)
            self._best_scores.clear()

            # Analyse des fixations déjà montées, avant tout cheminement : les
            # agents doivent partir de ce qui existe plutôt que d'en reposer
            # par-dessus. Un scan impossible n'interrompt jamais le lancement.
            self._post(self.view.set_status, self.t("routing.prepare.scan"), "info")
            # On passe la maquette déjà chargée, pas seulement son chemin : la
            # fusion est enregistrée en ``.vtk``, un format que le détecteur ne
            # sait pas relire — il rendrait un maillage vide sans rien dire, et
            # ne trouverait donc jamais aucune fixation.
            scan_result = fixation_scan.scan(
                self.extraction_summary.get("fusion_path") or paths.FUSED_MESH_PATH,
                rule_values.get("clamps_folder", ""),
                mesh=mesh,
            )
            self.scan_result = scan_result
            self._post(self.view.pages[2].show_fixation_scan, scan_result)
            self._post(self._draw_fixations, scan_result)

            # Le choix de l'utilisateur commande tout : s'il refuse d'emprunter
            # les fixations, les agents ne doivent ni y être attirés ni y être
            # retenus. Transmettre la liste « au cas où » reviendrait à lui
            # imposer un passage qu'il vient de refuser.
            # La question est posée à l'utilisateur, pas déduite : emprunter une
            # fixation contraint le tracé à passer là, ce qui peut le rallonger
            # ou l'écarter du meilleur passage. Il doit trancher en connaissance
            # de cause, et le voir sur la vue 3D avant de répondre.
            values["use_fixations"] = self._ask_use_fixations(
                scan_result, bool(values.get("use_fixations", True))
            )
            use_fixations = bool(values["use_fixations"])
            passages = self._passages_to_use(values)

            config["existing_crabes"] = [
                {
                    "name": f.name,
                    "position": list(f.position),
                    "score": f.score,
                    "routing_points": [
                        list(point)
                        for passage in f.passages
                        for point in (passage.p_in, passage.p_out)
                    ],
                }
                for f in scan_result.fixations
            ] if use_fixations else []

            # Points que le câble doit traverser, épinglés par les agents à
            # chaque itération. L'attraction par récompense ne suffit pas :
            # elle rapproche le câble de l'encoche sans garantir qu'il y passe.
            config["mandatory_points"] = [
                list(point)
                for passage in passages
                for point in (passage.p_in, passage.p_out)
            ]

            self.shared_state = {
                "is_playing": False,
                "is_running": True,
                "algos": {},
                "config": config,
            }

            self.orchestrator = Orchestrator(
                values["team_preset"], temperature=values["temperature"]
            )
            self.benchmark_algos = build_benchmark_algos(
                self.orchestrator.team, max_points=values["max_points"]
            )

            # La maquette est affichée dès maintenant : l'utilisateur voit ce
            # sur quoi il travaille pendant que le reste se prépare, au lieu
            # d'attendre devant une fenêtre inerte.
            self._post(self._setup_viewer)

            self._post(self.view.set_status, self.t("routing.prepare.path"), "info")
            waypoints = self._build_initial_path(mesh, rules, values)
            if waypoints is None:
                self.is_scanning = False
                self._post(self.view.pages[2].set_running_state, "idle")
                return

            specs = {spec.name: spec for spec in self.orchestrator.team}
            with self.data_lock:
                for name in self.benchmark_algos:
                    self.shared_state["algos"][name] = self._blank_agent_state(waypoints)

            self._post(self.view.set_status, self.t("routing.prepare.mesh"), "info")
            private_meshes = self._prepare_agent_meshes(mesh, len(self.benchmark_algos))

            for name, private in zip(self.benchmark_algos, private_meshes):
                threading.Thread(
                    target=algo_worker,
                    args=(
                        name, self.benchmark_algos, waypoints, private,
                        self.shared_state, self.data_lock, None,
                        self.point_a, self.point_b,
                    ),
                    kwargs={"rules": rules, "spec": specs[name]},
                    daemon=True,
                ).start()

            self.supervisor = TeamSupervisor(
                self.orchestrator, self.shared_state, self.data_lock,
                period_s=1.0, lang=self.view.t.lang,
            )
            self.supervisor.start()

            self.threads_started = True
            self.is_scanning = False

            with self.data_lock:
                self.shared_state["is_playing"] = True

            self._post(self._setup_charts)
            self._post(self.view.pages[2].set_running_state, "running")
            self._post(self.view.refresh_steps)
            self._post(self.view.set_status, "Cheminement en cours…", "info")
            self._post(self._schedule_refresh)

        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.is_scanning = False
            self._post(self.view.pages[2].set_running_state, "idle")
            self._post(self.view.set_status, f"Échec du lancement : {exc}", "danger")

    def _build_initial_path(self, mesh, rules, values):
        """Construit le chemin de départ selon la stratégie choisie.

        Le trajet est découpé en tronçons par les **points de passage imposés**
        — les encoches des peignes détectés, quand l'utilisateur demande à les
        emprunter. Chaque tronçon est planifié séparément, puis les tronçons
        sont recollés : c'est ce qui garantit que le chemin passe réellement
        par les fixations, et pas seulement à proximité.

        Renvoie ``None`` en cas d'échec, après avoir dit pourquoi. Aucun repli
        silencieux : c'est précisément ce qui masquait le fait que la
        géodésique ne fonctionnait pas sur une maquette faite de pièces
        disjointes.
        """
        nodes, forced_straight = self._route_nodes(values)
        n_points = values["initial_points"]

        # Le budget de points est réparti sur les tronçons au prorata de leur
        # longueur à vol d'oiseau : un tronçon de 2 m ne doit pas recevoir
        # autant de points qu'une traversée d'encoche de 5 cm.
        spans = [
            float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
            for a, b in zip(nodes[:-1], nodes[1:])
        ]
        total_span = sum(spans) or 1.0

        pieces = []
        messages = []
        for index, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
            share = max(4, int(round(n_points * spans[index] / total_span)))

            if index in forced_straight:
                # Traversée d'une encoche : une ligne droite, sans détour. Y
                # appliquer une recherche de chemin ferait contourner le peigne
                # au lieu de passer dedans.
                segment = np.linspace(a, b, share, dtype=np.float32)
            else:
                segment, message = self._plan_segment(mesh, rules, values, a, b, share)
                if segment is None:
                    self._post(self.view.set_status, message, "danger")
                    return None
                if message:
                    messages.append(message)
            pieces.append(np.asarray(segment, dtype=np.float32))

        # On ne duplique pas le point de jonction entre deux tronçons.
        points = pieces[0]
        for piece in pieces[1:]:
            points = np.vstack([points, piece[1:]])

        if messages:
            self._post(self.view.set_status, messages[0], "ok")
        return points.astype(np.float32)

    def _route_nodes(self, values):
        """Étapes imposées du trajet, de A à B, et tronçons à laisser droits.

        Sans fixations à emprunter, le trajet n'a que deux étapes. Avec, chaque
        encoche ajoute son couple entrée/sortie, ordonné le long de A→B et
        orienté dans le sens de la marche : entrer par la sortie d'une encoche
        obligerait le câble à faire demi-tour dedans.
        """
        a = np.asarray(self.point_a, dtype=np.float64)
        b = np.asarray(self.point_b, dtype=np.float64)

        passages = self._passages_to_use(values)
        if not passages:
            return [a, b], set()

        direction = b - a
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])

        ordered = sorted(
            passages,
            key=lambda p: float(np.dot(np.asarray(p.center) - a, direction)),
        )

        nodes = [a]
        forced_straight = set()
        for passage in ordered:
            p_in = np.asarray(passage.p_in, dtype=np.float64)
            p_out = np.asarray(passage.p_out, dtype=np.float64)
            # Le côté par lequel on entre est celui qu'on rencontre en premier.
            if np.dot(p_out - p_in, direction) < 0:
                p_in, p_out = p_out, p_in
            nodes.append(p_in)
            forced_straight.add(len(nodes) - 1)  # tronçon p_in -> p_out
            nodes.append(p_out)
        nodes.append(b)
        return nodes, forced_straight

    def _passages_to_use(self, values):
        """Passages imposés à emprunter, selon le choix de l'utilisateur."""
        if not values.get("use_fixations", False):
            return []
        result = getattr(self, "scan_result", None)
        return list(result.passages) if result is not None and result.ran else []

    def _plan_segment(self, mesh, rules, values, a, b, n_points):
        """Planifie un tronçon. Renvoie ``(points, message)``.

        ``points`` vaut ``None`` en cas d'échec ; ``message`` dit alors
        pourquoi.
        """
        from core.path_planner import PlannerSettings, plan_route

        strategy = values.get("start_path", "balanced")
        lang = self.view.t.lang

        if strategy == "geodesic":
            # Le chemin de surface remplace ``PolyData.geodesic`` : celle-ci
            # exige un chemin d'arêtes, dont une maquette faite de pièces
            # disjointes ne dispose jamais. Elle échouait donc et rendait une
            # corde tendue, ce que l'utilisateur voyait comme « la géodésique
            # ne marche pas ».
            from core.surface_path import surface_path

            # Le tracé est décollé de la structure avant d'être rendu. Sans
            # cela il est *sur* la surface, donc en interférence sur toute sa
            # longueur : les agents passeraient leurs premières centaines
            # d'itérations à faire ce qu'un décalage géométrique fait d'un
            # coup. La cible est prise dans le bas de la bande autorisée, le
            # propre d'un chemin de surface étant de longer la structure.
            band = rules.clearance.max_mm - rules.clearance.default_min_mm
            target = rules.clearance.default_min_mm + 0.25 * max(band, 0.0)

            result = surface_path(mesh, a, b, num_points=n_points, offset_mm=target)
            return (result.points, result.message(lang)) if result.success \
                else (None, result.message(lang))

        settings = PlannerSettings(
            voxel_mm=values.get("voxel_mm") or None,
        ).with_strategy(strategy)

        result = plan_route(
            mesh, a, b,
            rules.clearance.default_min_mm,
            rules.clearance.max_mm,
            settings,
            num_points=n_points,
        )
        return (result.points, result.message(lang)) if result.success \
            else (None, result.message(lang))

    @staticmethod
    def _prepare_agent_meshes(mesh, count: int) -> list:
        """Un maillage privé par agent, préparé en parallèle.

        Chaque agent a besoin de sa propre copie : les requêtes de proximité de
        trimesh partagent un index natif qui corrompt le tas dès que deux fils
        l'interrogent en même temps — reproduit ici, ce n'est pas une
        précaution théorique.

        En revanche il est inutile de refaire ``merge_vertices`` et
        ``fix_normals`` sur chaque copie : la source les a déjà subis, et les
        copies héritent des faces corrigées. Les supprimer, puis construire les
        arbres k-d en parallèle (scipy relâche le GIL), fait passer la
        préparation de 26,4 s à 2,7 s sur 800 000 triangles, pour une géométrie
        strictement identique.
        """
        import trimesh
        from concurrent.futures import ThreadPoolExecutor

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)

        def build(_index):
            private = trimesh.Trimesh(
                vertices=vertices.copy(), faces=faces.copy(), process=False
            )
            # Caches réchauffés ici, une fois pour toutes : sinon le premier
            # appel de chaque agent les construirait au milieu de la boucle.
            _ = private.face_normals
            _ = private.triangles
            _ = private.kdtree
            return private

        if count <= 1:
            return [build(0)]
        with ThreadPoolExecutor(max_workers=min(count, 8)) as pool:
            return list(pool.map(build, range(count)))

    @staticmethod
    def _blank_agent_state(waypoints) -> dict:
        return {
            "iteration": 0,
            "initial_waypoints": waypoints.copy(),
            "waypoints": waypoints.copy(),
            "density": len(waypoints),
            "collisions": 1,
            "reward": 0.0,
            "reward_history": [],
            "reward_details": {},
            "done": False,
            "stuck_msg": "",
            "min_dist": 0.0,
            "max_dist": 0.0,
            "mean_dist": 0.0,
            "dist_history": [],
            "collision_history": [],
            "point_count_history": [],
            "prev_disp": None,
            "crabes": [],
            "crabe_count": 0,
            "crabe_count_history": [],
        }

    def _load_mesh(self):
        """Charge le maillage fusionné produit à l'étape 1."""
        try:
            import pyvista as pv
            import trimesh
        except ImportError as exc:
            print(f"⚠️ Dépendances géométriques manquantes : {exc}")
            return None

        path = self.extraction_summary.get("fusion_path") or paths.FUSED_MESH_PATH
        if not os.path.exists(str(path)):
            return None

        pv_mesh = pv.read(str(path)).triangulate()
        if pv_mesh.n_cells == 0:
            return None

        mesh = trimesh.Trimesh(
            vertices=np.array(pv_mesh.points, copy=True),
            faces=np.array(pv_mesh.faces.reshape(-1, 4)[:, 1:], copy=True),
        )
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.fix_normals()
        _ = mesh.face_normals
        _ = mesh.kdtree
        self._pv_mesh = pv_mesh
        return mesh

    def set_temperature(self, value: float):
        """Applique le curseur Exploration ↔ Exploitation, y compris en cours de calcul."""
        if self.orchestrator is None:
            return
        team = self.orchestrator.set_temperature(value)
        specs = {spec.name: spec for spec in team}
        for name, entry in self.benchmark_algos.items():
            if name in specs:
                entry["spec"] = specs[name]
        self.view.set_status(
            f"{self.t('routing.explore')} : {self.orchestrator.policy.label(self.view.t.lang)}",
            "info",
        )

    def reset_routing(self):
        """Remet le cheminement à son état initial, agents compris."""
        if self.shared_state is None:
            return
        with self.data_lock:
            self.shared_state["is_playing"] = False
            self.shared_state["is_running"] = False
        if self.supervisor is not None:
            self.supervisor.stop()

        self.threads_started = False
        self.supervisor = None
        self.benchmark_algos = {}
        self.shared_state = None
        self._path_signatures.clear()
        self._best_scores.clear()
        self._clamp_signature = ()
        if self.viewer is not None:
            self.viewer.remove_prefix("traj_")
            self.viewer.remove_prefix("clamp_")
            self.viewer.render()

        if self.charts is not None:
            self.charts.reset()

        self.view.pages[2].set_running_state("idle")
        self.view.pages[2].update_live(
            {"report": None, "team": {}, "agents": [], "advice": []}
        )
        self.view.set_status(self.t("app.ready"), "neutral")

    # ==================================================================
    # Vue 3D
    # ==================================================================

    def _setup_charts(self):
        """Prépare les courbes de progression et y place les limites en vigueur."""
        from ui.charts import ProgressCharts

        page = self.view.pages[2]
        if self.charts is None:
            self.charts = ProgressCharts(page.charts_container, lang=self.view.t.lang)
            self.charts.start()
        else:
            self.charts.reset()

        if self.rules is not None:
            self.charts.set_limits(
                self.rules.clearance.default_min_mm,
                self.rules.clearance.max_mm,
                self.rules.harness.min_bend_radius_mm,
            )

    def _setup_viewer(self):
        """Ouvre la vue 3D et y place la maquette.

        ``Viewer3D.start`` rend la main immédiatement : la construction du
        contexte 3D — plus de treize secondes sur une grosse maquette — se fait
        dans un fil dédié. Les ordres émis ici sont mis en file et exécutés dès
        que ce contexte est prêt, ce qui évite de figer l'interface comme le
        faisait la version précédente.
        """
        from ui.viewer3d import MODE_UNAVAILABLE, Viewer3D

        page = self.view.pages[2]
        if self.viewer is None:
            self.viewer = Viewer3D(
                page.viewer_container,
                on_status=self._on_viewer_status,
                t=lambda key, default="": self.t(key) or default,
            )
            self.viewer.start()

        if self.viewer.mode == MODE_UNAVAILABLE:
            return

        pv_mesh = getattr(self, "_pv_mesh", None)
        if pv_mesh is not None:
            self.viewer.show_mesh(pv_mesh, "dmu")
            self.viewer.show_bbox(pv_mesh.bounds, "bbox")
            self.viewer.set_visible("bbox", False)

        if self.point_a is not None:
            self.viewer.show_sphere(self.point_a, "point_a", radius=25.0, color="#1E9E5A")
        if self.point_b is not None:
            self.viewer.show_sphere(self.point_b, "point_b", radius=25.0, color="#D93A45")
        self.viewer.reset_camera()
        self.viewer.render()

    def _ask_use_fixations(self, scan_result, default: bool) -> bool:
        """Demande s'il faut faire passer le câble par les fixations reconnues.

        La question n'est posée que lorsqu'elle se pose : sans passage détecté,
        il n'y a rien à emprunter et interrompre l'utilisateur serait gratuit.

        Le lancement se fait dans un fil de travail, la fenêtre doit s'ouvrir
        sur celui de Tk : on attend donc la réponse. L'attente est bornée —
        une fenêtre restée sans réponse ne doit pas immobiliser le calcul pour
        toujours, on retombe alors sur le réglage de la page.
        """
        if scan_result is None or not scan_result.ran or not scan_result.passages:
            return default

        def ask():
            try:
                return bool(self.view.ask_use_fixations(scan_result, default))
            except Exception:
                return default

        if threading.current_thread() is threading.main_thread():
            # Déjà sur le fil de Tk : passer par ``after`` puis attendre la
            # réponse serait un interblocage, la boucle d'évènements étant
            # justement celle qu'on bloquerait.
            answer = {"value": ask()}
        else:
            answer = {"value": default}
            answered = threading.Event()

            def ask_on_ui():
                try:
                    answer["value"] = ask()
                finally:
                    answered.set()

            self._post(ask_on_ui)
            answered.wait(timeout=ASK_TIMEOUT_S)

        # La page reflète la réponse : l'utilisateur retrouve son choix au même
        # endroit que s'il l'avait coché lui-même, et il est mémorisé.
        self._post(self.view.pages[2].set_use_fixations, answer["value"])
        return bool(answer["value"])

    def _draw_fixations(self, scan_result):
        """Place les fixations reconnues dans la vue 3D.

        Rien ne les dessinait : le viewer savait masquer les acteurs
        ``clamp_``, mais aucun n'était jamais créé. L'utilisateur n'avait donc
        aucun moyen de vérifier ce que le scan avait reconnu.

        Chaque fixation est un repère à sa position ; chaque passage de peigne
        est matérialisé par son entrée (vert), sa sortie (rouge) et le segment
        que le câble doit emprunter (jaune) — les mêmes couleurs que dans la
        liste affichée au-dessus de la vue.
        """
        if self.viewer is None or not self.viewer.is_available:
            return
        self.viewer.remove_prefix("fixation_")
        if scan_result is None or not scan_result.ran:
            self.viewer.render()
            return

        radius = max(6.0, self.rules.harness.radius_mm if self.rules else 20.0)
        for index, fixation in enumerate(scan_result.fixations):
            # La fixation est dessinée avec son propre modèle, recalé là où le
            # détecteur l'a trouvée. Un repère symbolique ne dirait rien de son
            # encombrement, qui est précisément ce qu'on cherche à voir.
            body = self._fixation_body(fixation)
            if body is not None:
                self.viewer.show_mesh(body, f"fixation_body_{index}",
                                      color="#8FA3B8", opacity=0.85, show_edges=False)
            elif fixation.position:
                # Modèle illisible : on montre au moins où elle se trouve,
                # plutôt que de la faire disparaître de la vue.
                self.viewer.show_sphere(
                    fixation.position, f"fixation_body_{index}",
                    radius=radius * 1.2, color="#8FA3B8",
                )

        for index, passage in enumerate(scan_result.passages):
            self.viewer.show_sphere(passage.p_in, f"fixation_in_{index}",
                                    radius=radius, color="#1E9E5A")
            self.viewer.show_sphere(passage.p_out, f"fixation_out_{index}",
                                    radius=radius, color="#D93A45")
            self.viewer.show_path([passage.p_in, passage.p_out],
                                  f"fixation_slot_{index}", color="#E5B300", width=9)
        self.viewer.render()

    @staticmethod
    def _fixation_body(fixation):
        """Maillage de la fixation reconnue, recalé sur la maquette.

        C'est ce que faisait déjà l'ancienne application : lire le STL du
        modèle et lui appliquer la matrice de recalage rendue par l'ICP.
        Renvoie ``None`` si le fichier est introuvable ou illisible — un
        modèle absent ne doit pas faire disparaître le reste de la vue.
        """
        if not getattr(fixation, "is_drawable", False):
            return None
        try:
            import numpy as np
            import pyvista as pv

            if not os.path.exists(fixation.file_path):
                return None
            mesh = pv.read(fixation.file_path)
            mesh.transform(np.asarray(fixation.transform, dtype=float), inplace=True)
            return mesh
        except Exception:
            return None

    def _draw_clamps(self, crabes):
        """Place les crabes posés par le meilleur agent, au fil du calcul.

        Le crabe est dessiné **avec sa géométrie réelle**, dans le repère où
        son absence de collision a été vérifiée : même matrice de rotation,
        même origine sur la structure. Un repère symbolique ne dirait rien de
        l'encombrement réel de la fixation, qui est précisément ce que
        l'intégrateur doit pouvoir juger à l'œil.

        Sans modèle STL chargeable, il n'y a rien à dessiner — et rien n'a été
        posé non plus, ``compute_crabes`` rendant une liste vide. L'onglet
        « Conseils » le signale ; on n'invente pas un marqueur à la place.
        """
        if self.viewer is None or not self.viewer.is_available:
            return

        signature = tuple(
            (round(float(c.get("arc_mm", 0.0)), 1), round(float(c.get("tilt_deg", 0.0)), 1))
            for c in crabes
        )
        if signature == self._clamp_signature:
            return
        self._clamp_signature = signature

        self.viewer.remove_prefix("clamp_")
        geometry = self._crabe_geometry()
        if geometry is None:
            self.viewer.render()
            return

        for index, crabe in enumerate(crabes[:MAX_DRAWN_CLAMPS]):
            body = self._clamp_body(crabe, geometry)
            if body is not None:
                self.viewer.show_mesh(body, f"clamp_{index}", color="#FFD166",
                                      opacity=1.0, show_edges=False)
        self.viewer.render()

    def _crabe_geometry(self):
        """Géométrie du crabe, chargée une fois et mémorisée."""
        path = self._crabe_stl_path
        if not path:
            return None
        try:
            from core.agent.tool import get_crabe_geometry

            return get_crabe_geometry(path)
        except Exception:
            return None

    @staticmethod
    def _clamp_body(crabe, geometry):
        """Maillage PyVista du crabe posé, ou ``None`` si indessinable."""
        try:
            import numpy as np
            import pyvista as pv

            from core.agent.tool import crabe_world_vertices

            vertices = crabe_world_vertices(crabe, geometry)
            if vertices is None:
                return None
            faces = np.asarray(geometry["faces"], dtype=np.int64)
            padded = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces])
            return pv.PolyData(np.asarray(vertices, dtype=float), padded.ravel())
        except Exception:
            return None

    def _on_viewer_status(self, mode: str, error: str | None):
        from ui.viewer3d import MODE_EMBEDDED, MODE_STARTING, MODE_UNAVAILABLE

        if mode == MODE_STARTING:
            self.view.set_status(self.t("routing.view.starting"), "info")
        elif mode == MODE_EMBEDDED:
            self.view.set_status(self.t("routing.view.ready"), "ok")
            self.view.pages[2].set_detached(self.viewer.is_detached if self.viewer else False)
        elif mode == MODE_UNAVAILABLE:
            self.view.set_status(
                f"{self.t('routing.view.unavailable')} {error}" if error
                else self.t("routing.view.unavailable"),
                "warn",
            )

    def update_3d_visibility(self, toggles: dict):
        if self.viewer is None:
            return
        self.viewer.set_visible("dmu", toggles.get("mesh", True))
        self.viewer.set_edges("dmu", toggles.get("edges", False))
        self.viewer.set_visible("bbox", toggles.get("bbox", False))
        self.viewer.set_visible_prefix("clamp_", toggles.get("clamps", True))
        self.viewer.render()

    def detach_3d(self):
        """Ouvre — ou referme — la fenêtre 3D interactive."""
        if self.viewer is None or not self.viewer.is_available:
            self.view.set_status(self.t("routing.view.none"), "warn")
            return
        detached = self.viewer.toggle_window()
        self._detached = detached
        self.view.pages[2].set_detached(detached)

    # ==================================================================
    # Rafraîchissement de l'interface
    # ==================================================================

    def _schedule_refresh(self):
        if self._refresh_job is not None:
            try:
                self.view.after_cancel(self._refresh_job)
            except Exception:
                pass
        self._refresh_job = self.view.after(REFRESH_RUNNING_MS, self._refresh)

    def _refresh(self):
        if self.shared_state is None:
            self._refresh_job = self.view.after(REFRESH_IDLE_MS, self._refresh)
            return

        with self.data_lock:
            playing = self.shared_state.get("is_playing", False)
            team = dict(self.shared_state.get("team", {}) or {})
            snapshot = {
                name: {
                    "iteration": state.get("iteration", 0),
                    "reward": state.get("reward", 0.0),
                    "report": state.get("report"),
                    "score": state.get("score"),
                    "waypoints": state.get("waypoints"),
                    "crabes": list(state.get("crabes", [])),
                    "done": state.get("done", False),
                    "migrations": state.get("migrations", 0),
                }
                for name, state in self.shared_state.get("algos", {}).items()
            }

        self._draw_paths(snapshot)
        self._update_page(snapshot, team)
        if self.charts is not None and playing:
            colors = {n: e.get("color", "#2D7FF9") for n, e in self.benchmark_algos.items()}
            self.charts.update(snapshot, colors)

        delay = REFRESH_RUNNING_MS if playing else REFRESH_IDLE_MS
        self._refresh_job = self.view.after(delay, self._refresh)

    def _draw_paths(self, snapshot: dict):
        """Redessine les trajectoires qui ont bougé, et elles seules.

        Reconstruire les cinq tubes à chaque rafraîchissement coûtait 20 ms par
        appel, soit 8 % du fil principal à 4 Hz, pour redessiner le plus
        souvent des trajectoires identiques : un agent ne bouge pas à chaque
        cycle d'affichage.
        """
        if self.viewer is None or not self.viewer.is_available:
            return
        changed = False
        for name, state in snapshot.items():
            points = state.get("waypoints")
            if points is None:
                continue
            signature = (state.get("iteration"), len(points))
            if self._path_signatures.get(name) == signature:
                continue
            self._path_signatures[name] = signature
            entry = self.benchmark_algos.get(name, {})
            self.viewer.show_path(
                points, f"traj_{name}",
                color=entry.get("color", "#2D7FF9"), width=7,
            )
            changed = True
        if changed:
            self.viewer.render()

    def _update_page(self, snapshot: dict, team: dict):
        ranking = team.get("ranking") or list(snapshot)
        best = ranking[0] if ranking else None

        # Conseils : tirés du meilleur agent, et seulement de lui. Moyenner les
        # rapports de cinq agents produirait un diagnostic qui ne correspond à
        # aucune route réellement obtenue.
        best_state = snapshot.get(best, {}) if best else {}
        advice = self._advise(best_state)
        self._draw_clamps(best_state.get("crabes") or [])
        lang = self.view.t.lang

        agents = []
        for name in ranking:
            state = snapshot.get(name, {})
            spec = self.benchmark_algos.get(name, {}).get("spec")
            role = ROLES.get(getattr(spec, "role", ""), None)
            report = state.get("report")

            if report is None:
                detail = "Démarrage…"
                badge, badge_color = "", None
            else:
                k = report.kpis
                radius = k.get("min_bend_radius_mm", float("inf"))
                detail = (
                    f"{k.get('n_clashes', 0)} interférence(s) · "
                    f"{'∞' if radius == float('inf') else f'{radius:.0f} mm'} de cintrage · "
                    f"{k.get('straight_ratio', 0) * 100:.0f} % droit"
                )
                badge = "conforme" if report.is_compliant else ""
                badge_color = None

            migrations = state.get("migrations", 0)
            if migrations:
                detail += f" · {migrations} reprise(s)"

            agents.append(
                {
                    "name": name,
                    "label": role.label(lang) if role else name,
                    "color": self.benchmark_algos.get(name, {}).get("color", "#2D7FF9"),
                    "rank": ranking.index(name) + 1,
                    "state": detail,
                    "badge": badge,
                    "badge_color": badge_color,
                }
            )

        self.view.pages[2].update_live(
            {
                "report": best_state.get("report"),
                "iteration": best_state.get("iteration"),
                "team": team,
                "agents": agents,
                "advice": advice,
            }
        )

    @staticmethod
    def _check_clamp_model(path: str) -> bool:
        """Le fichier STL du crabe est-il réellement chargeable ?

        Un chemin renseigné mais illisible — le cas le plus courant, un chemin
        hérité d'un autre poste — se traduisait par un « AUCUN crabe possible »
        répété à chaque itération dans la console, invisible depuis l'interface.
        """
        if not path:
            return False
        try:
            import trimesh

            mesh = trimesh.load_mesh(str(path), force="mesh")
            return len(getattr(mesh, "faces", [])) > 0
        except Exception:
            return False

    def _advise(self, best_state: dict) -> list:
        """Conseils sur l'état du meilleur agent, ou liste vide s'il est trop tôt.

        L'historique des scores sert à repérer la stagnation : conseiller de
        relâcher une règle alors que le score progresse encore reviendrait à
        inciter l'utilisateur à baisser ses exigences au premier obstacle.
        """
        if self.rules is None:
            return []

        report = best_state.get("report")
        score = best_state.get("score")
        if score is not None:
            self._best_scores.append(score)
            # On ne garde que ce qui sert à détecter la stagnation.
            window = diagnostics.STAGNATION_WINDOW * 3
            if len(self._best_scores) > window:
                del self._best_scores[:-window]

        return diagnostics.analyse(
            report,
            self.rules,
            iterations=int(best_state.get("iteration", 0)),
            stagnant=diagnostics.is_stagnant(self._best_scores),
            clamp_model_ok=self._clamp_model_ok,
        )

    def apply_suggestion(self, suggestion):
        """Reporte la valeur conseillée dans la page « Règles ».

        Le réglage n'est pas appliqué à chaud : les agents ont recopié les
        règles au démarrage, et n'en changer qu'une partie donnerait un mélange
        incohérent entre ce qui récompense et ce qui est mesuré. On écrit donc
        le réglage et on invite explicitement à relancer.
        """
        if not getattr(suggestion, "is_applicable", False):
            return False

        page = self.view.pages[1]
        fields = {
            "min_margin": getattr(page, "f_min", None),
            "max_margin": getattr(page, "f_max", None),
            "bend_radius_factor": getattr(page, "f_bend_factor", None),
            "fixation_pitch": getattr(page, "f_pitch", None),
            "fixation_parallel_tol": getattr(page, "f_parallel", None),
        }
        field = fields.get(suggestion.setting)
        if field is None:
            self.view.set_status(self.t("advice.not_settable"), "warn")
            return False

        field.set(suggestion.value)
        if suggestion.setting in ("bend_radius_factor",):
            page._refresh_bend()
        self.view.remember(**{suggestion.setting: suggestion.value})
        self.view.set_status(self.t("advice.applied"), "ok")
        return True

    def _publish_reports(self):
        """Alimente la page de rapport avec l'état courant des agents."""
        if self.shared_state is None:
            self.view.pages[3].update_reports({}, [], {})
            return

        with self.data_lock:
            team = dict(self.shared_state.get("team", {}) or {})
            reports = {
                name: state["report"]
                for name, state in self.shared_state.get("algos", {}).items()
                if state.get("report") is not None
            }

        lang = self.view.t.lang
        labels = {}
        for name in reports:
            spec = self.benchmark_algos.get(name, {}).get("spec")
            role = ROLES.get(getattr(spec, "role", ""), None)
            labels[name] = role.label(lang) if role else name

        self.view.pages[3].update_reports(reports, team.get("ranking") or list(reports), labels)

    # ==================================================================
    # Exports
    # ==================================================================

    def export(self, agent_name: str, kind: str) -> str:
        """Exporte le tracé d'un agent. Renvoie le nom du fichier écrit."""
        if self.shared_state is None:
            return ""

        with self.data_lock:
            state = self.shared_state.get("algos", {}).get(agent_name)
            if state is None:
                return ""
            waypoints = np.asarray(state["waypoints"], dtype=np.float64).copy()
            crabes = list(state.get("crabes", []))
            report = state.get("report")

        if len(waypoints) < 2:
            return ""

        try:
            if kind == "csv":
                return self._export_csv(agent_name, waypoints, crabes)
            if kind == "report":
                return self._export_report(agent_name, report)
            if kind == "catia":
                self._send_to_catia(agent_name, waypoints)
                return ""
            return self._export_stl(agent_name, waypoints)
        except Exception as exc:
            self.view.set_status(f"Échec de l'export : {exc}", "danger")
            return ""

    def _export_stl(self, agent_name: str, waypoints) -> str:
        from tkinter import filedialog

        import pyvista as pv

        path = filedialog.asksaveasfilename(
            title=self.t("report.export.stl"),
            defaultextension=".stl",
            initialfile=f"cheminement_{agent_name}.stl",
            filetypes=[("Fichiers STL", "*.stl")],
        )
        if not path:
            return ""

        radius = self.rules.harness.radius_mm if self.rules else 20.0
        pv.lines_from_points(waypoints).tube(radius=radius, n_sides=24).save(path)
        return os.path.basename(path)

    def _export_csv(self, agent_name: str, waypoints, crabes) -> str:
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            title=self.t("report.export.csv"),
            defaultextension=".csv",
            initialfile=f"cheminement_{agent_name}.csv",
            filetypes=[("Fichiers CSV", "*.csv")],
        )
        if not path:
            return ""

        clamp_indices = {int(c.get("index", -1)) for c in crabes}
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("index;X_mm;Y_mm;Z_mm;fixation\n")
            for i, point in enumerate(waypoints):
                fixation = "oui" if i in clamp_indices else ""
                handle.write(
                    f"{i};{point[0]:.3f};{point[1]:.3f};{point[2]:.3f};{fixation}\n"
                )
        return os.path.basename(path)

    def _send_to_catia(self, agent_name: str, waypoints):
        """Écrit le tube dans le cache, puis l'insère dans le CATIA ouvert.

        L'insertion passe par une macro exécutée par CATIA : c'est bloquant, et
        cela peut prendre plusieurs secondes. On le fait donc dans un fil
        séparé, et on rend compte à l'utilisateur quand c'est fini — plutôt que
        de figer l'interface sans explication.
        """
        import pyvista as pv

        radius = self.rules.harness.radius_mm if self.rules else 20.0
        paths.ensure_cache_folders()
        stl_path = paths.RUNS_DIR / f"faisceau_{agent_name}.stl"
        pv.lines_from_points(waypoints).tube(radius=radius, n_sides=24).save(str(stl_path))

        def run():
            try:
                from core.catia_handler import load_path_in_catia

                load_path_in_catia(stl_path)
                self._post(self.view.pages[3].report_catia_result, True, "")
            except Exception as exc:
                self._post(self.view.pages[3].report_catia_result, False, str(exc))

        threading.Thread(target=run, daemon=True).start()

    def _export_report(self, agent_name: str, report) -> str:
        if report is None:
            return ""

        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            title=self.t("report.export.report"),
            defaultextension=".json",
            initialfile=f"rapport_{agent_name}.json",
            filetypes=[("Fichiers JSON", "*.json")],
        )
        if not path:
            return ""

        payload = report.to_dict()
        payload["agent"] = agent_name
        payload["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.rules is not None:
            payload["rules"] = {
                "harness_diameter_mm": self.rules.harness.diameter_mm,
                "min_bend_radius_mm": self.rules.harness.min_bend_radius_mm,
                "clearance_min_mm": self.rules.clearance.default_min_mm,
                "clearance_max_mm": self.rules.clearance.max_mm,
                "fixation_pitch_mm": self.rules.fixation_pitch_mm,
            }

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return os.path.basename(path)

    # ==================================================================
    # Divers
    # ==================================================================

    def _post(self, callback, *args):
        """Exécute un appel sur le fil de l'interface."""
        try:
            self.view.after(0, lambda: callback(*args))
        except Exception:
            pass

    def shutdown(self):
        """Arrête proprement agents, superviseur et processus d'extraction."""
        if self.shared_state is not None:
            with self.data_lock:
                self.shared_state["is_running"] = False
                self.shared_state["is_playing"] = False
        if self.supervisor is not None:
            self.supervisor.stop()
        if self.extraction_process and self.extraction_process.is_alive():
            self.extraction_process.terminate()
        if self.viewer is not None:
            self.viewer.close()
