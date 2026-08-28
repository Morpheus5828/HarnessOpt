"""Contrôleur : relie l'interface au moteur de cheminement.

Responsabilités :

* piloter l'extraction du DMU (processus séparé) ;
* construire le jeu de règles à partir de ce que l'utilisateur a saisi ;
* lancer, mettre en pause et réinitialiser l'équipe d'agents ;
* rapatrier périodiquement l'état partagé vers l'interface ;
* exporter le résultat.

L'interface n'importe jamais ni PyTorch ni trimesh : ces dépendances lourdes
sont chargées ici, à la demande, ce qui garde le démarrage de l'application
rapide et permet d'afficher un message clair si l'une d'elles manque.
"""

from __future__ import annotations

import json
import os
import threading
import time
from multiprocessing import Process, Queue
from tkinter import filedialog

import numpy as np

from core import paths
from core.orchestrator import ROLES, Orchestrator
from core.routing_rules import ClearanceModel, HarnessSpec, RoutingRules

__all__ = ["AppController"]

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
        self._post(self.view.unlock_step, 1)
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

    def on_step_shown(self, index: int):
        """Appelé à chaque changement d'étape."""
        if index == 1 and self.extraction_summary:
            self.view.pages[1].show_families(sorted(self.extraction_summary.get("families", {})))
        elif index == 2:
            problems = self.view.pages[1].validate()
            if problems:
                self.view.set_status(problems[0], "warn")
            else:
                self.view.unlock_step(2)
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
            initial_points=values["initial_points"],
            max_points=values["max_points"],
            max_step_mm=values["max_step_mm"],
            iterations=values["iterations"],
        )

        page.set_running_state("scanning")
        self.is_scanning = True
        self.view.set_status("Préparation de l'environnement…", "info")
        threading.Thread(target=self._prepare_and_launch, args=(values,), daemon=True).start()

    def _prepare_and_launch(self, values: dict):
        """Charge la maquette, prépare l'équipe puis lance les agents."""
        try:
            import trimesh

            from core.agent.config import CONFIG
            from core.agent.tool import generate_dense_waypoints
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

            waypoints = generate_dense_waypoints(
                self.point_a, self.point_b, values["initial_points"], mesh=mesh
            )

            specs = {spec.name: spec for spec in self.orchestrator.team}
            with self.data_lock:
                for name in self.benchmark_algos:
                    self.shared_state["algos"][name] = self._blank_agent_state(waypoints)

            for name in self.benchmark_algos:
                private = trimesh.Trimesh(
                    vertices=np.array(mesh.vertices, copy=True),
                    faces=np.array(mesh.faces, copy=True),
                )
                private.merge_vertices()
                private.fix_normals()
                _ = private.face_normals
                _ = private.kdtree

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

            self._post(self._setup_viewer, waypoints)
            self._post(self._setup_charts)
            self._post(self.view.pages[2].set_running_state, "running")
            self._post(self.view.unlock_step, 3)
            self._post(self.view.set_status, "Cheminement en cours…", "info")
            self._post(self._schedule_refresh)

        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.is_scanning = False
            self._post(self.view.pages[2].set_running_state, "idle")
            self._post(self.view.set_status, f"Échec du lancement : {exc}", "danger")

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
        if self.viewer is not None:
            self.viewer.remove_prefix("traj_")
            self.viewer.remove_prefix("clamp_")
            self.viewer.render()

        if self.charts is not None:
            self.charts.reset()

        self.view.pages[2].set_running_state("idle")
        self.view.pages[2].update_live({"report": None, "team": {}, "agents": []})
        self.view.set_status(self.t("app.ready"), "neutral")

    # ==================================================================
    # Vue 3D
    # ==================================================================

    def _setup_charts(self):
        """Prépare les courbes de progression et y place les limites en vigueur."""
        from ui.charts import ProgressCharts

        page = self.view.pages[2]
        if self.charts is None:
            self.charts = ProgressCharts(page.charts_container)
            self.charts.start()
        else:
            self.charts.reset()

        if self.rules is not None:
            self.charts.set_limits(
                self.rules.clearance.default_min_mm,
                self.rules.clearance.max_mm,
                self.rules.harness.min_bend_radius_mm,
            )

    def _setup_viewer(self, waypoints):
        from ui.viewer3d import MODE_UNAVAILABLE, Viewer3D

        page = self.view.pages[2]
        if self.viewer is None:
            self.viewer = Viewer3D(page.viewer_container, on_status=self._on_viewer_status)
            self.viewer.start()

        if self.viewer.mode == MODE_UNAVAILABLE:
            return

        pv_mesh = getattr(self, "_pv_mesh", None)
        if pv_mesh is not None:
            self.viewer.show_mesh(pv_mesh, "dmu")
            self.viewer.show_bbox(pv_mesh.bounds, "bbox")
            self.viewer.set_visible("bbox", False)

        self.viewer.show_sphere(self.point_a, "point_a", radius=25.0, color="#1E9E5A")
        self.viewer.show_sphere(self.point_b, "point_b", radius=25.0, color="#D93A45")
        self.viewer.reset_camera()
        self.viewer.render()

    def _on_viewer_status(self, mode: str, error: str | None):
        from ui.viewer3d import MODE_EMBEDDED, MODE_WINDOW

        if mode == MODE_EMBEDDED:
            self.view.set_status("Vue 3D intégrée à la fenêtre.", "ok")
        elif mode == MODE_WINDOW:
            self.view.set_status(
                "Vue 3D en fenêtre séparée sur cette plateforme "
                "(utilisez « Ouvrir en grand »).",
                "info",
            )
        elif error:
            self.view.set_status(f"Vue 3D indisponible : {error}", "warn")

    def update_3d_visibility(self, toggles: dict):
        if self.viewer is None:
            return
        self.viewer.set_visible("dmu", toggles.get("mesh", True))
        self.viewer.set_edges("dmu", toggles.get("edges", False))
        self.viewer.set_visible("bbox", toggles.get("bbox", False))
        self.viewer.set_visible_prefix("clamp_", toggles.get("clamps", True))
        self.viewer.render()

    def detach_3d(self):
        if self.viewer is None or not self.viewer.is_available:
            self.view.set_status("Aucune vue 3D à ouvrir pour l'instant.", "warn")
            return
        self.viewer.show_window()
        self._detached = True
        self.view.pages[2].set_detached(True)

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
        if self.viewer is None or not self.viewer.is_available:
            return
        for name, state in snapshot.items():
            entry = self.benchmark_algos.get(name, {})
            self.viewer.show_path(
                state.get("waypoints"), f"traj_{name}",
                color=entry.get("color", "#2D7FF9"), width=7,
            )
        self.viewer.render()

    def _update_page(self, snapshot: dict, team: dict):
        ranking = team.get("ranking") or list(snapshot)
        best = ranking[0] if ranking else None
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

        best_state = snapshot.get(best, {}) if best else {}
        self.view.pages[2].update_live(
            {
                "report": best_state.get("report"),
                "iteration": best_state.get("iteration"),
                "team": team,
                "agents": agents,
            }
        )

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
