import time
import os
import threading
import numpy as np
import tkinter as tk
from tkinter import filedialog
import trimesh
import pyvista as pv
import customtkinter as ctk
from multiprocessing import Process, Queue

from core.catia_handler import BASE_CACHE

from core.agent_worker import (
    algo_worker, benchmark_algos, generate_dense_waypoints,
    shared_state, data_lock, get_rotation_matrix_from_vectors
)

from core.path_managment.fixation_detection import run_detection_for_agent

# 🔥 IMPORT DE NOTRE NOUVELLE FONCTION TOPOLOGIQUE DEPUIS TOOL.PY
from core.agent.tool import placer_lignes_dans_encoches_clamp

clamps_folder = r"C:\Users\a609568\Desktop\H160\H160-HarnessOpt-CLAMPS\H160-HarnessOpt-CLAMPS"


def parse_coords(text):
    try:
        return np.array([float(x.strip()) for x in text.split(',')], dtype=np.float32)
    except Exception as e:
        print(f"⚠️ Erreur de parsing des coordonnées ('{text}') : {e}")
        return None


class AppController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.extraction_process = None

        self.shared_state = shared_state
        self.data_lock = data_lock
        self.benchmark_algos = benchmark_algos

        self.mesh = None
        self.plotter = None
        self.initial_waypoints = None

        self.point_A = None
        self.point_B = None

        self.fig = None
        self.canvas = None
        self.axes_registry = {}
        self.lines_registry = {"reward": {}, "dist": {}, "col": {}, "cross": {}, "length": {}, "pts": {}, "crabes": {}}

        self.pv_mesh_base = None
        self.margin_actor_min = None
        self.margin_actor_max = None

        self.frame_count = 0
        self.agent_initialized = False
        self.threads_started = False

        self.is_3d_popped_out = False
        self.popout_window = None
        self.is_typing = False
        self.is_scanning = False



        self._render_suspended = False
        self._render_resume_after_id = None
        self._loop_busy = False
        self.view.bind("<Configure>", self._on_window_configure, add="+")

    def _on_window_configure(self, event):
        if event.widget is not self.view:
            return
        self._render_suspended = True
        if self._render_resume_after_id is not None:
            try:
                self.view.after_cancel(self._render_resume_after_id)
            except Exception:
                pass
        self._render_resume_after_id = self.view.after(300, self._resume_render)

    def _resume_render(self):
        self._render_suspended = False
        self._render_resume_after_id = None
        if self.plotter and hasattr(self.plotter, 'render_window'):
            try:
                self.plotter.render()
            except Exception:
                pass

    def _ensure_algo_widgets(self, page):
        if not hasattr(page, "algo_widgets"):
            page.algo_widgets = {}
        missing = [name for name in self.benchmark_algos.keys() if name not in page.algo_widgets]
        if missing:
            print(f"⚠️ page.algo_widgets ne connaît pas encore ces agents : {missing}. ")
            for name in missing:
                try:
                    lbl = ctk.CTkLabel(page, text="En attente...", text_color=("#50565E", "#8B949E"))
                    page.algo_widgets[name] = lbl
                except Exception:
                    pass

    def _init_agent_system(self, page):
        print("🤖 Initialisation de l'environnement 3D incrusté...")
        self.plotter = pv.Plotter(off_screen=False)

        if hasattr(self.plotter, 'disable_interactor'):
            self.plotter.disable_interactor()
        elif hasattr(self.plotter, 'iren') and self.plotter.iren is not None:
            if hasattr(self.plotter.iren, 'disable_interactor'):
                self.plotter.iren.disable_interactor()
            elif hasattr(self.plotter.iren, 'Disable'):
                self.plotter.iren.Disable()

        if page and hasattr(page, "vtk_container"):
            page.vtk_container.update_idletasks()
            hwnd = page.vtk_container.winfo_id()
            embedded_success = False
            for method_name in ["SetParentInfo", "SetWindowInfo", "SetParentId", "SetWindowId"]:
                if hasattr(self.plotter.render_window, method_name):
                    try:
                        if "Info" in method_name:
                            getattr(self.plotter.render_window, method_name)(str(hwnd))
                        else:
                            getattr(self.plotter.render_window, method_name)(hwnd)
                        embedded_success = True
                        print(f"🚀 Incrustation 3D réussie via : {method_name}")
                        break
                    except Exception:
                        continue
            if not embedded_success: print("⚠️ Impossible d'ancrer la fenêtre 3D.")

        vtk_path = os.path.join(BASE_CACHE, "fusion", "clipped_obstacles.vtk")

        if os.path.exists(vtk_path):

            pv_mesh = pv.read(vtk_path).triangulate()
            if getattr(pv_mesh, 'n_cells', getattr(pv_mesh, 'n_faces', 0)) > 0:
                self.mesh_actor = self.plotter.add_mesh(pv_mesh, color='lightgray', opacity=0.8, show_edges=False)
                self.plotter.add_axes()

                self.bbox_actor = self.plotter.add_mesh(pv.Box(bounds=pv_mesh.bounds), color='yellow',
                                                        style='wireframe',
                                                        line_width=2, label="BBox")
                self.bbox_actor.SetVisibility(False)

                self.pv_mesh_base = pv_mesh.compute_normals(point_normals=True, cell_normals=False,
                                                            auto_orient_normals=True)
                verts = np.array(pv_mesh.points, copy=True)
                faces = np.array(pv_mesh.faces.reshape(-1, 4)[:, 1:], copy=True)

                self.mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                self.mesh.merge_vertices(merge_tex=False, merge_norm=False)
                self.mesh.update_faces(self.mesh.unique_faces())
                self.mesh.fix_normals()

                print("🔥 Pré-calcul des propriétés du maillage pour le multithreading...")
                _ = self.mesh.face_normals
                _ = self.mesh.bounds
                _ = self.mesh.edges_unique
                _ = self.mesh.kdtree

                self.is_scanning = True
                if page:
                    page.btn_play.configure(state="disabled")
                    page.btn_text.set("⏳ RADAR EN COURS...")

                threading.Thread(target=self._run_fixation_detection, args=(self.mesh, clamps_folder),
                                 daemon=True).start()

            else:
                self.mesh = trimesh.creation.box(extents=[1, 1, 1])

        else:
            self.mesh = trimesh.creation.box(extents=[1, 1, 1])

        self.fig = page.fig
        self.canvas = page.canvas

        self._ensure_algo_widgets(page)
        self.update_graph_layouts()
        self._sync_parameters(page, force_3d_update=True)

        self.update_3d_visibility()

        if hasattr(self.plotter, 'iren') and not self.plotter.iren.initialized:
            self.plotter.iren.initialize()
        self.plotter.reset_camera()
        self.plotter.render()

        self.agent_initialized = True
        self.view.after(30, self._update_loop)

    def _sync_parameters(self, page, force_3d_update=False):
        try:
            def get_val(widget):
                return float(widget.get().replace(',', '.'))

            new_ns = get_val(page.entry_ns)
            new_nm = get_val(page.entry_nm)
            new_decay = get_val(page.entry_decay)
            new_shift = get_val(page.entry_shift)
            new_min = get_val(page.entry_min)
            new_max = get_val(page.entry_max)
            new_pts = int(get_val(page.entry_pts))
            tube_radius = float(page.entry_diam.get().replace(',', '.')) / 2.0 if hasattr(page, 'entry_diam') else 6.0

            with self.data_lock:
                cfg = self.shared_state["config"]
                margins_changed = (new_min != cfg.get("min_margin") or new_max != cfg.get("max_margin"))
                cfg["exploration_noise_start"] = new_ns
                cfg["exploration_noise_min"] = new_nm
                cfg["exploration_decay"] = new_decay
                cfg["local_max_shift"] = new_shift
                cfg["min_margin"] = new_min
                cfg["max_margin"] = new_max
                cfg["initial_points"] = new_pts
                cfg["tube_radius"] = tube_radius
                cfg["force_fixations"] = page.chk_force_fix.get() == 1

            if margins_changed or force_3d_update:
                self.update_graph_layouts()
                if self.margin_actor_min and self.plotter:
                    self.plotter.remove_actor(self.margin_actor_min)
                if self.margin_actor_max and self.plotter:
                    self.plotter.remove_actor(self.margin_actor_max)
        except Exception as e:
            print(f"⚠️ Erreur Sync : {e}")

    def _run_fixation_detection(self, current_mesh, clamps_folder):
        try:
            temp_stl_path = os.path.join(BASE_CACHE, "fusion", "temp_for_detection.stl")
            current_mesh.export(temp_stl_path)

            existing_clamps = run_detection_for_agent(temp_stl_path, clamps_folder)

            with self.data_lock:
                self.shared_state["config"]["existing_crabes"] = existing_clamps

            self.view.after(0, lambda: self._on_detection_complete(existing_clamps))

        except Exception as e:
            print(f"⚠️ Erreur lors de la détection des fixations: {e}")
            self.view.after(0, lambda: self._on_detection_complete([]))

    def _on_detection_complete(self, clamps):
        self.is_scanning = False
        print(f"\n✅ DÉTECTION TERMINÉE : {len(clamps)} fixations trouvées au total.")

        with self.data_lock:
            tube_radius = self.shared_state["config"].get("tube_radius", 6.0)

        if self.plotter and len(clamps) > 0:
            for i, clamp in enumerate(clamps):
                file_path = clamp.get("file_path")
                name = clamp.get("name", "")
                if not file_path or not os.path.exists(file_path):
                    continue

                transform_matrix = np.array(clamp["transform"])

                try:
                    # Affichage du maillage 3D de la fixation
                    clamp_mesh_pv = pv.read(file_path)
                    clamp_mesh_pv.transform(transform_matrix, inplace=True)
                    self.plotter.add_mesh(clamp_mesh_pv, color="lightblue", name=f"existing_clamp_{i}",
                                          show_edges=False, opacity=0.8)

                    # -------------------------------------------------------------
                    # 🔥 TRAITEMENT TOPOLOGIQUE POUR LES PEIGNES (CLAMPS)
                    # -------------------------------------------------------------
                    if name.upper().startswith("X"):
                        print(f"🪮 Peigne détecté ({name}) - Calcul topologique parfait IN/OUT...")

                        try:
                            # Chargement Trimesh natif pour l'analyse
                            mesh_trimesh = trimesh.load_mesh(file_path, force='mesh')

                            # Appel de notre nouvel outil de tool.py
                            passages, nb_detecte = placer_lignes_dans_encoches_clamp(mesh_trimesh, transform_matrix)

                            print(f"✅ {nb_detecte} encoche(s) détectée(s) géométriquement.")

                            routing_points = []
                            for pt_idx, p in enumerate(passages):
                                p_in = np.array(p['p_in'])
                                p_out = np.array(p['p_out'])

                                # Formatage aplati (in, out, in, out...) pour l'Agent RL
                                routing_points.extend([p_in, p_out])

                                # 🟢 AFFICHAGE GRAPHIQUE ÉCLATANT (Sphères + Lignes jaunes)
                                self.plotter.add_mesh(pv.Sphere(center=p_in, radius=tube_radius * 0.5), color="green",
                                                      name=f"peigne_{i}_in_{pt_idx}")
                                self.plotter.add_mesh(pv.Sphere(center=p_out, radius=tube_radius * 0.5), color="red",
                                                      name=f"peigne_{i}_out_{pt_idx}")

                                line = pv.Line(p_in, p_out)
                                self.plotter.add_mesh(line, color="yellow", line_width=4, render_lines_as_tubes=True,
                                                      name=f"peigne_{i}_line_{pt_idx}")

                            # On sauvegarde les points réels dans le dict global
                            clamp["routing_points"] = routing_points

                        except Exception as e:
                            print(f"⚠️ Erreur lors du calcul topologique pour {name} : {e}")

                except Exception as e:
                    print(f"⚠️ Impossible d'afficher la fixation {file_path} : {e}")

            if hasattr(self.plotter, 'render_window'):
                try:
                    self.plotter.render()
                    self.plotter.render_window.Render()
                except Exception as e:
                    print(f"⚠️ Erreur de rafraîchissement 3D : {e}")

        page = self.view.pages.get("agent")
        if page:
            page.btn_play.configure(state="normal")
            page.update_language(self.view.current_lang)

    def _get_crabe_clip_template(self, stl_path):
        if not stl_path:
            return None
        if not hasattr(self, "_crabe_clip_cache"):
            self._crabe_clip_cache = {}
        if stl_path in self._crabe_clip_cache:
            return self._crabe_clip_cache[stl_path]

        try:
            clip = pv.read(stl_path)
            clip = clip.compute_normals(cell_normals=True, point_normals=False, inplace=False)

            mesh_avec_surfaces = clip.compute_cell_sizes()
            surfaces = mesh_avec_surfaces.cell_data['Area']
            normales = clip.cell_normals
            normales_arrondies = np.round(normales, decimals=4)
            normales_uniques, _, inverses = np.unique(
                normales_arrondies, axis=0, return_index=True, return_inverse=True)
            surfaces_par_normale = np.zeros(len(normales_uniques))
            for i in range(len(surfaces)):
                surfaces_par_normale[inverses[i]] += surfaces[i]
            base_normal = normales_uniques[np.argmax(surfaces_par_normale)]

            R = get_rotation_matrix_from_vectors(base_normal, np.array([0.0, 0.0, -1.0]))
            T_rot = np.eye(4)
            T_rot[0:3, 0:3] = R
            clip.transform(T_rot, inplace=True)

            bounds = clip.bounds
            cx, cy, z_contact = (bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, bounds[4]
            clip.translate([-cx, -cy, -z_contact], inplace=True)

            self._crabe_clip_cache[stl_path] = clip
            print(f"🦀 Clip visuel chargé et normalisé : {stl_path}")
        except Exception as e:
            print(f"⚠️ Impossible de charger/normaliser le clip visuel '{stl_path}' : {e}")
            self._crabe_clip_cache[stl_path] = None

        return self._crabe_clip_cache[stl_path]

    def _build_crabe_polydata(self, crabes_list, stl_path):
        template = self._get_crabe_clip_template(stl_path)
        if template is None or not crabes_list:
            return pv.PolyData()

        meshes = []
        for crabe in crabes_list:
            T = np.eye(4)
            T[0:3, 0] = crabe["x_axis"]
            T[0:3, 1] = crabe["y_axis"]
            T[0:3, 2] = crabe["normal"]
            T[0:3, 3] = crabe["surface_position"]

            instance = template.copy()
            instance.transform(T, inplace=True)
            meshes.append(instance)

        if not meshes:
            return pv.PolyData()
        if len(meshes) == 1:
            return meshes[0]
        return meshes[0].merge(meshes[1:])

    def _update_loop(self):
        if not getattr(self, 'plotter', None) or not hasattr(self.plotter, 'render_window'): return

        if self._render_suspended or self._loop_busy:
            self.view.after(50, self._update_loop)
            return

        self._loop_busy = True
        try:
            self._update_loop_body()
        finally:
            self._loop_busy = False

    def _update_loop_body(self):
        page = self.view.pages.get("agent")

        current_focus = self.view.focus_get()
        is_typing = current_focus and ('entry' in str(current_focus).lower() or isinstance(current_focus, tk.Entry))
        if is_typing:
            self.view.after(100, self._update_loop)
            return

        self.frame_count += 1

        do_graphs = (self.frame_count % 10 == 0)
        snapshot = {}
        with self.data_lock:
            is_playing = self.shared_state.get("is_playing", False)
            crabe_stl_path = self.shared_state["config"].get("crabe_stl_path", "")
            if is_playing:
                for name in self.benchmark_algos.keys():
                    if "algos" not in self.shared_state or name not in self.shared_state["algos"]: continue
                    st = self.shared_state["algos"][name]
                    snap = {
                        "waypoints": st["waypoints"].copy(),
                        "iteration": st["iteration"], "collisions": st["collisions"],
                        "details": dict(st["reward_details"]), "done": st["done"], "msg": st["stuck_msg"],
                        "min_dist": st["min_dist"], "max_dist": st["max_dist"], "mean_dist": st["mean_dist"],
                        "crabe_count": st.get("crabe_count", 0),
                    }
                    if do_graphs:
                        snap["crabes"] = list(st.get("crabes", []))
                        snap["reward_history"] = list(st["reward_history"])
                        snap["dist_history"] = list(st["dist_history"])
                        snap["collision_history"] = list(st["collision_history"])
                        snap["point_count_history"] = list(st["point_count_history"])
                        snap["crabe_count_history"] = list(st.get("crabe_count_history", []))
                        snap["crossing_history"] = list(st.get("crossing_history", []))
                        snap["length_history"] = list(st.get("length_history", []))
                    snapshot[name] = snap

        if is_playing:
            for name, snap in snapshot.items():
                new_wps = snap["waypoints"]

                self.plotter.add_mesh(pv.lines_from_points(new_wps),
                                      name=f"traj_{name}",
                                      color=self.benchmark_algos[name]["color"],
                                      line_width=8, label=name)

                if do_graphs:
                    crabe_poly = self._build_crabe_polydata(snap["crabes"], crabe_stl_path)
                    if crabe_poly.n_points > 0:
                        self.plotter.add_mesh(crabe_poly,
                                              name=f"crabes_{name}",
                                              color=self.benchmark_algos[name]["color"],
                                              opacity=0.9, show_edges=True, edge_color="black")
                    else:
                        if f"crabes_{name}" in self.plotter.actors:
                            self.plotter.remove_actor(f"crabes_{name}")

                details = snap["details"]
                det_txt = (f"Marg:{details.get('Margin', 0):.1f} | Fuit:{details.get('Towards', 0):.1f} | "
                           f"Liss:{details.get('Smooth', 0):.1f} | Cra:{details.get('Crash', 0):.1f} | "
                           f"Cap:{details.get('Sensor', 0):.1f} | 🦀:{snap['crabe_count']}\n"
                           f"Dist -> Min: {snap['min_dist']:.1f}mm | Max: {snap['max_dist']:.1f}mm | "
                           f"Moy: {snap['mean_dist']:.1f}mm")
                color_police = "#28A745" if snap["collisions"] == 0 else "#DC3545"

                if page and name in page.algo_widgets:
                    if snap["done"]:
                        page.algo_widgets[name].configure(text=f"✅ Convergé (it: {snap['iteration']})",
                                                          text_color="#28A745")
                    else:
                        page.algo_widgets[name].configure(
                            text=f"Iter:{snap['iteration']} | Vrais Clashs (Raw):{snap['collisions']}"
                                 f"{snap['msg']}\n{det_txt}",
                            text_color=color_police)

                if do_graphs:
                    if "reward" in self.axes_registry: self.lines_registry["reward"][name].set_data(
                        range(len(snap["reward_history"])), snap["reward_history"])
                    if "dist" in self.axes_registry: self.lines_registry["dist"][name].set_data(
                        range(len(snap["dist_history"])), snap["dist_history"])
                    if "col" in self.axes_registry:
                        self.lines_registry["col"][name].set_data(
                            range(len(snap["collision_history"])), snap["collision_history"])
                        if name in self.lines_registry.get("cross", {}):
                            self.lines_registry["cross"][name].set_data(
                                range(len(snap["crossing_history"])), snap["crossing_history"])
                    if "length" in self.axes_registry: self.lines_registry["length"][name].set_data(
                        range(len(snap["length_history"])), snap["length_history"])
                    if "pts" in self.axes_registry: self.lines_registry["pts"][name].set_data(
                        range(len(snap["point_count_history"])), snap["point_count_history"])
                    if "crabes" in self.axes_registry: self.lines_registry["crabes"][name].set_data(
                        range(len(snap["crabe_count_history"])), snap["crabe_count_history"])

        if do_graphs and page and is_playing:
            for key, ax in self.axes_registry.items():
                hists = []
                max_iterations = 0

                for name, snap in snapshot.items():
                    if key == "reward":
                        hist = snap["reward_history"]
                    elif key == "dist":
                        hist = snap["dist_history"]
                    elif key == "col":
                        hist = snap["collision_history"]
                        if snap.get("crossing_history"):
                            hists.append(snap["crossing_history"])
                    elif key == "length":
                        hist = snap["length_history"]
                    elif key == "pts":
                        hist = snap["point_count_history"]
                    elif key == "crabes":
                        hist = snap["crabe_count_history"]
                    else:
                        hist = []

                    if len(hist) > 0:
                        hists.append(hist)
                        max_iterations = max(max_iterations, len(hist))

                if hists:
                    ax.set_xlim(0, max_iterations + 5)
                    ymin = min(min(h) for h in hists)
                    ymax = max(max(h) for h in hists)
                    padding = max(1.0, (ymax - ymin) * 0.15)
                    ax.set_ylim(ymin - padding, ymax + padding)

            self.canvas.draw_idle()

        if not self._render_suspended and getattr(self, 'plotter', None) is not None:
            try:
                if hasattr(self.plotter, 'render_window') and is_playing:
                    self.plotter.render()
            except Exception as e:
                pass

        self.view.after(30 if is_playing else 100, self._update_loop)

    def update_graph_layouts(self):
        page = self.view.pages.get("agent")
        if not page or self.fig is None: return

        active_plots = []
        if page.chk_reward.get(): active_plots.append("reward")
        if page.chk_dist.get(): active_plots.append("dist")
        if page.chk_col.get(): active_plots.append("col")
        if hasattr(page, "chk_length") and page.chk_length.get(): active_plots.append("length")
        if page.chk_pts.get(): active_plots.append("pts")
        if page.chk_crabes.get(): active_plots.append("crabes")

        self.fig.clear()
        self.axes_registry.clear()
        self.lines_registry = {"reward": {}, "dist": {}, "col": {}, "cross": {}, "length": {}, "pts": {}, "crabes": {}}

        n = len(active_plots)
        if n == 0:
            self.canvas.draw_idle()
            return

        for idx, key in enumerate(active_plots):
            ax = self.fig.add_subplot(n, 1, idx + 1)
            ax.tick_params(colors="#8B949E", labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.3)

            if key == "reward":
                ax.set_title("Évolution des Récompenses", fontsize=9, color="#8B949E", pad=4)
                ax.set_ylabel("Reward Mean", fontsize=8, color="#8B949E")
            elif key == "dist":
                ax.set_title("Distance Moyenne au Maillage (mm)", fontsize=9, color="#8B949E", pad=4)
                ax.set_ylabel("Distance (mm)", fontsize=8, color="#8B949E")
                try:
                    ax.axhline(float(page.entry_min.get().replace(',', '.')), color='red', linestyle=':', alpha=0.5)
                    ax.axhline(float(page.entry_max.get().replace(',', '.')), color='lightblue', linestyle=':',
                               alpha=0.5)
                except Exception:
                    pass
            elif key == "col":
                ax.set_title("Violations de marge (plein) / Traversées de faces - test exact (pointillé)",
                             fontsize=9, color="#8B949E", pad=4)
                ax.set_ylabel("Nb Anomalies", fontsize=8, color="#8B949E")
            elif key == "length":
                ax.set_title("Longueur de la Trajectoire (mm)", fontsize=9, color="#8B949E", pad=4)
                ax.set_ylabel("Longueur (mm)", fontsize=8, color="#8B949E")
            elif key == "pts":
                ax.set_title("Nombre de Points sur la Trajectoire", fontsize=9, color="#8B949E", pad=4)
                ax.set_ylabel("Points", fontsize=8, color="#8B949E")
            elif key == "crabes":
                ax.set_title("Nombre de Fixations (Crabes) Posées", fontsize=9, color="#8B949E", pad=4)
                ax.set_ylabel("Nb Fixations", fontsize=8, color="#8B949E")

            self.axes_registry[key] = ax
            for name, config in self.benchmark_algos.items():
                line, = ax.plot([], [], label=name, color=config["color"], alpha=0.8)
                self.lines_registry[key][name] = line
                if key == "col":
                    cross_line, = ax.plot([], [], color=config["color"], alpha=0.9,
                                          linestyle="--", linewidth=1.0)
                    self.lines_registry["cross"][name] = cross_line

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()

    def update_3d_visibility(self):
        if getattr(self, 'plotter', None) is None: return
        page = self.view.pages.get("agent")
        if not page: return

        try:
            show_mesh = page.chk_mesh.get()
            show_edges = page.chk_edges.get()
            show_bbox = page.chk_bbox.get()

            if hasattr(self, 'mesh_actor') and self.mesh_actor:
                self.mesh_actor.SetVisibility(show_mesh)
                self.mesh_actor.prop.show_edges = show_edges

            if hasattr(self, 'bbox_actor') and self.bbox_actor:
                self.bbox_actor.SetVisibility(show_bbox)

            if hasattr(self.plotter, 'render_window'):
                self.plotter.render()
        except Exception as e:
            print(f"⚠️ Erreur d'affichage 3D : {e}")

    def toggle_popout_3d(self):
        page = self.view.pages.get("agent")
        if not page: return

        if not self.is_3d_popped_out:
            self.is_3d_popped_out = True
            self.popout_window = ctk.CTkToplevel(self.view)
            self.popout_window.title("Vue 3D - Mode Plein Écran")
            self.popout_window.geometry("1024x768")
            self.popout_window.protocol("WM_DELETE_WINDOW", self.toggle_popout_3d)

            page.vtk_container.pack_forget()
            page.vtk_container.pack(in_=self.popout_window, fill="both", expand=True, padx=2, pady=2)
        else:
            self.is_3d_popped_out = False
            page.vtk_container.pack_forget()
            page.vtk_container.pack(in_=page.view_3d_frame, fill="both", expand=True, padx=2, pady=2)
            if self.popout_window:
                self.popout_window.destroy()
                self.popout_window = None

        page.update_language(self.view.current_lang)

    def toggle_agent_training(self):
        if getattr(self, 'is_scanning', False):
            print("⚠️ Action bloquée : Veuillez attendre la fin du scan du radar Open3D.")
            return

        page = self.view.pages.get("agent")
        if not page: return

        if not self.agent_initialized:
            print("🧊 Chargement de la 3D et lancement du Radar...")
            self._init_agent_system(page)
            return

        self._ensure_algo_widgets(page)

        will_play = not self.shared_state["is_playing"]
        if will_play and not self.threads_started:
            self._sync_parameters(page)
            coords_src = parse_coords(page.entry_src.get())
            coords_tgt = parse_coords(page.entry_tgt.get())
            if coords_src is not None: self.point_A = coords_src
            if coords_tgt is not None: self.point_B = coords_tgt

            if hasattr(self, 'src_actor') and self.src_actor: self.plotter.remove_actor(self.src_actor)
            if hasattr(self, 'tgt_actor') and self.tgt_actor: self.plotter.remove_actor(self.tgt_actor)
            self.src_actor = self.plotter.add_mesh(pv.Sphere(radius=30, center=self.point_A), color='green')
            self.tgt_actor = self.plotter.add_mesh(pv.Sphere(radius=30, center=self.point_B), color='red')

            cfg = self.shared_state["config"]
            inter_pts = []

            # 🔥 CORRECTION : SELECTION INTELLIGENTE DE L'ENCOCHE
            if cfg.get("force_fixations", False):
                existing_crabes = cfg.get("existing_crabes", [])

                # Vecteur global du trajet pour orienter le passage
                A = np.array(self.point_A)
                B = np.array(self.point_B)
                vec_AB = B - A
                len_AB_sq = np.dot(vec_AB, vec_AB) + 1e-8

                for c in existing_crabes:
                    if "routing_points" in c and c["routing_points"]:
                        # C'est un peigne : on regroupe les points par paires (Vert -> Rouge)
                        pts = c["routing_points"]
                        pairs = [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]

                        best_pair = None
                        min_dist = float('inf')

                        # 1. On cherche l'encoche la plus proche du trajet naturel A -> B
                        for p_in, p_out in pairs:
                            center = (p_in + p_out) / 2.0
                            t = max(0, min(1, np.dot(center - A, vec_AB) / len_AB_sq))
                            proj = A + t * vec_AB
                            dist = np.linalg.norm(center - proj)

                            if dist < min_dist:
                                min_dist = dist
                                best_pair = (p_in, p_out)

                        # 2. On ajoute uniquement LA MEILLEURE encoche, dans le bon sens !
                        if best_pair:
                            p_in, p_out = best_pair
                            vec_slot = p_out - p_in
                            # Si le segment de l'encoche est à l'envers par rapport à A->B, on l'inverse
                            if np.dot(vec_AB, vec_slot) < 0:
                                inter_pts.extend([p_out, p_in])  # Traversée inversée
                            else:
                                inter_pts.extend([p_in, p_out])  # Traversée normale
                    else:
                        # Fixation classique (Crabe)
                        inter_pts.append(c["position"])

            self.initial_waypoints = generate_dense_waypoints(
                self.point_A, self.point_B, cfg["initial_points"],
                mesh=self.mesh, intermediate_points=inter_pts
            )

            with self.data_lock:
                if "algos" not in self.shared_state: self.shared_state["algos"] = {}
                for name in self.benchmark_algos.keys():
                    self.shared_state["algos"][name] = {
                        "iteration": 0, "initial_waypoints": self.initial_waypoints.copy(),
                        "waypoints": self.initial_waypoints.copy(), "density": cfg["initial_points"],
                        "collisions": 1, "reward": 0.0, "reward_history": [],
                        "reward_details": {"Margin": 0, "Towards": 0, "Smooth": 0, "Crash": 0, "Sensor": 0,
                                           "Flat": 0, "Ahead": 0, "Crabe": 0},
                        "done": False, "stuck_msg": "", "min_dist": 0.0, "max_dist": 0.0, "mean_dist": 0.0,
                        "dist_history": [], "collision_history": [], "point_count_history": [],
                        "prev_disp": None, "crabes": [], "crabe_count": 0, "crabe_count_history": []
                    }

            for name, config in self.benchmark_algos.items():
                self.plotter.add_mesh(pv.lines_from_points(self.initial_waypoints),
                                      name=f"traj_{name}",
                                      color=config["color"], line_width=8, label=name)

            self.plotter.add_legend(bcolor=None, face=None, size=(0.15, 0.15))

            print("🧊 Préparation des environnements 100% isolés pour chaque agent...")
            agent_meshes = {}
            for name in self.benchmark_algos.keys():
                m_copy = trimesh.Trimesh(
                    vertices=np.array(self.mesh.vertices, copy=True),
                    faces=np.array(self.mesh.faces, copy=True)
                )
                m_copy.merge_vertices(merge_tex=False, merge_norm=False)
                m_copy.fix_normals()

                _ = m_copy.face_normals
                _ = m_copy.kdtree
                pq = trimesh.proximity.ProximityQuery(m_copy)
                _ = pq.on_surface(np.array([[0.0, 0.0, 0.0]]))
                agent_meshes[name] = m_copy

            for name in self.benchmark_algos.keys():
                threading.Thread(target=algo_worker, args=(
                    name,
                    self.benchmark_algos,
                    self.initial_waypoints,
                    agent_meshes[name],
                    self.shared_state,
                    self.data_lock,
                    None,
                    self.point_A,
                    self.point_B
                ), daemon=True).start()

            self.threads_started = True

        with self.data_lock:
            self.shared_state["is_playing"] = will_play
            page.update_language(self.view.current_lang)

        self.view.focus_set()

    def reset_agent_training(self):
        if getattr(self, 'is_scanning', False): return

        page = self.view.pages.get("agent")
        if not page or not self.agent_initialized: return
        self._ensure_algo_widgets(page)
        self._sync_parameters(page, force_3d_update=True)
        coords_src = parse_coords(page.entry_src.get())
        coords_tgt = parse_coords(page.entry_tgt.get())
        if coords_src is not None: self.point_A = coords_src
        if coords_tgt is not None: self.point_B = coords_tgt

        with self.data_lock:
            self.shared_state["is_playing"] = False
            n_init = int(self.shared_state["config"]["initial_points"])
            cfg = self.shared_state["config"]

            inter_pts = []

            # 🔥 CORRECTION : LA MÊME QUE POUR LE TOGGLE
            if cfg.get("force_fixations", False):
                existing_crabes = cfg.get("existing_crabes", [])

                A = np.array(self.point_A)
                B = np.array(self.point_B)
                vec_AB = B - A
                len_AB_sq = np.dot(vec_AB, vec_AB) + 1e-8

                for c in existing_crabes:
                    if "routing_points" in c and c["routing_points"]:
                        pts = c["routing_points"]
                        pairs = [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]

                        best_pair = None
                        min_dist = float('inf')

                        for p_in, p_out in pairs:
                            center = (p_in + p_out) / 2.0
                            t = max(0, min(1, np.dot(center - A, vec_AB) / len_AB_sq))
                            proj = A + t * vec_AB
                            dist = np.linalg.norm(center - proj)

                            if dist < min_dist:
                                min_dist = dist
                                best_pair = (p_in, p_out)

                        if best_pair:
                            p_in, p_out = best_pair
                            vec_slot = p_out - p_in
                            if np.dot(vec_AB, vec_slot) < 0:
                                inter_pts.extend([p_out, p_in])
                            else:
                                inter_pts.extend([p_in, p_out])
                    else:
                        inter_pts.append(c["position"])

        self.initial_waypoints = generate_dense_waypoints(self.point_A, self.point_B, n_init, mesh=self.mesh,
                                                          intermediate_points=inter_pts)
        if hasattr(self, 'src_actor') and self.src_actor: self.plotter.remove_actor(self.src_actor)
        if hasattr(self, 'tgt_actor') and self.tgt_actor: self.plotter.remove_actor(self.tgt_actor)
        self.src_actor = self.plotter.add_mesh(pv.Sphere(radius=30, center=self.point_A), color='green')
        self.tgt_actor = self.plotter.add_mesh(pv.Sphere(radius=30, center=self.point_B), color='red')

        with self.data_lock:
            cfg = self.shared_state["config"]
            for name in self.benchmark_algos.keys():
                if name in self.shared_state.get("algos", {}):
                    self.shared_state["algos"][name].update({
                        "iteration": 0, "initial_waypoints": self.initial_waypoints.copy(),
                        "waypoints": self.initial_waypoints.copy(),
                        "density": cfg["initial_points"], "collisions": 1, "reward": 0.0, "reward_history": [],
                        "reward_details": {"Margin": 0, "Towards": 0, "Smooth": 0, "Crash": 0, "Sensor": 0,
                                           "Flat": 0, "Ahead": 0, "Crabe": 0},
                        "done": False, "stuck_msg": "",
                        "min_dist": 0.0, "max_dist": 0.0, "mean_dist": 0.0, "dist_history": [], "collision_history": [],
                        "point_count_history": [], "prev_disp": None,
                        "crabes": [], "crabe_count": 0, "crabe_count_history": []
                    })

                    self.plotter.add_mesh(pv.lines_from_points(self.initial_waypoints),
                                          name=f"traj_{name}",
                                          color=self.benchmark_algos[name]["color"], line_width=8, label=name)
                    if f"crabes_{name}" in self.plotter.actors:
                        self.plotter.remove_actor(f"crabes_{name}")

            self.plotter.add_legend(bcolor=None, face=None, size=(0.15, 0.15))

        for name in self.benchmark_algos.keys():
            if name in page.algo_widgets:
                page.algo_widgets[name].configure(text="En attente...", text_color=("#50565E", "#8B949E"))

        self.update_graph_layouts()

        try:
            self.plotter.render()
        except Exception:
            pass

        page.update_language(self.view.current_lang)

        self.view.focus_set()

    def save_trajectories_to_stl(self):
        page = self.view.pages.get("agent")
        if not page: return
        folder = filedialog.askdirectory(title="📍 Dossier d'exportation .stl")
        if not folder: return
        with self.data_lock:
            crabe_stl_path = self.shared_state["config"].get("crabe_stl_path", "")
            export_data = {}
            for name in self.benchmark_algos.keys():
                if name not in self.shared_state.get("algos", {}): continue
                export_data[name] = (
                    self.shared_state["algos"][name]["waypoints"].copy(),
                    list(self.shared_state["algos"][name].get("crabes", [])),
                )

        for name, (waypoints, crabes) in export_data.items():
            if len(waypoints) < 2: continue
            try:
                tube_3d = pv.lines_from_points(waypoints).tube(radius=6.0, n_sides=16)

                if crabes:
                    crabe_poly = self._build_crabe_polydata(crabes, crabe_stl_path)
                    if crabe_poly.n_points > 0:
                        tube_3d = tube_3d + crabe_poly

                export_path = os.path.join(folder, f"Routage_Complet_{name}.stl")
                tube_3d.save(export_path)
                print(f"✅ Export complet sauvegardé : {export_path}")
            except Exception as e:
                print(f"⚠️ Échec Export pour {name} : {e}")

    def update_step_visual(self, step_index):
        print(f"Passage à l'étape {step_index}")

    def handle_extraction(self, params):
        page = self.view.pages["extraction"]
        page.show_progress(True)
        page.btn_run.configure(state="disabled")
        page.btn_cancel.configure(state="normal")

        def run_task():
            q = Queue()
            target_dir = params["stl_folder"]
            use_catia = not params["use_existing"]
            exclude_filter = params["exclude_filter"]

            from core.mesh_processor import extraction_worker
            self.extraction_process = Process(target=extraction_worker, args=(target_dir, use_catia, exclude_filter, q))
            self.extraction_process.start()

            start_time = time.time()
            last_msg_prefix = ""
            last_ui_update = 0

            while True:
                try:
                    res = q.get(timeout=0.1)
                except Exception:
                    if self.extraction_process and not self.extraction_process.is_alive(): break
                    continue

                status = res[0]
                if status == "UPDATE":
                    now = time.time()
                    if now - last_ui_update < 0.06:
                        continue
                    last_ui_update = now

                    msg, current, total, pct = res[1], res[2], res[3], res[4]
                    current_msg_prefix = msg.split(":")[0] if ":" in msg else msg

                    if current_msg_prefix != last_msg_prefix:
                        start_time = now
                        last_msg_prefix = current_msg_prefix

                    elapsed = now - start_time
                    if current > 0 and elapsed > 0:
                        it_s = current / elapsed
                        rem_s = (total - current) / it_s
                        min_e, sec_e = divmod(int(elapsed), 60)
                        min_r, sec_r = divmod(int(rem_s), 60)
                        tqdm_txt = f"[{min_e:02}:{sec_e:02}<{min_r:02}:{sec_r:02}, {it_s:.2f}it/s]"
                        status_msg = f"{msg}\n📦 {current}/{total} {tqdm_txt}"
                    else:
                        status_msg = msg

                    self.view.after(0, lambda m=status_msg, v=pct: page.update_status(m, v))

                elif status == "SUCCESS":
                    _, df, n_points, n_cells, bounds, fusion_path = res
                    self.view.after(0, lambda: page.display_results(df, n_points, n_cells, bounds, fusion_path))
                    break

                elif status == "ERROR":
                    err_msg = res[1]
                    self.view.after(0, lambda: page.update_status(f"❌ Erreur: {err_msg}", 0))
                    break

            if self.extraction_process:
                self.extraction_process.join()

            self.view.after(0, lambda: page.btn_cancel.configure(state="disabled"))
            self.view.after(0, lambda: page.btn_run.configure(state="normal"))

        threading.Thread(target=run_task, daemon=True).start()

    def cancel_extraction(self):
        if self.extraction_process and self.extraction_process.is_alive():
            self.extraction_process.terminate()
            page = self.view.pages["extraction"]
            page.update_status("🛑 Processus annulé par l'utilisateur", 0)
            page.show_progress(False)
            page.btn_run.configure(state="normal")
            page.btn_cancel.configure(state="disabled")
