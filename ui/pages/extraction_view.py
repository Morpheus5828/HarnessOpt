import os

import customtkinter as ctk
import tkinter as tk
import time
import config as cfg
import pyvista as pv
import traceback
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class ExtractionPage(ctk.CTkFrame):
    def __init__(self, parent, main_window):
        super().__init__(parent, fg_color="transparent")
        self.main_window = main_window

        self.df_result = None
        self.bounds_result = None
        self.fusion_path = None
        self.start_time = None

        self.columnconfigure(0, weight=1)

        self.card = ctk.CTkFrame(
            self,
            fg_color=cfg.COLOR_CARD_BG,
            border_color=cfg.COLOR_CARD_BORDER,
            border_width=1,
            corner_radius=15
        )
        self.card.pack(padx=50, pady=20, fill="both", expand=True)

        self.lbl_title = ctk.CTkLabel(
            self.card,
            text="📂 DMU EXTRACTION & FUSION",
            font=("Inter", 20, "bold"),
            text_color=cfg.COLOR_ACCENT
        )
        self.lbl_title.pack(pady=(30, 5))

        self.stats_summary_label = ctk.CTkLabel(
            self.card,
            text="📊 Pièces: 0 | 🏗️ Noeuds: 0 | 📐 Faces: 0",
            font=("Inter", 13, "bold"),
            text_color=cfg.COLOR_TEXT_SUB
        )
        self.stats_summary_label.pack(pady=(0, 10))


        self.chk_existing_var = tk.BooleanVar(value=False)
        self.chk_existing = ctk.CTkCheckBox(
            self.card,
            text="J'ai déjà mes STL (Ignorer l'export CATIA)",
            variable=self.chk_existing_var,
            font=("Inter", 12),
            text_color=cfg.COLOR_TEXT_MAIN,
            fg_color=cfg.COLOR_ACCENT,
            command=self._toggle_mode
        )
        self.chk_existing.pack(pady=(15, 10))

        # --- CONTENEUR POUR L'EXPORT CATIA (S'affiche par défaut) ---
        self.frame_catia_export = ctk.CTkFrame(self.card, fg_color="transparent")
        self.frame_catia_export.pack(pady=5, fill="x")

        self.entry_exclude = ctk.CTkEntry(
            self.frame_catia_export,
            placeholder_text="Filtres d'exclusion (ex: U258*)",
            width=450,
            height=35,
            fg_color=cfg.COLOR_ENTRY_BG,
            border_color=cfg.COLOR_CARD_BORDER,
            text_color=cfg.COLOR_TEXT_MAIN
        )
        self.entry_exclude.pack(pady=5)

        # --- CONTENEUR POUR LE CHEMIN STL (Masqué par défaut) ---
        self.frame_existing_stl = ctk.CTkFrame(self.card, fg_color="transparent")

        self.entry_stl_folder = ctk.CTkEntry(
            self.frame_existing_stl,
            placeholder_text="Chemin du dossier STL",
            width=450,
            height=35,
            fg_color=cfg.COLOR_ENTRY_BG,
            border_color=cfg.COLOR_CARD_BORDER,
            text_color=cfg.COLOR_TEXT_MAIN
        )
        self.entry_stl_folder.insert(0, r"C:\Users\a609568\Desktop\STL fuselage H160")
        self.entry_stl_folder.pack(pady=5)
        # ==========================================

        # --- ZONE BOUTONS (RUN & CANCEL) ---
        self.btn_container = ctk.CTkFrame(self.card, fg_color="transparent")
        self.btn_container.pack(pady=15, padx=40, fill="x")

        self.btn_run = ctk.CTkButton(
            self.btn_container, text="🚀 RUN LOAD & FUSION", command=self.on_run_click,
            height=45, font=("Inter", 14, "bold"), fg_color=cfg.COLOR_ACCENT
        )
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_cancel = ctk.CTkButton(
            self.btn_container, text="🛑 ANNULER", command=self.on_cancel_click,
            height=45, font=("Inter", 14, "bold"), fg_color="#E63946", hover_color="#C0392B", state="disabled"
        )
        self.btn_cancel.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # --- ZONE DE PROGRESSION CENTRÉE ---
        self.status_container = ctk.CTkFrame(self.card, fg_color="transparent")

        self.lbl_timer = ctk.CTkLabel(
            self.status_container, text="⏱️ 00:00  |  ETA: --:--",
            font=("JetBrains Mono", 20, "bold"), text_color=cfg.COLOR_TEXT_MAIN
        )
        self.lbl_timer.pack(pady=(10, 5))

        self.progress = ctk.CTkProgressBar(
            self.status_container,
            width=500,
            mode="determinate",
            fg_color=cfg.COLOR_INACTIVE,
            progress_color=cfg.COLOR_ACCENT
        )
        self.progress.pack(pady=5)
        self.progress.set(0)

        self.lbl_step = ctk.CTkLabel(
            self.status_container, text="Prêt pour l'extraction",
            font=("Inter", 14, "bold"), text_color=cfg.COLOR_TEXT_SUB
        )
        self.lbl_step.pack(pady=(5, 10))

        # --- ZONE ACTIONS POST-TRAITEMENT ---
        self.post_action_frame = ctk.CTkFrame(self.card, fg_color="transparent")

        self.btn_show_colors = ctk.CTkButton(
            self.post_action_frame,
            text="📊 Afficher Stats Couleurs",
            command=self.show_colors_chart,
            width=200,
            fg_color=cfg.COLOR_INACTIVE,
            text_color=cfg.COLOR_TEXT_MAIN
        )
        self.btn_show_colors.grid(row=0, column=0, padx=10)

        self.btn_show_mesh = ctk.CTkButton(
            self.post_action_frame,
            text="👁️ Afficher Rendu 3D",
            command=self.show_3d_mesh,
            width=200,
            fg_color=cfg.COLOR_INACTIVE,
            text_color=cfg.COLOR_TEXT_MAIN
        )
        self.btn_show_mesh.grid(row=0, column=1, padx=10)

        # --- ZONE DE RÉSULTAT (Graphiques) ---
        self.result_zone = ctk.CTkFrame(self.card, fg_color="transparent")
        self.result_zone.pack(fill="both", expand=True, pady=10, padx=20)

        # 💡 SOLUTION: On retire la grille rigide. Les frames sont juste créées, elles seront "packed" dynamiquement
        self.frame_pie = ctk.CTkFrame(self.result_zone, fg_color="transparent")
        self.frame_mesh = ctk.CTkFrame(self.result_zone, fg_color="transparent")

    def update_language(self, lang):
        if lang == "EN":
            self.lbl_title.configure(text="📂 DMU EXTRACTION & FUSION")
            self.entry_exclude.configure(placeholder_text="Exclusion filters (e.g., U258*)")
            self.chk_existing.configure(text="I already have my STLs (Skip CATIA export)")
            self.entry_stl_folder.configure(placeholder_text="STL folder path")

            if "NOUVELLE" in self.btn_run.cget("text"):
                self.btn_run.configure(text="🚀 NEW EXTRACTION")
            elif "COURS" not in self.btn_run.cget("text"):
                self.btn_run.configure(text="🚀 RUN LOAD & FUSION")

            self.btn_cancel.configure(text="🛑 CANCEL")

            if self.lbl_step.cget("text") == "Prêt pour l'extraction":
                self.lbl_step.configure(text="Ready for extraction")

            self.btn_show_colors.configure(text="📊 Show Color Stats")
            self.btn_show_mesh.configure(text="👁️ Show 3D Render")

            if hasattr(self, 'lbl_bbox_title') and self.lbl_bbox_title.winfo_exists():
                self.lbl_bbox_title.configure(text="DETECTED GEOMETRIC BOUNDS (BBOX)")
        else:
            self.lbl_title.configure(text="📂 DMU EXTRACTION & FUSION")
            self.entry_exclude.configure(placeholder_text="Filtres d'exclusion (ex: U258*)")
            self.chk_existing.configure(text="J'ai déjà mes STL (Ignorer l'export CATIA)")
            self.entry_stl_folder.configure(placeholder_text="Chemin du dossier STL")

            if "NEW" in self.btn_run.cget("text"):
                self.btn_run.configure(text="🚀 NOUVELLE EXTRACTION")
            elif "PROCESSING" not in self.btn_run.cget("text"):
                self.btn_run.configure(text="🚀 RUN LOAD & FUSION")

            self.btn_cancel.configure(text="🛑 ANNULER")

            if self.lbl_step.cget("text") == "Ready for extraction":
                self.lbl_step.configure(text="Prêt pour l'extraction")

            self.btn_show_colors.configure(text="📊 Afficher Stats Couleurs")
            self.btn_show_mesh.configure(text="👁️ Afficher Rendu 3D")

            if hasattr(self, 'lbl_bbox_title') and self.lbl_bbox_title.winfo_exists():
                self.lbl_bbox_title.configure(text="LIMITES GÉOMÉTRIQUES DÉTECTÉES (BBOX)")

    def on_cancel_click(self):
        if self.main_window.controller:
            self.main_window.controller.cancel_extraction()

    def update_accent_color(self, hex_color):
        self.lbl_title.configure(text_color=hex_color)
        self.btn_run.configure(fg_color=hex_color)
        self.progress.configure(progress_color=hex_color)
        if hasattr(self, 'lbl_bbox_title') and self.lbl_bbox_title.winfo_exists():
            self.lbl_bbox_title.configure(text_color=hex_color)

    def _toggle_mode(self):
        if self.chk_existing_var.get():
            self.frame_catia_export.pack_forget()
            self.frame_existing_stl.pack(pady=5, fill="x", before=self.btn_container)
        else:
            self.frame_existing_stl.pack_forget()
            self.frame_catia_export.pack(pady=5, fill="x", before=self.btn_container)

    def on_run_click(self):
        # 💡 On nettoie ET on masque les frames de résultats pour remettre à zéro l'UI
        for w in self.frame_pie.winfo_children(): w.destroy()
        self.frame_pie.pack_forget()

        for w in self.frame_mesh.winfo_children(): w.destroy()
        self.frame_mesh.pack_forget()

        if hasattr(self, 'bbox_card'): self.bbox_card.pack_forget()
        self.post_action_frame.pack_forget()

        stl_path = self.entry_stl_folder.get().strip()
        nb_stl = len([f for f in os.listdir(stl_path) if f.endswith('.stl')]) if os.path.exists(stl_path) else 0

        if self.main_window.controller:
            params = {
                "use_existing": self.chk_existing_var.get(),
                "exclude_filter": self.entry_exclude.get(),
                "stl_folder": stl_path,
                "nb_stl": nb_stl
            }

            self.start_time = time.time()
            self.show_progress(True)

            is_en = getattr(self.main_window, 'current_lang', 'FR') == "EN"
            self.btn_run.configure(state="disabled", text="⏳ PROCESSING..." if is_en else "⏳ EN COURS...")
            self.btn_cancel.configure(state="normal")

            self.main_window.controller.handle_extraction(params)

    def show_progress(self, show=True):
        is_en = getattr(self.main_window, 'current_lang', 'FR') == "EN"
        if show:
            self.status_container.pack(fill="x", pady=10)
            self.progress.set(0)
            self.lbl_timer.configure(text="⏱️ 00:00  |  ETA: --:--")
            self.lbl_step.configure(text="Initialization..." if is_en else "Initialisation...")
        else:
            self.status_container.pack_forget()

    def update_status(self, text, progress_ratio=None):
        self.lbl_step.configure(text=text)

        if progress_ratio is not None:
            self.progress.set(progress_ratio)

            if self.start_time and progress_ratio > 0:
                elapsed_time = time.time() - self.start_time

                if progress_ratio > 0.01:
                    total_estimated_time = elapsed_time / progress_ratio
                    eta_seconds = total_estimated_time - elapsed_time

                    def format_time(seconds):
                        if seconds < 0: return "00:00"
                        mins, secs = divmod(int(seconds), 60)
                        return f"{mins:02d}:{secs:02d}"

                    elapsed_str = format_time(elapsed_time)
                    eta_str = format_time(eta_seconds)

                    self.update_timer(elapsed_str, eta_str)

    def update_timer(self, elapsed_str, eta_str="--:--"):
        self.lbl_timer.configure(text=f"⏱️ {elapsed_str}  |  ETA: {eta_str}")

    def show_colors_chart(self):
        if self.df_result is None: return

        # 💡 On "pack" le conteneur du graphique. S'il est tout seul, il prendra 100% de la largeur !
        self.frame_pie.pack(side="left", expand=True, fill="both", padx=5)

        for w in self.frame_pie.winfo_children(): w.destroy()

        is_en = getattr(self.main_window, 'current_lang', 'FR') == "EN"
        self.btn_show_colors.configure(state="disabled", text="✅ Stats Displayed" if is_en else "✅ Stats Affichées")

        try:


            clean_colors = self.df_result['Color'].fillna("Inconnu").astype(str)
            stats = clean_colors.value_counts()

            is_dark = ctk.get_appearance_mode() == "Dark"

            def resolve_color(c):
                if isinstance(c, (list, tuple)):
                    return c[1] if is_dark else c[0]
                if c == "transparent":
                    return "#1a1a1a" if is_dark else "#f0f0f0"
                return c

            bg_color = resolve_color(self.card.cget("fg_color"))
            accent_color = resolve_color(cfg.COLOR_ACCENT)
            text_color = "white" if is_dark else "black"

            fig_pie, ax_pie = plt.subplots(figsize=(5, 5), facecolor=bg_color)

            wedges, texts, autotexts = ax_pie.pie(
                stats,
                labels=stats.index,
                startangle=90,
                autopct=lambda p: '{:.0f}'.format(p * sum(stats) / 100),
                pctdistance=0.82,
                textprops={'color': text_color, 'weight': 'bold', 'fontsize': 9},
                wedgeprops={'linewidth': 2, 'edgecolor': bg_color}
            )

            centre_circle = plt.Circle((0, 0), 0.65, fc=bg_color)
            ax_pie.add_artist(centre_circle)

            total_parts = sum(stats)
            ax_pie.text(0, 0, f"TOTAL\n{total_parts}", ha='center', va='center',
                        fontsize=12, weight='bold', color=accent_color)

            plt.tight_layout()
            canvas_pie = FigureCanvasTkAgg(fig_pie, master=self.frame_pie)

            # 💡 Centrage parfait dans le conteneur
            canvas_pie.get_tk_widget().pack(expand=True, anchor="center")
            canvas_pie.draw()

        except Exception as e:

            print(f"Erreur UI Couleurs: {e}")
            print(traceback.format_exc())

    def display_results(self, df, n_points, n_cells, bounds, fusion_path):
        self.df_result = df
        self.bounds_result = bounds
        self.fusion_path = fusion_path

        is_en = getattr(self.main_window, 'current_lang', 'FR') == "EN"

        self.show_progress(False)
        self.btn_run.configure(state="normal", text="🚀 NEW EXTRACTION" if is_en else "🚀 NOUVELLE EXTRACTION")
        self.btn_cancel.configure(state="disabled")

        self.stats_summary_label.configure(
            text=f"✅ DONE | 📊 {n_cells} Parts Checked" if is_en else f"✅ TERMINÉ | 📊 {n_cells} Pièces Vérifiées",
            text_color=cfg.COLOR_SUCCESS
        )

        if not hasattr(self, 'bbox_card'):
            self.bbox_card = ctk.CTkFrame(
                self.card, fg_color=cfg.COLOR_ENTRY_BG,
                corner_radius=10, border_width=1, border_color=cfg.COLOR_ACCENT
            )
        self.bbox_card.pack(pady=20, padx=40, fill="x")

        for child in self.bbox_card.winfo_children(): child.destroy()

        self.lbl_bbox_title = ctk.CTkLabel(
            self.bbox_card,
            text="DETECTED GEOMETRIC BOUNDS (BBOX)" if is_en else "LIMITES GÉOMÉTRIQUES DÉTECTÉES (BBOX)",
            font=("Inter", 10, "bold"), text_color=cfg.COLOR_ACCENT
        )
        self.lbl_bbox_title.pack(pady=(10, 5))

        x_min, x_max, y_min, y_max, z_min, z_max = [round(b, 2) for b in bounds]
        val_txt = f"X: {x_min} ↔ {x_max}   |   Y: {y_min} ↔ {y_max}   |   Z: {z_min} ↔ {z_max}"

        val_entry = ctk.CTkEntry(
            self.bbox_card,
            font=("JetBrains Mono", 13, "bold"),
            text_color=cfg.COLOR_TEXT_MAIN,
            fg_color="transparent",
            border_width=0,
            justify="center",
            width=550
        )
        val_entry.pack(pady=(0, 10))

        val_entry.insert(0, val_txt)
        val_entry.configure(state="readonly")

        if hasattr(self, 'btn_show_mesh'):
            self.btn_show_mesh.grid_remove()

        self.post_action_frame.pack(pady=10)

    def show_3d_mesh(self):
        if not self.fusion_path or not os.path.exists(self.fusion_path): return

        # 💡 On "pack" le conteneur 3D. S'il est appelé après le graphique, ils se diviseront l'espace 50/50 automatiquement
        self.frame_mesh.pack(side="left", expand=True, fill="both", padx=5)

        for w in self.frame_mesh.winfo_children(): w.destroy()

        is_en = getattr(self.main_window, 'current_lang', 'FR') == "EN"
        self.btn_show_mesh.configure(state="disabled", text="⏳ Generating..." if is_en else "⏳ Génération...")
        self.update()

        try:

            mesh = pv.read(self.fusion_path)
            mesh_tri = mesh.decimate(0.7).triangulate() if mesh.n_cells > 5000 else mesh.triangulate()
            verts, faces = mesh_tri.points, mesh_tri.faces.reshape(-1, 4)[:, 1:4]

            is_dark = ctk.get_appearance_mode() == "Dark"
            plot_bg = "#23272D" if is_dark else "white"
            edge_col = "#58A6FF" if is_dark else "#0077B6"

            fig_mesh = plt.figure(figsize=(5, 5), facecolor=plot_bg)
            ax_mesh = fig_mesh.add_subplot(111, projection='3d')
            ax_mesh.set_facecolor(plot_bg)
            ax_mesh.set_axis_off()

            poly = Poly3DCollection(verts[faces], alpha=0.3, facecolor='#A0B2C6', edgecolor=edge_col, linewidths=0.1)
            ax_mesh.add_collection3d(poly)

            ax_mesh.set_xlim([self.bounds_result[0], self.bounds_result[1]])
            ax_mesh.set_ylim([self.bounds_result[2], self.bounds_result[3]])
            ax_mesh.set_zlim([self.bounds_result[4], self.bounds_result[5]])

            canvas_mesh = FigureCanvasTkAgg(fig_mesh, master=self.frame_mesh)

            # 💡 Centrage parfait dans le conteneur
            canvas_mesh.get_tk_widget().pack(expand=True, anchor="center")
            canvas_mesh.draw()

            self.btn_show_mesh.configure(text="✅ Displayed" if is_en else "✅ Affiché")

        except Exception as e:
            print(f"Erreur UI Mesh: {e}")
            self.btn_show_mesh.configure(text="❌ Error" if is_en else "❌ Erreur")
