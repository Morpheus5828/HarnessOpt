import customtkinter as ctk
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.agent_worker import benchmark_algos


class AgentPage(ctk.CTkFrame):
    def __init__(self, parent, window):
        super().__init__(parent, fg_color="transparent")
        self.window = window
        self.accent_color = window.accent_color

        # --- TRANSLATIONS (Includes 3D view toggles) ---
        self.translations = {
            "FR": {
                "title": "🚁 Hub Multi-Agent RL | Routage Harnais",
                "desc": "Optimisation GPU : Explorateurs (Recherche 3D) vs Optimiseurs (Lissage Extrême).",
                "show_config": "▼ Afficher description & paramètres",
                "hide_config": "▲ Masquer description & paramètres",
                "src_lbl": "📍 Point Source (X,Y,Z):",
                "tgt_lbl": "🎯 Point Cible (X,Y,Z):",
                "play": "▶ LECTURE",
                "pause": "⏸ PAUSE",
                "reset": "🔄 RÉINITIALISER",
                "save": "💾 SAUVEGARDER STL",
                "popout": "🗖 DÉTACHER 3D",
                "popin": "🗗 INCRUSTER 3D",
                "chk_reward": "🏆 Récompenses",
                "chk_dist": "📏 Distances",
                "chk_col": "💥 Collisions",
                "chk_length": "📏 Longueur Trajectoire",
                "chk_pts": "📏 Nb Points",
                "chk_fixations": "🦀 Fixations",
                "chk_mesh": "🧊 Maillage",
                "chk_edges": "🕸️ Arêtes",
                "chk_bbox": "📦 BBox"
            },
            "EN": {
                "title": "🚁 Multi-Agent RL Hub | Harness Routing",
                "desc": "GPU Optimization: Explorers (3D Search) vs Optimizers (Extreme Smoothing).",
                "show_config": "▼ Show description & settings",
                "hide_config": "▲ Hide description & settings",
                "src_lbl": "📍 Source Point (X,Y,Z):",
                "tgt_lbl": "🎯 Target Point (X,Y,Z):",
                "play": "▶ PLAY",
                "pause": "⏸ PAUSE",
                "reset": "🔄 RESET",
                "save": "💾 SAVE TO STL",
                "popout": "🗖 POP-OUT 3D",
                "popin": "🗗 EMBED 3D",
                "chk_reward": "🏆 Rewards",
                "chk_dist": "📏 Distances",
                "chk_col": "💥 Collisions",
                "chk_length": "📏 Trajectory Length",
                "chk_pts": "📏 Nb Points",
                "chk_fixations": "🦀 Fixations",
                "chk_mesh": "🧊 Mesh",
                "chk_edges": "🕸️ Edges",
                "chk_bbox": "📦 BBox"
            }
        }

        self.config_visible = False
        self._build_ui()
        self.update_language(window.current_lang)

    def _build_ui(self):
        # --- EN-TÊTE ---
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(10, 5))

        self.title_lbl = ctk.CTkLabel(self.header_frame, text="", font=("Inter", 22, "bold"))
        self.title_lbl.pack(side="left")

        self.btn_toggle_config = ctk.CTkButton(
            self.header_frame, text="", width=20, height=28,
            fg_color="transparent", text_color=("#6A737D", "#8B949E"),
            hover_color=("#E1E4E8", "#3F444C"), font=("Inter", 12, "bold"),
            command=self._on_toggle_config
        )
        self.btn_toggle_config.pack(side="left", padx=15)

        # --- CONTENEUR RÉTRACTABLE ---
        self.config_container = ctk.CTkFrame(self, fg_color="transparent")

        self.desc_lbl = ctk.CTkLabel(self.config_container, text="", font=("Inter", 13),
                                     text_color=("#6A737D", "#8B949E"))
        self.desc_lbl.pack(anchor="w", pady=(0, 15))

        self.config_frame = ctk.CTkFrame(self.config_container, fg_color="transparent")
        self.config_frame.pack(fill="x", pady=(0, 5))

        self.src_lbl = ctk.CTkLabel(self.config_frame, text="", font=("Inter", 12, "bold"))
        self.src_lbl.pack(side="left", padx=(0, 5))
        self.entry_src = ctk.CTkEntry(self.config_frame, width=180, font=("Consolas", 12))
        self.entry_src.insert(0, "3960, 1135, 1197")
        self.entry_src.pack(side="left", padx=(0, 25))

        self.tgt_lbl = ctk.CTkLabel(self.config_frame, text="", font=("Inter", 12, "bold"))
        self.tgt_lbl.pack(side="left", padx=(0, 5))
        self.entry_tgt = ctk.CTkEntry(self.config_frame, width=180, font=("Consolas", 12))
        self.entry_tgt.insert(0, "3857, 992, 2663")
        self.entry_tgt.pack(side="left")

        self.params_frame = ctk.CTkFrame(self.config_container, fg_color="transparent")
        self.params_frame.pack(fill="x", pady=(0, 5))

        self.lbl_ns = ctk.CTkLabel(self.params_frame, text="Bruit Init:", font=("Inter", 12, "bold"))
        self.lbl_ns.pack(side="left", padx=(0, 5))
        self.entry_ns = ctk.CTkEntry(self.params_frame, width=50, font=("Consolas", 12))
        self.entry_ns.insert(0, "0.5")
        self.entry_ns.pack(side="left", padx=(0, 15))

        self.lbl_nm = ctk.CTkLabel(self.params_frame, text="Bruit Min:", font=("Inter", 12, "bold"))
        self.lbl_nm.pack(side="left", padx=(0, 5))
        self.entry_nm = ctk.CTkEntry(self.params_frame, width=50, font=("Consolas", 12))
        self.entry_nm.insert(0, "0.05")
        self.entry_nm.pack(side="left", padx=(0, 15))

        self.lbl_decay = ctk.CTkLabel(self.params_frame, text="Decay:", font=("Inter", 12, "bold"))
        self.lbl_decay.pack(side="left", padx=(0, 5))
        self.entry_decay = ctk.CTkEntry(self.params_frame, width=60, font=("Consolas", 12))
        self.entry_decay.insert(0, "0.998")
        self.entry_decay.pack(side="left", padx=(0, 15))

        self.lbl_shift = ctk.CTkLabel(self.params_frame, text="Vitesse (mm):", font=("Inter", 12, "bold"))
        self.lbl_shift.pack(side="left", padx=(0, 5))
        self.entry_shift = ctk.CTkEntry(self.params_frame, width=50, font=("Consolas", 12))
        self.entry_shift.insert(0, "3.0")
        self.entry_shift.pack(side="left")

        self.margins_frame = ctk.CTkFrame(self.config_container, fg_color="transparent")
        self.margins_frame.pack(fill="x", pady=(0, 15))

        self.lbl_min = ctk.CTkLabel(self.margins_frame, text="🛑 Marge Min (Crash):", font=("Inter", 12, "bold"),
                                    text_color="#FF6B6B")
        self.lbl_min.pack(side="left", padx=(0, 5))
        self.entry_min = ctk.CTkEntry(self.margins_frame, width=50, font=("Consolas", 12))
        self.entry_min.insert(0, "10.0")
        self.entry_min.pack(side="left", padx=(0, 15))

        self.lbl_max = ctk.CTkLabel(self.margins_frame, text="⚠️ Marge Max (Fuite):", font=("Inter", 12, "bold"),
                                    text_color="#F6AD55")
        self.lbl_max.pack(side="left", padx=(0, 5))
        self.entry_max = ctk.CTkEntry(self.margins_frame, width=50, font=("Consolas", 12))
        self.entry_max.insert(0, "15.0")
        self.entry_max.pack(side="left", padx=(0, 25))

        self.lbl_pts = ctk.CTkLabel(self.margins_frame, text="📏 Nb Pts Init:", font=("Inter", 12, "bold"),
                                    text_color="#4DABF7")
        self.lbl_pts.pack(side="left", padx=(0, 5))
        self.entry_pts = ctk.CTkEntry(self.margins_frame, width=50, font=("Consolas", 12))
        self.entry_pts.insert(0, "350")
        self.entry_pts.pack(side="left")

        self.chk_force_fix = ctk.CTkCheckBox(self.margins_frame, text="🧲 Forcer Fixations",
                                             font=("Inter", 11, "bold"), text_color="#28A745")
        self.chk_force_fix.select()  # Cochée par défaut
        self.chk_force_fix.pack(side="left", padx=(20, 0))

        entries = [self.entry_src, self.entry_tgt, self.entry_ns, self.entry_nm,
                   self.entry_decay, self.entry_shift, self.entry_min, self.entry_max, self.entry_pts]
        for entry in entries:
            entry.bind("<FocusIn>", self._on_focus_in)
            entry.bind("<FocusOut>", self._on_focus_out)

        # --- CONTENEUR BOUTONS ---
        self.actions_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.actions_frame.pack(fill="x", pady=(0, 15))

        self.btn_text = tk.StringVar(value="▶ LECTURE")
        self.btn_play = ctk.CTkButton(self.actions_frame, textvariable=self.btn_text, font=("Inter", 13, "bold"),
                                      fg_color=self.accent_color, height=40, command=self._on_toggle_play)
        self.btn_play.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_reset_text = tk.StringVar(value="🔄 RÉINITIALISER")
        self.btn_reset = ctk.CTkButton(self.actions_frame, textvariable=self.btn_reset_text, font=("Inter", 13, "bold"),
                                       fg_color="#D9534F", hover_color="#C9302C", height=40, command=self._on_reset)
        self.btn_reset.pack(side="left", fill="x", expand=True, padx=(5, 5))

        self.btn_save_text = tk.StringVar(value="💾 SAUVEGARDER STL")
        self.btn_save = ctk.CTkButton(self.actions_frame, textvariable=self.btn_save_text, font=("Inter", 13, "bold"),
                                      fg_color="#28A745", hover_color="#218838", height=40, command=self._on_save_stl)
        self.btn_save.pack(side="left", fill="x", expand=True, padx=(5, 5))

        self.btn_popout_text = tk.StringVar(value="🗖 DÉTACHER 3D")
        self.btn_popout = ctk.CTkButton(self.actions_frame, textvariable=self.btn_popout_text,
                                        font=("Inter", 13, "bold"), fg_color="#17A2B8", hover_color="#138496",
                                        height=40, command=self._on_popout)
        self.btn_popout.pack(side="right", fill="x", expand=True, padx=(5, 0))

        self.cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        algos = list(benchmark_algos.keys())
        self.algo_widgets = {}
        for idx, name in enumerate(algos):
            lbl_status = ctk.CTkLabel(self.cards_frame, text="En attente...")
            self.algo_widgets[name] = lbl_status

        # --- ZONE INFÉRIEURE (Graphiques & 3D) ---
        self.bottom_container = ctk.CTkFrame(self, fg_color="transparent")
        self.bottom_container.pack(fill="both", expand=True, pady=5)
        self.bottom_container.grid_columnconfigure(0, weight=1, uniform="bottom_split")
        self.bottom_container.grid_columnconfigure(1, weight=1, uniform="bottom_split")
        self.bottom_container.grid_rowconfigure(0, weight=1)

        self.left_plots_frame = ctk.CTkFrame(self.bottom_container, fg_color="transparent")
        self.left_plots_frame.grid(row=0, column=0, padx=5, sticky="nsew")

        self.selector_frame = ctk.CTkFrame(self.left_plots_frame, border_width=1, border_color=("#E1E4E8", "#3F444C"))
        self.selector_frame.pack(fill="x", pady=(0, 5))

        self.chk_reward = ctk.CTkCheckBox(self.selector_frame, text="", font=("Inter", 11, "bold"),
                                          command=self._on_graph_filter_changed)
        self.chk_reward.select()
        self.chk_reward.pack(side="left", padx=15, pady=6)

        self.chk_dist = ctk.CTkCheckBox(self.selector_frame, text="", font=("Inter", 11, "bold"),
                                        command=self._on_graph_filter_changed)
        self.chk_dist.select()
        self.chk_dist.pack(side="left", padx=15, pady=6)

        self.chk_col = ctk.CTkCheckBox(self.selector_frame, text="", font=("Inter", 11, "bold"),
                                       command=self._on_graph_filter_changed)
        self.chk_col.select()
        self.chk_col.pack(side="left", padx=15, pady=6)

        self.chk_length = ctk.CTkCheckBox(self.selector_frame, text="", font=("Inter", 11, "bold"),
                                          command=self._on_graph_filter_changed)
        self.chk_length.select()
        self.chk_length.pack(side="left", padx=15, pady=6)

        self.chk_pts = ctk.CTkCheckBox(self.selector_frame, text="", font=("Inter", 11, "bold"),
                                       command=self._on_graph_filter_changed)
        self.chk_pts.select()
        self.chk_pts.pack(side="left", padx=15, pady=6)

        self.chk_crabes = ctk.CTkCheckBox(self.selector_frame, text="", font=("Inter", 11, "bold"),
                                          command=self._on_graph_filter_changed)
        self.chk_crabes.select()
        self.chk_crabes.pack(side="left", padx=15, pady=6)

        self.graph_frame = ctk.CTkFrame(self.left_plots_frame, border_width=1, border_color=("#E1E4E8", "#3F444C"),
                                        fg_color=("#FFFFFF", "#161B22"))
        self.graph_frame.pack(fill="both", expand=True)

        self.fig = plt.figure(facecolor='none', figsize=(4, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # --- NOUVEAU : CADRE 3D ET OPTIONS VISUELLES ---
        self.view_3d_frame = ctk.CTkFrame(self.bottom_container, border_width=1, border_color=("#E1E4E8", "#3F444C"))
        self.view_3d_frame.grid(row=0, column=1, padx=5, sticky="nsew")

        self.options_3d_frame = ctk.CTkFrame(self.view_3d_frame, fg_color="transparent")
        self.options_3d_frame.pack(fill="x", padx=5, pady=(5, 0))

        self.chk_mesh = ctk.CTkCheckBox(self.options_3d_frame, text="", font=("Inter", 11, "bold"),
                                        command=self._on_3d_display_changed)
        self.chk_mesh.select()  # Maillage visible par défaut
        self.chk_mesh.pack(side="left", padx=(5, 15))

        self.chk_edges = ctk.CTkCheckBox(self.options_3d_frame, text="", font=("Inter", 11, "bold"),
                                         command=self._on_3d_display_changed)
        self.chk_edges.deselect()  # Arêtes invisibles par défaut
        self.chk_edges.pack(side="left", padx=15)

        self.chk_bbox = ctk.CTkCheckBox(self.options_3d_frame, text="", font=("Inter", 11, "bold"),
                                        command=self._on_3d_display_changed)
        self.chk_bbox.deselect()  # BBox invisible par défaut
        self.chk_bbox.pack(side="left", padx=15)

        self.vtk_container = tk.Frame(self.view_3d_frame, bg="#0D1117")
        self.vtk_container.pack(fill="both", expand=True, padx=2, pady=2)

    def _on_toggle_config(self):
        self.config_visible = not self.config_visible
        lang_code = getattr(self.window, 'current_lang', "EN")
        t = self.translations.get(lang_code, self.translations["EN"])
        if self.config_visible:
            self.btn_toggle_config.configure(text=t["hide_config"])
            self.config_container.pack(fill="x", after=self.header_frame)
        else:
            self.btn_toggle_config.configure(text=t["show_config"])
            self.config_container.pack_forget()

    def _on_focus_in(self, event):
        if self.window.controller: self.window.controller.is_typing = True

    def _on_focus_out(self, event):
        if self.window.controller: self.window.controller.is_typing = False

    def _on_graph_filter_changed(self):
        if self.window.controller and self.window.controller.agent_initialized:
            self.window.controller.update_graph_layouts()

    def _on_3d_display_changed(self):
        """Déclenche la mise à jour visuelle du mesh/bbox dans le contrôleur."""
        if self.window.controller and self.window.controller.agent_initialized:
            self.window.controller.update_3d_visibility()

    def _on_toggle_play(self):
        if self.window.controller: self.window.controller.toggle_agent_training()

    def _on_reset(self):
        if self.window.controller: self.window.controller.reset_agent_training()

    def _on_save_stl(self):
        if self.window.controller: self.window.controller.save_trajectories_to_stl()

    def _on_popout(self):
        if self.window.controller: self.window.controller.toggle_popout_3d()

    def update_language(self, lang_code):
        t = self.translations.get(lang_code, self.translations["EN"])
        self.title_lbl.configure(text=t["title"])
        self.desc_lbl.configure(text=t["desc"])
        self.src_lbl.configure(text=t["src_lbl"])
        self.tgt_lbl.configure(text=t["tgt_lbl"])
        self.btn_reset_text.set(t["reset"])
        self.btn_save_text.set(t["save"])

        self.chk_reward.configure(text=t["chk_reward"])
        self.chk_dist.configure(text=t["chk_dist"])
        self.chk_col.configure(text=t["chk_col"])
        self.chk_length.configure(text=t["chk_length"])
        self.chk_pts.configure(text=t["chk_pts"])
        self.chk_crabes.configure(text=t["chk_fixations"])

        self.chk_mesh.configure(text=t["chk_mesh"])
        self.chk_edges.configure(text=t["chk_edges"])
        self.chk_bbox.configure(text=t["chk_bbox"])

        if self.config_visible:
            self.btn_toggle_config.configure(text=t["hide_config"])
        else:
            self.btn_toggle_config.configure(text=t["show_config"])

        if self.window.controller and self.window.controller.shared_state.get("is_playing", False):
            self.btn_text.set(t["pause"])
        else:
            self.btn_text.set(t["play"])

        if self.window.controller and getattr(self.window.controller, 'is_3d_popped_out', False):
            self.btn_popout_text.set(t["popin"])
        else:
            self.btn_popout_text.set(t["popout"])

    def update_accent_color(self, color):
        self.accent_color = color
        self.btn_play.configure(fg_color=color)
