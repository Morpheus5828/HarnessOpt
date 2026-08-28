import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from ui.pages.extraction_view import ExtractionPage
from ui.pages.agent_view import AgentPage
import os
import shutil


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🚁 HARNESSOPT")
        self.geometry("1400x900")
        self.after(0, lambda: self.state('zoomed'))

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=("#F0F2F5", "#1A1C1E"))

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.controller = None
        self.pages = {}
        self.current_page = "extraction"
        self.accent_color = "#58A6FF"
        self.current_lang = "FR"

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._check_startup_cache()

        self._build_menu()
        self._build_sidebar()

        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self._init_pages()
        self.show_page("extraction", 0)

    def _build_menu(self):
        if hasattr(self, 'menubar'):
            self.menubar.destroy()

        self.menubar = tk.Menu(self)

        translations = {
            "FR": {
                "file": "Fichier", "cache": "Paramètres de Cache", "quit": "Quitter",
                "edit": "Edition", "undo": "Annuler", "redo": "Rétablir",
                "opt": "Options", "lang": "Langues", "set": "Paramètres",
                "c_def": "🎨 Couleurs par défaut", "c_dal": "👁️ Mode Daltonien",
                "clear_cache": "🧹 Vider le cache"
            },
            "EN": {
                "file": "File", "cache": "Cache Settings", "quit": "Quit",
                "edit": "Edit", "undo": "Undo", "redo": "Redo",
                "opt": "Options", "lang": "Language", "set": "Settings",
                "c_def": "🎨 Default Colors", "c_dal": "👁️ Colorblind Mode",
                "clear_cache": "🧹 Clear Cache"
            },
            "DE": {
                "file": "Datei", "cache": "Cache-Einstellungen", "quit": "Beenden",
                "edit": "Bearbeiten", "undo": "Rückgängig", "redo": "Wiederholen",
                "opt": "Optionen", "lang": "Sprache", "set": "Einstellungen",
                "c_def": "🎨 Standardfarben", "c_dal": "👁️ Farbenblind-Modus",
                "clear_cache": "🧹 Cache leeren"
            },
            "ES": {
                "file": "Archivo", "cache": "Ajustes de Caché", "quit": "Salir",
                "edit": "Edición", "undo": "Deshacer", "redo": "Rehacer",
                "opt": "Opciones", "lang": "Idioma", "set": "Ajustes",
                "c_def": "🎨 Colores predeterminados", "c_dal": "👁️ Modo Daltónico",
                "clear_cache": "🧹 Vaciar caché"
            }
        }

        t = translations.get(self.current_lang, translations["EN"])

        menu_fichier = tk.Menu(self.menubar, tearoff=0)
        menu_fichier.add_command(label=t["cache"], state="disabled")
        menu_fichier.add_separator()
        menu_fichier.add_command(label=t["quit"], command=self.on_closing)
        self.menubar.add_cascade(label=t["file"], menu=menu_fichier)

        menu_edition = tk.Menu(self.menubar, tearoff=0)
        menu_edition.add_command(label=t["undo"], state="disabled")
        menu_edition.add_command(label=t["redo"], state="disabled")
        menu_edition.add_separator()
        menu_edition.add_command(label=t["clear_cache"], command=self._user_request_clear_cache)
        self.menubar.add_cascade(label=t["edit"], menu=menu_edition)

        menu_options = tk.Menu(self.menubar, tearoff=0)
        menu_langue = tk.Menu(menu_options, tearoff=0)
        menu_langue.add_command(label="🇫🇷 Français", command=lambda: self.set_language("FR"))
        menu_langue.add_command(label="🇬🇧 English", command=lambda: self.set_language("EN"))
        menu_langue.add_command(label="🇩🇪 Deutsch", command=lambda: self.set_language("DE"))
        menu_langue.add_command(label="🇪🇸 Español", command=lambda: self.set_language("ES"))
        menu_options.add_cascade(label=t["lang"], menu=menu_langue)
        self.menubar.add_cascade(label=t["opt"], menu=menu_options)

        menu_settings = tk.Menu(self.menubar, tearoff=0)
        menu_settings.add_command(label=t["c_def"], command=self.set_default_colors)
        menu_settings.add_command(label=t["c_dal"], command=self.set_colorblind_mode)
        self.menubar.add_cascade(label=t["set"], menu=menu_settings)

        self.config(menu=self.menubar)

    def _user_request_clear_cache(self):
        titles = {"FR": "Confirmation", "EN": "Confirmation", "DE": "Bestätigung", "ES": "Confirmación"}
        msgs = {
            "FR": "Voulez-vous vraiment vider tout le cache ?\nCela supprimera vos calculs actuels.",
            "EN": "Do you really want to clear all cache?\nThis will delete your current calculations.",
            "DE": "Möchten Sie wirklich den gesamten Cache leeren?",
            "ES": "¿Realmente de seas vaciar toute la caché?"
        }

        title = titles.get(self.current_lang, "Confirmation")
        msg = msgs.get(self.current_lang, msgs["EN"])

        if messagebox.askyesno(title, msg):
            self._clear_cache()
            messagebox.showinfo("Cache", "Cache vidé avec succès !" if self.current_lang == "FR" else "Cache cleared!")

    def set_language(self, lang_code):
        self.current_lang = lang_code
        self._build_menu()

        sb = {
            "FR": ["📂  Extraction DMU", "🤖  Agent", "Quitter", "Mode Sombre", "Mode Clair"],
            "EN": ["📂  DMU Extraction", "🤖  Agent", "Quit", "Dark Mode", "Light Mode"],
            "DE": ["📂  DMU-Extraktion", "🤖  Agent", "Beenden", "Dunkelmodus", "Hellmodus"],
            "ES": ["📂  Extracción DMU", "🤖  Agent", "Salir", "Modo Oscuro", "Modo Claro"]
        }[lang_code]

        self.btn_ext.configure(text=sb[0])
        self.btn_agt.configure(text=sb[1])
        self.btn_quit.configure(text=sb[2])
        self.theme_switch.configure(text=sb[3] if self.theme_switch_var.get() == "dark" else sb[4])

        for page in self.pages.values():
            if hasattr(page, 'update_language'):
                page.update_language(lang_code)

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color=("#FFFFFF", "#0D1117"), border_width=1, border_color=("#E1E4E8", "#3F444C"))
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="HARNESSOPT", font=("Inter", 24, "bold"), text_color=self.accent_color)
        self.logo_label.pack(pady=(40, 40))

        line = ctk.CTkFrame(self.sidebar, height=1, fg_color=("#E1E4E8", "#3F444C"))
        line.pack(fill="x", padx=20, pady=(0, 20))

        self.btn_ext = self._add_sidebar_button("📂  Extraction DMU", lambda: self.show_page("extraction", 0))
        self.btn_agt = self._add_sidebar_button("🤖  Agent", lambda: self.show_page("agent", 1))

        bottom_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom_frame.pack(side="bottom", fill="x", pady=20, padx=20)

        self.theme_switch_var = ctk.StringVar(value="light")
        self.theme_switch = ctk.CTkSwitch(bottom_frame, text="Mode Sombre", font=("Inter", 12), variable=self.theme_switch_var, onvalue="dark", offvalue="light", command=self._toggle_theme, progress_color=self.accent_color)
        self.theme_switch.pack(anchor="w", pady=(0, 15))

        self.btn_quit = ctk.CTkButton(bottom_frame, text="Quitter", fg_color="transparent", text_color=("#E63946", "#FF6B6B"), hover_color=("#FDECEC", "#3F1D24"), font=("Inter", 12, "bold"), command=self.on_closing, height=30)
        self.btn_quit.pack(fill="x")

        self.version_lbl = ctk.CTkLabel(bottom_frame, text="v2.4.0-desktop", font=("Inter", 10), text_color=("#6A737D", "#8B949E"))
        self.version_lbl.pack(pady=(10, 0))

        self.nav_indicator = ctk.CTkFrame(self.sidebar, width=4, height=35, fg_color=self.accent_color, corner_radius=2)
        self.nav_indicator.place(x=2, y=-100)

    def _update_sidebar_style(self, active_page):
        color = getattr(self, 'accent_color', "#58A6FF")
        buttons = {"extraction": self.btn_ext, "agent": self.btn_agt}
        for name, btn in buttons.items():
            if name == active_page:
                btn.configure(fg_color=("#E6F0FF", "#1F2328"), text_color=color, font=("Inter", 13, "bold"))
                if btn.winfo_y() > 0: self._animate_indicator(btn.winfo_y())
            else:
                btn.configure(fg_color="transparent", text_color=("#50565E", "#8B949E"), font=("Inter", 13, "normal"))

    def _animate_indicator(self, target_y):
        current_y = self.nav_indicator.winfo_y()
        step = (target_y - current_y) / 5
        if abs(step) > 0.5:
            self.nav_indicator.place(x=2, y=current_y + step)
            self.after(10, lambda: self._animate_indicator(target_y))
        else:
            self.nav_indicator.place(x=2, y=target_y)

    def _toggle_theme(self):
        mode = self.theme_switch_var.get()
        ctk.set_appearance_mode(mode)
        is_en = self.current_lang == "EN"
        if mode == "light":
            self.theme_switch.configure(text="Light Mode" if is_en else "Mode Clair")
        else:
            self.theme_switch.configure(text="Dark Mode" if is_en else "Mode Sombre")

    def _add_sidebar_button(self, text, command):
        btn = ctk.CTkButton(self.sidebar, text=text, anchor="w", fg_color="transparent", text_color=("#50565E", "#8B949E"), font=("Inter", 13, "normal"), height=45, hover_color=("#F3F4F6", "#1F2328"), command=command, corner_radius=8)
        btn.pack(fill="x", pady=2, padx=15)
        return btn

    def _init_pages(self):
        self.pages["extraction"] = ExtractionPage(self.container, self)
        self.pages["agent"] = AgentPage(self.container, self)

    def show_page(self, page_name, step_index):
        self.current_page = page_name
        self.container.grid_remove()
        for page in self.pages.values(): page.pack_forget()

        if page_name in self.pages:
            target_page = self.pages[page_name]
            target_page.pack(fill="both", expand=True)
            self.after(50, lambda: self.container.grid())

        self._update_sidebar_style(page_name)
        if self.controller: self.controller.update_step_visual(step_index)

    def set_controller(self, controller):
        self.controller = controller

    def set_colorblind_mode(self):
        self.accent_color = "#F39C12"
        self._update_all_colors()

    def set_default_colors(self):
        self.accent_color = "#58A6FF"
        self._update_all_colors()

    def _update_all_colors(self):
        self.logo_label.configure(text_color=self.accent_color)
        self.nav_indicator.configure(fg_color=self.accent_color)
        self.theme_switch.configure(progress_color=self.accent_color)
        self._update_sidebar_style(self.current_page)
        for page in self.pages.values():
            if hasattr(page, 'update_accent_color'):
                page.update_accent_color(self.accent_color)

    def _check_startup_cache(self):
        cache_path = r"C:\Temp\HarnessOpt_cache"
        has_content = False

        if os.path.exists(cache_path):
            for root, dirs, files in os.walk(cache_path):
                if files:
                    has_content = True
                    break

        if has_content:
            title = "Reprise de session" if self.current_lang == "FR" else "Resume session"
            msg = (
                "Un cache contenant des données d'une session précédente a été détecté.\n\n"
                "Voulez-vous CONSERVER ces données pour reprendre là où vous en étiez ?\n"
                "• OUI : Garder le cache.\n"
                "• NON : Vider tout le cache et recommencer à zéro."
            ) if self.current_lang == "FR" else (
                "A cache containing data from a previous session was detected.\n\n"
                "Do you want to KEEP this data to resume where you left off?\n"
                "• YES: Keep the cache.\n"
                "• NO: Clear the cache completely and start fresh."
            )

            keep_cache = messagebox.askyesno(title, msg)
            if not keep_cache:
                self._clear_cache()
            else:
                print("💾 Conservation du cache existant demandée par l'utilisateur.")
                self._ensure_cache_folders()
        else:
            self._ensure_cache_folders()

    def _ensure_cache_folders(self):
        cache_path = r"C:\Temp\HarnessOpt_cache"
        subfolders = ["", "stl", "color", "fusion", "sphere_generations", "graphs", "paths"]
        for sub in subfolders:
            try:
                os.makedirs(os.path.join(cache_path, sub), exist_ok=True)
            except:
                pass

    def _clear_cache(self):
        cache_path = r"C:\Temp\HarnessOpt_cache"
        if os.path.exists(cache_path):
            try:
                shutil.rmtree(cache_path)
                print(f"🧹 Cache vidé avec succès : {cache_path}")
            except Exception as e:
                print(f"⚠️ Impossible de vider complètement le cache : {e}")
        self._ensure_cache_folders()

    def on_closing(self):
        print("🛑 Demande de fermeture de l'application...")
        title = "Quitter" if self.current_lang == "FR" else "Quit"
        msg = (
            "Voulez-vous VIDER LE CACHE avant de quitter l'application afin de libérer de l'espace disque ?"
        ) if self.current_lang == "FR" else (
            "Do you want to CLEAR THE CACHE before quitting to free up disk space?"
        )

        if messagebox.askyesno(title, msg):
            self._clear_cache()

        self.quit()
        self.destroy()
