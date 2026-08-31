"""Vue 3D de la maquette et des trajectoires.

Les deux versions précédentes ne montraient rien du tout, pour deux raisons
distinctes et cumulées :

* le ``pyvista.Plotter`` était construit sur le fil principal de Tk. Sur une
  maquette d'hélicoptère cette construction demande plusieurs secondes — 13,6 s
  mesurées sur 800 000 triangles — pendant lesquelles l'application entière est
  figée, sans le moindre curseur d'attente ;
* la fenêtre de rendu n'était ensuite jamais ni affichée ni « pompée ». En mode
  incrusté personne n'appelait le rendu ; en mode fenêtre, rien ne s'ouvrait
  tant que l'utilisateur ne cliquait pas « Ouvrir en grand ». Le cadre réservé
  à la 3D restait donc vide, sans explication.

Le principe retenu ici est simple : **tout ce qui touche à VTK vit dans un fil
de rendu dédié, et rien d'autre.** Ce fil possède un plotter hors écran, reçoit
des ordres par une file, et renvoie des images. L'interface se contente
d'afficher la dernière image reçue et de traduire les gestes de souris en
ordres de caméra. Aucun appel VTK n'a lieu sur le fil de Tk, donc l'interface
ne peut plus se figer ; et comme le rendu est une simple image, il s'incruste
dans la fenêtre sur toutes les plateformes, sans manipuler d'identifiant de
fenêtre natif.

Ce choix a un coût assumé : l'image n'est pas rafraîchie à la fréquence d'un
pilote 3D natif. Pour l'inspection fine, « Ouvrir en grand » ouvre une vraie
fenêtre VTK interactive, dont les évènements sont cette fois réellement
traités.
"""

from __future__ import annotations

import queue
import threading
import time

import customtkinter as ctk

from ui.theme import FONT, SPACE, current

__all__ = [
    "Viewer3D",
    "MODE_EMBEDDED",
    "MODE_WINDOW",
    "MODE_UNAVAILABLE",
    "MODE_STARTING",
    "orbit_delta",
    "pan_delta",
    "zoom_factor",
]

#: Image incrustée dans la fenêtre, pilotée par le fil de rendu.
MODE_EMBEDDED = "embedded"
#: Fenêtre VTK native ouverte à côté de l'application.
MODE_WINDOW = "window"
#: Aucun rendu possible (PyVista absent, pas de contexte OpenGL…).
MODE_UNAVAILABLE = "unavailable"
#: Le fil de rendu démarre encore.
MODE_STARTING = "starting"

#: Sensibilité des gestes de souris.
ORBIT_DEG_PER_PIXEL = 0.4
PAN_FRACTION_PER_PIXEL = 0.0022
ZOOM_PER_NOTCH = 1.12

#: Délai de regroupement des ordres avant de relancer un rendu, en secondes.
COALESCE_S = 0.03
#: Taille de rendu par défaut, avant que le cadre ne connaisse la sienne.
DEFAULT_SIZE = (900, 600)
#: Bornes de la taille de rendu, pour ne pas demander une image absurde.
MIN_SIZE, MAX_SIZE = 160, 2400


# ----------------------------------------------------------------------
# Traduction des gestes en ordres de caméra — fonctions pures, testables
# sans écran ni VTK.
# ----------------------------------------------------------------------

def orbit_delta(dx: int, dy: int) -> tuple[float, float]:
    """Déplacement souris → (azimut, élévation) en degrés.

    L'azimut est inversé pour que la maquette suive le doigt : tirer vers la
    droite fait tourner la scène vers la droite.
    """
    return (-dx * ORBIT_DEG_PER_PIXEL, dy * ORBIT_DEG_PER_PIXEL)


def pan_delta(dx: int, dy: int, width: int, height: int) -> tuple[float, float]:
    """Déplacement souris → translation, en fraction de la hauteur visible.

    On rapporte les deux axes à la hauteur : un déplacement diagonal de la
    souris doit produire une translation diagonale à l'écran, ce qui suppose
    la même échelle en x et en y.
    """
    ref = max(1, int(height))
    return (-dx / ref, dy / ref)


def zoom_factor(notches: float) -> float:
    """Crans de molette → facteur de zoom multiplicatif (toujours > 0)."""
    return float(ZOOM_PER_NOTCH ** notches)


def clamp_size(width: int, height: int) -> tuple[int, int]:
    """Taille de rendu bornée, jamais nulle."""
    w = max(MIN_SIZE, min(MAX_SIZE, int(width or 0)))
    h = max(MIN_SIZE, min(MAX_SIZE, int(height or 0)))
    return w, h


# ----------------------------------------------------------------------
# Le fil de rendu
# ----------------------------------------------------------------------

class _RenderThread(threading.Thread):
    """Propriétaire exclusif du plotter PyVista.

    Toute méthode publique de :class:`Viewer3D` se contente de déposer un
    ordre dans ``self.orders``. Les objets VTK ne sont jamais touchés depuis
    un autre fil : c'est la seule discipline qui rende l'ensemble sûr, VTK
    n'offrant aucune garantie de réentrance.
    """

    def __init__(self, on_frame, on_ready, size, on_window_closed=None):
        super().__init__(daemon=True, name="viewer3d")
        self.orders: queue.Queue = queue.Queue()
        self._on_frame = on_frame
        self._on_ready = on_ready
        self._on_window_closed = on_window_closed or (lambda: None)
        self._size = size
        self._stop_event = threading.Event()
        self.plotter = None
        self.error: str | None = None
        # Acteurs connus, pour répondre sans interroger VTK depuis un autre fil.
        self.actor_names: set[str] = set()
        # Géométrie et style de chaque acteur, pour pouvoir reconstruire la
        # scène dans la fenêtre détachée. Partager les acteurs VTK entre deux
        # rendus est possible mais fragile ; les rejouer ne l'est pas.
        self.recipes: dict[str, tuple] = {}
        self._window = None

    # -- boucle ------------------------------------------------------

    def run(self):
        if not self._build_plotter():
            self._on_ready(False, self.error)
            return
        self._on_ready(True, None)

        dirty = False
        last = 0.0
        while not self._stop_event.is_set():
            # Une fenêtre détachée ouverte doit voir ses évènements traités à
            # chaque tour, sinon le système la marque « ne répond pas ». C'est
            # exactement ce qui manquait à la version précédente.
            timeout = 0.01 if self._window is not None else 0.05
            try:
                order = self.orders.get(timeout=timeout)
            except queue.Empty:
                order = None

            if order is not None:
                if order[0] == "__stop__":
                    break
                dirty = self._apply(order) or dirty
                # On vide la file avant de rendre : dix ordres successifs ne
                # doivent produire qu'une image, sinon le zoom saccade.
                continue

            self._pump_window()

            if dirty and (time.monotonic() - last) >= COALESCE_S:
                self._emit_frame()
                dirty = False
                last = time.monotonic()

        self._teardown()

    def _pump_window(self):
        """Traite les évènements de la fenêtre détachée, si elle est ouverte."""
        if self._window is None:
            return
        try:
            self._window.update()
        except Exception:
            # Fenêtre fermée par l'utilisateur : on repasse en incrusté.
            self._window = None
            self._on_window_closed()

    def _build_plotter(self) -> bool:
        try:
            import pyvista as pv
        except ImportError as exc:
            self.error = f"PyVista n'est pas installé ({exc})."
            return False
        try:
            self.plotter = pv.Plotter(off_screen=True, window_size=list(self._size))
            self.plotter.set_background("white", top="#DCE3EC")
            self.plotter.add_axes()
            self.plotter.enable_lightkit()
        except Exception as exc:  # contexte OpenGL absent, pilote incomplet…
            self.plotter = None
            self.error = str(exc)
            return False
        return True

    def _teardown(self):
        if self._window is not None:
            try:
                self._window.close()
            except Exception:
                pass
            self._window = None
        if self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:
                pass
        self.plotter = None

    def stop(self):
        self._stop_event.set()
        self.orders.put(("__stop__",))

    # -- exécution des ordres ---------------------------------------

    def _apply(self, order) -> bool:
        """Exécute un ordre. Renvoie True s'il faut redessiner."""
        name, args, kwargs = order[0], order[1], order[2]
        handler = getattr(self, f"_do_{name}", None)
        if handler is None:
            return False
        try:
            return bool(handler(*args, **kwargs))
        except Exception:
            # Un ordre invalide ne doit pas emporter le fil de rendu : la 3D
            # est un confort, le cheminement doit continuer.
            return False

    def _do_add(self, geometry, name, **style):
        self.plotter.add_mesh(geometry, name=name, **style)
        self.actor_names.add(name)
        self.recipes[name] = (geometry, dict(style))
        if self._window is not None:
            try:
                self._window.add_mesh(geometry, name=name, **style)
            except Exception:
                pass
        return True

    def _do_remove(self, name):
        try:
            self.plotter.remove_actor(name)
        except Exception:
            pass
        if self._window is not None:
            try:
                self._window.remove_actor(name)
            except Exception:
                pass
        self.actor_names.discard(name)
        self.recipes.pop(name, None)
        return True

    def _do_visible(self, name, visible):
        actor = self.plotter.actors.get(name)
        if actor is None:
            return False
        actor.SetVisibility(bool(visible))
        return True

    def _do_edges(self, name, show):
        actor = self.plotter.actors.get(name)
        if actor is None:
            return False
        actor.prop.show_edges = bool(show)
        return True

    def _do_resize(self, width, height):
        self.plotter.window_size = [int(width), int(height)]
        return True

    def _do_reset_camera(self):
        self.plotter.reset_camera()
        return True

    def _do_orbit(self, azimuth, elevation):
        cam = self.plotter.camera
        cam.Azimuth(float(azimuth))
        cam.Elevation(float(elevation))
        cam.OrthogonalizeViewUp()
        return True

    def _do_pan(self, fx, fy):
        """Translate la caméra dans son plan image.

        ``fx``/``fy`` sont exprimés en fraction de la hauteur visible, ce qui
        rend le geste indépendant de la distance à la scène.
        """
        import numpy as np

        cam = self.plotter.camera
        pos = np.array(cam.position, dtype=float)
        foc = np.array(cam.focal_point, dtype=float)
        up = np.array(cam.up, dtype=float)

        forward = foc - pos
        distance = float(np.linalg.norm(forward))
        if distance <= 0:
            return False
        forward /= distance
        right = np.cross(forward, up)
        norm = np.linalg.norm(right)
        if norm <= 0:
            return False
        right /= norm
        true_up = np.cross(right, forward)

        # Hauteur réellement visible à la distance du point de mire.
        height = 2.0 * distance * np.tan(np.radians(float(cam.view_angle)) / 2.0)
        shift = right * (fx * height) + true_up * (fy * height)
        cam.position = tuple(pos + shift)
        cam.focal_point = tuple(foc + shift)
        return True

    def _do_zoom(self, factor):
        self.plotter.camera.Zoom(float(factor))
        return True

    def _do_render(self):
        return True

    def _do_show_window(self):
        """Ouvre une vraie fenêtre VTK interactive, à côté de la vue incrustée.

        La scène y est rejouée depuis les recettes mémorisées. La fenêtre est
        ensuite pompée par la boucle principale du fil : ses évènements sont
        donc réellement traités, ce qui n'était pas le cas auparavant.
        """
        import pyvista as pv

        if self._window is not None:
            return False
        window = pv.Plotter(window_size=[1100, 800])
        window.set_background("white", top="#DCE3EC")
        for name, (geometry, style) in list(self.recipes.items()):
            try:
                window.add_mesh(geometry, name=name, **style)
            except Exception:
                continue
        window.add_axes()
        window.reset_camera()
        window.show(interactive_update=True, auto_close=False)
        self._window = window
        return False

    def _do_close_window(self):
        if self._window is None:
            return False
        try:
            self._window.close()
        except Exception:
            pass
        self._window = None
        self._on_window_closed()
        return False

    def _emit_frame(self):
        try:
            image = self.plotter.screenshot(return_img=True)
        except Exception:
            return
        if image is not None:
            self._on_frame(image)


# ----------------------------------------------------------------------
# La façade utilisée par l'interface
# ----------------------------------------------------------------------

class Viewer3D:
    """Vue 3D incrustée, pilotée par un fil de rendu.

    L'API est celle de la version précédente — ``show_mesh``, ``show_path``,
    ``set_visible``… — de sorte que le contrôleur n'a pas à savoir comment le
    rendu est produit. Toutes ces méthodes sont non bloquantes : elles
    déposent un ordre et rendent la main immédiatement.
    """

    def __init__(self, container, on_status=None, t=None):
        self.container = container
        self._on_status = on_status
        self._t = t or (lambda key, default="": default or key)

        self.mode = MODE_UNAVAILABLE
        self.error: str | None = None

        self._thread: _RenderThread | None = None
        self._actors: set[str] = set()
        self._label: ctk.CTkLabel | None = None
        self._placeholder: ctk.CTkLabel | None = None
        self._photo = None
        self._size = DEFAULT_SIZE
        self._drag: tuple[int, int] | None = None
        self._drag_mode = "orbit"
        self._detached = False
        self._closed = False

    # -- cycle de vie ---------------------------------------------------

    def start(self) -> str:
        """Démarre le fil de rendu. **Ne bloque pas.**

        Renvoie ``MODE_STARTING`` : le mode définitif est communiqué plus tard
        par la fonction de statut, une fois le contexte 3D construit.
        """
        if self._thread is not None:
            return self.mode

        self._show_placeholder(self._t("routing.view.starting",
                                       "Préparation de la vue 3D…"))
        self.mode = MODE_STARTING
        self._thread = _RenderThread(
            on_frame=self._post_frame,
            on_ready=self._post_ready,
            size=self._size,
            on_window_closed=self._post_window_closed,
        )
        self._thread.start()
        return self.mode

    def _post_ready(self, ok: bool, error: str | None):
        self._post(self._on_ready, ok, error)

    def _on_ready(self, ok: bool, error: str | None):
        if self._closed:
            return
        if not ok:
            self.mode = MODE_UNAVAILABLE
            self.error = error
            self._show_placeholder(
                self._t("routing.view.unavailable", "Vue 3D indisponible sur ce poste.")
                + (f"\n{error}" if error else "")
                + "\n"
                + self._t("routing.view.still_running",
                          "Le calcul du cheminement, lui, continue normalement.")
            )
            self._notify()
            return

        self.mode = MODE_EMBEDDED
        self._build_canvas()
        self._notify()
        self.render()

    def _build_canvas(self):
        """Remplace le message d'attente par la zone d'image interactive."""
        if self._placeholder is not None:
            self._placeholder.destroy()
            self._placeholder = None
        if self._label is not None:
            return

        self._label = ctk.CTkLabel(self.container, text="", fg_color="transparent")
        self._label.pack(fill="both", expand=True)

        self._label.bind("<Configure>", self._on_configure)
        self._label.bind("<ButtonPress-1>", lambda e: self._begin_drag(e, "orbit"))
        self._label.bind("<ButtonPress-2>", lambda e: self._begin_drag(e, "pan"))
        self._label.bind("<ButtonPress-3>", lambda e: self._begin_drag(e, "pan"))
        self._label.bind("<Shift-ButtonPress-1>", lambda e: self._begin_drag(e, "pan"))
        for seq in ("<B1-Motion>", "<B2-Motion>", "<B3-Motion>", "<Shift-B1-Motion>"):
            self._label.bind(seq, self._on_drag)
        for seq in ("<ButtonRelease-1>", "<ButtonRelease-2>", "<ButtonRelease-3>"):
            self._label.bind(seq, self._end_drag)
        # Windows et macOS envoient <MouseWheel>, X11 des boutons 4 et 5.
        self._label.bind("<MouseWheel>", self._on_wheel)
        self._label.bind("<Button-4>", lambda e: self._zoom(1))
        self._label.bind("<Button-5>", lambda e: self._zoom(-1))

    def close(self):
        self._closed = True
        if self._thread is not None:
            self._thread.stop()
            self._thread = None
        self._actors.clear()
        self.mode = MODE_UNAVAILABLE

    @property
    def is_available(self) -> bool:
        """Vrai dès que des ordres peuvent être acceptés.

        Vrai aussi pendant le démarrage : les ordres émis à ce moment-là sont
        mis en file et exécutés dès que le contexte 3D est prêt. C'est ce qui
        permet au contrôleur de charger la maquette sans attendre.
        """
        return self._thread is not None and self.mode in (MODE_STARTING, MODE_EMBEDDED)

    # -- envoi d'ordres --------------------------------------------------

    def _order(self, name, *args, **kwargs):
        if self._thread is None or self._closed:
            return
        self._thread.orders.put((name, args, kwargs))

    # -- contenu ---------------------------------------------------------

    def show_mesh(self, mesh, name: str = "dmu", **style):
        """Affiche la maquette. ``style`` est transmis tel quel à PyVista."""
        if mesh is None:
            return None
        style.setdefault("color", "#B9C2CC")
        style.setdefault("opacity", 0.55)
        style.setdefault("show_edges", False)
        return self._add(mesh, name, **style)

    def show_path(self, points, name: str, color: str = "#2D7FF9",
                  width: int = 8, tube_radius: float | None = None):
        """Affiche une trajectoire, en tube si un rayon est fourni."""
        if points is None or len(points) < 2:
            return None
        try:
            import numpy as np
            import pyvista as pv

            line = pv.lines_from_points(np.asarray(points, dtype=float))
            if tube_radius and tube_radius > 0:
                return self._add(line.tube(radius=tube_radius, n_sides=16), name,
                                 color=color, smooth_shading=True)
            return self._add(line, name, color=color, line_width=width,
                             render_lines_as_tubes=True)
        except Exception:
            return None

    def show_sphere(self, center, name: str, radius: float = 25.0, color: str = "#1E9E5A"):
        if center is None:
            return None
        try:
            import pyvista as pv

            return self._add(pv.Sphere(radius=radius, center=center), name, color=color)
        except Exception:
            return None

    def show_bbox(self, bounds, name: str = "bbox", color: str = "#E08A00"):
        if bounds is None:
            return None
        try:
            import pyvista as pv

            return self._add(pv.Box(bounds=bounds), name, color=color,
                             style="wireframe", line_width=2)
        except Exception:
            return None

    def _add(self, geometry, name: str, **style):
        self._actors.add(name)
        self._order("add", geometry, name, **style)
        return name

    def remove(self, name: str):
        self._actors.discard(name)
        self._order("remove", name)

    def remove_prefix(self, prefix: str):
        for name in [n for n in self._actors if n.startswith(prefix)]:
            self.remove(name)

    def set_visible(self, name: str, visible: bool):
        self._order("visible", name, bool(visible))

    def set_visible_prefix(self, prefix: str, visible: bool):
        for name in list(self._actors):
            if name.startswith(prefix):
                self.set_visible(name, visible)

    def set_edges(self, name: str, show: bool):
        self._order("edges", name, bool(show))

    def has_actor(self, name: str) -> bool:
        return name in self._actors

    # -- caméra et rendu -------------------------------------------------

    def reset_camera(self):
        self._order("reset_camera")

    def render(self):
        self._order("render")

    def show_window(self):
        """Ouvre la maquette dans une fenêtre VTK réellement interactive."""
        if not self.is_available:
            return
        self._detached = True
        self._order("show_window")

    def close_window(self):
        """Referme la fenêtre détachée ; la vue incrustée reprend seule."""
        if not self.is_available:
            return
        self._order("close_window")

    def toggle_window(self) -> bool:
        """Ouvre ou referme la fenêtre détachée. Renvoie son nouvel état."""
        if self._detached:
            self.close_window()
        else:
            self.show_window()
        return self._detached

    @property
    def is_detached(self) -> bool:
        return self._detached

    def _post_window_closed(self):
        self._post(self._on_window_closed)

    def _on_window_closed(self):
        self._detached = False
        if self._on_status is not None:
            self._on_status(self.mode, self.error)

    # -- gestes de souris ------------------------------------------------

    def _begin_drag(self, event, mode: str):
        self._drag = (event.x, event.y)
        self._drag_mode = mode

    def _on_drag(self, event):
        if self._drag is None:
            return
        dx, dy = event.x - self._drag[0], event.y - self._drag[1]
        self._drag = (event.x, event.y)
        if dx == 0 and dy == 0:
            return
        if self._drag_mode == "pan":
            fx, fy = pan_delta(dx, dy, *self._size)
            self._order("pan", fx, fy)
        else:
            az, el = orbit_delta(dx, dy)
            self._order("orbit", az, el)

    def _end_drag(self, _event=None):
        self._drag = None

    def _on_wheel(self, event):
        # Windows : delta multiple de 120. macOS : petites valeurs signées.
        delta = getattr(event, "delta", 0)
        self._zoom(1 if delta > 0 else -1)

    def _zoom(self, notches: float):
        self._order("zoom", zoom_factor(notches))

    def _on_configure(self, event):
        size = clamp_size(event.width, event.height)
        if size == self._size:
            return
        self._size = size
        self._order("resize", *size)
        self._order("render")

    # -- réception des images --------------------------------------------

    def _post_frame(self, image):
        self._post(self._show_frame, image)

    def _show_frame(self, image):
        """Affiche une image reçue du fil de rendu. Toujours sur le fil Tk."""
        if self._closed or self._label is None:
            return
        try:
            from PIL import Image
        except ImportError:
            return
        try:
            pil = Image.fromarray(image)
            self._photo = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
            self._label.configure(image=self._photo, text="")
        except Exception:
            pass

    def _post(self, callback, *args):
        """Repasse sur le fil de Tk, seul autorisé à toucher aux widgets."""
        if self._closed:
            return
        try:
            self.container.after(0, lambda: callback(*args))
        except Exception:
            pass

    # -- messages ---------------------------------------------------------

    def _show_placeholder(self, message: str):
        theme = current()
        if self._placeholder is None:
            self._placeholder = ctk.CTkLabel(
                self.container, text=message, font=FONT.BODY,
                text_color=theme.TEXT_SOFT, justify="center", wraplength=420,
            )
            self._placeholder.pack(expand=True, padx=SPACE.LG, pady=SPACE.LG)
        else:
            self._placeholder.configure(text=message)

    def _notify(self):
        if self._on_status is not None:
            self._on_status(self.mode, self.error)
