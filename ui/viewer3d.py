"""Vue 3D de la maquette et des trajectoires.

**Une seule vue, et c'est une vraie fenêtre VTK.** La version précédente en
tenait deux : une image incrustée dans la page, produite par capture d'écran
d'un plotter hors écran, et une fenêtre détachée optionnelle. L'incrustée
coûtait cher pour ce qu'elle montrait — une image figée, une émulation
maison de l'orbite, du panoramique et du zoom, et un pipeline de capture qui
se dispute le pilote graphique avec la fenêtre. Elle est supprimée.

Ce qui reste tient en une phrase : **tout ce qui touche à VTK vit dans un fil
de rendu dédié, et rien d'autre.** Le fil possède le plotter, reçoit des
ordres par une file, et pompe les évènements de la fenêtre. Aucun appel VTK
n'a lieu sur le fil de Tk, donc l'interface ne peut pas se figer — la
construction du contexte 3D prend treize secondes sur une grosse maquette.

La scène est décrite par des **recettes** (géométrie + style), pas par des
acteurs VTK. C'est ce qui permet de fermer la fenêtre, de la rouvrir, et d'y
retrouver la scène intacte, sans jamais partager d'objet VTK entre deux fils.

Deux gestes sont rendus à l'utilisateur, tous deux impossibles sur une image
capturée : **cliquer** un repère de la scène — pour désigner l'encoche qu'il
veut emprunter — et **déplacer** une poignée pour imposer un point de passage
au tracé.
"""

from __future__ import annotations

import queue
import threading

import customtkinter as ctk

from ui.theme import FONT, SPACE, current

__all__ = [
    "Viewer3D",
    "MODE_WINDOW",
    "MODE_CLOSED",
    "MODE_UNAVAILABLE",
    "MODE_STARTING",
    "nearest_target",
    "DEFAULT_PICK_TOLERANCE_MM",
]

#: Fenêtre VTK ouverte.
MODE_WINDOW = "window"
#: Fil de rendu prêt, fenêtre fermée. La scène reste mémorisée.
MODE_CLOSED = "closed"
#: Aucun rendu possible (PyVista absent, pas de contexte OpenGL…).
MODE_UNAVAILABLE = "unavailable"
#: Le fil de rendu démarre encore.
MODE_STARTING = "starting"

#: Distance maximale entre un clic et un repère pour que le clic le désigne.
#: Au-delà, l'utilisateur visait autre chose, et deviner à sa place est pire
#: que ne rien faire.
DEFAULT_PICK_TOLERANCE_MM = 120.0

#: Taille de la fenêtre 3D.
DEFAULT_SIZE = (1280, 860)


# ----------------------------------------------------------------------
# Fonction pure, testable sans écran ni VTK
# ----------------------------------------------------------------------

def nearest_target(point, targets, tolerance_mm: float = DEFAULT_PICK_TOLERANCE_MM):
    """Repère désigné par un clic, ou ``None`` si le clic est trop loin.

    ``targets`` est une liste de couples ``(clé, position)``. Un clic dans le
    vide, ou sur la maquette loin de tout repère, ne doit **rien** désigner :
    prendre le repère le plus proche quoi qu'il arrive ferait basculer un
    choix à l'autre bout de la maquette sur un clic de rotation manqué.
    """
    import numpy as np

    if point is None or not targets:
        return None
    origin = np.asarray(point, dtype=float)
    if origin.shape != (3,):
        return None

    best, best_distance = None, float("inf")
    for key, position in targets:
        distance = float(np.linalg.norm(np.asarray(position, dtype=float) - origin))
        if distance < best_distance:
            best, best_distance = key, distance
    return best if best_distance <= float(tolerance_mm) else None


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

    def __init__(self, on_ready, on_window_state, on_pick, on_handle_move, size):
        super().__init__(daemon=True, name="viewer3d")
        self.orders: queue.Queue = queue.Queue()
        self._on_ready = on_ready
        self._on_window_state = on_window_state
        self._on_pick = on_pick
        self._on_handle_move = on_handle_move
        self._size = size
        self._stop_event = threading.Event()

        self.plotter = None
        self.error: str | None = None
        #: Géométrie et style de chaque acteur. La scène vit ici, pas dans
        #: VTK : c'est ce qui permet de refermer la fenêtre sans la perdre.
        self.recipes: dict[str, tuple] = {}
        self.actor_names: set[str] = set()

        self._exited = False
        self._pick_targets: list = []
        self._pick_tolerance = DEFAULT_PICK_TOLERANCE_MM
        self._handles: list = []
        self._handle_radius = 30.0

    # -- boucle ------------------------------------------------------

    def run(self):
        if not self._probe():
            self._on_ready(False, self.error)
            return
        self._on_ready(True, None)

        while not self._stop_event.is_set():
            # Une fenêtre ouverte doit voir ses évènements traités à chaque
            # tour, sinon le système la marque « ne répond pas ».
            timeout = 0.01 if self.plotter is not None else 0.05
            try:
                order = self.orders.get(timeout=timeout)
            except queue.Empty:
                order = None

            if order is not None:
                if order[0] == "__stop__":
                    break
                self._apply(order)
                continue

            self._pump()

        self._teardown()

    @staticmethod
    def _probe_import():
        import pyvista  # noqa: F401

    def _probe(self) -> bool:
        """PyVista est-il seulement installé ? Le contexte OpenGL, lui, ne se
        vérifie qu'à l'ouverture de la fenêtre — c'est là qu'il est créé."""
        try:
            self._probe_import()
        except Exception as exc:
            self.error = f"PyVista n'est pas installé ({exc})."
            return False
        return True

    def _pump(self):
        """Traite les évènements de la fenêtre, si elle est encore là."""
        if self.plotter is None:
            return
        if not self._window_alive():
            # Fermée par l'utilisateur. Continuer à la rafraîchir, c'est
            # dessiner dans un contexte OpenGL détruit : VTK réclame alors un
            # shader qu'il ne peut plus compiler et noie la console
            # d'« ERR| Could not create shader object ».
            self._drop_window()
            return
        try:
            self.plotter.update()
        except Exception:
            self._drop_window()

    def _window_alive(self) -> bool:
        """La fenêtre est-elle encore utilisable ?

        ``Plotter.update()`` appelle ``render()`` sans vérifier quoi que ce
        soit : une fenêtre fermée par l'utilisateur ne lève donc aucune
        exception, elle rend simplement dans le vide. Il faut le demander à
        VTK, et de plusieurs façons — aucune n'est fiable seule d'une
        plateforme à l'autre.
        """
        plotter = self.plotter
        if plotter is None or self._exited:
            return False
        try:
            if getattr(plotter, "_closed", False):
                return False
            if plotter.render_window is None:
                return False
            interactor = getattr(getattr(plotter, "iren", None), "interactor", None)
            if interactor is not None and interactor.GetDone():
                return False
        except Exception:
            return False
        return True

    def _drop_window(self, notify: bool = True):
        plotter, self.plotter = self.plotter, None
        self._exited = False
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                pass
        if notify:
            self._on_window_state(False)

    def _teardown(self):
        self._drop_window(notify=False)

    def stop(self):
        self._stop_event.set()
        self.orders.put(("__stop__",))

    # -- exécution des ordres ---------------------------------------

    def _apply(self, order):
        name, args, kwargs = order[0], order[1], order[2]
        handler = getattr(self, f"_do_{name}", None)
        if handler is None:
            return
        try:
            handler(*args, **kwargs)
        except Exception:
            # Un ordre invalide ne doit pas emporter le fil de rendu : la 3D
            # est un confort, le cheminement doit continuer.
            pass

    # -- contenu -----------------------------------------------------

    def _do_add(self, geometry, name, **style):
        self.recipes[name] = (geometry, dict(style))
        self.actor_names.add(name)
        if self._window_alive():
            try:
                self.plotter.add_mesh(geometry, name=name, **style)
            except Exception:
                pass

    def _do_remove(self, name):
        self.recipes.pop(name, None)
        self.actor_names.discard(name)
        if self._window_alive():
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass

    def _do_visible(self, name, visible):
        if not self._window_alive():
            return
        actor = self.plotter.actors.get(name)
        if actor is not None:
            actor.SetVisibility(bool(visible))

    def _do_edges(self, name, show):
        if not self._window_alive():
            return
        actor = self.plotter.actors.get(name)
        if actor is not None:
            actor.prop.show_edges = bool(show)

    def _do_reset_camera(self):
        if self._window_alive():
            self.plotter.reset_camera()

    def _do_render(self):
        return

    # -- fenêtre -----------------------------------------------------

    def _do_open_window(self):
        if self._window_alive():
            return
        self._drop_window(notify=False)

        import pyvista as pv

        plotter = pv.Plotter(window_size=list(self._size), title="HarnessOpt — vue 3D")
        plotter.set_background("white", top="#DCE3EC")
        for name, (geometry, style) in list(self.recipes.items()):
            try:
                plotter.add_mesh(geometry, name=name, **style)
            except Exception:
                continue
        plotter.add_axes()
        plotter.reset_camera()
        plotter.show(interactive_update=True, auto_close=False)

        self.plotter = plotter
        self._exited = False
        self._observe_exit()
        self._install_picking()
        self._install_handles()
        self._on_window_state(True)

    def _do_close_window(self):
        if self.plotter is None:
            return
        self._drop_window()

    def _observe_exit(self):
        """La fermeture par l'utilisateur passe par ``ExitEvent``."""
        try:
            self.plotter.iren.add_observer("ExitEvent", self._mark_exited)
        except Exception:
            pass

    def _mark_exited(self, *_args):
        self._exited = True

    # -- désignation par clic ----------------------------------------

    def _do_set_pick_targets(self, targets, tolerance):
        self._pick_targets = list(targets or [])
        self._pick_tolerance = float(tolerance)
        self._install_picking()

    def _install_picking(self):
        if not self._window_alive() or not self._pick_targets:
            return
        try:
            self.plotter.enable_point_picking(
                callback=self._picked, left_clicking=True, show_message=False,
                show_point=False,
            )
        except Exception:
            pass

    def _picked(self, point, *_args):
        key = nearest_target(point, self._pick_targets, self._pick_tolerance)
        if key is not None:
            self._on_pick(key)

    # -- poignées déplaçables ----------------------------------------

    def _do_set_handles(self, points, radius):
        self._handles = [list(p) for p in (points or [])]
        self._handle_radius = float(radius)
        self._install_handles()

    def _install_handles(self):
        if not self._window_alive():
            return
        try:
            self.plotter.clear_sphere_widgets()
        except Exception:
            pass
        if not self._handles:
            return
        try:
            import numpy as np

            self.plotter.add_sphere_widget(
                self._handle_moved,
                center=np.asarray(self._handles, dtype=float),
                radius=self._handle_radius, color="#E5B300",
                selected_color="#D93A45", test_callback=False,
            )
        except Exception:
            pass

    def _handle_moved(self, center, index=0):
        self._on_handle_move(int(index), tuple(float(c) for c in center))


# ----------------------------------------------------------------------
# La façade utilisée par l'interface
# ----------------------------------------------------------------------

class Viewer3D:
    """Vue 3D pilotée par un fil de rendu.

    Toutes les méthodes de contenu — ``show_mesh``, ``show_path``,
    ``show_sphere``… — sont non bloquantes et acceptées **même fenêtre
    fermée** : elles alimentent la scène mémorisée, qui sera rejouée à
    l'ouverture. Le contrôleur n'a donc jamais à savoir si une fenêtre est
    ouverte pour décrire ce qu'il veut montrer.
    """

    def __init__(self, container, on_status=None, t=None):
        self.container = container
        self._on_status = on_status
        self._t = t or (lambda key, default="": default or key)

        self.mode = MODE_UNAVAILABLE
        self.error: str | None = None

        self._thread: _RenderThread | None = None
        self._actors: set[str] = set()
        self._placeholder: ctk.CTkLabel | None = None
        self._closed = False
        self._open = False
        self._on_pick = None
        self._on_handle_move = None

    # -- cycle de vie ---------------------------------------------------

    def start(self) -> str:
        """Démarre le fil de rendu. **Ne bloque pas.**"""
        if self._thread is not None:
            return self.mode

        self._show_placeholder(self._t("routing.view.starting",
                                       "Préparation de la vue 3D…"))
        self.mode = MODE_STARTING
        self._thread = _RenderThread(
            on_ready=self._post_ready,
            on_window_state=self._post_window_state,
            on_pick=self._post_pick,
            on_handle_move=self._post_handle_move,
            size=DEFAULT_SIZE,
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
        else:
            self.mode = MODE_CLOSED
            self._show_placeholder(
                self._t("routing.view.closed",
                        "La vue 3D s'ouvre dans sa propre fenêtre.")
            )
        self._notify()

    def close(self):
        self._closed = True
        if self._thread is not None:
            self._thread.stop()
            self._thread = None
        self._actors.clear()
        self._open = False
        self.mode = MODE_UNAVAILABLE

    @property
    def is_available(self) -> bool:
        """Vrai dès que des ordres peuvent être acceptés.

        Vrai fenêtre fermée : les ordres alimentent la scène mémorisée. Vrai
        aussi pendant le démarrage, ce qui permet au contrôleur de charger la
        maquette sans attendre.
        """
        return self._thread is not None and self.mode in (
            MODE_STARTING, MODE_CLOSED, MODE_WINDOW
        )

    @property
    def is_open(self) -> bool:
        """La fenêtre 3D est-elle ouverte ?"""
        return self._open

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

    # -- fenêtre ---------------------------------------------------------

    def open_window(self):
        """Ouvre la fenêtre 3D et y rejoue la scène mémorisée."""
        if not self.is_available:
            return
        self._order("open_window")

    def close_window(self):
        if not self.is_available:
            return
        self._order("close_window")

    def toggle_window(self) -> bool:
        """Ouvre ou referme la fenêtre. Renvoie l'état visé."""
        if self._open:
            self.close_window()
            return False
        self.open_window()
        return True

    def _post_window_state(self, is_open: bool):
        self._post(self._on_window_state, is_open)

    def _on_window_state(self, is_open: bool):
        if self._closed:
            return
        self._open = bool(is_open)
        self.mode = MODE_WINDOW if is_open else MODE_CLOSED
        self._show_placeholder(
            self._t("routing.view.opened", "Vue 3D ouverte dans sa fenêtre.")
            if is_open else
            self._t("routing.view.closed", "La vue 3D s'ouvre dans sa propre fenêtre.")
        )
        self._notify()

    # -- désignation par clic --------------------------------------------

    def set_pick_targets(self, targets, tolerance_mm: float = DEFAULT_PICK_TOLERANCE_MM):
        """Repères cliquables : une liste de couples ``(clé, position)``."""
        self._order("set_pick_targets", list(targets or []), float(tolerance_mm))

    def set_on_pick(self, callback):
        self._on_pick = callback

    def _post_pick(self, key):
        self._post(self._deliver_pick, key)

    def _deliver_pick(self, key):
        if self._on_pick is not None and not self._closed:
            self._on_pick(key)

    # -- poignées déplaçables --------------------------------------------

    def set_handles(self, points, radius: float = 30.0):
        """Place des poignées déplaçables aux points donnés."""
        self._order("set_handles", list(points or []), float(radius))

    def clear_handles(self):
        self.set_handles([])

    def set_on_handle_move(self, callback):
        self._on_handle_move = callback

    def _post_handle_move(self, index, point):
        self._post(self._deliver_handle_move, index, point)

    def _deliver_handle_move(self, index, point):
        if self._on_handle_move is not None and not self._closed:
            self._on_handle_move(index, point)

    # -- messages ---------------------------------------------------------

    def _post(self, callback, *args):
        """Repasse sur le fil de Tk, seul autorisé à toucher aux widgets."""
        if self._closed:
            return
        try:
            self.container.after(0, lambda: callback(*args))
        except Exception:
            pass

    def _show_placeholder(self, message: str):
        theme = current()
        if self._placeholder is None:
            try:
                self._placeholder = ctk.CTkLabel(
                    self.container, text=message, font=FONT.BODY,
                    text_color=theme.TEXT_SOFT, justify="center", wraplength=420,
                )
                self._placeholder.pack(expand=True, padx=SPACE.LG, pady=SPACE.LG)
            except Exception:
                self._placeholder = None
        else:
            try:
                self._placeholder.configure(text=message)
            except Exception:
                pass

    def _notify(self):
        if self._on_status is not None:
            self._on_status(self.mode, self.error)
