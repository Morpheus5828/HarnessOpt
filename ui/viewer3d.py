"""Vue 3D de la maquette et des trajectoires.

L'ancienne version incrustait la fenêtre VTK dans le cadre Tk en appelant
directement ``SetParentInfo`` avec un identifiant de fenêtre natif. Cela ne
fonctionne que sous Windows, échoue silencieusement ailleurs, et laissait
l'utilisateur devant un rectangle noir sans explication.

Cette classe essaie les modes d'affichage du plus intégré au plus simple et
dit toujours lequel a été retenu :

1. **incrusté** dans l'interface (Windows, via l'identifiant de fenêtre) ;
2. **fenêtre séparée** pilotée par l'application (partout ailleurs) ;
3. **indisponible** : un message explique quoi faire, sans planter.
"""

from __future__ import annotations

import sys

import customtkinter as ctk

from ui.theme import FONT, SPACE, current

__all__ = ["Viewer3D"]

MODE_EMBEDDED = "embedded"
MODE_WINDOW = "window"
MODE_UNAVAILABLE = "unavailable"


class Viewer3D:
    """Enveloppe autour d'un ``pyvista.Plotter``.

    L'appelant ne manipule que des noms d'acteurs : ``show_mesh``,
    ``show_path``, ``show_points``. La classe se charge du mode d'affichage et
    ne lève jamais d'exception vers l'interface — une 3D indisponible ne doit
    pas empêcher de calculer un cheminement.
    """

    def __init__(self, container: ctk.CTkFrame, on_status=None):
        self.container = container
        self.plotter = None
        self.mode = MODE_UNAVAILABLE
        self.error: str | None = None
        self._on_status = on_status
        self._actors: dict[str, object] = {}
        self._placeholder: ctk.CTkLabel | None = None

    # -- cycle de vie ---------------------------------------------------

    def start(self) -> str:
        """Ouvre la vue 3D et renvoie le mode retenu."""
        try:
            import pyvista as pv
        except ImportError as exc:
            self.error = f"PyVista n'est pas installé ({exc})."
            self._show_placeholder(
                "Vue 3D indisponible : PyVista n'est pas installé.\n"
                "Le calcul du cheminement reste possible."
            )
            return self.mode

        try:
            self.plotter = pv.Plotter(off_screen=False)
            self.plotter.set_background("white", top="lightgrey")
            self.plotter.add_axes()
        except Exception as exc:
            self.error = str(exc)
            self.plotter = None
            self._show_placeholder(
                "Vue 3D indisponible sur ce poste.\n"
                f"Détail : {exc}\n"
                "Le calcul du cheminement reste possible."
            )
            return self.mode

        self.mode = MODE_WINDOW
        if sys.platform.startswith("win") and self._try_embed():
            self.mode = MODE_EMBEDDED

        self._notify()
        return self.mode

    def _try_embed(self) -> bool:
        """Tente l'incrustation dans le cadre Tk (Windows uniquement)."""
        render_window = getattr(self.plotter, "render_window", None)
        if render_window is None:
            return False
        try:
            self.container.update_idletasks()
            handle = self.container.winfo_id()
        except Exception:
            return False

        for method in ("SetParentInfo", "SetWindowInfo", "SetParentId", "SetWindowId"):
            setter = getattr(render_window, method, None)
            if setter is None:
                continue
            try:
                setter(str(handle) if "Info" in method else handle)
                return True
            except Exception:
                continue
        return False

    def _show_placeholder(self, message: str):
        theme = current()
        if self._placeholder is None:
            self._placeholder = ctk.CTkLabel(
                self.container,
                text=message,
                font=FONT.BODY,
                text_color=theme.TEXT_SOFT,
                justify="center",
                wraplength=420,
            )
            self._placeholder.pack(expand=True, padx=SPACE.LG, pady=SPACE.LG)
        else:
            self._placeholder.configure(text=message)
        self._notify()

    def _notify(self):
        if self._on_status is not None:
            self._on_status(self.mode, self.error)

    @property
    def is_available(self) -> bool:
        return self.plotter is not None

    def close(self):
        if self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:
                pass
        self.plotter = None
        self._actors.clear()
        self.mode = MODE_UNAVAILABLE

    # -- contenu --------------------------------------------------------

    def show_mesh(self, mesh, name: str = "dmu", **style):
        """Affiche la maquette. ``style`` est passé tel quel à PyVista."""
        if not self.is_available:
            return None
        style.setdefault("color", "#B9C2CC")
        style.setdefault("opacity", 0.75)
        style.setdefault("show_edges", False)
        return self._add(mesh, name, **style)

    def show_path(self, points, name: str, color: str = "#2D7FF9",
                  width: int = 8, tube_radius: float | None = None):
        """Affiche une trajectoire, en tube si un rayon est fourni."""
        if not self.is_available or points is None or len(points) < 2:
            return None
        try:
            import pyvista as pv

            line = pv.lines_from_points(points)
            if tube_radius and tube_radius > 0:
                return self._add(line.tube(radius=tube_radius, n_sides=16),
                                 name, color=color, smooth_shading=True)
            return self._add(line, name, color=color, line_width=width,
                             render_lines_as_tubes=True)
        except Exception:
            return None

    def show_sphere(self, center, name: str, radius: float = 25.0, color: str = "#1E9E5A"):
        if not self.is_available or center is None:
            return None
        try:
            import pyvista as pv

            return self._add(pv.Sphere(radius=radius, center=center), name, color=color)
        except Exception:
            return None

    def show_bbox(self, bounds, name: str = "bbox", color: str = "#E08A00"):
        if not self.is_available or bounds is None:
            return None
        try:
            import pyvista as pv

            return self._add(pv.Box(bounds=bounds), name, color=color,
                             style="wireframe", line_width=2)
        except Exception:
            return None

    def _add(self, geometry, name: str, **style):
        try:
            actor = self.plotter.add_mesh(geometry, name=name, **style)
            self._actors[name] = actor
            return actor
        except Exception:
            return None

    def remove(self, name: str):
        if not self.is_available:
            return
        try:
            self.plotter.remove_actor(name)
        except Exception:
            pass
        self._actors.pop(name, None)

    def remove_prefix(self, prefix: str):
        """Retire tous les acteurs dont le nom commence par ``prefix``."""
        for name in [n for n in self._actors if n.startswith(prefix)]:
            self.remove(name)

    def set_visible(self, name: str, visible: bool):
        actor = self._actors.get(name)
        if actor is None:
            return
        try:
            actor.SetVisibility(bool(visible))
        except Exception:
            pass

    def set_visible_prefix(self, prefix: str, visible: bool):
        for name in list(self._actors):
            if name.startswith(prefix):
                self.set_visible(name, visible)

    def set_edges(self, name: str, show: bool):
        actor = self._actors.get(name)
        if actor is None:
            return
        try:
            actor.prop.show_edges = bool(show)
        except Exception:
            pass

    def has_actor(self, name: str) -> bool:
        return name in self._actors

    # -- rendu ----------------------------------------------------------

    def reset_camera(self):
        if self.is_available:
            try:
                self.plotter.reset_camera()
            except Exception:
                pass

    def render(self):
        """Rafraîchit l'image. Silencieux en cas d'échec (fenêtre fermée)."""
        if not self.is_available:
            return
        try:
            self.plotter.render()
        except Exception:
            pass

    def show_window(self):
        """Ouvre la fenêtre 3D séparée (mode non incrusté)."""
        if not self.is_available or self.mode == MODE_EMBEDDED:
            return
        try:
            self.plotter.show(interactive_update=True, auto_close=False)
        except Exception as exc:
            self.error = str(exc)
            self._show_placeholder(f"Impossible d'ouvrir la vue 3D : {exc}")
