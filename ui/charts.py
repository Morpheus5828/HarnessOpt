"""Courbes de progression de l'apprentissage.

Trois courbes suffisent à suivre une session : la conformité (nombre
d'interférences), la distance moyenne au DMU rapportée à la bande autorisée, et
le rayon de cintrage le plus serré comparé au rayon admissible. Chaque agent a
la couleur de sa trajectoire dans la vue 3D, ce qui permet de relier une courbe
à un agent sans légende à déchiffrer.

Les anciennes courbes traçaient la récompense brute : un nombre sans unité, que
seul l'auteur de la fonction de récompense sait interpréter. On trace ici des
grandeurs physiques, avec leur limite en pointillés.
"""

from __future__ import annotations

import customtkinter as ctk

__all__ = ["ProgressCharts"]


class ProgressCharts:
    """Trois graphiques empilés, rafraîchis pendant le calcul."""

    #: Nombre de points conservés par courbe : au-delà, on sous-échantillonne
    #: pour que le tracé reste fluide sur une longue session.
    MAX_POINTS = 400

    def __init__(self, container: ctk.CTkFrame):
        self.container = container
        self.available = False
        self.figure = None
        self.canvas = None
        self._axes: dict = {}
        self._lines: dict = {}
        self._limits: dict = {}
        self._history: dict[str, dict[str, list]] = {}

    def start(self) -> bool:
        """Prépare la figure. Renvoie False si matplotlib est indisponible."""
        try:
            import matplotlib

            matplotlib.use("TkAgg", force=False)
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except Exception as exc:
            ctk.CTkLabel(
                self.container,
                text=f"Courbes indisponibles : {exc}",
                wraplength=400,
            ).pack(expand=True)
            return False

        self.figure = Figure(figsize=(5, 3.2), dpi=96)
        self.figure.patch.set_alpha(0.0)

        specs = [
            ("clashes", "Interférences", "nombre"),
            ("distance", "Distance au DMU", "mm"),
            ("bend", "Rayon de cintrage", "mm"),
        ]
        for i, (key, title, unit) in enumerate(specs):
            ax = self.figure.add_subplot(len(specs), 1, i + 1)
            ax.set_title(title, fontsize=8, loc="left", color="#6C7684")
            ax.set_ylabel(unit, fontsize=7, color="#6C7684")
            ax.tick_params(labelsize=7, colors="#8A94A3")
            ax.grid(True, linestyle=":", alpha=0.35)
            ax.patch.set_alpha(0.0)
            for side in ax.spines.values():
                side.set_color("#DDE3EA")
            self._axes[key] = ax

        self.figure.tight_layout(pad=1.1)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.available = True
        return True

    def set_limits(self, min_margin: float, max_margin: float, min_bend_radius: float):
        """Trace les limites réglementaires en pointillés."""
        if not self.available:
            return

        for line in self._limits.values():
            try:
                line.remove()
            except Exception:
                pass
        self._limits.clear()

        ax_dist = self._axes.get("distance")
        if ax_dist is not None:
            self._limits["dmin"] = ax_dist.axhline(
                min_margin, color="#D93A45", linestyle="--", linewidth=1, alpha=0.7
            )
            self._limits["dmax"] = ax_dist.axhline(
                max_margin, color="#E08A00", linestyle="--", linewidth=1, alpha=0.7
            )

        ax_bend = self._axes.get("bend")
        if ax_bend is not None:
            self._limits["bend"] = ax_bend.axhline(
                min_bend_radius, color="#D93A45", linestyle="--", linewidth=1, alpha=0.7
            )

    def reset(self):
        """Efface l'historique (nouveau calcul)."""
        self._history.clear()
        for lines in self._lines.values():
            for line in lines.values():
                try:
                    line.remove()
                except Exception:
                    pass
        self._lines.clear()
        if self.available:
            self.canvas.draw_idle()

    def update(self, snapshot: dict, colors: dict):
        """Ajoute un point à chaque courbe.

        ``snapshot`` : ``{nom d'agent: {"report": RouteReport, ...}}``.
        """
        if not self.available:
            return

        for name, state in snapshot.items():
            report = state.get("report")
            if report is None:
                continue

            k = report.kpis
            radius = k.get("min_bend_radius_mm", float("inf"))
            history = self._history.setdefault(
                name, {"clashes": [], "distance": [], "bend": []}
            )
            history["clashes"].append(k.get("n_clashes", 0))
            history["distance"].append(k.get("mean_distance_mm", 0.0))
            # Un rayon infini (tracé parfaitement droit) ne se trace pas :
            # on le plafonne pour garder une échelle lisible.
            history["bend"].append(min(radius, 5000.0))

            for series in history.values():
                if len(series) > self.MAX_POINTS:
                    del series[: len(series) - self.MAX_POINTS]

            color = colors.get(name, "#2D7FF9")
            for key, ax in self._axes.items():
                line = self._lines.setdefault(key, {}).get(name)
                if line is None:
                    (line,) = ax.plot([], [], color=color, linewidth=1.4, alpha=0.9, label=name)
                    self._lines[key][name] = line
                values = history[key]
                line.set_data(range(len(values)), values)

        for key, ax in self._axes.items():
            series = [h[key] for h in self._history.values() if h[key]]
            if not series:
                continue
            longest = max(len(s) for s in series)
            ax.set_xlim(0, max(10, longest))
            low = min(min(s) for s in series)
            high = max(max(s) for s in series)
            if key in self._limits or key == "distance":
                for limit_line in self._limits.values():
                    if limit_line.axes is ax:
                        value = limit_line.get_ydata()[0]
                        low, high = min(low, value), max(high, value)
            pad = max(1.0, (high - low) * 0.15)
            ax.set_ylim(low - pad, high + pad)

        self.canvas.draw_idle()
