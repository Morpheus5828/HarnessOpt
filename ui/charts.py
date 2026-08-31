"""Courbes de progression de l'apprentissage.

Quatre courbes suivent une session :

* la **récompense** de chaque agent, c'est-à-dire ce qu'il cherche réellement à
  maximiser. Elle ne dit rien de la conformité, mais elle est la seule à dire
  si l'apprentissage progresse ou piétine ;
* trois grandeurs **physiques** — interférences, distance au DMU, rayon de
  cintrage — avec leur limite tracée en pointillés, qui disent si la route est
  livrable.

Les deux registres sont nécessaires et ne se remplacent pas : une récompense
qui monte pendant que le rayon de cintrage stagne sous la limite signale une
pondération mal réglée, ce qu'aucune des deux courbes ne montre seule.

L'axe des abscisses porte le **numéro d'itération**, pas l'indice du point dans
l'historique. La distinction compte dès que l'historique est sous-échantillonné :
l'axe affichait sinon 0 à 400 indéfiniment, alors que les agents en étaient à
plusieurs milliers d'itérations.
"""

from __future__ import annotations

import customtkinter as ctk

__all__ = ["ProgressCharts"]

#: Séries tracées : clé interne, titre, unité.
SERIES = (
    ("reward", "Récompense", "score"),
    ("clashes", "Interférences", "nombre"),
    ("distance", "Distance au DMU", "mm"),
    ("bend", "Rayon de cintrage", "mm"),
)

SERIES_EN = {
    "reward": ("Reward", "score"),
    "clashes": ("Interferences", "count"),
    "distance": ("Distance to DMU", "mm"),
    "bend": ("Bend radius", "mm"),
}

#: Rayon au-delà duquel un tracé est « droit » : au-delà l'échelle devient
#: illisible, un segment parfaitement droit ayant un rayon infini.
BEND_CAP_MM = 5000.0


class ProgressCharts:
    """Graphiques empilés, rafraîchis pendant le calcul."""

    #: Nombre de points conservés par courbe. Au-delà, on ne garde qu'un point
    #: sur deux : la courbe reste fluide et couvre toujours toute la session,
    #: au lieu d'oublier son début.
    MAX_POINTS = 400

    def __init__(self, container: ctk.CTkFrame, lang: str = "FR"):
        self.container = container
        self.lang = lang
        self.available = False
        self.figure = None
        self.canvas = None
        self._axes: dict = {}
        self._lines: dict = {}
        self._limits: dict = {}
        self._history: dict[str, dict[str, list]] = {}
        #: Pas d'échantillonnage courant, par agent (1 = toutes les itérations).
        self._stride: dict[str, int] = {}

    # -- construction ----------------------------------------------------

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

        self.figure = Figure(figsize=(5, 4.4), dpi=96)
        self.figure.patch.set_alpha(0.0)

        english = str(self.lang).upper().startswith("EN")
        for index, (key, title, unit) in enumerate(SERIES):
            if english:
                title, unit = SERIES_EN[key]
            ax = self.figure.add_subplot(len(SERIES), 1, index + 1)
            ax.set_title(title, fontsize=8, loc="left", color="#6C7684")
            ax.set_ylabel(unit, fontsize=7, color="#6C7684")
            ax.tick_params(labelsize=7, colors="#8A94A3")
            ax.grid(True, linestyle=":", alpha=0.35)
            ax.patch.set_alpha(0.0)
            for side in ax.spines.values():
                side.set_color("#DDE3EA")
            self._axes[key] = ax

        # Seul le dernier graphique porte le libellé de l'axe : répété quatre
        # fois, il mangerait la hauteur utile.
        self._axes[SERIES[-1][0]].set_xlabel(
            "Iteration" if english else "Itération", fontsize=7, color="#6C7684"
        )

        self.figure.tight_layout(pad=1.0)
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
        self._stride.clear()
        for lines in self._lines.values():
            for line in lines.values():
                try:
                    line.remove()
                except Exception:
                    pass
        self._lines.clear()
        if self.available:
            self.canvas.draw_idle()

    # -- alimentation ------------------------------------------------------

    def update(self, snapshot: dict, colors: dict):
        """Ajoute un point à chaque courbe.

        ``snapshot`` : ``{nom d'agent: {"iteration": int, "reward": float,
        "report": RouteReport}}``.
        """
        if not self.available:
            return

        for name, state in snapshot.items():
            report = state.get("report")
            if report is None:
                continue
            self._append(name, state, report)

        self._redraw(colors)

    def _append(self, name: str, state: dict, report):
        """Enregistre un point, en sous-échantillonnant si l'historique déborde."""
        kpis = report.kpis
        iteration = int(state.get("iteration", 0))

        history = self._history.setdefault(
            name, {"iteration": [], "reward": [], "clashes": [], "distance": [], "bend": []}
        )
        stride = self._stride.setdefault(name, 1)

        # Une fois l'historique décimé d'un facteur N, on n'enregistre plus
        # qu'une itération sur N : sans cela le début de session serait effacé
        # au profit de la fin, et la courbe perdrait justement la phase où l'on
        # cherche à savoir si l'apprentissage a démarré.
        if history["iteration"] and iteration - history["iteration"][-1] < stride:
            return

        radius = kpis.get("min_bend_radius_mm", float("inf"))
        history["iteration"].append(iteration)
        history["reward"].append(float(state.get("reward", 0.0)))
        history["clashes"].append(kpis.get("n_clashes", 0))
        history["distance"].append(kpis.get("mean_distance_mm", 0.0))
        history["bend"].append(min(radius, BEND_CAP_MM))

        if len(history["iteration"]) > self.MAX_POINTS:
            for series in history.values():
                del series[1::2]
            self._stride[name] = stride * 2

    def _redraw(self, colors: dict):
        for name, history in self._history.items():
            steps = history["iteration"]
            if not steps:
                continue
            color = colors.get(name, "#2D7FF9")
            for key in self._axes:
                line = self._lines.setdefault(key, {}).get(name)
                if line is None:
                    ax = self._axes[key]
                    (line,) = ax.plot([], [], color=color, linewidth=1.4, alpha=0.9, label=name)
                    self._lines[key][name] = line
                line.set_data(steps, history[key])

        self._rescale()
        self.canvas.draw_idle()

    def _rescale(self):
        """Ajuste les échelles. L'abscisse porte le numéro d'itération."""
        steps = [h["iteration"] for h in self._history.values() if h["iteration"]]
        if not steps:
            return
        first = min(s[0] for s in steps)
        last = max(s[-1] for s in steps)
        # Une seule itération donnerait un intervalle nul, que matplotlib
        # refuse : on garde une largeur minimale.
        right = max(last, first + 10)

        for key, ax in self._axes.items():
            ax.set_xlim(first, right)

            series = [h[key] for h in self._history.values() if h[key]]
            if not series:
                continue
            low = min(min(s) for s in series)
            high = max(max(s) for s in series)

            # La limite réglementaire doit rester visible même quand toutes les
            # valeurs sont d'un seul côté : sinon l'utilisateur ne voit pas de
            # quoi il s'approche.
            for limit_line in self._limits.values():
                if limit_line.axes is ax:
                    value = limit_line.get_ydata()[0]
                    low, high = min(low, value), max(high, value)

            pad = max(1.0, (high - low) * 0.15)
            ax.set_ylim(low - pad, high + pad)

    def update_language(self, lang: str):
        """Retraduit les titres sans perdre l'historique tracé."""
        self.lang = lang
        if not self.available:
            return
        english = str(lang).upper().startswith("EN")
        for key, title, unit in SERIES:
            if english:
                title, unit = SERIES_EN[key]
            self._axes[key].set_title(title, fontsize=8, loc="left", color="#6C7684")
            self._axes[key].set_ylabel(unit, fontsize=7, color="#6C7684")
        self._axes[SERIES[-1][0]].set_xlabel(
            "Iteration" if english else "Itération", fontsize=7, color="#6C7684"
        )
        self.canvas.draw_idle()
