"""L'humain dans la boucle : désigner une encoche, déplacer un point.

Deux gestes, une seule mécanique. Un point désigné à la souris devient un
**point imposé** : les agents le replacent à chaque itération puis cessent de
le déplacer. C'est exactement ce qui existait déjà pour les encoches de
peigne ; l'édition manuelle n'invente rien, elle ouvre la même porte à
l'utilisateur.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh non installé")

from controller.app_controller import MAX_HANDLES, AppController  # noqa: E402
from core.agent.tool import snap_mandatory_points  # noqa: E402


class _Traducteur:
    lang = "FR"
    is_english = False

    def __call__(self, key, **kwargs):
        return key


class _Vue:
    def __init__(self):
        self.t = _Traducteur()
        self.messages = []
        self.pages = {2: _Page()}

    def after(self, _delay, callback):
        callback()

    def set_status(self, message, tone="neutral"):
        self.messages.append((tone, message))


class _Page:
    def __init__(self):
        self.editing = None
        self.detached = None
        self.use_fixations = None

    def set_manual_editing(self, active):
        self.editing = active

    def set_detached(self, value):
        self.detached = value

    def set_use_fixations(self, value):
        self.use_fixations = value

    def show_fixation_scan(self, *_a, **_k):
        pass


class _Viewer:
    """Vue 3D factice : on n'observe que ce qui lui est demandé."""

    def __init__(self, available=True):
        self.is_available = available
        self.is_open = False
        self.handles = None
        self.handle_radius = None
        self.actors = {}
        self.pick_targets = None
        self.opened = 0

    def set_handles(self, points, radius=30.0):
        self.handles = [list(p) for p in points]
        self.handle_radius = radius

    def clear_handles(self):
        self.set_handles([])

    def set_pick_targets(self, targets, tolerance_mm=120.0):
        self.pick_targets = list(targets)

    def open_window(self):
        self.opened += 1
        self.is_open = True

    def show_sphere(self, center, name, **_kwargs):
        self.actors[name] = center

    def show_path(self, points, name, **_kwargs):
        self.actors[name] = points

    def show_mesh(self, mesh, name, **_kwargs):
        self.actors[name] = mesh

    def remove_prefix(self, prefix):
        for name in [n for n in self.actors if n.startswith(prefix)]:
            del self.actors[name]

    def render(self):
        pass


def controleur(available=True):
    view = _Vue()
    controller = AppController(view)
    controller.viewer = _Viewer(available)
    controller._initial_path = np.linspace([0, 0, 0], [1000, 0, 0], 40)
    return controller, view


# ----------------------------------------------------------------------
# Les poignées
# ----------------------------------------------------------------------

def test_l_edition_pose_des_poignees_sur_le_trace():
    controller, _ = controleur()
    controller.set_manual_editing(True)
    assert controller.viewer.handles


def test_les_poignees_sont_echantillonnees():
    """Une par point serait illisible, et deux voisines se recouvriraient."""
    controller, _ = controleur()
    controller._initial_path = np.linspace([0, 0, 0], [1000, 0, 0], 200)
    controller.set_manual_editing(True)
    assert 0 < len(controller.viewer.handles) <= MAX_HANDLES


def test_les_extremites_ne_recoivent_pas_de_poignee():
    """A et B appartiennent aux équipements : ils ne se négocient pas."""
    controller, _ = controleur()
    controller.set_manual_editing(True)
    assert 0 not in controller._handle_indices
    assert (len(controller._initial_path) - 1) not in controller._handle_indices


def test_desarmer_retire_les_poignees():
    controller, _ = controleur()
    controller.set_manual_editing(True)
    controller.set_manual_editing(False)
    assert controller.viewer.handles == []


def test_desarmer_ne_libere_pas_les_points_imposes():
    """Ils ont été posés délibérément : les perdre ferait tout recommencer."""
    controller, _ = controleur()
    controller.set_manual_editing(True)
    controller._on_handle_moved(2, (500.0, 60.0, 0.0))
    controller.set_manual_editing(False)
    assert controller.pinned_points


def test_liberer_les_points_est_une_action_a_part():
    controller, _ = controleur()
    controller.set_manual_editing(True)
    controller._on_handle_moved(2, (500.0, 60.0, 0.0))
    controller.clear_pinned_points()
    assert controller.pinned_points == {}


def test_sans_trace_l_edition_le_dit():
    controller, view = controleur()
    controller._initial_path = None
    controller.set_manual_editing(True)
    assert view.messages[-1][0] == "warn"


def test_sans_vue_3d_l_edition_se_desarme():
    """Poser des poignées invisibles laisserait croire qu'elles existent."""
    controller, view = controleur(available=False)
    controller.set_manual_editing(True)
    assert controller.manual_editing is False
    assert view.pages[2].editing is False


# ----------------------------------------------------------------------
# Le point imposé
# ----------------------------------------------------------------------

def test_une_poignee_deplacee_impose_son_point():
    controller, _ = controleur()
    controller.set_manual_editing(True)
    controller._on_handle_moved(3, (400.0, 90.0, 10.0))
    assert list(controller.pinned_points.values()) == [[400.0, 90.0, 10.0]]


def test_deplacer_la_meme_poignee_remplace_son_point():
    """Sinon chaque micro-mouvement laisserait une contrainte derrière lui."""
    controller, _ = controleur()
    controller.set_manual_editing(True)
    controller._on_handle_moved(3, (400.0, 90.0, 0.0))
    controller._on_handle_moved(3, (400.0, 120.0, 0.0))
    assert len(controller.pinned_points) == 1
    assert list(controller.pinned_points.values())[0][1] == 120.0


def test_deux_poignees_donnent_deux_points():
    controller, _ = controleur()
    controller.set_manual_editing(True)
    controller._on_handle_moved(3, (400.0, 90.0, 0.0))
    controller._on_handle_moved(7, (700.0, -60.0, 0.0))
    assert len(controller.pinned_points) == 2


def test_le_point_impose_est_dessine():
    controller, _ = controleur()
    controller.set_manual_editing(True)
    controller._on_handle_moved(3, (400.0, 90.0, 0.0))
    assert "pinned_0" in controller.viewer.actors


def test_un_deplacement_hors_edition_est_ignore():
    """Une poignée résiduelle ne doit pas imposer un point à l'insu de tous."""
    controller, _ = controleur()
    controller._on_handle_moved(3, (400.0, 90.0, 0.0))
    assert controller.pinned_points == {}


def test_les_points_imposes_partent_vers_les_agents():
    controller, _ = controleur()
    controller.shared_state = {"config": {}}
    controller.set_manual_editing(True)
    controller._on_handle_moved(3, (400.0, 90.0, 0.0))
    assert controller.shared_state["config"]["pinned_points"] == [[400.0, 90.0, 0.0]]


def test_liberer_vide_aussi_la_liste_transmise():
    controller, _ = controleur()
    controller.shared_state = {"config": {}}
    controller.set_manual_editing(True)
    controller._on_handle_moved(3, (400.0, 90.0, 0.0))
    controller.clear_pinned_points()
    assert controller.shared_state["config"]["pinned_points"] == []


# ----------------------------------------------------------------------
# Ce que les agents en font
# ----------------------------------------------------------------------

def test_l_agent_replace_un_point_impose():
    """La mécanique est celle des encoches : épingler, puis geler."""
    trace = np.linspace([0, 0, 0], [1000, 0, 0], 21).astype(np.float32)
    cible = np.array([[500.0, 80.0, 0.0]], dtype=np.float32)
    locked = snap_mandatory_points(trace, cible)
    assert len(locked) == 1
    index = next(iter(locked))
    assert np.allclose(trace[index], cible[0])


def test_un_point_impose_ne_prend_pas_la_place_d_une_encoche():
    """Les encoches passent d'abord : elles ne sont pas négociables."""
    trace = np.linspace([0, 0, 0], [1000, 0, 0], 21).astype(np.float32)
    reserve = {10, 11}
    locked = snap_mandatory_points(trace, np.array([[500.0, 80.0, 0.0]], dtype=np.float32),
                                   used=reserve)
    assert reserve <= locked
    assert len(locked - reserve) == 1


def test_le_terme_est_branche_sur_l_agent():
    """Un point imposé non relu par l'agent ne contraindrait rien."""
    source = open("core/agent_worker.py", encoding="utf-8").read()
    assert 'cfg.get("pinned_points")' in source
    assert "snap_mandatory_points(\n                wp_current, pinned_points" in source
