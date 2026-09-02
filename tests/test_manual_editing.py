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
    assert controller.pinned_points == []


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

def test_une_poignee_deplacee_pose_un_point():
    """Une poignée du tracé déplacée crée un point de passage."""
    controller, _ = controleur()
    controller.set_manual_editing(True)
    controller._on_handle_moved(3, (400.0, 90.0, 10.0))
    assert controller.pinned_points == [[400.0, 90.0, 10.0]]


def test_reprendre_un_point_pose_le_deplace_au_lieu_d_en_ajouter():
    """Les premières poignées *sont* les points posés : les rouvrir les bouge.

    Sinon chaque retouche laisserait une contrainte de plus derrière elle, et
    le tracé finirait cloué par une dizaine d'intentions successives.
    """
    controller, _ = controleur()
    controller.set_manual_editing(True)
    controller._on_handle_moved(5, (400.0, 90.0, 0.0))     # poignée du tracé
    controller._on_handle_moved(0, (400.0, 120.0, 0.0))    # le point posé
    assert controller.pinned_points == [[400.0, 120.0, 0.0]]


def test_deux_poignees_du_trace_donnent_deux_points():
    controller, _ = controleur()
    controller.set_manual_editing(True)
    controller._on_handle_moved(6, (400.0, 90.0, 0.0))
    controller._on_handle_moved(9, (700.0, -60.0, 0.0))
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
    assert controller.pinned_points == []


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

def test_un_point_pose_est_une_zone_et_non_un_sommet_fige():
    """Le défaut signalé : l'édition manuelle défaisait le travail des agents.

    Un sommet exigé au millimètre puis gelé ne peut plus être lissé, et le
    tracé se plie autour au lieu de s'améliorer. La contrainte est donc une
    zone : le câble doit passer *à portée*, pas exactement dessus.
    """
    from core.safety import project_anchors

    trace = np.linspace([0, 0, 0], [1000, 0, 0], 21).astype(np.float32)
    ancre = np.array([[500.0, 200.0, 0.0]], dtype=np.float32)
    bouges, ecart = project_anchors(trace, ancre, tolerance_mm=30.0)

    assert bouges == 1
    assert ecart == pytest.approx(30.0), "à portée, pas dessus"
    assert np.linalg.norm(trace - ancre[0], axis=1).min() == pytest.approx(30.0)


def test_un_cable_deja_a_portee_n_est_pas_touche():
    """L'agent garde la main tant qu'il respecte la zone."""
    from core.safety import project_anchors

    trace = np.linspace([0, 0, 0], [1000, 0, 0], 21).astype(np.float32)
    avant = trace.copy()
    bouges, _ = project_anchors(trace, np.array([[500.0, 10.0, 0.0]], dtype=np.float32),
                                tolerance_mm=30.0)
    assert bouges == 0
    assert np.allclose(trace, avant)


def test_une_ancre_ne_gele_aucun_point():
    """C'est tout le sujet : les agents doivent pouvoir continuer à lisser."""
    source = open("core/agent_worker.py", encoding="utf-8").read()
    assert "safety.project_anchors(" in source
    assert "anchors=pinned_points" in source
    assert "frozen_indices = snap_comb_passages(" in source


def test_deux_ancres_ne_se_disputent_pas_le_meme_point():
    from core.safety import project_anchors

    trace = np.linspace([0, 0, 0], [1000, 0, 0], 21).astype(np.float32)
    ancres = np.array([[300.0, 200.0, 0.0], [320.0, 200.0, 0.0]], dtype=np.float32)
    assert project_anchors(trace, ancres, tolerance_mm=30.0)[0] == 2


def test_les_extremites_ne_servent_jamais_d_ancrage():
    from core.safety import project_anchors

    trace = np.linspace([0, 0, 0], [1000, 0, 0], 21).astype(np.float32)
    depart, arrivee = trace[0].copy(), trace[-1].copy()
    project_anchors(trace, np.array([[0.0, 100.0, 0.0], [1000.0, 100.0, 0.0]],
                                    dtype=np.float32), tolerance_mm=5.0)
    assert np.allclose(trace[0], depart) and np.allclose(trace[-1], arrivee)
