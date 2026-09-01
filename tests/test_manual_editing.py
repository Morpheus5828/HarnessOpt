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


# ----------------------------------------------------------------------
# Désigner une encoche au clic
# ----------------------------------------------------------------------

def peigne(nom, x, n=3):
    from core.fixation_scan import Passage

    return [Passage(index=i, p_in=(x, i * 40.0, 0.0), p_out=(x, i * 40.0, 20.0),
                    comb=nom)
            for i in range(n)]


def test_chaque_encoche_devient_cliquable():
    controller, _ = controleur()
    controller._publish_pick_targets([peigne("A", 500.0), peigne("B", 700.0)])
    cles = [key for key, _ in controller.viewer.pick_targets]
    assert cles == [("A", 0), ("A", 1), ("A", 2), ("B", 0), ("B", 1), ("B", 2)]


def test_la_cible_cliquable_est_le_centre_de_l_encoche():
    """On vise une encoche, pas une entrée ou une sortie."""
    controller, _ = controleur()
    controller._publish_pick_targets([peigne("A", 500.0, n=1)])
    _, position = controller.viewer.pick_targets[0]
    assert tuple(position) == (500.0, 0.0, 10.0)


def test_un_clic_sans_question_en_cours_ne_fait_rien():
    controller, _ = controleur()
    controller._pick_handler = None
    controller._on_viewer_pick(("A", 1))          # ne doit pas lever


def test_un_clic_est_transmis_au_choix_en_cours():
    controller, _ = controleur()
    vus = []
    controller._pick_handler = lambda comb, index: vus.append((comb, index))
    controller._on_viewer_pick(("A", 2))
    assert vus == [("A", 2)]


def test_une_cle_mal_formee_ne_leve_pas():
    controller, _ = controleur()
    controller._pick_handler = lambda *_a: pytest.fail("ne devrait pas être appelé")
    controller._on_viewer_pick("A")
    controller._on_viewer_pick(None)


def scan_deux_peignes():
    from core.fixation_scan import summarise

    peignes = []
    for nom, x in (("avant.stl", 400.0), ("arriere.stl", 700.0)):
        points = []
        for i in range(3):
            points += [[x, i * 40.0, 0.0], [x, i * 40.0, 20.0]]
        peignes.append({"name": nom, "position": [x, 40.0, 10.0], "score": 0.9,
                        "routing_points": points})
    return summarise(peignes)


def controleur_avec_scan():
    controller, view = controleur()
    controller.point_a = np.array([0.0, 40.0, 10.0], dtype=np.float32)
    controller.point_b = np.array([1100.0, 40.0, 10.0], dtype=np.float32)
    controller.scan_result = scan_deux_peignes()
    return controller, view


def pose_la_question(controller, view, reponse=None):
    """Ouvre le choix sans fenêtre Tk, et rend la main sur le clic branché."""
    capture = {}

    def faux_dialogue(combs, selection=None, scan_message="", on_change=None,
                      on_ready=None):
        capture["selection"] = dict(selection or {})
        capture["on_change"] = on_change
        capture["handler"] = controller._pick_handler
        return reponse if reponse is not None else dict(selection or {})

    view.ask_fixation_choice = faux_dialogue
    controller._ask_fixation_choice(controller.scan_result, {"use_fixations": True})
    return capture


def test_le_choix_rend_les_encoches_cliquables():
    controller, view = controleur_avec_scan()
    pose_la_question(controller, view)
    cles = [key for key, _ in controller.viewer.pick_targets]
    assert ("avant.stl", 0) in cles and ("arriere.stl", 2) in cles


def encoches_allumees(controller):
    """Rangs des encoches dessinées en couleur, donc réellement empruntées."""
    return {int(name.rsplit("_", 1)[1])
            for name in controller.viewer.actors if name.startswith("fixation_in_")}


def test_un_clic_retient_l_encoche_pour_son_peigne():
    """Cliquer une encoche la désigne ; recliquer la même écarte le peigne."""
    controller, view = controleur_avec_scan()
    capture = pose_la_question(controller, view)
    controller._pick_handler = capture["handler"]

    # Les passages sont listés peigne par peigne : 0-2 pour « avant », 3-5
    # pour « arriere ». Cliquer la troisième encoche d'« avant » l'allume.
    controller._on_viewer_pick(("avant.stl", 2))
    assert 2 in encoches_allumees(controller)
    assert not ({0, 1} & encoches_allumees(controller))

    controller._on_viewer_pick(("avant.stl", 2))
    assert not ({0, 1, 2} & encoches_allumees(controller)), \
        "recliquer l'encoche retenue écarte le peigne"
    assert encoches_allumees(controller), "l'autre peigne reste emprunté"


def test_le_clic_est_debranche_une_fois_la_question_tranchee():
    """La scène reste cliquable ; la décision, non."""
    controller, view = controleur_avec_scan()
    capture = pose_la_question(controller, view)
    assert capture["handler"] is not None, "branché pendant la question"
    assert controller._pick_handler is None, "débranché après"


def test_le_choix_propose_couvre_chaque_peigne():
    controller, view = controleur_avec_scan()
    capture = pose_la_question(controller, view)
    assert set(capture["selection"]) == {"avant.stl", "arriere.stl"}


def test_la_vue_3d_s_ouvre_pour_la_question():
    controller, _ = controleur_avec_scan()
    controller._open_viewer_window()
    assert controller.viewer.opened == 1
    controller._open_viewer_window()
    assert controller.viewer.opened == 1, "déjà ouverte : on ne la rouvre pas"
