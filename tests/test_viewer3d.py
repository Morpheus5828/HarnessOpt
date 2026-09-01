"""Vue 3D : fil de rendu, fermeture de fenêtre, clic et poignées.

Le fil de rendu est testé avec un plotter factice. C'est volontaire : ce qu'on
veut vérifier ici n'est pas que VTK sait dessiner — il sait — mais que les
ordres arrivent, qu'un ordre invalide n'emporte pas le fil, qu'aucun appel 3D
ne se produit sur le fil appelant, et surtout **qu'on cesse de dessiner dans
une fenêtre fermée**. Ce dernier point est celui qui noyait la console
d'« ERR| Could not create shader object » : ``Plotter.update()`` appelle
``render()`` sans rien vérifier, et une fenêtre fermée par l'utilisateur ne
lève aucune exception — elle rend dans un contexte OpenGL détruit.
"""

import time

import pytest

pytest.importorskip("tkinter", reason="tkinter absent de cet interpréteur")
pytest.importorskip("customtkinter", reason="customtkinter non installé")

from ui import viewer3d as v3  # noqa: E402


# ----------------------------------------------------------------------
# Désignation par clic — fonction pure
# ----------------------------------------------------------------------

CIBLES = [("a", (0.0, 0.0, 0.0)), ("b", (500.0, 0.0, 0.0)), ("c", (0.0, 500.0, 0.0))]


def test_un_clic_designe_le_repere_le_plus_proche():
    assert v3.nearest_target((480.0, 10.0, 0.0), CIBLES) == "b"


def test_un_clic_loin_de_tout_ne_designe_rien():
    """Un clic de rotation manqué ne doit pas basculer un choix."""
    assert v3.nearest_target((5000.0, 5000.0, 0.0), CIBLES) is None


def test_la_tolerance_commande_la_portee():
    point = (100.0, 0.0, 0.0)
    assert v3.nearest_target(point, CIBLES, tolerance_mm=50.0) is None
    assert v3.nearest_target(point, CIBLES, tolerance_mm=150.0) == "a"


def test_sans_repere_rien_n_est_designe():
    assert v3.nearest_target((0.0, 0.0, 0.0), []) is None
    assert v3.nearest_target(None, CIBLES) is None


def test_un_point_mal_forme_ne_leve_pas():
    assert v3.nearest_target((0.0, 0.0), CIBLES) is None


# ----------------------------------------------------------------------
# Le fil de rendu, avec un plotter factice
# ----------------------------------------------------------------------

class FakeActor:
    def __init__(self):
        self.visible = True
        self.prop = type("P", (), {"show_edges": False})()

    def SetVisibility(self, value):
        self.visible = bool(value)


class FakeInteractor:
    def __init__(self):
        self.done = False
        self.observers = []

    def GetDone(self):
        return self.done


class FakeIren:
    def __init__(self):
        self.interactor = FakeInteractor()
        self.observers = []

    def add_observer(self, event, call):
        self.observers.append((event, call))


class FakePlotter:
    """Plotter factice qui sait être fermé, comme le vrai."""

    def __init__(self):
        self.actors = {}
        self.render_window = object()
        self.iren = FakeIren()
        self._closed = False
        self.reset_calls = 0
        self.updates = 0
        self.shown = False
        self.widgets = None
        self.picking = None

    # -- ce que le fil appelle -------------------------------------
    def set_background(self, *_a, **_k):
        pass

    def add_axes(self):
        pass

    def add_mesh(self, geometry, name=None, **style):
        self.actors[name] = FakeActor()
        return self.actors[name]

    def remove_actor(self, name):
        self.actors.pop(name, None)

    def reset_camera(self):
        self.reset_calls += 1

    def show(self, **_kwargs):
        self.shown = True

    def update(self):
        if self._closed:
            raise RuntimeError("la fenêtre est fermée")
        self.updates += 1

    def close(self):
        self._closed = True

    def enable_point_picking(self, callback=None, **_kwargs):
        self.picking = callback

    def clear_sphere_widgets(self):
        self.widgets = None

    def add_sphere_widget(self, callback, center=None, **_kwargs):
        self.widgets = (callback, center)

    # -- pour les tests --------------------------------------------
    def ferme_par_l_utilisateur(self):
        """Ce que fait VTK : l'interacteur passe à « done », rien ne lève."""
        self.iren.interactor.done = True


@pytest.fixture
def thread(monkeypatch):
    """Fil de rendu réel, branché sur un plotter factice."""
    ready, states, picks, moves = [], [], [], []

    monkeypatch.setattr(v3._RenderThread, "_probe_import", staticmethod(lambda: None))

    def fake_open(self):
        if self._window_alive():
            return
        self._drop_window(notify=False)
        self.plotter = FakePlotter()
        for name, (geometry, style) in list(self.recipes.items()):
            self.plotter.add_mesh(geometry, name=name, **style)
        self.plotter.show()
        self._exited = False
        self._observe_exit()
        self._install_picking()
        self._install_handles()
        self._on_window_state(True)

    monkeypatch.setattr(v3._RenderThread, "_do_open_window", fake_open)
    th = v3._RenderThread(
        on_ready=lambda ok, err: ready.append((ok, err)),
        on_window_state=states.append,
        on_pick=picks.append,
        on_handle_move=lambda i, p: moves.append((i, p)),
        size=(320, 240),
    )
    th.start()
    for _ in range(200):
        if ready:
            break
        time.sleep(0.01)
    yield th, ready, states, picks, moves
    th.stop()
    th.join(timeout=2)


def drain(th, timeout=2.0):
    """Attend que la file d'ordres soit vide."""
    end = time.time() + timeout
    while time.time() < end and not th.orders.empty():
        time.sleep(0.01)
    time.sleep(0.1)


def ouvre(th):
    th.orders.put(("open_window", (), {}))
    drain(th)
    return th.plotter


def test_le_fil_signale_qu_il_est_pret(thread):
    th, ready, _, _, _ = thread
    assert ready and ready[0][0] is True


def test_la_scene_vit_sans_fenetre(thread):
    """Les ordres sont acceptés fenêtre fermée : la scène est mémorisée."""
    th, _, _, _, _ = thread
    th.orders.put(("add", (object(), "dmu"), {"color": "grey"}))
    drain(th)
    assert "dmu" in th.recipes
    assert th.plotter is None


def test_la_scene_est_rejouee_a_l_ouverture(thread):
    th, _, _, _, _ = thread
    th.orders.put(("add", (object(), "dmu"), {"color": "grey"}))
    th.orders.put(("add", (object(), "traj"), {"color": "blue"}))
    drain(th)
    plotter = ouvre(th)
    assert set(plotter.actors) == {"dmu", "traj"}


def test_un_acteur_retire_disparait_des_deux_cotes(thread):
    th, _, _, _, _ = thread
    th.orders.put(("add", (object(), "dmu"), {}))
    drain(th)
    plotter = ouvre(th)
    th.orders.put(("remove", ("dmu",), {}))
    drain(th)
    assert "dmu" not in th.recipes
    assert "dmu" not in plotter.actors


def test_un_ordre_inconnu_ne_tue_pas_le_fil(thread):
    th, _, _, _, _ = thread
    th.orders.put(("nexiste_pas", (), {}))
    th.orders.put(("add", (object(), "dmu"), {}))
    drain(th)
    assert th.is_alive() and "dmu" in th.recipes


def test_un_ordre_qui_echoue_ne_tue_pas_le_fil(thread):
    th, _, _, _, _ = thread
    ouvre(th)
    th.orders.put(("add", (object(),), {}))      # nom manquant
    th.orders.put(("add", (object(), "ok"), {}))
    drain(th)
    assert th.is_alive() and "ok" in th.recipes


# ----------------------------------------------------------------------
# Fermeture de la fenêtre — le défaut signalé
# ----------------------------------------------------------------------

def test_une_fenetre_fermee_n_est_plus_rafraichie(thread):
    """Le défaut : VTK dessine dans un contexte détruit et crache des shaders."""
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    for _ in range(50):
        if plotter.updates > 0:
            break
        time.sleep(0.01)
    assert plotter.updates > 0, "la fenêtre ouverte doit être pompée"

    plotter.ferme_par_l_utilisateur()
    time.sleep(0.2)
    avant = plotter.updates
    time.sleep(0.3)
    assert plotter.updates == avant, "plus aucun rendu après fermeture"


def test_la_fermeture_est_detectee_sans_exception(thread):
    """VTK ne lève pas : c'est à nous de le demander."""
    th, _, states, _, _ = thread
    plotter = ouvre(th)
    plotter.ferme_par_l_utilisateur()
    end = time.time() + 2
    while time.time() < end and states[-1] is not False:
        time.sleep(0.02)
    assert states[-1] is False
    assert th.plotter is None


def test_les_autres_signaux_de_fermeture_sont_lus(thread):
    th, _, _, _, _ = thread

    plotter = ouvre(th)
    plotter._closed = True
    assert not th._window_alive()

    plotter = ouvre(th)
    plotter.render_window = None
    assert not th._window_alive()

    plotter = ouvre(th)
    th._exited = True
    assert not th._window_alive()


def test_l_evenement_de_sortie_est_observe(thread):
    """C'est le signal que VTK émet quand l'utilisateur clique la croix."""
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    evenements = [event for event, _ in plotter.iren.observers]
    assert "ExitEvent" in evenements
    for event, call in plotter.iren.observers:
        if event == "ExitEvent":
            call()
    assert not th._window_alive()


def test_la_fenetre_se_rouvre_avec_sa_scene(thread):
    """Fermer n'est pas perdre : la scène est décrite, pas dessinée."""
    th, _, _, _, _ = thread
    th.orders.put(("add", (object(), "dmu"), {}))
    drain(th)
    plotter = ouvre(th)
    plotter.ferme_par_l_utilisateur()
    time.sleep(0.3)
    encore = ouvre(th)
    assert encore is not plotter
    assert "dmu" in encore.actors


def test_l_arret_ferme_le_plotter(thread):
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    th.stop()
    th.join(timeout=2)
    assert not th.is_alive()
    assert plotter._closed is True


def test_pyvista_absent_est_signale_sans_exception(monkeypatch):
    def refuse():
        raise ImportError("pas de pyvista")

    monkeypatch.setattr(v3._RenderThread, "_probe_import", staticmethod(refuse))
    ready = []
    th = v3._RenderThread(on_ready=lambda ok, err: ready.append((ok, err)),
                          on_window_state=lambda _s: None,
                          on_pick=lambda _k: None,
                          on_handle_move=lambda _i, _p: None,
                          size=(320, 240))
    th.start()
    th.join(timeout=2)
    assert ready and ready[0][0] is False
    assert "pyvista" in ready[0][1].lower()
    assert not th.is_alive()


# ----------------------------------------------------------------------
# Clic et poignées
# ----------------------------------------------------------------------

def test_un_clic_sur_un_repere_est_transmis(thread):
    th, _, _, picks, _ = thread
    th.orders.put(("set_pick_targets", (CIBLES, 120.0), {}))
    drain(th)
    plotter = ouvre(th)
    assert plotter.picking is not None
    plotter.picking((10.0, 0.0, 0.0))
    assert picks == ["a"]


def test_un_clic_dans_le_vide_ne_transmet_rien(thread):
    th, _, _, picks, _ = thread
    th.orders.put(("set_pick_targets", (CIBLES, 120.0), {}))
    drain(th)
    ouvre(th).picking((9000.0, 0.0, 0.0))
    assert picks == []


def test_les_poignees_sont_posees_sur_les_points_demandes(thread):
    th, _, _, _, _ = thread
    points = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
    th.orders.put(("set_handles", (points, 25.0), {}))
    drain(th)
    plotter = ouvre(th)
    assert plotter.widgets is not None
    _, centres = plotter.widgets
    assert len(centres) == 2


def test_une_poignee_deplacee_est_transmise(thread):
    th, _, _, _, moves = thread
    th.orders.put(("set_handles", ([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], 25.0), {}))
    drain(th)
    callback, _ = ouvre(th).widgets
    callback((5.0, 6.0, 7.0), 1)
    assert moves == [(1, (5.0, 6.0, 7.0))]


def test_vider_les_poignees_les_retire(thread):
    th, _, _, _, _ = thread
    th.orders.put(("set_handles", ([[0.0, 0.0, 0.0]], 25.0), {}))
    drain(th)
    plotter = ouvre(th)
    th.orders.put(("set_handles", ([], 25.0), {}))
    drain(th)
    assert plotter.widgets is None


# ----------------------------------------------------------------------
# La façade : ce sont ces garanties qui empêchent le gel
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def root():
    ctk = pytest.importorskip("customtkinter")
    try:
        window = ctk.CTk()
    except Exception as exc:  # pas d'affichage disponible
        pytest.skip(f"aucun serveur graphique disponible : {exc}")
    yield window
    window.destroy()


@pytest.fixture
def slow_probe(monkeypatch):
    """Simule un import lent — treize secondes en conditions réelles."""
    monkeypatch.setattr(v3._RenderThread, "_probe_import",
                        staticmethod(lambda: time.sleep(0.6)))


def test_le_demarrage_rend_la_main_immediatement(root, slow_probe):
    """Le cœur de la panne signalée : l'application se figeait ici."""
    import customtkinter as ctk

    frame = ctk.CTkFrame(root)
    viewer = v3.Viewer3D(frame)
    try:
        started = time.perf_counter()
        mode = viewer.start()
        elapsed = time.perf_counter() - started
        assert elapsed < 0.2, f"start() a bloqué {elapsed:.2f} s"
        assert mode == v3.MODE_STARTING
    finally:
        viewer.close()
        frame.destroy()


def test_les_ordres_emis_avant_la_fin_du_demarrage_sont_rejoues(root, slow_probe):
    """Charger la maquette ne doit pas attendre que la 3D soit prête."""
    import customtkinter as ctk

    frame = ctk.CTkFrame(root)
    viewer = v3.Viewer3D(frame)
    try:
        viewer.start()
        assert viewer.is_available, "les ordres doivent être acceptés dès le départ"
        viewer.show_mesh(object(), "dmu")

        end = time.time() + 5
        while time.time() < end:
            thread = viewer._thread
            if thread is not None and "dmu" in thread.recipes:
                break
            root.update()
            time.sleep(0.02)
        assert "dmu" in viewer._thread.recipes
    finally:
        viewer.close()
        frame.destroy()


def test_les_ordres_ne_touchent_pas_a_vtk_sur_le_fil_appelant(root, slow_probe):
    """Aucun appel 3D ne doit avoir lieu depuis le fil de l'interface."""
    import customtkinter as ctk

    frame = ctk.CTkFrame(root)
    viewer = v3.Viewer3D(frame)
    try:
        viewer.start()
        started = time.perf_counter()
        for index in range(200):
            viewer.show_path([[0, 0, 0], [index + 1, 0, 0]], f"traj_{index}")
        viewer.reset_camera()
        viewer.render()
        elapsed = time.perf_counter() - started
        assert elapsed < 0.5, f"les ordres ont coûté {elapsed:.2f} s au fil de l'interface"
    finally:
        viewer.close()
        frame.destroy()


def test_la_vue_reste_utilisable_apres_fermeture(root, slow_probe):
    """Fermer puis continuer d'émettre des ordres ne doit rien casser."""
    import customtkinter as ctk

    frame = ctk.CTkFrame(root)
    viewer = v3.Viewer3D(frame)
    viewer.start()
    viewer.close()
    viewer.show_mesh(object(), "dmu")
    viewer.render()
    viewer.set_handles([[0, 0, 0]])
    viewer.open_window()
    assert not viewer.is_available
    frame.destroy()


def test_la_facade_suit_l_etat_de_la_fenetre(root, slow_probe):
    import customtkinter as ctk

    frame = ctk.CTkFrame(root)
    viewer = v3.Viewer3D(frame)
    try:
        viewer.start()
        assert viewer.is_open is False
        viewer._on_window_state(True)
        assert viewer.is_open is True and viewer.mode == v3.MODE_WINDOW
        viewer._on_window_state(False)
        assert viewer.is_open is False and viewer.mode == v3.MODE_CLOSED
    finally:
        viewer.close()
        frame.destroy()


def test_le_basculement_annonce_l_etat_vise(root, slow_probe):
    import customtkinter as ctk

    frame = ctk.CTkFrame(root)
    viewer = v3.Viewer3D(frame)
    try:
        viewer.start()
        assert viewer.toggle_window() is True
        viewer._on_window_state(True)
        assert viewer.toggle_window() is False
    finally:
        viewer.close()
        frame.destroy()
