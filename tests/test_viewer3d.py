"""Vue 3D : gestes de souris et fil de rendu.

Le fil de rendu est testé avec un plotter factice. C'est volontaire : ce qu'on
veut vérifier ici n'est pas que VTK sait dessiner — il sait — mais que les
ordres arrivent dans le bon ordre, qu'un ordre invalide n'emporte pas le fil,
et surtout qu'aucun appel 3D ne se produit sur le fil appelant. C'est cette
dernière propriété qui a manqué à la version précédente et gelait
l'application treize secondes durant.
"""

import time

import pytest

pytest.importorskip("tkinter", reason="tkinter absent de cet interpréteur")
pytest.importorskip("customtkinter", reason="customtkinter non installé")

from ui import viewer3d as v3  # noqa: E402


# ----------------------------------------------------------------------
# Traduction des gestes — fonctions pures
# ----------------------------------------------------------------------

def test_orbit_suit_le_doigt():
    """Tirer vers la droite fait tourner la scène vers la droite."""
    az, _ = v3.orbit_delta(10, 0)
    assert az < 0
    az2, _ = v3.orbit_delta(-10, 0)
    assert az2 > 0


def test_orbit_est_proportionnel():
    a1, e1 = v3.orbit_delta(10, 10)
    a2, e2 = v3.orbit_delta(20, 20)
    assert a2 == pytest.approx(2 * a1)
    assert e2 == pytest.approx(2 * e1)


def test_orbit_immobile_ne_bouge_pas():
    assert v3.orbit_delta(0, 0) == (0.0, 0.0)


def test_pan_utilise_la_meme_echelle_sur_les_deux_axes():
    """Un geste diagonal doit produire une translation diagonale.

    Rapporter x à la largeur et y à la hauteur déformerait le déplacement sur
    une vue qui n'est pas carrée.
    """
    fx, fy = v3.pan_delta(50, 50, 1600, 400)
    assert abs(fx) == pytest.approx(abs(fy))


def test_pan_est_inverse_en_x_et_direct_en_y():
    fx, fy = v3.pan_delta(10, 10, 800, 600)
    assert fx < 0 and fy > 0


def test_pan_supporte_une_hauteur_nulle():
    """Un <Configure> peut arriver avant que le cadre ait une taille."""
    fx, fy = v3.pan_delta(10, 10, 0, 0)
    assert all(map(lambda x: x == x, (fx, fy)))  # pas de NaN, pas d'exception


def test_zoom_avant_et_arriere_sont_reciproques():
    assert v3.zoom_factor(1) * v3.zoom_factor(-1) == pytest.approx(1.0)


def test_zoom_est_toujours_positif():
    for n in (-5, -1, 0, 1, 5):
        assert v3.zoom_factor(n) > 0


def test_taille_de_rendu_bornee():
    assert v3.clamp_size(0, 0) == (v3.MIN_SIZE, v3.MIN_SIZE)
    assert v3.clamp_size(99999, 99999) == (v3.MAX_SIZE, v3.MAX_SIZE)
    assert v3.clamp_size(800, 600) == (800, 600)


def test_taille_de_rendu_accepte_none():
    assert v3.clamp_size(None, None) == (v3.MIN_SIZE, v3.MIN_SIZE)


# ----------------------------------------------------------------------
# Le fil de rendu, avec un plotter factice
# ----------------------------------------------------------------------

class FakeCamera:
    def __init__(self):
        self.position = (0.0, 0.0, 10.0)
        self.focal_point = (0.0, 0.0, 0.0)
        self.up = (0.0, 1.0, 0.0)
        self.view_angle = 30.0
        self.calls = []

    def Azimuth(self, a):
        self.calls.append(("azimuth", a))

    def Elevation(self, e):
        self.calls.append(("elevation", e))

    def OrthogonalizeViewUp(self):
        self.calls.append(("ortho",))

    def Zoom(self, f):
        self.calls.append(("zoom", f))


class FakeActor:
    def __init__(self):
        self.visible = True
        self.prop = type("P", (), {"show_edges": False})()

    def SetVisibility(self, value):
        self.visible = bool(value)


class FakePlotter:
    def __init__(self):
        self.actors = {}
        self.camera = FakeCamera()
        self.window_size = [900, 600]
        self.reset_calls = 0
        self.closed = False

    def add_mesh(self, geometry, name=None, **style):
        self.actors[name] = FakeActor()
        return self.actors[name]

    def remove_actor(self, name):
        self.actors.pop(name, None)

    def reset_camera(self):
        self.reset_calls += 1

    def screenshot(self, return_img=True):
        import numpy as np

        return np.zeros((4, 4, 3), dtype="uint8")

    def close(self):
        self.closed = True


@pytest.fixture
def thread(monkeypatch):
    """Fil de rendu réel, mais branché sur un plotter factice."""
    frames, ready = [], []

    def fake_build(self):
        self.plotter = FakePlotter()
        return True

    monkeypatch.setattr(v3._RenderThread, "_build_plotter", fake_build)
    th = v3._RenderThread(on_frame=frames.append,
                          on_ready=lambda ok, err: ready.append((ok, err)),
                          size=(320, 240))
    th.start()
    for _ in range(200):
        if ready:
            break
        time.sleep(0.01)
    yield th, frames, ready
    th.stop()
    th.join(timeout=2)


def drain(th, timeout=2.0):
    """Attend que la file d'ordres soit vide."""
    end = time.time() + timeout
    while time.time() < end and not th.orders.empty():
        time.sleep(0.01)
    time.sleep(0.1)


def test_le_fil_signale_qu_il_est_pret(thread):
    _, _, ready = thread
    assert ready and ready[0][0] is True


def test_un_ordre_ajoute_un_acteur(thread):
    th, _, _ = thread
    th.orders.put(("add", (object(), "dmu"), {"color": "#fff"}))
    drain(th)
    assert "dmu" in th.plotter.actors
    assert "dmu" in th.actor_names


def test_un_acteur_retire_disparait(thread):
    th, _, _ = thread
    th.orders.put(("add", (object(), "dmu"), {}))
    th.orders.put(("remove", ("dmu",), {}))
    drain(th)
    assert "dmu" not in th.plotter.actors
    assert "dmu" not in th.actor_names


def test_la_visibilite_est_appliquee(thread):
    th, _, _ = thread
    th.orders.put(("add", (object(), "dmu"), {}))
    th.orders.put(("visible", ("dmu", False), {}))
    drain(th)
    assert th.plotter.actors["dmu"].visible is False


def test_un_ordre_inconnu_ne_tue_pas_le_fil(thread):
    th, _, _ = thread
    th.orders.put(("ordre_qui_n_existe_pas", (), {}))
    th.orders.put(("add", (object(), "dmu"), {}))
    drain(th)
    assert th.is_alive()
    assert "dmu" in th.plotter.actors


def test_un_ordre_qui_echoue_ne_tue_pas_le_fil(thread):
    """Régler la visibilité d'un acteur absent ne doit rien casser."""
    th, _, _ = thread
    th.orders.put(("visible", ("inexistant", True), {}))
    th.orders.put(("add", (object(), "dmu"), {}))
    drain(th)
    assert th.is_alive()
    assert "dmu" in th.plotter.actors


def test_une_image_est_produite_apres_un_ordre(thread):
    th, frames, _ = thread
    th.orders.put(("add", (object(), "dmu"), {}))
    end = time.time() + 2
    while time.time() < end and not frames:
        time.sleep(0.01)
    assert frames, "aucune image produite par le fil de rendu"


def test_les_ordres_successifs_sont_regroupes(thread):
    """Vingt ordres ne doivent pas produire vingt images.

    Sans regroupement, un simple glisser de souris déclencherait un rendu par
    pixel parcouru.
    """
    th, frames, _ = thread
    for i in range(20):
        th.orders.put(("orbit", (1.0, 0.0), {}))
    drain(th)
    assert len(frames) < 20


def test_le_zoom_est_transmis_a_la_camera(thread):
    th, _, _ = thread
    th.orders.put(("zoom", (1.5,), {}))
    drain(th)
    assert ("zoom", 1.5) in th.plotter.camera.calls


def test_l_orbite_est_transmise_puis_reorthogonalisee(thread):
    """Sans réorthogonalisation, la scène finit par se coucher."""
    th, _, _ = thread
    th.orders.put(("orbit", (10.0, 5.0), {}))
    drain(th)
    calls = th.plotter.camera.calls
    assert ("azimuth", 10.0) in calls
    assert ("elevation", 5.0) in calls
    assert ("ortho",) in calls


def test_le_pan_deplace_position_et_point_de_mire_ensemble(thread):
    """Déplacer la caméra sans son point de mire ferait pivoter la vue."""
    th, _, _ = thread
    before_pos = th.plotter.camera.position
    before_foc = th.plotter.camera.focal_point
    th.orders.put(("pan", (0.1, 0.0), {}))
    drain(th)
    cam = th.plotter.camera
    moved_pos = tuple(a - b for a, b in zip(cam.position, before_pos))
    moved_foc = tuple(a - b for a, b in zip(cam.focal_point, before_foc))
    assert moved_pos != (0.0, 0.0, 0.0)
    assert moved_pos == pytest.approx(moved_foc)


def test_le_redimensionnement_change_la_taille_de_rendu(thread):
    th, _, _ = thread
    th.orders.put(("resize", (640, 480), {}))
    drain(th)
    assert list(th.plotter.window_size) == [640, 480]


def test_l_arret_ferme_le_plotter(monkeypatch):
    def fake_build(self):
        self.plotter = FakePlotter()
        return True

    monkeypatch.setattr(v3._RenderThread, "_build_plotter", fake_build)
    th = v3._RenderThread(on_frame=lambda _i: None, on_ready=lambda *_: None, size=(320, 240))
    th.start()
    time.sleep(0.2)
    plotter = th.plotter
    th.stop()
    th.join(timeout=2)
    assert not th.is_alive()
    assert plotter.closed is True


def test_un_contexte_3d_absent_est_signale_sans_exception(monkeypatch):
    """PyVista manquant doit produire un message, pas une pile d'appels."""
    def fail(self):
        self.error = "pas de contexte OpenGL"
        return False

    monkeypatch.setattr(v3._RenderThread, "_build_plotter", fail)
    ready = []
    th = v3._RenderThread(on_frame=lambda _i: None,
                          on_ready=lambda ok, err: ready.append((ok, err)),
                          size=(320, 240))
    th.start()
    th.join(timeout=2)
    assert ready == [(False, "pas de contexte OpenGL")]
    assert not th.is_alive()


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
def slow_plotter(monkeypatch):
    """Simule une construction 3D lente — treize secondes en conditions réelles."""
    def slow_build(self):
        time.sleep(0.6)
        self.plotter = FakePlotter()
        return True

    monkeypatch.setattr(v3._RenderThread, "_build_plotter", slow_build)


def test_le_demarrage_rend_la_main_immediatement(root, slow_plotter):
    """Le cœur de la panne signalée : l'application se figeait ici.

    ``pv.Plotter()`` demandait plusieurs secondes sur le fil de Tk. Le
    démarrage doit désormais coûter quelques millisecondes, quelle que soit la
    lenteur du contexte 3D.
    """
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


def test_les_ordres_emis_avant_la_fin_du_demarrage_sont_rejoues(root, slow_plotter):
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
            if thread is not None and "dmu" in thread.actor_names:
                break
            root.update()
            time.sleep(0.02)
        assert "dmu" in viewer._thread.actor_names
    finally:
        viewer.close()
        frame.destroy()


def test_les_ordres_ne_touchent_pas_a_vtk_sur_le_fil_appelant(root, slow_plotter):
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


def test_la_vue_reste_utilisable_apres_fermeture(root, slow_plotter):
    """Fermer puis continuer d'émettre des ordres ne doit rien casser."""
    import customtkinter as ctk

    frame = ctk.CTkFrame(root)
    viewer = v3.Viewer3D(frame)
    viewer.start()
    viewer.close()
    viewer.show_mesh(object(), "dmu")
    viewer.render()
    assert not viewer.is_available
    frame.destroy()
