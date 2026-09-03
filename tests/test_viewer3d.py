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
# Le fil de rendu, avec un plotter factice
# ----------------------------------------------------------------------

class FakeActor:
    def __init__(self):
        self.visible = True
        self.prop = type("P", (), {"show_edges": False})()

    def SetVisibility(self, value):
        self.visible = bool(value)

    def GetVisibility(self):
        return self.visible


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
        self.window_size = (320, 240)
        self.render_window = object()
        self.iren = FakeIren()
        self._closed = False
        self.reset_calls = 0
        self.updates = 0
        self.shown = False
        self.widgets = None
        #: Textes posés en surimpression, par nom d'acteur.
        self.texts: dict = {}
        #: Case à cocher installée dans la fenêtre : (rappel, valeur).
        self.button = None

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
        self.texts.pop(name, None)

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

    def add_text(self, text, position=None, name=None, **_kwargs):
        self.texts[name] = (text, position)
        self.actors[name] = FakeActor()
        return self.actors[name]

    def clear_button_widgets(self):
        self.button = None

    def add_checkbox_button_widget(self, callback, value=False, **_kwargs):
        self.button = (callback, bool(value))
        return self.button

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
    ready, states, moves, bests = [], [], [], []

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
        self._install_handles()
        self._install_metrics()
        self._install_legend()
        self._install_best_button()
        self._apply_path_filter()
        self._on_window_state(True)

    monkeypatch.setattr(v3._RenderThread, "_do_open_window", fake_open)
    th = v3._RenderThread(
        on_ready=lambda ok, err: ready.append((ok, err)),
        on_window_state=states.append,
        on_handle_move=lambda i, p: moves.append((i, p)),
        on_best_only=bests.append,
        size=(320, 240),
    )
    th.start()
    for _ in range(200):
        if ready:
            break
        time.sleep(0.01)
    yield th, ready, states, moves, bests
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
    # Les surimpressions — tableau, légende, libellé du bouton — sont de
    # l'habillage : la scène rejouée est celle des géométries.
    scene = {n for n in plotter.actors if not n.startswith("__")}
    assert scene == {"dmu", "traj"}


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
                          on_handle_move=lambda _i, _p: None,
                          on_best_only=lambda _f: None,
                          size=(320, 240))
    th.start()
    th.join(timeout=2)
    assert ready and ready[0][0] is False
    assert "pyvista" in ready[0][1].lower()
    assert not th.is_alive()


# ----------------------------------------------------------------------
# Poignées déplaçables
# ----------------------------------------------------------------------

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
    th, _, _, moves, _ = thread
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


# ----------------------------------------------------------------------
# Surimpressions : tableau des métriques, légende, filtre
# ----------------------------------------------------------------------
#
# Les trois se posent sur la fenêtre sans faire partie de la scène : elles
# survivent donc à une fermeture-réouverture, que la scène rejoue de son côté.

def test_le_tableau_des_metriques_va_en_haut_a_droite(thread):
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    th.orders.put(("set_metrics", ("recompense  1  2  3",), {}))
    drain(th)
    texte, position = plotter.texts[v3.METRICS_ACTOR]
    assert "recompense" in texte
    assert position == "upper_right"


def test_un_tableau_vide_retire_l_acteur(thread):
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    th.orders.put(("set_metrics", ("quelque chose",), {}))
    drain(th)
    assert v3.METRICS_ACTOR in plotter.texts
    th.orders.put(("set_metrics", ("",), {}))
    drain(th)
    assert v3.METRICS_ACTOR not in plotter.texts


def test_la_legende_ecrit_une_ligne_par_agent(thread):
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    th.orders.put(("set_legend", ([("1. TD3", "#2D7FF9"), ("2. SAC", "#E08A00")],), {}))
    drain(th)
    lignes = {n: t for n, t in plotter.texts.items() if n.startswith(v3.LEGEND_PREFIX)}
    assert len(lignes) == 2
    assert any("TD3" in texte for texte, _ in lignes.values())


def test_une_legende_plus_courte_efface_les_lignes_en_trop(thread):
    """Sinon un agent disparu resterait écrit dans la légende."""
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    th.orders.put(("set_legend", ([("A", "#111111"), ("B", "#222222"),
                                   ("C", "#333333")],), {}))
    drain(th)
    th.orders.put(("set_legend", ([("A", "#111111")],), {}))
    drain(th)
    restantes = [n for n in plotter.texts if n.startswith(v3.LEGEND_PREFIX)]
    assert restantes == [f"{v3.LEGEND_PREFIX}0"]


def visibilites(plotter):
    return {n: a.visible for n, a in plotter.actors.items()
            if n.startswith(v3.PATH_PREFIX)}


def trois_trajectoires(th):
    for nom in ("alpha", "beta", "gamma"):
        th.orders.put(("add", (object(), f"{v3.PATH_PREFIX}{nom}"), {}))
    drain(th)


def test_le_filtre_ne_laisse_que_le_meilleur(thread):
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    trois_trajectoires(th)
    th.orders.put(("set_best_agent", ("beta",), {}))
    th.orders.put(("set_best_only", (True,), {}))
    drain(th)
    assert visibilites(plotter) == {
        f"{v3.PATH_PREFIX}alpha": False,
        f"{v3.PATH_PREFIX}beta": True,
        f"{v3.PATH_PREFIX}gamma": False,
    }


def test_relacher_le_filtre_rend_tout_le_monde(thread):
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    trois_trajectoires(th)
    th.orders.put(("set_best_agent", ("beta",), {}))
    th.orders.put(("set_best_only", (True,), {}))
    drain(th)
    th.orders.put(("set_best_only", (False,), {}))
    drain(th)
    assert all(visibilites(plotter).values())


def test_le_filtre_survit_au_redessin_d_une_trajectoire(thread):
    """Un acteur réajouté est visible d'office : le filtre doit se réappliquer."""
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    trois_trajectoires(th)
    th.orders.put(("set_best_agent", ("beta",), {}))
    th.orders.put(("set_best_only", (True,), {}))
    drain(th)
    th.orders.put(("add", (object(), f"{v3.PATH_PREFIX}alpha"), {}))
    drain(th)
    assert visibilites(plotter)[f"{v3.PATH_PREFIX}alpha"] is False


def test_le_filtre_epargne_ce_qui_n_est_pas_une_trajectoire(thread):
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    th.orders.put(("add", (object(), "dmu"), {}))
    trois_trajectoires(th)
    th.orders.put(("set_best_agent", ("beta",), {}))
    th.orders.put(("set_best_only", (True,), {}))
    drain(th)
    assert plotter.actors["dmu"].visible is True


def test_sans_meilleur_designe_le_filtre_ne_masque_rien(thread):
    """Masquer tout le monde parce qu'aucun agent n'est encore classé serait pire."""
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    trois_trajectoires(th)
    th.orders.put(("set_best_only", (True,), {}))
    drain(th)
    assert all(visibilites(plotter).values())


def test_le_bouton_est_pose_dans_la_fenetre(thread):
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    assert plotter.button is not None
    assert "__best_label__" in plotter.texts


def test_le_bouton_bascule_le_filtre_et_previent(thread):
    th, _, _, _, bests = thread
    plotter = ouvre(th)
    trois_trajectoires(th)
    th.orders.put(("set_best_agent", ("beta",), {}))
    drain(th)

    rappel, _valeur = plotter.button
    rappel(True)
    assert th._best_only is True
    assert visibilites(plotter)[f"{v3.PATH_PREFIX}alpha"] is False
    assert bests == [True]

    rappel(False)
    assert all(visibilites(plotter).values())
    assert bests == [True, False]


def test_le_libelle_du_bouton_dit_ce_qu_il_fera(thread):
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    avant = plotter.texts["__best_label__"][0]
    plotter.button[0](True)
    apres = plotter.texts["__best_label__"][0]
    assert avant != apres


def test_les_surimpressions_survivent_a_une_reouverture(thread):
    """Elles ne font pas partie de la scène : rien ne les rejouerait sinon."""
    th, _, _, _, _ = thread
    ouvre(th)
    th.orders.put(("set_metrics", ("cintrage 1 2 3",), {}))
    th.orders.put(("set_legend", ([("1. TD3", "#2D7FF9")],), {}))
    th.orders.put(("set_best_agent", ("TD3",), {}))
    th.orders.put(("set_best_only", (True,), {}))
    drain(th)

    th.orders.put(("close_window", (), {}))
    drain(th)
    plotter = ouvre(th)

    assert v3.METRICS_ACTOR in plotter.texts
    assert f"{v3.LEGEND_PREFIX}0" in plotter.texts
    assert plotter.button is not None
    assert th._best_only is True


def test_les_ordres_de_surimpression_passent_par_la_facade(root, slow_probe):
    """La façade ne doit toucher aucun objet VTK : elle dépose des ordres."""
    import customtkinter as ctk

    frame = ctk.CTkFrame(root)
    viewer = v3.Viewer3D(frame)
    try:
        viewer.start()
        viewer.set_metrics("x")
        viewer.set_legend([("A", "#111111")])
        viewer.set_best_agent("A")
        viewer.set_best_only(True)
        assert viewer.best_only is True
        noms = []
        while not viewer._thread.orders.empty():
            noms.append(viewer._thread.orders.get()[0])
        assert {"set_metrics", "set_legend", "set_best_agent", "set_best_only"} <= set(noms)
    finally:
        viewer.close()
        frame.destroy()


def test_la_legende_reste_au_dessus_du_triedre(thread):
    """Le bas gauche appartient aux axes : la légende s'ancre en haut."""
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    th.orders.put(("set_legend", ([("A", "#111111"), ("B", "#222222")],), {}))
    drain(th)
    hauteur = plotter.window_size[1]
    ordonnees = [position[1] for nom, (_, position) in plotter.texts.items()
                 if nom.startswith(v3.LEGEND_PREFIX)]
    assert all(y > hauteur / 2 for y in ordonnees), "légende dans la moitié basse"
    # La première ligne est la plus haute : la lecture va de haut en bas.
    assert ordonnees[0] > ordonnees[1]


def test_la_legende_suit_le_redimensionnement(thread):
    """Ancrée depuis le bas, elle glisserait au milieu d'une fenêtre agrandie."""
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    th.orders.put(("set_legend", ([("A", "#111111")],), {}))
    drain(th)
    petite = plotter.texts[f"{v3.LEGEND_PREFIX}0"][1][1]

    plotter.window_size = (1600, 1000)
    th.orders.put(("set_legend", ([("A", "#111111")],), {}))
    drain(th)
    grande = plotter.texts[f"{v3.LEGEND_PREFIX}0"][1][1]
    assert grande > petite


def test_sans_taille_de_fenetre_la_legende_se_pose_quand_meme(thread):
    """Un plotter qui ne sait pas dire sa taille ne doit pas la faire disparaître."""
    th, _, _, _, _ = thread
    plotter = ouvre(th)
    del plotter.window_size
    th.orders.put(("set_legend", ([("A", "#111111")],), {}))
    drain(th)
    assert f"{v3.LEGEND_PREFIX}0" in plotter.texts
