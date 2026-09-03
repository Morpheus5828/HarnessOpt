"""Projection des contraintes rédhibitoires, et verrou de sortie.

Une grosse pénalité de récompense rend une trajectoire coûteuse, pas
impossible : rien n'empêche un agent de converger vers un optimum qui viole
une contrainte critique si le gain ailleurs la compense. Les contraintes dures
sont donc *projetées*, pas seulement punies — et projetées plutôt que
rejetées, pour que l'agent reparte d'un point valide au lieu de perdre
l'itération.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh non installé")

from core import geometry_metrics as gm  # noqa: E402
from core.safety import (  # noqa: E402
    ProjectionReport,
    project,
    project_bend_radius,
    project_clearance,
)


@pytest.fixture(scope="module")
def mur():
    piece = trimesh.creation.box(extents=(1200.0, 40.0, 600.0))
    piece.apply_translation((600.0, 0.0, 0.0))
    return piece


def marges(mesh, points):
    from trimesh.proximity import ProximityQuery

    pts = np.asarray(points, dtype=np.float64)
    closest, distance, faces = ProximityQuery(mesh).on_surface(pts)
    dehors = np.einsum("ij,ij->i", pts - closest, mesh.face_normals[faces]) >= 0
    return np.where(dehors, distance, -distance)


def route_traversante():
    """Une trajectoire qui pénètre le mur et fait un coude vif au milieu."""
    return gm.resample_by_arclength(
        np.array([[0.0, -200.0, 0.0], [300.0, -60.0, 0.0], [600.0, 5.0, 0.0],
                  [900.0, -60.0, 0.0], [1200.0, -200.0, 0.0]]), 25
    )


# ----------------------------------------------------------------------
# Distance et pénétration
# ----------------------------------------------------------------------

def test_un_trace_qui_penetre_ressort(mur):
    route = route_traversante()
    assert marges(mur, route).min() < 0.0, "le cas de départ doit bien pénétrer"
    project_clearance(route, mur, 25.0)
    assert marges(mur, route).min() >= 0.0


def test_la_distance_exigee_est_atteinte(mur):
    route = route_traversante()
    project_clearance(route, mur, 25.0)
    assert marges(mur, route).min() == pytest.approx(25.0, abs=1e-6)


def test_un_trace_deja_conforme_n_est_pas_touche(mur):
    route = gm.resample_by_arclength(
        np.array([[0.0, -300.0, 0.0], [600.0, -300.0, 0.0], [1200.0, -300.0, 0.0]]), 20
    )
    avant = route.copy()
    moved, _ = project_clearance(route, mur, 25.0)
    assert moved == 0
    assert np.allclose(route, avant)


def test_la_distance_peut_varier_point_par_point(mur):
    """Les familles de couleur exigent des distances différentes."""
    route = route_traversante()
    exigences = np.full(len(route), 25.0)
    exigences[10:15] = 80.0
    project_clearance(route, mur, exigences)
    obtenues = marges(mur, route)
    assert obtenues[10:15].min() >= 79.0
    assert obtenues.min() >= 24.0


# ----------------------------------------------------------------------
# Rayon de cintrage
# ----------------------------------------------------------------------

def test_un_coude_trop_serre_est_assoupli():
    route = route_traversante()
    avant = gm.min_bend_radius(route)
    _, apres = project_bend_radius(route, min_radius_mm=avant * 2.0)
    assert apres > avant


def test_les_portions_droites_ne_sont_pas_lissees():
    """Lisser tout le tracé effacerait les longues droites, qu'on cherche."""
    route = gm.resample_by_arclength(
        np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]), 20
    )
    avant = route.copy()
    moved, _ = project_bend_radius(route, min_radius_mm=200.0)
    assert moved == 0
    assert np.allclose(route, avant)


def test_un_rayon_non_demande_ne_change_rien():
    route = route_traversante()
    avant = route.copy()
    for limite in (0.0, None, float("inf")):
        project_bend_radius(route, min_radius_mm=limite or 0.0)
    assert np.allclose(route, avant)


# ----------------------------------------------------------------------
# Ce que la projection ne doit jamais toucher
# ----------------------------------------------------------------------

def test_les_extremites_ne_bougent_jamais(mur):
    """Elles appartiennent aux équipements : ce sont des données, pas des variables."""
    route = route_traversante()
    depart, arrivee = route[0].copy(), route[-1].copy()
    project(route, mesh=mur, required_mm=25.0, min_radius_mm=150.0)
    assert np.allclose(route[0], depart)
    assert np.allclose(route[-1], arrivee)


def test_un_point_gele_ne_bouge_pas(mur):
    """Encoche de peigne ou point posé à la main : une décision, pas une variable."""
    route = route_traversante()
    fige = route[12].copy()
    project(route, mesh=mur, required_mm=25.0, min_radius_mm=150.0, frozen={12})
    assert np.allclose(route[12], fige)


def test_plusieurs_points_geles_sont_respectes(mur):
    route = route_traversante()
    geles = {5, 12, 19}
    avant = {i: route[i].copy() for i in geles}
    project(route, mesh=mur, required_mm=25.0, min_radius_mm=150.0, frozen=geles)
    for index, position in avant.items():
        assert np.allclose(route[index], position)


# ----------------------------------------------------------------------
# Le rapport
# ----------------------------------------------------------------------

def test_la_projection_rend_le_trace_admissible(mur):
    route = route_traversante()
    rapport = project(route, mesh=mur, required_mm=25.0, min_radius_mm=120.0)
    assert rapport.feasible
    assert rapport.n_clearance_left == 0 and rapport.n_bend_left == 0
    assert rapport.n_moved > 0


def test_le_rapport_dit_ce_qui_resiste(mur):
    """Une contrainte inatteignable est rapportée, pas annoncée comme tenue."""
    route = route_traversante()
    # Un rayon plus grand que la géométrie ne le permet, à points gelés.
    rapport = project(route, mesh=mur, required_mm=25.0, min_radius_mm=50000.0,
                      frozen=set(range(1, len(route) - 1)))
    assert not rapport.feasible
    assert rapport.n_bend_left > 0


def test_la_projection_ne_leve_jamais(mur):
    """Un tracé impossible est rapporté, pas fatal."""
    for route in (np.zeros((0, 3)), np.zeros((1, 3)), np.zeros((2, 3))):
        rapport = project(route, mesh=mur, required_mm=25.0, min_radius_mm=100.0)
        assert isinstance(rapport, ProjectionReport)
        assert rapport.n_moved == 0


def test_sans_maillage_seule_la_courbure_est_projetee(mur):
    route = route_traversante()
    rapport = project(route, mesh=None, required_mm=None, min_radius_mm=200.0)
    assert rapport.n_clearance_moved == 0
    assert rapport.n_bend_moved > 0


def test_la_projection_est_idempotente(mur):
    """Rappelée à chaque itération, elle ne doit pas dériver."""
    route = route_traversante()
    project(route, mesh=mur, required_mm=25.0, min_radius_mm=120.0)
    stable = route.copy()
    rapport = project(route, mesh=mur, required_mm=25.0, min_radius_mm=120.0)
    assert rapport.n_moved == 0
    assert np.allclose(route, stable)


# ----------------------------------------------------------------------
# Branchement
# ----------------------------------------------------------------------

def test_la_projection_est_branchee_sur_l_agent():
    """Une projection non appelée ne garantit rien."""
    source = open("core/agent_worker.py", encoding="utf-8").read()
    assert "safety.project(" in source
    assert "frozen=frozen_indices" in source, "les points imposés doivent être gelés"
    assert "query=local_pq" in source, "les requêtes trimesh ne sont pas réentrantes"


def test_l_action_est_exprimee_dans_le_repere_local():
    """Observer en local et agir en global oblige à réapprendre la rotation."""
    source = open("core/agent_worker.py", encoding="utf-8").read()
    assert "actions_batch[:, 0:1] * frame_u" in source
    assert "actions_batch[:, 1:2] * frame_v" in source
    assert "actions_batch[:, 2:3] * frame_t" in source


def test_le_repere_local_est_orthonorme():
    """Sinon le changement de repère modifierait l'amplitude du déplacement."""
    from core.agent.tool import build_local_frame

    rng = np.random.default_rng(0)
    tangentes = rng.normal(size=(200, 3))
    tangentes /= np.linalg.norm(tangentes, axis=1, keepdims=True)
    u, v = build_local_frame(tangentes)

    for a, b in ((u, tangentes), (v, tangentes), (u, v)):
        assert np.abs(np.einsum("ij,ij->i", a, b)).max() < 1e-6
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_le_changement_de_repere_conserve_l_amplitude():
    from core.agent.tool import build_local_frame

    rng = np.random.default_rng(1)
    tangentes = rng.normal(size=(200, 3))
    tangentes /= np.linalg.norm(tangentes, axis=1, keepdims=True)
    u, v = build_local_frame(tangentes)
    actions = rng.uniform(-1.0, 1.0, size=(200, 3))

    monde = actions * 7.0
    local = (actions[:, 0:1] * u + actions[:, 1:2] * v
             + actions[:, 2:3] * tangentes) * 7.0
    assert np.allclose(np.linalg.norm(monde, axis=1), np.linalg.norm(local, axis=1))


def test_une_tangente_verticale_ne_degenere_pas():
    """Un câble vertical ne doit pas produire un repère nul."""
    from core.agent.tool import build_local_frame

    u, v = build_local_frame(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


# ----------------------------------------------------------------------
# Interdire le retour en arrière
# ----------------------------------------------------------------------

# project_anchors est couvert par tests/test_manual_editing.py : c'est là que
# vit la fonctionnalité qu'il sert.
from core.safety import (  # noqa: E402
    arc_positions,
    project_progress,
    prune_redundant_points,
)


def reference_droite():
    return np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])


def route_qui_recule():
    return np.array([[0.0, 0.0, 0.0], [200.0, 10.0, 0.0], [400.0, 10.0, 0.0],
                     [250.0, 20.0, 0.0],            # le repli
                     [600.0, 10.0, 0.0], [800.0, 5.0, 0.0], [1000.0, 0.0, 0.0]])


def test_l_avancement_se_mesure_le_long_de_la_reference():
    s, _ = arc_positions([[250.0, 40.0, 0.0]], reference_droite())
    assert s[0] == pytest.approx(250.0)


def test_un_recul_est_detecte():
    s, _ = arc_positions(route_qui_recule(), reference_droite())
    assert np.count_nonzero(np.diff(s) < 0) == 1


def test_un_recul_est_supprime():
    """Le défaut signalé : marches avec recul, chaînes non simples."""
    route = route_qui_recule()
    bouges, restants = project_progress(route, reference_droite())
    assert bouges == 1 and restants == 0
    s, _ = arc_positions(route, reference_droite())
    assert np.all(np.diff(s) > 0)


def test_un_trajet_en_L_n_est_pas_cassé():
    """Il recule franchement le long de A→B sans revenir sur ses pas.

    C'est pour cela que l'avancement se mesure sur la référence et non sur la
    droite A→B : l'interdire supprimerait des trajets parfaitement valides.
    """
    forme = gm.resample_by_arclength(
        np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [1000.0, 800.0, 0.0]]), 30
    )
    avant = forme.copy()
    bouges, _ = project_progress(forme, avant.copy())
    assert bouges == 0
    assert np.allclose(forme, avant)


def test_les_extremites_ne_sont_jamais_repoussees():
    route = route_qui_recule()
    depart, arrivee = route[0].copy(), route[-1].copy()
    project_progress(route, reference_droite())
    assert np.allclose(route[0], depart) and np.allclose(route[-1], arrivee)


def test_un_point_gele_sert_de_plancher_sans_bouger():
    route = route_qui_recule()
    fige = route[3].copy()
    project_progress(route, reference_droite(), frozen={3})
    assert np.allclose(route[3], fige)


def test_un_trace_trop_court_ne_leve_pas():
    for n in (0, 1, 2):
        assert project_progress(np.zeros((n, 3)), reference_droite()) == (0, 0)


def test_sans_reference_rien_n_est_touche():
    route = route_qui_recule()
    avant = route.copy()
    assert project_progress(route, None) == (0, 0)
    assert np.allclose(route, avant)


# ----------------------------------------------------------------------
# Retirer des points, pas seulement en ajouter
# ----------------------------------------------------------------------

def dense(n=40):
    """Un tracé dense qui longe le mur sans le toucher."""
    return gm.resample_by_arclength(
        np.array([[0.0, -300.0, 0.0], [500.0, -280.0, 0.0], [1000.0, -300.0, 0.0]]), n
    )


def test_un_trace_trop_dense_est_elague(mur):
    avant = dense()
    apres, retires = prune_redundant_points(avant, mesh=mur, required_mm=25.0,
                                            min_radius_mm=100.0)
    assert len(retires) > 0
    assert len(apres) == len(avant) - len(retires)


def test_l_elagage_raccourcit_et_tend_la_courbe(mur):
    avant = dense()
    apres, _ = prune_redundant_points(avant, mesh=mur, required_mm=25.0,
                                      min_radius_mm=100.0)
    assert gm.path_length(apres) <= gm.path_length(avant) + 1e-6


def test_l_elagage_ne_casse_jamais_le_cintrage(mur):
    """Un point qui sert à tenir un coude n'est jamais candidat."""
    # Un quart de cercle : le rayon réalisable y est celui de l'arc, et un
    # coude franc de polyligne ne saurait pas l'atteindre — (c/2)/tan(45°) ne
    # vaut que la moitié du pas d'échantillonnage.
    angles = np.linspace(0.0, np.pi / 2.0, 40)
    coude = np.stack([600.0 * np.cos(angles),
                      -400.0 - 600.0 * np.sin(angles),
                      np.zeros_like(angles)], axis=1)
    limite = 150.0
    assert gm.min_bend_radius(coude) >= limite, "le cas de départ doit être conforme"
    apres, _ = prune_redundant_points(coude, mesh=mur, required_mm=25.0,
                                      min_radius_mm=limite)
    assert gm.min_bend_radius(apres) >= limite


def test_l_elagage_ne_casse_jamais_la_distance(mur):
    """Le segment de remplacement est vérifié sur toute sa longueur."""
    frolant = gm.resample_by_arclength(
        np.array([[0.0, -60.0, 0.0], [500.0, -200.0, 0.0], [1000.0, -60.0, 0.0]]), 40
    )
    apres, _ = prune_redundant_points(frolant, mesh=mur, required_mm=25.0)

    from trimesh.proximity import ProximityQuery

    echantillons = gm.resample_by_arclength(apres, 200)
    closest, distance, faces = ProximityQuery(mur).on_surface(echantillons)
    dehors = np.einsum("ij,ij->i", echantillons - closest,
                       mur.face_normals[faces]) >= 0
    assert np.where(dehors, distance, -distance).min() >= 25.0 - 1e-6


def test_les_extremites_ne_sont_jamais_retirees(mur):
    avant = dense()
    apres, _ = prune_redundant_points(avant, mesh=mur, required_mm=25.0)
    assert np.allclose(apres[0], avant[0]) and np.allclose(apres[-1], avant[-1])


def test_un_point_impose_n_est_jamais_retire(mur):
    avant = dense()
    fige = avant[10].copy()
    apres, retires = prune_redundant_points(avant, mesh=mur, required_mm=25.0,
                                            frozen={10})
    assert 10 not in retires
    assert any(np.allclose(p, fige) for p in apres)


def test_l_elagage_est_borne(mur):
    _, retires = prune_redundant_points(dense(), mesh=mur, required_mm=25.0,
                                        max_removals=2)
    assert len(retires) <= 2


def test_un_trace_deja_court_n_est_pas_touche(mur):
    court = np.linspace([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], 4)
    apres, retires = prune_redundant_points(court, mesh=mur, required_mm=25.0)
    assert retires == [] and len(apres) == 4


def test_l_elagage_est_branche_sur_l_agent():
    source = open("core/agent_worker.py", encoding="utf-8").read()
    assert "safety.prune_redundant_points(" in source
    assert "adaptive_prune_max_per_event" in source


# ----------------------------------------------------------------------
# Retirer les replis
# ----------------------------------------------------------------------

from core.safety import remove_backtracking  # noqa: E402


def trace_repliee():
    """Un aller-retour franc au milieu d'un trajet par ailleurs droit."""
    return np.array([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0], [400.0, 0.0, 0.0],
                     [250.0, 0.0, 0.0],           # le repli
                     [600.0, 0.0, 0.0], [800.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])


def test_un_repli_est_retire():
    apres, retires = remove_backtracking(trace_repliee())
    assert len(retires) >= 1
    assert len(apres) == 7 - len(retires)


def test_le_trace_deplie_n_a_plus_d_inversion():
    apres, _ = remove_backtracking(trace_repliee())
    seg = np.diff(apres, axis=0)
    unit = seg / np.linalg.norm(seg, axis=1, keepdims=True)
    assert np.all(np.einsum("ij,ij->i", unit[:-1], unit[1:]) >= 0.0)


def test_un_trace_simple_n_est_pas_touche():
    droit = np.linspace([0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], 12)
    apres, retires = remove_backtracking(droit)
    assert retires == [] and np.allclose(apres, droit)


def test_un_coude_franc_n_est_pas_un_repli():
    """Un virage à 90° avance encore : seul le demi-tour est visé."""
    coude = np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0], [500.0, 500.0, 0.0]])
    _, retires = remove_backtracking(coude)
    assert retires == []


def test_le_repli_epargne_les_extremites():
    apres, _ = remove_backtracking(trace_repliee())
    assert np.allclose(apres[0], [0.0, 0.0, 0.0])
    assert np.allclose(apres[-1], [1000.0, 0.0, 0.0])


def test_un_point_a_conserver_est_conserve():
    apres, retires = remove_backtracking(trace_repliee(), keep={3})
    assert 3 not in retires


def test_le_retrait_est_borne():
    _, retires = remove_backtracking(trace_repliee(), max_removals=1)
    assert len(retires) <= 1


def test_les_indices_rendus_sont_ceux_d_origine():
    """Sinon l'appelant ne saurait pas quoi décaler chez lui."""
    _, retires = remove_backtracking(trace_repliee())
    assert all(0 < i < 6 for i in retires)


def test_un_trace_trop_court_traverse_le_filtre_de_replis():
    for n in (0, 1, 2):
        apres, retires = remove_backtracking(np.zeros((n, 3)))
        assert retires == [] and len(apres) == n


# --- progression stricte vers la cible ---------------------------------
#
# Un repli est un demi-tour franc ; un recul est autre chose. Un tracé peut
# n'avoir aucun demi-tour et pourtant revenir vers son point de départ en
# décrivant une boucle large. Les deux filtres sont donc distincts, et le
# second ne se déduit pas du premier.

from core.safety import forward_only  # noqa: E402


def trace_reculante():
    """Un arc de 290°, échantillonné tous les 10°.

    Chaque raccord n'ouvre que de 10° : aucun demi-tour, ``cos`` vaut 0,985
    partout. Le tracé n'en revient pas moins vers son point de départ sur la
    seconde moitié de l'arc. C'est exactement le cas qu'un filtre de replis ne
    voit pas.
    """
    angles = np.radians(np.arange(-90.0, 200.1, 10.0))
    return np.stack([
        100.0 + 100.0 * np.cos(angles),
        100.0 + 100.0 * np.sin(angles),
        np.zeros_like(angles),
    ], axis=1)


def avance(points, source=None, cible=None):
    pts = np.asarray(points, dtype=np.float64)
    a = pts[0] if source is None else np.asarray(source, dtype=np.float64)
    b = pts[-1] if cible is None else np.asarray(cible, dtype=np.float64)
    axe = (b - a) / np.linalg.norm(b - a)
    return (pts - a) @ axe


def test_une_boucle_large_recule_sans_faire_demi_tour():
    """Le fixture doit bien piéger remove_backtracking, sinon le test ne prouve rien."""
    trace = trace_reculante()
    _, replis = remove_backtracking(trace)
    assert replis == []
    assert (np.diff(avance(trace)) < 0).any()


def test_le_recul_est_retire():
    trace = trace_reculante()
    apres, retires = forward_only(trace)
    assert retires, "aucun recul retiré sur un arc qui revient sur lui-même"
    assert (np.diff(avance(apres, trace[0], trace[-1])) >= -1e-9).all()


def test_un_trace_qui_avance_n_est_pas_touche():
    droit = np.array([[float(x), 0.0, 0.0] for x in (0, 30, 60, 90, 120)])
    apres, retires = forward_only(droit)
    assert retires == []
    assert np.allclose(apres, droit)


def test_aller_de_cote_n_est_pas_reculer():
    """Contourner un obstacle fait stagner la progression, pas décroître."""
    contournement = np.array([
        [0.0, 0.0, 0.0],
        [50.0, 0.0, 0.0],
        [50.0, 80.0, 0.0],    # plein travers : progression inchangée
        [50.0, 160.0, 0.0],
        [200.0, 160.0, 0.0],
    ])
    _, retires = forward_only(contournement, [0.0, 0.0, 0.0], [200.0, 160.0, 0.0])
    assert retires == []


def test_depasser_la_cible_puis_revenir_est_un_recul():
    """La cible ne peut pas être retirée : c'est le sommet qui la dépasse qui saute."""
    trace = np.array([
        [0.0, 0.0, 0.0],
        [60.0, 0.0, 0.0],
        [140.0, 0.0, 0.0],   # au-delà de la cible
        [100.0, 0.0, 0.0],
    ])
    apres, retires = forward_only(trace)
    assert retires == [2]
    assert len(apres) == 3


def test_les_extremites_survivent_au_filtre():
    trace = trace_reculante()
    apres, _ = forward_only(trace)
    assert np.allclose(apres[0], trace[0])
    assert np.allclose(apres[-1], trace[-1])


def test_un_point_a_conserver_echappe_au_filtre():
    _, retires = forward_only(trace_reculante(), keep={20})
    assert 20 not in retires


def test_la_tolerance_laisse_passer_un_petit_recul():
    serre, _ = forward_only(trace_reculante())
    large, _ = forward_only(trace_reculante(), tolerance_mm=15.0)
    assert len(large) > len(serre)


def test_un_trace_trop_court_traverse_le_filtre():
    for n in (0, 1, 2):
        apres, retires = forward_only(np.zeros((n, 3)))
        assert retires == [] and len(apres) == n


def test_source_et_cible_confondues_ne_levent_pas():
    trace = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    apres, retires = forward_only(trace)
    assert retires == [] and len(apres) == 3
