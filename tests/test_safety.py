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
