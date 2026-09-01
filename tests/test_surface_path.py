"""Chemin suivant la surface, y compris entre pièces disjointes.

Le défaut corrigé : ``pyvista.PolyData.geodesic`` cherche un chemin d'arêtes.
Sur une maquette faite de pièces qui ne se touchent pas — le cas de tout DMU
fusionné — il n'en existe aucun, et l'ancien code rendait une corde tendue en
prétendant fournir une géodésique.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh non installé")
pytest.importorskip("scipy", reason="scipy non installé")

from core.surface_path import (  # noqa: E402
    NO_MESH,
    NO_PATH,
    SurfacePathResult,
    offset_from_surface,
    surface_path,
)


@pytest.fixture(scope="module")
def sphere():
    """Maillage connexe : une vraie géodésique existe."""
    return trimesh.creation.icosphere(subdivisions=3, radius=100)


@pytest.fixture(scope="module")
def deux_pieces():
    """Deux plaques séparées : aucun chemin d'arêtes de l'une à l'autre."""
    a = trimesh.creation.box(extents=(200, 200, 20))
    a.apply_translation((-150, 0, 0))
    b = trimesh.creation.box(extents=(200, 200, 20))
    b.apply_translation((150, 0, 0))
    mesh = trimesh.util.concatenate([a, b])
    mesh.merge_vertices()
    return mesh


def ecart_a_la_corde(points):
    droite = np.linspace(points[0], points[-1], len(points))
    return float(np.abs(points - droite).max())


# ----------------------------------------------------------------------
# Maillage connexe : la géodésique classique
# ----------------------------------------------------------------------

def test_une_sphere_donne_un_arc_et_non_une_corde(sphere):
    result = surface_path(sphere, [-100, 0, 0], [100, 0, 0], num_points=60)
    assert result.success
    # Un demi grand cercle vaut pi/2 fois la corde ; une corde vaudrait 1.
    assert result.total_length_mm / 200.0 > 1.3
    assert ecart_a_la_corde(result.points) > 50.0


def test_un_maillage_connexe_ne_saute_jamais(sphere):
    result = surface_path(sphere, [-100, 0, 0], [100, 0, 0])
    assert result.n_bridges == 0
    assert result.surface_ratio == pytest.approx(1.0)


def test_le_chemin_reste_sur_la_surface(sphere):
    from trimesh.proximity import ProximityQuery

    result = surface_path(sphere, [-100, 0, 0], [100, 0, 0], num_points=80)
    distances = np.abs(ProximityQuery(sphere).signed_distance(result.points.astype(float)))
    # Les points sont des sommets du maillage : ils sont sur la surface. Le
    # rééchantillonnage coupe légèrement les cordes entre sommets voisins.
    assert distances.mean() < 5.0


# ----------------------------------------------------------------------
# Pièces disjointes : le cas qui échouait
# ----------------------------------------------------------------------

def test_deux_pieces_disjointes_sont_reliees(deux_pieces):
    result = surface_path(deux_pieces, [-240, 0, 10], [240, 0, 10], num_points=60)
    assert result.success, result.message("FR")
    assert result.n_bridges >= 1


def test_le_trajet_reste_majoritairement_sur_la_surface(deux_pieces):
    result = surface_path(deux_pieces, [-240, 0, 10], [240, 0, 10])
    assert result.surface_ratio > 0.5


def test_le_saut_est_compte_et_mesure(deux_pieces):
    result = surface_path(deux_pieces, [-240, 0, 10], [240, 0, 10])
    assert result.bridge_length_mm > 0.0
    assert result.total_length_mm == pytest.approx(
        result.surface_length_mm + result.bridge_length_mm
    )


def test_une_penalite_forte_privilegie_la_surface(deux_pieces):
    faible = surface_path(deux_pieces, [-240, 0, 10], [240, 0, 10], bridge_penalty=1.0)
    forte = surface_path(deux_pieces, [-240, 0, 10], [240, 0, 10], bridge_penalty=20.0)
    assert forte.surface_ratio >= faible.surface_ratio


def test_un_saut_trop_court_interdit_rend_le_trajet_impossible(deux_pieces):
    """L'écart entre les deux plaques vaut 100 mm."""
    result = surface_path(deux_pieces, [-240, 0, 10], [240, 0, 10], max_bridge_mm=5.0)
    assert not result.success
    assert result.reason == NO_PATH


# ----------------------------------------------------------------------
# Extrémités et rééchantillonnage
# ----------------------------------------------------------------------

def test_les_extremites_demandees_sont_respectees(deux_pieces):
    a, b = [-240.0, 0.0, 10.0], [240.0, 0.0, 10.0]
    result = surface_path(deux_pieces, a, b, num_points=40)
    assert np.allclose(result.points[0], a, atol=1e-3)
    assert np.allclose(result.points[-1], b, atol=1e-3)


def test_le_nombre_de_points_demande_est_rendu(deux_pieces):
    for n in (10, 40, 120):
        result = surface_path(deux_pieces, [-240, 0, 10], [240, 0, 10], num_points=n)
        assert len(result.points) == n


def test_sans_reechantillonnage_le_chemin_garde_ses_sommets(deux_pieces):
    result = surface_path(deux_pieces, [-240, 0, 10], [240, 0, 10])
    assert len(result.points) >= 2


# ----------------------------------------------------------------------
# Corridor
# ----------------------------------------------------------------------

def test_le_corridor_ne_change_pas_le_resultat_sur_un_petit_maillage(deux_pieces):
    large = surface_path(deux_pieces, [-240, 0, 10], [240, 0, 10], corridor_factor=0)
    borne = surface_path(deux_pieces, [-240, 0, 10], [240, 0, 10], corridor_factor=3.5)
    assert borne.success and large.success
    assert borne.total_length_mm == pytest.approx(large.total_length_mm, rel=0.25)


def test_le_corridor_reduit_le_graphe(sphere):
    """Sur un trajet court, il ne garde que la zone concernée."""
    complet = surface_path(sphere, [0, 0, 100], [30, 0, 95], corridor_factor=0)
    borne = surface_path(sphere, [0, 0, 100], [30, 0, 95], corridor_factor=2.0)
    assert borne.graph_nodes < complet.graph_nodes


def test_un_corridor_trop_serre_ne_vide_pas_le_graphe(sphere):
    """Il doit se rouvrir plutôt que rendre un graphe inexploitable."""
    result = surface_path(sphere, [-100, 0, 0], [100, 0, 0], corridor_factor=0.01)
    assert result.success


# ----------------------------------------------------------------------
# Cas dégradés
# ----------------------------------------------------------------------

def test_un_maillage_absent_est_signale():
    result = surface_path(None, [0, 0, 0], [1, 1, 1])
    assert not result.success
    assert result.reason == NO_MESH
    assert result.message("FR").strip()
    assert result.message("EN").strip()


def test_les_messages_sont_bilingues(deux_pieces):
    result = surface_path(deux_pieces, [-240, 0, 10], [240, 0, 10])
    assert result.message("FR") != result.message("EN")
    assert "saut" in result.message("FR")
    assert "hop" in result.message("EN")


def test_un_trajet_sans_saut_le_dit(sphere):
    message = surface_path(sphere, [-100, 0, 0], [100, 0, 0]).message("FR")
    assert "saut" not in message
    assert "entièrement" in message


def test_un_resultat_vide_a_des_valeurs_sures():
    result = SurfacePathResult()
    assert result.total_length_mm == 0.0
    assert result.surface_ratio == 1.0
    assert len(result.points) == 0


# ----------------------------------------------------------------------
# Décollement de la surface
# ----------------------------------------------------------------------

def clearance(mesh, points):
    """Distance signée à la structure : négative = à l'intérieur."""
    from trimesh.proximity import ProximityQuery

    closest, distance, faces = ProximityQuery(mesh).on_surface(np.asarray(points))
    dehors = np.einsum("ij,ij->i", np.asarray(points) - closest,
                       mesh.face_normals[faces]) >= 0
    return np.where(dehors, distance, -distance)


def test_sur_la_surface_le_trace_est_en_interference(sphere):
    """Le défaut signalé : un tracé qui rase la structure clashe dès le départ."""
    trace = surface_path(sphere, (0, 0, 100), (0, 0, -100), num_points=40)
    assert trace.success
    assert clearance(sphere, trace.points).max() < 1.0, \
        "sans décalage, le tracé colle au maillage"


def test_le_trace_decale_respecte_la_distance_demandee(sphere):
    trace = surface_path(sphere, (0, 0, 100), (0, 0, -100),
                         num_points=40, offset_mm=25.0)
    assert trace.success
    assert clearance(sphere, trace.points).min() > 24.0


def test_la_distance_atteinte_est_rendue(sphere):
    trace = surface_path(sphere, (0, 0, 100), (0, 0, -100),
                         num_points=40, offset_mm=25.0)
    assert trace.offset_mm == pytest.approx(25.0)
    # Le tracé est rendu en float32, comme le manipulent les agents : la marge
    # annoncée et la marge remesurée diffèrent donc de quelques nanomètres.
    assert trace.min_clearance_mm == pytest.approx(
        float(clearance(sphere, trace.points).min()), abs=1e-3)


@pytest.mark.parametrize("cible", [5.0, 20.0, 60.0])
def test_plusieurs_distances_sont_atteintes(sphere, cible):
    decale, marge = offset_from_surface(sphere, sphere.vertices[::40], cible)
    assert marge >= cible - 1e-6


def test_un_decalage_nul_ne_bouge_rien(sphere):
    points = np.array(sphere.vertices[::40], dtype=float)
    decale, _ = offset_from_surface(sphere, points, 0.0)
    assert np.allclose(decale, points)


def test_le_decalage_pousse_vers_l_exterieur(sphere):
    """Décaler vers l'intérieur enfoncerait le câble dans la structure."""
    points = np.array(sphere.vertices[::40], dtype=float)
    decale, _ = offset_from_surface(sphere, points, 20.0)
    assert np.linalg.norm(decale, axis=1).min() > np.linalg.norm(points, axis=1).max()


def test_un_point_deja_assez_loin_n_est_pas_ramene(sphere):
    """On ne colle pas au maillage un point qui respirait déjà."""
    loin = np.array([[0.0, 0.0, 300.0]])
    decale, _ = offset_from_surface(sphere, loin, 20.0)
    assert np.allclose(decale, loin)


def test_aucun_point_ne_donne_aucun_point(sphere):
    decale, marge = offset_from_surface(sphere, np.zeros((0, 3)), 20.0)
    assert len(decale) == 0 and marge == 0.0


def test_le_decalage_marche_sur_des_pieces_disjointes(deux_pieces):
    """Le cas réel : une maquette fusionnée n'est jamais connexe."""
    trace = surface_path(deux_pieces, (-150, 0, 10), (150, 0, 10),
                         num_points=60, offset_mm=15.0)
    assert trace.success
    assert trace.min_clearance_mm > 14.0
