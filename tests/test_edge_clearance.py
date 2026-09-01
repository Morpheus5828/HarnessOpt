"""Distance minimale au bord de tôle.

Le défaut : le tracé longe le chant de la structure. Aucune règle ne s'y
opposait, et pour une raison précise — la distance à la structure est
*satisfaite* le long d'un bord, puisque la matière est bien là, juste à côté.
Or un chant use la gaine, et surtout ne peut recevoir aucune fixation :
router le long d'un bord, c'est router là où rien ne tiendra le faisceau.
"""

import numpy as np
import pytest

from core.geometry_metrics import boundary_points, edge_distances
from core.reward_terms import edge_clearance_penalty
from core.routing_rules import RULE_IDS, RoutingRules, evaluate_route


@pytest.fixture(scope="module")
def tole():
    """Une tôle plane : ses quatre côtés sont des bords libres."""
    trimesh = pytest.importorskip("trimesh")

    xs = np.linspace(0.0, 1000.0, 11)
    ys = np.linspace(0.0, 600.0, 7)
    vertices = [[x, y, 0.0] for y in ys for x in xs]
    faces = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            a, b = j * len(xs) + i, j * len(xs) + i + 1
            c, d = (j + 1) * len(xs) + i, (j + 1) * len(xs) + i + 1
            faces += [[a, b, c], [b, d, c]]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# ----------------------------------------------------------------------
# Trouver les bords
# ----------------------------------------------------------------------

def test_une_piece_fermee_n_a_aucun_bord_libre():
    """Une boîte n'a que des arêtes partagées : aucune contrainte de bord."""
    trimesh = pytest.importorskip("trimesh")

    assert len(boundary_points(trimesh.creation.box(extents=(100, 100, 100)))) == 0


def test_une_tole_expose_son_contour(tole):
    assert len(boundary_points(tole)) > 0


def test_les_bords_sont_echantillonnes_et_pas_seulement_les_sommets(tole):
    """Deux sommets de bord peuvent être distants de plusieurs centimètres."""
    sommets = boundary_points(tole, samples_per_edge=2)
    fins = boundary_points(tole, samples_per_edge=6)
    assert len(fins) > len(sommets)


def test_la_distance_au_bord_est_exacte(tole):
    bord = boundary_points(tole)
    d = edge_distances([[500.0, 300.0, 0.0], [500.0, 20.0, 0.0], [10.0, 300.0, 0.0]], bord)
    assert d[0] == pytest.approx(300.0, abs=1.0)
    assert d[1] == pytest.approx(20.0, abs=1.0)
    assert d[2] == pytest.approx(10.0, abs=1.0)


def test_sans_bord_la_distance_est_infinie():
    """Une pièce fermée n'impose rien : rendre zéro condamnerait tout le tracé."""
    d = edge_distances([[0.0, 0.0, 0.0]], np.zeros((0, 3)))
    assert np.isinf(d).all()


def test_un_trace_vide_ne_leve_pas(tole):
    assert len(edge_distances(np.zeros((0, 3)), boundary_points(tole))) == 0


# ----------------------------------------------------------------------
# La sanction
# ----------------------------------------------------------------------

def test_un_trace_a_bonne_distance_ne_paie_rien():
    assert np.all(edge_clearance_penalty([100.0, 50.0, 25.0], min_mm=25.0) == 0.0)


def test_longer_un_bord_coute():
    assert edge_clearance_penalty([5.0], min_mm=25.0)[0] < 0.0


def test_la_sanction_croit_quand_on_se_rapproche():
    proche = edge_clearance_penalty([20.0], min_mm=25.0)[0]
    colle = edge_clearance_penalty([2.0], min_mm=25.0)[0]
    assert colle < proche < 0.0


def test_la_sanction_est_bornee():
    assert edge_clearance_penalty([0.0], min_mm=25.0, weight=70.0)[0] == pytest.approx(-70.0)


def test_une_piece_fermee_ne_sanctionne_rien():
    assert edge_clearance_penalty([np.inf], min_mm=25.0)[0] == 0.0


def test_une_distance_nulle_demandee_neutralise_la_regle():
    assert edge_clearance_penalty([0.0], min_mm=0.0)[0] == 0.0


# ----------------------------------------------------------------------
# La règle, dans le rapport
# ----------------------------------------------------------------------

def test_la_regle_est_au_catalogue():
    assert "edge_clearance" in RULE_IDS


def test_un_trace_le_long_du_bord_echoue_a_la_regle():
    route = np.linspace([0, 0, 0], [1000, 0, 0], 20)
    report = evaluate_route(route, RoutingRules(), edge_distances=np.full(20, 3.0))
    check = next(c for c in report.checks if c.rule_id == "edge_clearance")
    assert not check.passed
    assert report.kpis["n_edge_violations"] == 20


def test_un_trace_ecarte_du_bord_passe():
    route = np.linspace([0, 0, 0], [1000, 0, 0], 20)
    report = evaluate_route(route, RoutingRules(), edge_distances=np.full(20, 80.0))
    check = next(c for c in report.checks if c.rule_id == "edge_clearance")
    assert check.passed
    assert report.kpis["min_edge_distance_mm"] == pytest.approx(80.0)


def test_sans_mesure_la_regle_est_ignoree_et_non_supposee_bonne():
    route = np.linspace([0, 0, 0], [1000, 0, 0], 20)
    report = evaluate_route(route, RoutingRules(), edge_distances=None)
    assert not [c for c in report.checks if c.rule_id == "edge_clearance"]


def test_la_regle_decochee_disparait_du_rapport():
    route = np.linspace([0, 0, 0], [1000, 0, 0], 20)
    rules = RoutingRules().with_rules(set(RULE_IDS) - {"edge_clearance"})
    report = evaluate_route(route, rules, edge_distances=np.full(20, 3.0))
    assert not [c for c in report.checks if c.rule_id == "edge_clearance"]


# ----------------------------------------------------------------------
# Le chemin de départ
# ----------------------------------------------------------------------

def test_le_chemin_de_surface_evite_les_bords(tole):
    """Le plus court chemin le long d'une tôle passe volontiers par son chant.

    Les deux extrémités sont imposées et restent où elles sont : c'est le
    trajet **entre** elles qui doit s'écarter, on ne juge donc que lui.
    """
    pytest.importorskip("scipy")
    from core.surface_path import surface_path

    bord = boundary_points(tole)
    # Un trajet qui longe le grand côté : en ligne droite, il reste à 50 mm.
    depart, arrivee = (100.0, 50.0, 0.0), (900.0, 50.0, 0.0)

    libre = surface_path(tole, depart, arrivee, num_points=40)
    ecarte = surface_path(tole, depart, arrivee, num_points=40, edge_clearance_mm=150.0)
    assert libre.success and ecarte.success

    def au_milieu(result):
        return edge_distances(result.points[5:-5], bord).min()

    assert au_milieu(libre) < 60.0, "sans contrainte, le tracé longe le bord"
    assert au_milieu(ecarte) > au_milieu(libre)


def test_une_contrainte_infaisable_ne_vide_pas_le_graphe(tole):
    """Mieux vaut un tracé perfectible qu'aucun tracé."""
    pytest.importorskip("scipy")
    from core.surface_path import surface_path

    result = surface_path(tole, (100.0, 300.0, 0.0), (900.0, 300.0, 0.0),
                          num_points=20, edge_clearance_mm=100000.0)
    assert result.success


def test_le_terme_est_branche_sur_l_agent():
    source = open("core/agent_worker.py", encoding="utf-8").read()
    assert "rwt.edge_clearance_penalty(" in source
    assert "+ w_edge * R_edge" in source
    assert "edge_distances=" in source


@pytest.fixture(scope="module")
def panneau():
    """Une tôle percée d'une ouverture : la bande utile est étroite."""
    trimesh = pytest.importorskip("trimesh")

    xs, ys = np.linspace(0, 1800, 37), np.linspace(0, 1100, 23)
    vertices = [[x, y, 0.0] for y in ys for x in xs]
    faces = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if 250 < cx < 1150 and 200 < cy < 850:
                continue                        # l'ouverture
            a, b = j * len(xs) + i, j * len(xs) + i + 1
            c, d = (j + 1) * len(xs) + i, (j + 1) * len(xs) + i + 1
            faces += [[a, b, c], [b, d, c]]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_une_ouverture_compte_aussi_comme_bord(panneau):
    """Le chant d'une découpe n'est pas plus tenable que celui du contour."""
    bord = boundary_points(panneau)
    au_bord_de_l_ouverture = edge_distances([[700.0, 210.0, 0.0]], bord)[0]
    assert au_bord_de_l_ouverture < 20.0


def test_un_ecart_trop_ambitieux_est_relache_et_annonce(panneau):
    """Refuser tout tracé serait pire ; l'annoncer sans l'avoir est pire encore."""
    pytest.importorskip("scipy")
    from core.surface_path import surface_path

    depart, arrivee = (150.0, 1000.0, 0.0), (1650.0, 150.0, 0.0)
    result = surface_path(panneau, depart, arrivee, num_points=60,
                          edge_clearance_mm=120.0)
    assert result.success
    assert 0.0 < result.edge_clearance_mm < 120.0, "relâché, et dit"
    bord = boundary_points(panneau)
    assert edge_distances(result.points[3:-3], bord).min() >= result.edge_clearance_mm


def test_un_ecart_impossible_est_annonce_comme_non_obtenu(panneau):
    """Une contrainte levée faute de place ne doit pas se lire comme tenue."""
    pytest.importorskip("scipy")
    from core.surface_path import surface_path

    result = surface_path(panneau, (150.0, 1000.0, 0.0), (1650.0, 150.0, 0.0),
                          num_points=60, edge_clearance_mm=5000.0)
    assert result.success
    assert result.edge_clearance_mm == 0.0


def test_l_ecart_obtenu_figure_dans_le_message(panneau):
    pytest.importorskip("scipy")
    from core.surface_path import surface_path

    result = surface_path(panneau, (150.0, 1000.0, 0.0), (1650.0, 150.0, 0.0),
                          num_points=60, edge_clearance_mm=60.0)
    assert "60" in result.message("FR") and "bord" in result.message("FR")
    assert "60" in result.message("EN") and "edge" in result.message("EN")
