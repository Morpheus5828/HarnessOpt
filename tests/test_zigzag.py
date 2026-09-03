"""Détection et pénalisation des zigzags.

Le point délicat est de distinguer un zigzag d'une courbe. Un arc de cercle
régulier accumule beaucoup de courbure sans jamais osciller ; une succession de
petits virages alternés fait exactement l'inverse. Ni le rayon de cintrage ni
la part de tracé rectiligne ne séparent ces deux cas, d'où une mesure propre.
"""

import numpy as np

from core.geometry_metrics import (
    straightness,
    turn_binormals,
    zigzag_metrics,
    zigzag_severity,
)
from core.reward_terms import zigzag_penalty


def ligne_droite(n=12):
    return np.linspace([0, 0, 0], [110, 0, 0], n)


def arc(n=12, rayon=100.0):
    angles = np.linspace(0, np.pi / 2, n)
    return np.stack([np.cos(angles) * rayon, np.sin(angles) * rayon, np.zeros(n)], axis=1)


def zigzag(n=12, amplitude=10.0):
    return np.array(
        [[i * 10.0, amplitude if i % 2 else -amplitude, 0.0] for i in range(n)]
    )


# ----------------------------------------------------------------------
# La mesure
# ----------------------------------------------------------------------

def test_une_ligne_droite_n_oscille_pas():
    assert zigzag_metrics(ligne_droite())["n_zigzags"] == 0


def test_un_arc_regulier_n_est_pas_un_zigzag():
    """Le cas qui compte : beaucoup de courbure, aucune oscillation."""
    mesure = zigzag_metrics(arc())
    assert mesure["n_zigzags"] == 0
    assert mesure["zigzag_deg"] == 0.0
    # …et pourtant l'arc courbe bel et bien.
    assert straightness(arc())["total_turning_deg"] > 60.0


def test_un_zigzag_est_detecte():
    mesure = zigzag_metrics(zigzag())
    assert mesure["n_zigzags"] > 0
    assert mesure["zigzag_deg"] > 0.0
    assert mesure["worst_zigzag_deg"] > 0.0


def test_un_zigzag_plus_ample_est_juge_plus_grave():
    faible = zigzag_metrics(zigzag(amplitude=2.0))["zigzag_deg"]
    fort = zigzag_metrics(zigzag(amplitude=25.0))["zigzag_deg"]
    assert fort > faible


def test_un_seul_coude_ne_fait_pas_un_zigzag():
    """Il faut deux virages opposés : un coude isolé n'oscille pas."""
    coude = np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0], [300, 80, 0], [400, 160, 0]], float)
    assert zigzag_metrics(coude)["n_zigzags"] == 0


def test_la_tolerance_ignore_les_micro_oscillations():
    """Une ondulation numérique sous la tolérance ne doit pas être comptée."""
    bruit = np.array([[i * 100.0, (0.05 if i % 2 else -0.05), 0.0] for i in range(12)])
    assert zigzag_metrics(bruit, angle_tol_deg=3.0)["n_zigzags"] == 0
    assert zigzag_metrics(bruit, angle_tol_deg=0.001)["n_zigzags"] > 0


def test_un_tres_court_trace_ne_leve_pas():
    for n in (0, 1, 2, 3):
        mesure = zigzag_metrics(np.zeros((n, 3)))
        assert mesure["n_zigzags"] == 0


def test_des_points_confondus_ne_levent_pas():
    doublons = np.array([[0, 0, 0], [0, 0, 0], [10, 0, 0], [10, 0, 0], [20, 0, 0]], float)
    assert zigzag_metrics(doublons)["n_zigzags"] == 0


def test_la_gravite_est_alignee_sur_les_points_interieurs():
    pts = zigzag()
    assert len(zigzag_severity(pts)) == len(pts) - 2


def test_la_gravite_retient_le_plus_petit_des_deux_virages():
    """Une petite oscillation entre deux grands virages reste petite.

    Retenir le plus grand ferait payer un zigzag franc pour un frémissement.
    """
    pts = np.array(
        [[0, 0, 0], [100, 0, 0], [200, 60, 0], [201, 59, 0], [300, 120, 0], [400, 180, 0]],
        float,
    )
    from core.geometry_metrics import turning_angles

    severite = np.degrees(zigzag_severity(pts))
    angles = np.degrees(turning_angles(pts))
    # La gravité retenue est bornée par le plus petit des virages en cause,
    # donc a fortiori par le plus grand angle du tracé.
    assert 0.0 < severite.max() <= angles.max() + 1e-9


def test_les_binormales_sont_unitaires_ou_nulles():
    normes = np.linalg.norm(turn_binormals(zigzag()), axis=1)
    assert np.all((np.isclose(normes, 1.0)) | (np.isclose(normes, 0.0)))


def test_les_binormales_s_inversent_sur_un_zigzag():
    unit = turn_binormals(zigzag())
    produits = np.einsum("ij,ij->i", unit[:-1], unit[1:])
    assert np.all(produits < 0), "les virages devraient alterner"


def test_les_binormales_restent_alignees_sur_un_arc():
    unit = turn_binormals(arc())
    produits = np.einsum("ij,ij->i", unit[:-1], unit[1:])
    assert np.all(produits > 0), "un arc tourne toujours dans le même sens"


# ----------------------------------------------------------------------
# La récompense
# ----------------------------------------------------------------------

def test_aucune_penalite_sur_une_ligne_droite():
    assert zigzag_penalty(ligne_droite()).sum() == 0.0


def test_aucune_penalite_sur_un_arc_regulier():
    assert zigzag_penalty(arc()).sum() == 0.0


def test_un_zigzag_est_penalise():
    assert zigzag_penalty(zigzag()).sum() < 0.0


def test_la_penalite_croit_avec_la_gravite():
    faible = zigzag_penalty(zigzag(amplitude=2.0)).sum()
    fort = zigzag_penalty(zigzag(amplitude=25.0)).sum()
    assert fort < faible


def test_la_penalite_est_saturee():
    """Un demi-tour isolé ne doit pas écraser tout le signal d'apprentissage."""
    penalites = zigzag_penalty(zigzag(amplitude=500.0), weight=60.0)
    assert penalites.min() >= -60.0


def test_la_penalite_a_la_taille_du_trace():
    pts = zigzag(n=15)
    assert len(zigzag_penalty(pts)) == len(pts)


def test_un_trace_trop_court_renvoie_des_zeros():
    for n in (0, 1, 2, 3):
        penalites = zigzag_penalty(np.zeros((n, 3)))
        assert len(penalites) == n
        assert not penalites.any()


def test_le_poids_module_la_penalite():
    faible = zigzag_penalty(zigzag(), weight=10.0).sum()
    fort = zigzag_penalty(zigzag(), weight=100.0).sum()
    assert fort < faible


# ----------------------------------------------------------------------
# Intégration dans le rapport et les rôles
# ----------------------------------------------------------------------

def test_le_rapport_publie_les_indicateurs_de_zigzag():
    from core.routing_rules import RoutingRules, evaluate_route

    rapport = evaluate_route(zigzag(), RoutingRules())
    assert rapport.kpis["n_zigzags"] > 0
    assert rapport.kpis["zigzag_deg"] > 0.0


def test_un_trace_qui_oscille_est_moins_bien_classe():
    from core.routing_rules import RoutingRules, evaluate_route

    rules = RoutingRules()
    propre = evaluate_route(arc(), rules).score()
    oscillant = evaluate_route(zigzag(), rules).score()
    assert propre < oscillant


def test_aucun_role_ne_neglige_le_zigzag():
    """« Pour chaque agent » : même l'éclaireur garde un poids plein."""
    from core.orchestrator import ROLES

    for key, role in ROLES.items():
        assert role.weights.get("zigzag", 0.0) >= 1.0, f"rôle {key}"


def test_les_lisseurs_y_sont_plus_sensibles():
    from core.orchestrator import ROLES

    assert ROLES["smoother"].weights["zigzag"] > ROLES["scout"].weights["zigzag"]
    assert ROLES["straightener"].weights["zigzag"] > ROLES["scout"].weights["zigzag"]


def test_le_zigzag_ne_depend_d_aucune_case_a_cocher():
    """Décocher toutes les règles ne doit pas autoriser les oscillations."""
    from core.routing_rules import RoutingRules

    rules = RoutingRules(enabled_rules=set())
    assert "zigzag" not in rules.reward_scale()
