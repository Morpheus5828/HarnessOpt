"""Verrou de sortie : ce qui est rendu est la meilleure trajectoire admissible.

``is_deliverable`` existait et ne servait qu'à colorer un badge : rien ne
refusait une trajectoire enfreignant une règle rédhibitoire, et le résultat
rendu était celui du meilleur **score**, conforme ou non. Les agents
continuent d'explorer librement ; c'est la sortie qui est verrouillée, pas la
recherche.
"""

import numpy as np

from controller.app_controller import VALID_ROUTE, AppController
from core.routing_rules import RULE_IDS, RoutingRules, evaluate_route


class _Traducteur:
    lang = "FR"
    is_english = False

    def __call__(self, key, **kwargs):
        return key


class _Vue:
    def __init__(self):
        self.t = _Traducteur()
        self.messages = []

    def after(self, _delay, callback):
        callback()

    def set_status(self, message, tone="neutral"):
        self.messages.append((tone, message))


def controleur():
    return AppController(_Vue())


def trace(n=20, y=0.0):
    return np.linspace([0.0, y, 0.0], [1000.0, y, 0.0], n).astype(np.float32)


def rapport(conforme=True, n=20):
    """Un rapport réel, admissible ou non selon la distance mesurée."""
    distances = np.full(n, 50.0 if conforme else 1.0)
    return evaluate_route(trace(n), RoutingRules(), distances=distances,
                          inside_mask=np.zeros(n, dtype=bool))


def etat(score, conforme=True, iteration=1, points=None):
    return {
        "waypoints": trace() if points is None else points,
        "report": rapport(conforme),
        "score": score,
        "iteration": iteration,
        "crabes": [],
    }


# ----------------------------------------------------------------------
# Ce qui est retenu
# ----------------------------------------------------------------------

def test_le_rapport_de_reference_distingue_bien_les_deux_cas():
    """Sans quoi tous les tests suivants ne prouveraient rien."""
    assert rapport(conforme=True).is_deliverable
    assert not rapport(conforme=False).is_deliverable


def test_une_trajectoire_admissible_est_retenue():
    controller = controleur()
    controller._track_valid({"TD3": etat(score=10.0, conforme=True)})
    assert controller.valid_route() is not None
    assert controller.valid_route()["agent"] == "TD3"


def test_une_trajectoire_inadmissible_n_est_pas_retenue():
    controller = controleur()
    controller._track_valid({"TD3": etat(score=1.0, conforme=False)})
    assert controller.valid_route() is None


def test_un_meilleur_score_inadmissible_ne_supplante_pas_l_admissible():
    """C'est exactement le défaut : le meilleur score gagnait, conforme ou non."""
    controller = controleur()
    controller._track_valid({"A": etat(score=10.0, conforme=True)})
    controller._track_valid({"B": etat(score=1.0, conforme=False)})
    assert controller.valid_route()["agent"] == "A"


def test_le_meilleur_admissible_l_emporte():
    controller = controleur()
    controller._track_valid({"A": etat(score=10.0, conforme=True)})
    controller._track_valid({"B": etat(score=4.0, conforme=True)})
    assert controller.valid_route()["agent"] == "B"


def test_un_moins_bon_admissible_ne_remplace_pas_le_meilleur():
    controller = controleur()
    controller._track_valid({"A": etat(score=4.0, conforme=True)})
    controller._track_valid({"B": etat(score=9.0, conforme=True)})
    assert controller.valid_route()["agent"] == "A"


def test_une_solution_sans_score_est_ignoree():
    controller = controleur()
    state = etat(score=1.0, conforme=True)
    state["score"] = None
    controller._track_valid({"A": state})
    assert controller.valid_route() is None


# ----------------------------------------------------------------------
# La copie
# ----------------------------------------------------------------------

def test_la_trajectoire_retenue_est_recopiee():
    """L'agent continue de la modifier : une référence la laisserait pourrir."""
    controller = controleur()
    points = trace()
    controller._track_valid({"A": etat(score=5.0, conforme=True, points=points)})
    retenue = controller.valid_route()["waypoints"].copy()

    points[5] = [999.0, 999.0, 999.0]          # l'agent poursuit son travail
    assert np.allclose(controller.valid_route()["waypoints"], retenue)


def test_les_crabes_retenus_sont_recopies():
    controller = controleur()
    state = etat(score=5.0, conforme=True)
    state["crabes"] = [{"position": [1.0, 2.0, 3.0]}]
    controller._track_valid({"A": state})
    state["crabes"][0]["position"][0] = 999.0
    assert controller.valid_route()["crabes"][0]["position"][0] == 1.0


# ----------------------------------------------------------------------
# Ce qui est dit
# ----------------------------------------------------------------------

def test_l_absence_de_solution_est_annoncee():
    """Un badge rouge se devine ; une phrase se lit."""
    controller = controleur()
    texte = controller._valid_summary()
    assert "ucune" in texte and texte.strip()


def test_la_solution_retenue_est_nommee():
    controller = controleur()
    controller._track_valid({"TD3": etat(score=5.0, conforme=True, iteration=42)})
    texte = controller._valid_summary()
    assert "TD3" in texte and "42" in texte


def test_le_resume_est_bilingue():
    controller = controleur()
    controller.view.t.is_english = True
    try:
        assert controller._valid_summary().strip()
        controller._track_valid({"TD3": etat(score=5.0, conforme=True)})
        assert "TD3" in controller._valid_summary()
    finally:
        controller.view.t.is_english = False


# ----------------------------------------------------------------------
# L'export
# ----------------------------------------------------------------------

def test_exporter_sans_solution_ne_produit_rien_et_le_dit():
    controller = controleur()
    assert controller.export(VALID_ROUTE, "csv") == ""
    assert controller.view.messages[-1][0] == "warn"


def test_l_export_de_la_solution_retenue_ne_depend_pas_des_agents():
    """Elle doit rester exportable même après l'arrêt du cheminement."""
    controller = controleur()
    controller._track_valid({"TD3": etat(score=5.0, conforme=True)})
    controller.shared_state = None                 # cheminement arrêté
    assert controller.valid_route() is not None


# ----------------------------------------------------------------------
# Cycle de vie
# ----------------------------------------------------------------------

def test_une_relance_oublie_la_solution_precedente():
    """Elle a été calculée sur d'autres règles, voire une autre maquette."""
    controller = controleur()
    controller._track_valid({"A": etat(score=5.0, conforme=True)})
    assert controller.valid_route() is not None
    controller.best_valid = None                   # ce que fait le lancement
    assert controller.valid_route() is None


def test_le_verrou_est_branche_sur_le_rafraichissement():
    """Le verrou tourne à chaque rafraîchissement, pas seulement à l'export.

    Le résumé, lui, n'apparaît plus sur la page de cheminement : elle ne porte
    que l'avancement et les courbes. Il reste dit au moment où il engage —
    quand on demande la trajectoire et qu'il n'y en a aucune d'admissible.
    """
    source = open("controller/app_controller.py", encoding="utf-8").read()
    assert "self._track_valid(snapshot)" in source
    assert "self.view.set_status(self._valid_summary()" in source


def test_le_verrou_porte_sur_les_regles_redhibitoires():
    """Ni le score, ni la conformité totale : les règles bloquantes."""
    source = open("controller/app_controller.py", encoding="utf-8").read()
    assert "report.is_deliverable" in source
    # Les trois règles bloquantes du catalogue.
    from core.routing_rules import RULE_CATALOG, Severity

    bloquantes = {i.rule_id for i in RULE_CATALOG if i.severity == Severity.BLOCKING}
    assert bloquantes == {"clash", "clearance_min", "bend_radius"}
    assert bloquantes <= set(RULE_IDS)
