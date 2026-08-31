"""Courbes de progression : contenu et échelle des axes.

Le défaut signalé était visuel mais bien réel : l'abscisse portait l'indice du
point dans l'historique, plafonné à 400, et non le numéro d'itération. Passé
400 relevés, la courbe annonçait donc éternellement « 400 » alors que les
agents en étaient à plusieurs milliers.
"""

import pytest

pytest.importorskip("tkinter", reason="tkinter absent de cet interpréteur")
pytest.importorskip("customtkinter", reason="customtkinter non installé")

from ui.charts import SERIES, ProgressCharts  # noqa: E402


class FakeReport:
    def __init__(self, **kpis):
        self.kpis = {
            "n_clashes": 0,
            "mean_distance_mm": 30.0,
            "min_bend_radius_mm": 250.0,
        }
        self.kpis.update(kpis)


def feed(charts, name="scout", count=10, start=0, step=1, reward=1.0):
    """Injecte des relevés sans passer par matplotlib."""
    for index in range(count):
        iteration = start + index * step
        charts._append(
            name,
            {"iteration": iteration, "reward": reward + iteration},
            FakeReport(),
        )


@pytest.fixture
def charts():
    return ProgressCharts(container=None)


# ----------------------------------------------------------------------
# La récompense fait partie des courbes
# ----------------------------------------------------------------------

def test_la_recompense_est_une_des_series():
    assert "reward" in [key for key, _title, _unit in SERIES]


def test_les_quatre_series_attendues_sont_presentes():
    keys = {key for key, _title, _unit in SERIES}
    assert keys == {"reward", "clashes", "distance", "bend"}


def test_la_recompense_est_enregistree(charts):
    feed(charts, count=5)
    assert charts._history["scout"]["reward"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_une_recompense_absente_vaut_zero(charts):
    charts._append("scout", {"iteration": 0}, FakeReport())
    assert charts._history["scout"]["reward"] == [0.0]


def test_la_recompense_negative_est_conservee(charts):
    """Une récompense négative est le cas normal en début d'apprentissage."""
    charts._append("scout", {"iteration": 0, "reward": -120.5}, FakeReport())
    assert charts._history["scout"]["reward"] == [-120.5]


# ----------------------------------------------------------------------
# L'abscisse porte le numéro d'itération
# ----------------------------------------------------------------------

def test_l_abscisse_est_le_numero_d_iteration(charts):
    charts._append("scout", {"iteration": 1500, "reward": 3.0}, FakeReport())
    assert charts._history["scout"]["iteration"] == [1500]


def test_l_abscisse_suit_les_iterations_reelles_meme_espacees(charts):
    """Le rafraîchissement n'est pas synchrone : les itérations sautent."""
    for iteration in (0, 7, 31, 128, 999):
        charts._append("scout", {"iteration": iteration, "reward": 1.0}, FakeReport())
    assert charts._history["scout"]["iteration"] == [0, 7, 31, 128, 999]


def test_l_historique_est_decime_et_non_tronque(charts):
    """Le début de session doit survivre : c'est là qu'on voit si ça démarre."""
    feed(charts, count=charts.MAX_POINTS + 60)
    steps = charts._history["scout"]["iteration"]
    assert len(steps) <= charts.MAX_POINTS
    assert steps[0] == 0, "le premier relevé a été perdu"


def test_la_decimation_conserve_la_derniere_iteration(charts):
    feed(charts, count=charts.MAX_POINTS + 60)
    steps = charts._history["scout"]["iteration"]
    assert steps[-1] >= charts.MAX_POINTS


def test_toutes_les_series_restent_alignees_apres_decimation(charts):
    feed(charts, count=charts.MAX_POINTS * 3)
    history = charts._history["scout"]
    lengths = {key: len(values) for key, values in history.items()}
    assert len(set(lengths.values())) == 1, f"séries désalignées : {lengths}"


def test_le_pas_d_echantillonnage_augmente_apres_decimation(charts):
    assert charts._stride.get("scout") is None
    feed(charts, count=charts.MAX_POINTS + 10)
    assert charts._stride["scout"] > 1


def test_apres_decimation_les_iterations_trop_proches_sont_ignorees(charts):
    """Sinon l'historique redéborderait immédiatement."""
    feed(charts, count=charts.MAX_POINTS + 10)
    before = len(charts._history["scout"]["iteration"])
    last = charts._history["scout"]["iteration"][-1]
    charts._append("scout", {"iteration": last, "reward": 0.0}, FakeReport())
    assert len(charts._history["scout"]["iteration"]) == before


def test_l_abscisse_depasse_largement_400_sur_une_longue_session(charts):
    """Le défaut signalé : l'axe restait bloqué à 400."""
    feed(charts, count=2000)
    assert charts._history["scout"]["iteration"][-1] > 400


# ----------------------------------------------------------------------
# Plusieurs agents
# ----------------------------------------------------------------------

def test_chaque_agent_a_son_propre_historique(charts):
    feed(charts, name="scout", count=5)
    feed(charts, name="smoother", count=3)
    assert len(charts._history["scout"]["iteration"]) == 5
    assert len(charts._history["smoother"]["iteration"]) == 3


def test_les_agents_sont_decimes_independamment(charts):
    """Un agent rapide ne doit pas décimer l'historique d'un agent lent."""
    feed(charts, name="rapide", count=charts.MAX_POINTS + 50)
    feed(charts, name="lent", count=20)
    assert charts._stride["rapide"] > 1
    assert charts._stride["lent"] == 1


def test_la_remise_a_zero_efface_tout(charts):
    feed(charts, count=charts.MAX_POINTS + 50)
    charts.reset()
    assert charts._history == {}
    assert charts._stride == {}


# ----------------------------------------------------------------------
# Valeurs particulières
# ----------------------------------------------------------------------

def test_un_rayon_infini_est_plafonne(charts):
    """Un tracé parfaitement droit a un rayon infini, intraçable tel quel."""
    from ui.charts import BEND_CAP_MM

    charts._append("scout", {"iteration": 0, "reward": 1.0},
                   FakeReport(min_bend_radius_mm=float("inf")))
    assert charts._history["scout"]["bend"] == [BEND_CAP_MM]


def test_un_rapport_absent_n_est_pas_enregistre(charts):
    charts.available = True
    charts.canvas = type("C", (), {"draw_idle": lambda self: None})()
    charts._redraw = lambda _colors: None
    charts.update({"scout": {"iteration": 3, "reward": 1.0, "report": None}}, {})
    assert "scout" not in charts._history
