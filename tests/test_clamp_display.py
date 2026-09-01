"""Affichage des crabes posés, avec leur géométrie réelle.

Un repère symbolique ne dit rien de l'encombrement d'une fixation, qui est
précisément ce que l'intégrateur doit juger à l'œil. Le crabe est donc dessiné
avec son modèle STL, **dans le repère où son absence de collision a été
vérifiée** : même matrice de rotation, même origine sur la structure. Dessiner
ailleurs que là où le test de collision a eu lieu donnerait une vue rassurante
et fausse.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh non installé")

from controller.app_controller import AppController  # noqa: E402
from core.agent.tool import (  # noqa: E402
    crabe_transform,
    crabe_world_vertices,
    is_crabe_clash_free,
    load_crabe_clamp,
)


@pytest.fixture(scope="module")
def geometry(tmp_path_factory):
    chemin = str(tmp_path_factory.mktemp("crabe") / "crabe.stl")
    trimesh.creation.box(extents=(30.0, 20.0, 12.0)).export(chemin)
    return load_crabe_clamp(chemin)


def crabe(seat=(0.0, 0.0, 0.0), x=(1, 0, 0), y=(0, 1, 0), n=(0, 0, 1)):
    return {
        "arc_mm": 250.0, "tilt_deg": 0.0,
        "position": np.asarray(seat, dtype=float) + np.asarray(n, dtype=float) * 20.0,
        "surface_position": np.asarray(seat, dtype=float),
        "x_axis": np.asarray(x, dtype=float),
        "y_axis": np.asarray(y, dtype=float),
        "normal": np.asarray(n, dtype=float),
    }


# ----------------------------------------------------------------------
# La géométrie normalisée
# ----------------------------------------------------------------------

def test_la_geometrie_complete_est_disponible(geometry):
    """``check_points`` est un échantillon aléatoire : indessinable tel quel."""
    assert "vertices" in geometry and "faces" in geometry
    assert len(geometry["vertices"]) > 0
    assert len(geometry["faces"]) > 0


def test_le_plan_de_contact_est_a_l_origine(geometry):
    assert geometry["vertices"][:, 2].min() == pytest.approx(0.0, abs=1e-5)


def test_le_modele_est_centre_en_x_et_y(geometry):
    for axis in (0, 1):
        milieu = (geometry["vertices"][:, axis].min()
                  + geometry["vertices"][:, axis].max()) / 2.0
        assert milieu == pytest.approx(0.0, abs=1e-5)


def test_les_dimensions_annoncees_correspondent(geometry):
    """Un pavé de 30 x 20 x 12 donne des demi-largeurs de 15 et 10."""
    assert geometry["dx"] == pytest.approx(15.0, abs=1e-3)
    assert geometry["dy"] == pytest.approx(10.0, abs=1e-3)
    assert geometry["height"] == pytest.approx(12.0, abs=1e-3)


# ----------------------------------------------------------------------
# Le placement
# ----------------------------------------------------------------------

def test_le_crabe_repose_sur_la_structure(geometry):
    monde = crabe_world_vertices(crabe(seat=(100.0, 50.0, 10.0)), geometry)
    assert monde[:, 2].min() == pytest.approx(10.0, abs=1e-4)


def test_le_crabe_est_centre_sur_son_point_d_appui(geometry):
    monde = crabe_world_vertices(crabe(seat=(100.0, 50.0, 10.0)), geometry)
    assert monde[:, 0].mean() == pytest.approx(100.0, abs=1e-4)
    assert monde[:, 1].mean() == pytest.approx(50.0, abs=1e-4)


def test_le_crabe_suit_l_orientation_demandee(geometry):
    """Normale horizontale : le corps s'étend en x, pas en z."""
    couche = crabe(seat=(0.0, 0.0, 0.0), x=(0, 1, 0), y=(0, 0, 1), n=(1, 0, 0))
    monde = crabe_world_vertices(couche, geometry)
    assert monde[:, 0].max() == pytest.approx(geometry["height"], abs=1e-4)
    assert monde[:, 2].max() == pytest.approx(geometry["dy"], abs=1e-4)


def test_le_repere_est_celui_du_test_de_collision(geometry):
    """La convention doit être unique, sinon on dessine ailleurs qu'on vérifie."""
    c = crabe(seat=(7.0, -3.0, 2.0))
    rotation, origine = crabe_transform(
        c["surface_position"], c["x_axis"], c["y_axis"], c["normal"]
    )
    attendu = origine + geometry["check_points"] @ rotation.T

    monde = crabe_world_vertices(c, geometry)
    # Les points de contrôle sont un sous-ensemble des sommets : chacun doit
    # se retrouver dans la géométrie dessinée.
    for point in attendu[:20]:
        assert np.linalg.norm(monde - point, axis=1).min() < 1e-4


def test_le_crabe_dessine_est_bien_celui_teste_en_collision(geometry):
    """Vérification croisée sur une vraie structure.

    Un crabe posé à plat sur une plaque ne doit pas entrer dedans, et la
    géométrie dessinée doit rester du bon côté de la surface.
    """
    from trimesh.proximity import ProximityQuery

    plaque = trimesh.creation.box(extents=(400.0, 400.0, 20.0))
    plaque.merge_vertices()
    plaque.fix_normals()
    pq = ProximityQuery(plaque)

    c = crabe(seat=(0.0, 0.0, 10.0))          # sur la face supérieure
    assert is_crabe_clash_free(c["surface_position"], c["x_axis"], c["y_axis"],
                               c["normal"], geometry, pq, plaque, tolerance=0.5)

    monde = crabe_world_vertices(c, geometry)
    closest, dist, faces = pq.on_surface(monde)
    dedans = np.einsum("ij,ij->i", monde - closest, plaque.face_normals[faces]) < 0
    assert not np.any(dedans & (dist > 0.5)), "le crabe dessiné entre dans la structure"


# ----------------------------------------------------------------------
# Cas dégradés
# ----------------------------------------------------------------------

def test_sans_geometrie_rien_n_est_dessine():
    assert crabe_world_vertices(crabe(), None) is None


def test_une_geometrie_sans_sommets_ne_leve_pas():
    assert crabe_world_vertices(crabe(), {"vertices": np.zeros((0, 3))}) is None


def test_un_crabe_sans_appui_retombe_sur_sa_position(geometry):
    """``surface_position`` peut manquer : on se rabat sur le point du câble."""
    c = crabe(seat=(0.0, 0.0, 0.0))
    c.pop("surface_position")
    c["position"] = np.array([5.0, 5.0, 5.0])
    monde = crabe_world_vertices(c, geometry)
    assert monde is not None
    assert monde[:, 2].min() == pytest.approx(5.0, abs=1e-4)


def test_un_crabe_totalement_vide_ne_leve_pas(geometry):
    assert crabe_world_vertices({}, geometry) is None


# ----------------------------------------------------------------------
# Le maillage rendu à la vue 3D
# ----------------------------------------------------------------------

def test_le_corps_est_un_maillage_exploitable(geometry):
    pytest.importorskip("pyvista", reason="pyvista non installé")

    corps = AppController._clamp_body(crabe(seat=(0.0, 0.0, 10.0)), geometry)
    assert corps is not None
    assert corps.n_points == len(geometry["vertices"])
    assert corps.n_cells == len(geometry["faces"])


def test_le_corps_est_place_au_bon_endroit(geometry):
    pytest.importorskip("pyvista", reason="pyvista non installé")

    corps = AppController._clamp_body(crabe(seat=(100.0, 50.0, 10.0)), geometry)
    points = np.asarray(corps.points)
    assert points[:, 2].min() == pytest.approx(10.0, abs=1e-3)
    assert points[:, 0].mean() == pytest.approx(100.0, abs=1e-3)


def test_un_crabe_indessinable_rend_none(geometry):
    pytest.importorskip("pyvista", reason="pyvista non installé")

    assert AppController._clamp_body({}, geometry) is None
    assert AppController._clamp_body(crabe(), None) is None


# ----------------------------------------------------------------------
# Rafraîchissement
# ----------------------------------------------------------------------

class _ViewerFactice:
    """Enregistre les acteurs, sans contexte 3D."""

    def __init__(self):
        self.actors = {}
        self.is_available = True
        self.renders = 0

    def show_mesh(self, mesh, name, **kwargs):
        self.actors[name] = mesh

    def show_sphere(self, center, name, **kwargs):
        self.actors[name] = center

    def show_path(self, points, name, **kwargs):
        self.actors[name] = points

    def remove_prefix(self, prefix):
        for name in [n for n in self.actors if n.startswith(prefix)]:
            del self.actors[name]

    def render(self):
        self.renders += 1


def _controleur(stl_path):
    """Contrôleur nu, sans interface : seul le dessin nous intéresse ici."""
    class _Vue:
        t = type("T", (), {"lang": "FR", "is_english": False})()

        def after(self, _delay, callback):
            callback()

        def set_status(self, *_a, **_k):
            pass

    controller = AppController(_Vue())
    controller.viewer = _ViewerFactice()
    controller._crabe_stl_path = stl_path
    return controller


@pytest.fixture
def stl_path(tmp_path):
    chemin = str(tmp_path / "crabe.stl")
    trimesh.creation.box(extents=(30.0, 20.0, 12.0)).export(chemin)
    return chemin


def test_chaque_crabe_pose_donne_un_maillage(stl_path):
    pytest.importorskip("pyvista", reason="pyvista non installé")

    controller = _controleur(stl_path)
    controller._draw_clamps([crabe(seat=(0.0, 0.0, 0.0)),
                             crabe(seat=(300.0, 0.0, 0.0))])
    assert set(controller.viewer.actors) == {"clamp_0", "clamp_1"}


def test_les_crabes_identiques_ne_sont_pas_redessines(stl_path):
    """Redessiner quatre fois par seconde ferait clignoter la vue."""
    pytest.importorskip("pyvista", reason="pyvista non installé")

    controller = _controleur(stl_path)
    crabes = [crabe(seat=(0.0, 0.0, 0.0))]
    controller._draw_clamps(crabes)
    avant = controller.viewer.renders
    controller._draw_clamps(crabes)
    assert controller.viewer.renders == avant


def test_un_crabe_deplace_est_redessine(stl_path):
    pytest.importorskip("pyvista", reason="pyvista non installé")

    controller = _controleur(stl_path)
    controller._draw_clamps([crabe(seat=(0.0, 0.0, 0.0))])
    avant = controller.viewer.renders
    bouge = crabe(seat=(0.0, 0.0, 0.0))
    bouge["arc_mm"] = 900.0
    controller._draw_clamps([bouge])
    assert controller.viewer.renders > avant


def test_sans_modele_aucun_crabe_n_est_dessine():
    """Rien n'a été posé non plus : on n'invente pas un marqueur."""
    controller = _controleur("")
    controller._draw_clamps([crabe()])
    assert not any(n.startswith("clamp_") for n in controller.viewer.actors)


def test_les_anciens_crabes_sont_effaces(stl_path):
    pytest.importorskip("pyvista", reason="pyvista non installé")

    controller = _controleur(stl_path)
    controller._draw_clamps([crabe(seat=(0.0, 0.0, 0.0)),
                             crabe(seat=(300.0, 0.0, 0.0))])
    controller._draw_clamps([crabe(seat=(0.0, 0.0, 0.0))])
    assert set(controller.viewer.actors) == {"clamp_0"}


# ----------------------------------------------------------------------
# Fixations reconnues : leur géométrie, pas une sphère
# ----------------------------------------------------------------------

def fixation_reconnue(file_path="", transform=(), position=(0.0, 0.0, 0.0)):
    from core.fixation_scan import DetectedFixation

    return DetectedFixation(
        name="XA453420.stl", position=position, score=0.9,
        file_path=file_path, transform=transform,
    )


def matrice(translation=(0.0, 0.0, 0.0)):
    m = np.eye(4)
    m[:3, 3] = translation
    return tuple(tuple(row) for row in m)


def test_le_detecteur_donne_de_quoi_dessiner_la_fixation():
    """Sans le fichier ni le recalage, il ne resterait qu'un point."""
    from core.fixation_scan import summarise

    fixation = summarise([{
        "name": "a.stl", "position": [1.0, 2.0, 3.0], "score": 0.8,
        "file_path": "/modeles/a.stl", "transform": np.eye(4).tolist(),
    }]).fixations[0]

    assert fixation.file_path == "/modeles/a.stl"
    assert len(fixation.transform) == 4
    assert fixation.is_drawable


def test_une_fixation_sans_modele_n_est_pas_dessinable():
    from core.fixation_scan import summarise

    fixation = summarise([{"name": "a.stl", "position": [0, 0, 0]}]).fixations[0]
    assert not fixation.is_drawable


def test_la_fixation_est_dessinee_avec_sa_geometrie(stl_path):
    pytest.importorskip("pyvista", reason="pyvista non installé")

    corps = AppController._fixation_body(fixation_reconnue(stl_path, matrice()))
    assert corps is not None
    assert corps.n_points > 0, "c'est un maillage, pas un point"


def test_la_fixation_est_recalee_la_ou_l_ICP_l_a_trouvee(stl_path):
    pytest.importorskip("pyvista", reason="pyvista non installé")

    corps = AppController._fixation_body(
        fixation_reconnue(stl_path, matrice((100.0, 0.0, 0.0)))
    )
    centre = np.asarray(corps.center)
    assert abs(centre[0] - 100.0) < 1e-3, "le recalage doit être appliqué"


def test_un_modele_absent_ne_leve_pas(tmp_path):
    pytest.importorskip("pyvista", reason="pyvista non installé")

    absent = str(tmp_path / "jamais_ecrit.stl")
    assert AppController._fixation_body(fixation_reconnue(absent, matrice())) is None


def test_sans_recalage_on_ne_dessine_pas_de_geometrie(stl_path):
    assert AppController._fixation_body(fixation_reconnue(stl_path, ())) is None


def test_une_fixation_indessinable_reste_reperee_par_une_sphere(stl_path):
    """Un modèle illisible ne doit pas faire disparaître la fixation de la vue."""
    from core.fixation_scan import ScanResult

    controller = _controleur(stl_path)
    controller._draw_fixations(ScanResult(
        fixations=[fixation_reconnue("", (), position=(5.0, 6.0, 7.0))]
    ))
    assert controller.viewer.actors["fixation_body_0"] == (5.0, 6.0, 7.0)


def test_une_fixation_reconnue_est_dessinee_en_maillage(stl_path):
    pytest.importorskip("pyvista", reason="pyvista non installé")
    from core.fixation_scan import ScanResult

    controller = _controleur(stl_path)
    controller._draw_fixations(ScanResult(
        fixations=[fixation_reconnue(stl_path, matrice())]
    ))
    corps = controller.viewer.actors["fixation_body_0"]
    assert hasattr(corps, "n_points"), "on attend un maillage, pas un centre"
