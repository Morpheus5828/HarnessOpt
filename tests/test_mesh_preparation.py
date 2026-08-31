"""Préparation des maillages de travail, un par agent.

Le point sensible n'est pas la performance mais l'isolement : les requêtes de
proximité de trimesh corrompent le tas si deux fils interrogent le même
maillage. Chaque agent doit donc recevoir sa propre copie, et cette copie doit
être géométriquement identique à la source — c'est sur elle que se mesurent
les distances au DMU.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh non installé")

from controller.app_controller import AppController  # noqa: E402


@pytest.fixture(scope="module")
def source():
    """Une maquette de test faite de pièces disjointes, comme un vrai DMU."""
    parts = []
    for i in range(4):
        box = trimesh.creation.box(extents=(200, 40, 150))
        box.apply_translation((i * 260 - 400, (-1) ** i * 180, 0))
        parts.append(box)
    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


def test_un_maillage_par_agent(source):
    meshes = AppController._prepare_agent_meshes(source, 5)
    assert len(meshes) == 5


def test_un_seul_agent_reste_supporte(source):
    assert len(AppController._prepare_agent_meshes(source, 1)) == 1


def test_les_copies_sont_reellement_distinctes(source):
    """Deux agents ne doivent jamais partager le même objet ni les mêmes tableaux.

    Un partage passerait inaperçu à la lecture et se manifesterait par une
    corruption mémoire sous charge.
    """
    a, b = AppController._prepare_agent_meshes(source, 2)
    assert a is not b
    assert a.vertices is not b.vertices
    assert a.faces is not b.faces
    assert not np.shares_memory(a.vertices, b.vertices)
    assert not np.shares_memory(a.vertices, source.vertices)


def test_la_geometrie_est_identique_a_la_source(source):
    """Supprimer merge_vertices/fix_normals ne doit rien changer au résultat."""
    copy = AppController._prepare_agent_meshes(source, 1)[0]
    assert copy.vertices.shape == source.vertices.shape
    assert copy.faces.shape == source.faces.shape
    assert np.allclose(copy.vertices, source.vertices)
    assert np.array_equal(copy.faces, source.faces)


def test_les_distances_mesurees_sont_identiques(source):
    """Le critère qui compte vraiment : la même distance au DMU."""
    from trimesh.proximity import ProximityQuery

    copy = AppController._prepare_agent_meshes(source, 1)[0]
    points = np.random.RandomState(0).uniform(-500, 500, (80, 3))
    reference = ProximityQuery(source).signed_distance(points)
    measured = ProximityQuery(copy).signed_distance(points)
    assert np.allclose(measured, reference, atol=1e-9)


def test_les_normales_de_faces_sont_preservees(source):
    """fix_normals ayant déjà corrigé l'orientation, elle doit être héritée."""
    copy = AppController._prepare_agent_meshes(source, 1)[0]
    assert np.allclose(copy.face_normals, source.face_normals)


def test_les_caches_couteux_sont_deja_chauds(source):
    """Sinon chaque agent les construirait au milieu de sa boucle."""
    copy = AppController._prepare_agent_meshes(source, 1)[0]
    assert "kdtree" in copy._cache.cache
    assert copy.kdtree is not None


def test_chaque_copie_a_son_propre_arbre(source):
    """L'arbre k-d partagé est précisément ce qui corrompt le tas."""
    a, b = AppController._prepare_agent_meshes(source, 2)
    assert a.kdtree is not b.kdtree


def test_les_copies_supportent_des_requetes_simultanees(source):
    """Reproduction directe du scénario applicatif : un fil par agent."""
    import threading

    from trimesh.proximity import ProximityQuery

    meshes = AppController._prepare_agent_meshes(source, 4)
    points = np.random.RandomState(1).uniform(-500, 500, (40, 3))
    results, errors = {}, []

    def query(index, mesh):
        try:
            results[index] = ProximityQuery(mesh).signed_distance(points)
        except Exception as exc:  # pragma: no cover - ne doit pas arriver
            errors.append(repr(exc))

    threads = [threading.Thread(target=query, args=(i, m)) for i, m in enumerate(meshes)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors
    assert len(results) == 4
    # Des maillages identiques doivent donner des distances identiques.
    for index in range(1, 4):
        assert np.allclose(results[index], results[0])
