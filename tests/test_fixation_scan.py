"""Analyse des fixations existantes.

Le détecteur lui-même repose sur Open3D, absent de beaucoup de postes. Ce qui
se teste ici — et qui doit tenir sans Open3D — est la frontière : la mise en
forme de sa sortie, et le fait qu'une détection impossible n'interrompe jamais
le cheminement mais dise pourquoi.
"""

import pytest

from core import fixation_scan as fs


def raw_comb(name="XA453420_peigne.stl", n_passages=3):
    points = []
    for i in range(n_passages):
        points.append([float(i * 20), 0.0, 0.0])
        points.append([float(i * 20), 50.0, 0.0])
    return {
        "name": name,
        "position": [10.0, 20.0, 30.0],
        "score": 0.93,
        "routing_points": points,
    }


def raw_clip(name="clip.stl"):
    return {"name": name, "position": [1.0, 2.0, 3.0], "score": 0.81}


# ----------------------------------------------------------------------
# Mise en forme
# ----------------------------------------------------------------------

def test_une_liste_vide_donne_un_resultat_vide():
    result = fs.summarise([])
    assert result.ran
    assert result.n_fixations == 0
    assert result.n_passages == 0


def test_none_est_tolere():
    assert fs.summarise(None).n_fixations == 0


def test_les_fixations_sont_comptees():
    result = fs.summarise([raw_comb(), raw_clip()])
    assert result.n_fixations == 2


def test_les_passages_sont_apparies_en_entree_sortie():
    """La sortie du détecteur est une liste aplatie : in, out, in, out…"""
    result = fs.summarise([raw_comb(n_passages=3)])
    assert result.n_passages == 3
    premier = result.passages[0]
    assert premier.p_in == (0.0, 0.0, 0.0)
    assert premier.p_out == (0.0, 50.0, 0.0)


def test_les_passages_sont_numerotes_a_partir_de_zero():
    result = fs.summarise([raw_comb(n_passages=3)])
    assert [p.index for p in result.passages] == [0, 1, 2]


def test_un_point_orphelin_est_ignore():
    """Un nombre impair de points signifie un passage tronqué.

    Mieux vaut l'ignorer que d'inventer le point manquant : un passage faux
    contraindrait le câble à traverser là où il n'y a pas d'encoche.
    """
    clamp = raw_comb(n_passages=2)
    clamp["routing_points"].append([99.0, 99.0, 99.0])
    result = fs.summarise([clamp])
    assert result.n_passages == 2


def test_un_clip_sans_passage_reste_compte_comme_fixation():
    result = fs.summarise([raw_clip()])
    assert result.n_fixations == 1
    assert result.n_passages == 0
    assert not result.fixations[0].is_comb


def test_un_peigne_est_reconnu_comme_tel():
    assert fs.summarise([raw_comb()]).fixations[0].is_comb


def test_le_centre_du_passage_est_le_milieu():
    passage = fs.summarise([raw_comb(n_passages=1)]).passages[0]
    assert passage.center == (0.0, 25.0, 0.0)


def test_l_ouverture_du_passage_est_sa_longueur():
    passage = fs.summarise([raw_comb(n_passages=1)]).passages[0]
    assert passage.width_mm == pytest.approx(50.0)


def test_le_score_et_la_position_sont_conserves():
    fixation = fs.summarise([raw_comb()]).fixations[0]
    assert fixation.score == pytest.approx(0.93)
    assert fixation.position == (10.0, 20.0, 30.0)


# ----------------------------------------------------------------------
# Points transmis à l'agent
# ----------------------------------------------------------------------

def test_les_points_pour_l_agent_sont_aplatis():
    result = fs.summarise([raw_comb(n_passages=2)])
    points = result.routing_points()
    assert len(points) == 4
    assert points[0] == [0.0, 0.0, 0.0]
    assert points[1] == [0.0, 50.0, 0.0]


def test_les_points_de_plusieurs_peignes_sont_concatenes():
    result = fs.summarise([raw_comb(n_passages=2), raw_comb(name="autre.stl", n_passages=1)])
    assert len(result.routing_points()) == 6


# ----------------------------------------------------------------------
# Messages
# ----------------------------------------------------------------------

def test_le_message_annonce_fixations_et_passages():
    message = fs.summarise([raw_comb(n_passages=3), raw_clip()]).message("FR")
    assert "2" in message and "3" in message


def test_le_message_est_bilingue():
    result = fs.summarise([raw_comb()])
    assert result.message("FR") != result.message("EN")
    assert result.message("EN").strip()


def test_aucune_fixation_se_dit_explicitement():
    assert "ucune" in fs.summarise([]).message("FR")


def test_chaque_raison_d_abandon_a_un_message():
    for reason in (fs.NO_FOLDER, fs.NO_OPEN3D, fs.NO_SCENE, fs.FAILED):
        result = fs.ScanResult(skipped_reason=reason)
        assert not result.ran
        assert result.message("FR").strip()
        assert result.message("EN").strip()


def test_un_passage_se_formate_avec_ses_deux_points():
    passage = fs.summarise([raw_comb(n_passages=1)]).passages[0]
    texte = passage.format("FR")
    assert "p_in" in texte and "p_out" in texte
    assert "n° 1" in texte, "les passages sont numérotés à partir de 1 à l'écran"


# ----------------------------------------------------------------------
# Dégradation
# ----------------------------------------------------------------------

def test_un_dossier_absent_n_est_pas_une_erreur():
    result = fs.scan("scene.stl", "")
    assert not result.ran
    assert result.skipped_reason == fs.NO_FOLDER


def test_un_dossier_inexistant_est_traite_comme_absent(tmp_path):
    result = fs.scan("scene.stl", str(tmp_path / "nexiste_pas"))
    assert result.skipped_reason == fs.NO_FOLDER


def test_une_maquette_absente_est_signalee(tmp_path):
    result = fs.scan(str(tmp_path / "absente.stl"), str(tmp_path))
    assert result.skipped_reason == fs.NO_SCENE


def test_open3d_absent_est_signale_sans_exception(tmp_path, monkeypatch):
    """Le cheminement doit démarrer même sans le détecteur."""
    scene = tmp_path / "scene.stl"
    scene.write_text("solid s\nendsolid s\n")

    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "open3d":
            raise ImportError("open3d absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    result = fs.scan(str(scene), str(tmp_path))
    assert result.skipped_reason == fs.NO_OPEN3D
    assert result.n_fixations == 0


def test_un_scan_qui_echoue_ne_leve_pas(tmp_path, monkeypatch):
    scene = tmp_path / "scene.stl"
    scene.write_text("solid s\nendsolid s\n")
    pytest.importorskip("open3d", reason="Open3D non installé, cas déjà couvert")

    import core.path_managment.fixation_detection as detection

    monkeypatch.setattr(detection, "run_detection_for_agent",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boum")))
    result = fs.scan(str(scene), str(tmp_path))
    assert result.skipped_reason == fs.FAILED


# ----------------------------------------------------------------------
# Numérotation
# ----------------------------------------------------------------------

def test_l_index_est_propre_a_chaque_peigne():
    """Deux peignes à une encoche portent tous deux l'index 0."""
    result = fs.summarise([raw_comb(name="a.stl", n_passages=1),
                           raw_comb(name="b.stl", n_passages=1)])
    assert [p.index for p in result.passages] == [0, 0]


def test_une_numerotation_continue_peut_etre_imposee():
    """Sans quoi une liste mélangeant les peignes afficherait trois « n° 1 »."""
    result = fs.summarise([raw_comb(name="a.stl", n_passages=1),
                           raw_comb(name="b.stl", n_passages=1),
                           raw_comb(name="c.stl", n_passages=1)])
    lignes = [p.format("FR", number=n) for n, p in enumerate(result.passages, 1)]
    assert "n° 1" in lignes[0]
    assert "n° 2" in lignes[1]
    assert "n° 3" in lignes[2]


def test_la_numerotation_continue_marche_aussi_en_anglais():
    passage = fs.summarise([raw_comb(n_passages=1)]).passages[0]
    assert "no. 7" in passage.format("EN", number=7)


# ----------------------------------------------------------------------
# Traçabilité en console
# ----------------------------------------------------------------------

def journal(*args, **kwargs):
    """Un log qui garde ses lignes, pour vérifier ce que le scan raconte."""
    lignes = []
    return lignes, lignes.append


def test_un_dossier_absent_le_dit_en_console():
    lignes, log = journal()
    fs.scan("scene.stl", "", log=log)
    assert any("Dossier de fixations absent" in ligne for ligne in lignes)


def test_le_scan_annonce_la_maquette_et_les_modeles(tmp_path, monkeypatch):
    """« Aucune fixation trouvée » doit être distinguable de « pas de scan »."""
    scene = tmp_path / "scene.stl"
    scene.write_text("solid s\nendsolid s\n")
    clamps = tmp_path / "clamps"
    clamps.mkdir()
    for name in ("a.stl", "b.stl"):
        (clamps / name).write_text("solid s\nendsolid s\n")

    lignes, log = journal()
    monkeypatch.setattr(fs, "_model_files", lambda folder: sorted(
        str(p) for p in (clamps).glob("*.stl")))
    fs.scan(str(scene), str(clamps), log=log)

    texte = "\n".join(lignes)
    assert str(scene) in texte, "la maquette réellement scannée doit être nommée"
    assert "2 modèle(s)" in texte
    assert "a.stl" in texte and "b.stl" in texte


def test_un_dossier_sans_modele_est_une_raison_a_part(tmp_path):
    """Un dossier vide n'est pas « zéro fixation » : il n'y a rien à chercher."""
    scene = tmp_path / "scene.stl"
    scene.write_text("solid s\nendsolid s\n")
    clamps = tmp_path / "vide"
    clamps.mkdir()

    lignes, log = journal()
    result = fs.scan(str(scene), str(clamps), log=log)
    assert result.skipped_reason == fs.NO_MODELS
    assert result.message("FR").strip() and result.message("EN").strip()
    assert any("Aucun modèle" in ligne for ligne in lignes)


def test_le_recapitulatif_detaille_chaque_fixation():
    lignes, log = journal()
    result = fs.summarise([raw_comb(n_passages=2)], scanned_files=4)
    fs._report(result, log)

    texte = "\n".join(lignes)
    assert "XA453420_peigne.stl" in texte
    assert "p_in" in texte and "p_out" in texte
    assert texte.count("p_in") == 2, "un passage par ligne"


def test_le_recapitulatif_signale_une_fixation_non_dessinable():
    lignes, log = journal()
    fs._report(fs.summarise([raw_clip()]), log)
    assert any("non réaffichable" in ligne for ligne in lignes)


def test_un_echec_de_scan_est_journalise(tmp_path, monkeypatch):
    scene = tmp_path / "scene.stl"
    scene.write_text("solid s\nendsolid s\n")
    pytest.importorskip("open3d", reason="Open3D non installé, cas déjà couvert")

    import core.path_managment.fixation_detection as detection

    monkeypatch.setattr(detection, "run_detection_for_agent",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boum")))
    lignes, log = journal()
    fs.scan(str(scene), str(tmp_path), log=log)
    assert any("boum" in ligne for ligne in lignes)


# ----------------------------------------------------------------------
# Maquette lisible par le détecteur
# ----------------------------------------------------------------------

def test_un_stl_est_donne_tel_quel(tmp_path):
    scene = tmp_path / "scene.stl"
    scene.write_text("solid s\nendsolid s\n")
    assert fs.scene_for_detection(str(scene), log=lambda *a: None) == str(scene)


def test_une_maquette_en_memoire_prime_sur_le_chemin(tmp_path, monkeypatch):
    """C'est la géométrie parcourue par les agents qui doit être scannée."""
    trimesh = pytest.importorskip("trimesh")
    cible = tmp_path / "temp_for_detection.stl"
    monkeypatch.setattr(fs, "_scene_export_path", lambda: str(cible))

    scene = tmp_path / "scene.stl"
    scene.write_text("solid s\nendsolid s\n")
    mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))

    obtenu = fs.scene_for_detection(str(scene), mesh=mesh, log=lambda *a: None)
    assert obtenu == str(cible)
    assert trimesh.load(obtenu).faces.shape[0] == 12


def test_un_vtk_est_converti_car_le_detecteur_ne_le_lit_pas(tmp_path, monkeypatch):
    """Open3D ne lit pas le ``.vtk`` : il rendrait un maillage vide en silence."""
    pv = pytest.importorskip("pyvista")
    trimesh = pytest.importorskip("trimesh")

    source = tmp_path / "clipped_obstacles.vtk"
    pv.wrap(trimesh.creation.box(extents=(10.0, 10.0, 10.0))).save(str(source))

    cible = tmp_path / "temp_for_detection.stl"
    monkeypatch.setattr(fs, "_scene_export_path", lambda: str(cible))

    obtenu = fs.scene_for_detection(str(source), log=lambda *a: None)
    assert obtenu == str(cible)
    assert obtenu.lower().endswith(fs.DETECTOR_SUFFIXES)
    assert trimesh.load(obtenu).faces.shape[0] == 12


def test_une_maquette_illisible_ne_leve_pas(tmp_path):
    source = tmp_path / "maquette.inconnu"
    source.write_text("ceci n'est pas un maillage")
    assert fs.scene_for_detection(str(source), log=lambda *a: None) is None


def test_le_vtk_de_la_fusion_n_est_pas_lisible_par_le_detecteur():
    """Le garde-fou du bug d'origine : la fusion est écrite dans ce format-là."""
    from core import paths

    assert not str(paths.FUSED_MESH_PATH).lower().endswith(fs.DETECTOR_SUFFIXES)


# ----------------------------------------------------------------------
# Les encoches, cherchées dans la matière
# ----------------------------------------------------------------------

def peigne_stl(tmp_path, n_dents=10, nom="XA453420_peigne.stl"):
    """Un peigne : une embase longue selon x, des dents selon z, des vides entre."""
    trimesh = pytest.importorskip("trimesh")

    parts = [trimesh.creation.box(extents=(200.0, 20.0, 6.0))]
    for i in range(n_dents):
        dent = trimesh.creation.box(extents=(9.0, 20.0, 30.0))
        dent.apply_translation((-90.0 + i * 20.0, 0.0, 18.0))
        parts.append(dent)
    mesh = trimesh.util.concatenate(parts)
    chemin = str(tmp_path / nom)
    mesh.export(chemin)
    return chemin, mesh


def directions(clamp):
    import numpy as np

    points = np.asarray(clamp["routing_points"], dtype=float)
    vecteurs = points[1::2] - points[0::2]
    return vecteurs / np.linalg.norm(vecteurs, axis=1)[:, None]


def brut_du_detecteur(chemin, n=13):
    """Ce que rend le détecteur : des tranches d'encombrement, pas des encoches."""
    import numpy as np

    identite = np.eye(4).tolist()
    points = [[float(i * 10), 0.0, 0.0] for i in range(2 * n)]
    return {"name": "XA453420_peigne.stl", "position": [0.0, 0.0, 0.0], "score": 0.9,
            "routing_points": points, "file_path": chemin, "transform": identite}


def test_les_encoches_sont_cherchees_dans_la_matiere(tmp_path):
    chemin, _ = peigne_stl(tmp_path)
    clamp = brut_du_detecteur(chemin)
    assert fs.refine_passages([clamp], log=lambda *_a: None) == 1
    assert len(clamp["routing_points"]) > 0


def test_les_segments_sont_paralleles_entre_eux(tmp_path):
    """Chaque encoche se traverse dans le même sens : celui des dents."""
    import numpy as np

    chemin, _ = peigne_stl(tmp_path)
    clamp = brut_du_detecteur(chemin)
    fs.refine_passages([clamp], log=lambda *_a: None)
    dirs = directions(clamp)
    assert np.allclose(np.abs(dirs @ dirs[0]), 1.0, atol=1e-6)


def test_les_segments_suivent_les_dents_et_non_l_epaisseur(tmp_path):
    """Le défaut : le détecteur traverse le peigne dans sa plus petite dimension.

    Les dents pointent selon z ; l'encombrement du peigne est le plus mince
    selon y. Un segment selon y est donc perpendiculaire au passage réel.
    """
    import numpy as np

    chemin, _ = peigne_stl(tmp_path)
    clamp = brut_du_detecteur(chemin)
    fs.refine_passages([clamp], log=lambda *_a: None)
    dirs = directions(clamp)
    assert abs(float(dirs[0] @ np.array([0.0, 0.0, 1.0]))) > 0.99, "parallèle aux dents"
    assert abs(float(dirs[0] @ np.array([0.0, 1.0, 0.0]))) < 0.01, "pas l'épaisseur"


def test_le_recalage_est_appliqué_aux_segments(tmp_path):
    """Les points doivent atterrir là où l'ICP a trouvé le peigne."""
    import numpy as np

    chemin, _ = peigne_stl(tmp_path)
    clamp = brut_du_detecteur(chemin)
    matrice = np.eye(4)
    matrice[:3, 3] = [1000.0, 0.0, 0.0]
    clamp["transform"] = matrice.tolist()
    fs.refine_passages([clamp], log=lambda *_a: None)
    points = np.asarray(clamp["routing_points"], dtype=float)
    assert points[:, 0].min() > 850.0


def test_le_nombre_d_encoches_vient_de_la_piece(tmp_path):
    """Et non d'un paramètre codé en dur dans le détecteur."""
    peu, _ = peigne_stl(tmp_path, n_dents=4, nom="peu.stl")
    beaucoup, _ = peigne_stl(tmp_path, n_dents=10, nom="beaucoup.stl")
    a, b = brut_du_detecteur(peu), brut_du_detecteur(beaucoup)
    fs.refine_passages([a, b], log=lambda *_a: None)
    assert len(a["routing_points"]) != len(b["routing_points"])


def test_une_fixation_sans_passage_n_est_pas_touchee(tmp_path):
    """Un clip n'a pas d'encoche : rien à y chercher."""
    chemin, _ = peigne_stl(tmp_path)
    clip = {"name": "clip.stl", "position": [0, 0, 0], "score": 0.8,
            "file_path": chemin, "transform": [[1, 0, 0, 0], [0, 1, 0, 0],
                                               [0, 0, 1, 0], [0, 0, 0, 1]]}
    assert fs.refine_passages([clip], log=lambda *_a: None) == 0
    assert "routing_points" not in clip


def test_un_modele_illisible_garde_les_points_du_detecteur(tmp_path):
    """Approximatifs vaut mieux qu'absents."""
    clamp = brut_du_detecteur(str(tmp_path / "jamais_ecrit.stl"))
    avant = list(clamp["routing_points"])
    lignes = []
    assert fs.refine_passages([clamp], log=lignes.append) == 0
    assert clamp["routing_points"] == avant
    assert any("non recalculées" in ligne for ligne in lignes)


def test_une_liste_vide_ne_leve_pas():
    assert fs.refine_passages([], log=lambda *_a: None) == 0
    assert fs.refine_passages(None, log=lambda *_a: None) == 0
