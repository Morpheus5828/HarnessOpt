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
