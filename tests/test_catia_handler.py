"""Vérifications du connecteur CATIA.

Le dialogue COM lui-même n'est pas testable hors Windows, mais l'essentiel
l'est : les chemins de travail et le texte des macros VBScript. C'est
précisément là que se logeaient les défauts, parce qu'aucun des deux ne produit
d'erreur visible quand il est faux.
"""

import pathlib

import pytest

from core import catia_handler as ch
from core.paths import BASE_CACHE, STL_DIR


class TestImportMultiplateforme:
    def test_le_module_simporte_partout(self):
        """pywin32 ne doit être exigé qu'à l'appel, pas au chargement."""
        assert callable(ch.run_catia_export_via_vba)
        assert callable(ch.load_path_in_catia)

    def test_sans_pywin32_lerreur_est_explicite(self):
        """Sur un poste sans CATIA, l'utilisateur doit savoir quoi faire."""
        try:
            import pythoncom  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("pywin32 présent : le cas d'erreur ne se produit pas")

        with pytest.raises(ch.CatiaError) as excinfo:
            ch.run_catia_export_via_vba()
        message = str(excinfo.value)
        assert "pywin32" in message
        assert "STL déjà exportés" in message


class TestChemins:
    def test_le_connecteur_partage_le_cache_de_lapplication(self):
        """Le défaut le plus coûteux : exporter là où personne ne relit.

        Le connecteur fixait « C:\\Temp\\HarnessOpt_cache » tandis que
        l'application lisait le cache de core.paths. L'export CATIA
        réussissait, puis l'étape suivante annonçait « aucun fichier .stl
        trouvé ».
        """
        assert ch.STL_FOLDER == STL_DIR
        assert ch.BASE_CACHE == BASE_CACHE

    def test_aucun_chemin_en_dur(self):
        text = pathlib.Path(ch.__file__).read_text(encoding="utf-8")
        assert "C:\\Temp\\HarnessOpt_cache" not in text


class TestMacroExport:
    def macro(self, exclude=""):
        return ch.build_export_macro("C:\\HarnessOpt\\stl\\", exclude)

    def test_les_antislashes_survivent(self):
        """Régression : la f-string doit rester brute.

        Sans le préfixe ``rf``, Python consommait les antislashes et la ligne
        d'assainissement devenait ``Replace(baseName, "", "_")`` — un
        no-op. Une pièce nommée « ASSY\\PART-1 » produisait alors un chemin
        vers un sous-dossier inexistant et disparaissait de l'export sans le
        moindre message.
        """
        line = next(l for l in self.macro().splitlines() if "baseName = Replace" in l)
        assert '"\\"' in line, "l'antislash a été consommé par Python"
        assert 'Replace(baseName, "", "_")' not in line

    def test_echappement_des_motifs_dexclusion(self):
        line = next(l for l in self.macro().splitlines() if 'Replace(pat, "\\"' in l)
        assert '"\\\\"' in line

    def test_le_dossier_dexport_est_interpole(self):
        line = next(l for l in self.macro().splitlines() if "EXPORT_FOLDER =" in l)
        assert line.strip() == 'Const EXPORT_FOLDER = "C:\\HarnessOpt\\stl\\"'

    def test_le_filtre_dexclusion_est_interpole(self):
        line = next(l for l in self.macro("U258*, *-DUMMY").splitlines() if "EXCLUDE_FILTER =" in l)
        assert line.strip() == 'Const EXCLUDE_FILTER = "U258*, *-DUMMY"'

    def test_filtre_vide_accepte(self):
        assert 'Const EXCLUDE_FILTER = ""' in self.macro()

    def test_la_macro_expose_bien_catmain(self):
        """ExecuteScript appelle CATMain : le point d'entrée doit exister."""
        assert "Sub CATMain()" in self.macro()

    def test_linteractivite_est_retablie(self):
        """La macro coupe l'interactivité de CATIA : elle doit la rendre."""
        macro = self.macro()
        assert "CATIA.Interactive = False" in macro
        assert macro.count("CATIA.Interactive = True") >= 2  # sorties anticipées incluses


class TestMacroImport:
    def test_le_chemin_est_interpole_tel_quel(self):
        macro = ch.build_import_macro("C:\\Sortie\\faisceau.stl")
        assert 'filesToInsert(0) = "C:\\Sortie\\faisceau.stl"' in macro

    def test_pas_de_boite_de_dialogue_bloquante(self):
        """Un MsgBox attendrait un clic dans CATIA, application figée côté Python."""
        assert "MsgBox" not in ch.build_import_macro("C:\\x.stl")

    def test_sortie_propre_sans_document_actif(self):
        assert "If doc Is Nothing Then Exit Sub" in ch.build_import_macro("C:\\x.stl")


class TestRemonteeDerreur:
    def test_lechec_catia_arrive_lisible_dans_linterface(self):
        """Une erreur attendue ne doit pas remonter sous forme de pile d'appels."""
        from multiprocessing import Queue

        from core.mesh_processor import extraction_worker

        try:
            import pythoncom  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("pywin32 présent : l'export serait réellement tenté")

        queue = Queue()
        extraction_worker("/dossier/inexistant", True, "", queue)

        # `Queue.empty()` n'est pas fiable : un fil d'alimentation en arrière-plan
        # transfère les éléments, et la file peut se déclarer vide juste après un
        # `put`. On draine donc avec un délai d'attente explicite.
        messages = []
        while True:
            try:
                messages.append(queue.get(timeout=2))
            except Exception:
                break

        errors = [m for m in messages if m[0] == "ERROR"]
        assert errors, "aucune erreur remontée"
        text = errors[0][1]
        assert "Traceback" not in text
        assert "pywin32" in text
