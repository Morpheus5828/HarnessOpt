"""Connecteur CATIA : export des pièces en STL et réimport du faisceau.

Le dialogue avec CATIA passe par une macro VBScript écrite à la volée puis
exécutée via ``SystemService.ExecuteScript``. Ce module ne fonctionne donc que
sous Windows, avec CATIA lancé et pywin32 installé.

Les dépendances Windows (``pythoncom``, ``win32com``) sont importées **à
l'appel** et non au chargement du module : le reste de l'application peut ainsi
importer ce fichier sur n'importe quelle plateforme et afficher un message
clair, au lieu de casser à l'import.

Attention à l'écriture des macros : le code VBScript est produit par des
f-strings **brutes** (``rf"..."``). VBScript n'a pas de caractère d'échappement
dans ses littéraux, donc un antislash doit arriver tel quel dans la macro.
Avec une f-string ordinaire, Python consomme ces antislashes avant que la macro
ne soit écrite.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

# Une seule définition des dossiers de travail, partagée avec le reste de
# l'application. Auparavant ce module fixait « C:\\Temp\\HarnessOpt_cache »
# tandis que l'application lisait le cache renvoyé par core.paths : l'export
# CATIA déposait donc les STL dans un dossier que personne n'allait relire.
from core.paths import BASE_CACHE, COLOR_DIR, FUSION_DIR, STL_DIR, ensure_cache_folders

__all__ = [
    "CatiaError",
    "BASE_CACHE",
    "STL_FOLDER",
    "COLOR_DIR",
    "FUSION_DIR",
    "COLOR_PATH",
    "build_export_macro",
    "build_import_macro",
    "run_catia_export_via_vba",
    "load_path_in_catia",
]

STL_FOLDER: Path = STL_DIR
COLOR_PATH: Path = COLOR_DIR / "dmu_color_parts.xlsx"


class CatiaError(RuntimeError):
    """CATIA est injoignable, ou la macro a échoué."""


def _com():
    """Importe pythoncom et win32com à la demande.

    Raises:
        CatiaError: si pywin32 n'est pas disponible (autre système, ou paquet
            absent), avec un message exploitable par l'utilisateur.
    """
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise CatiaError(
            "Le pilotage de CATIA nécessite Windows et le paquet pywin32.\n\n"
            f"Détail : {exc}\n\n"
            "Choisissez « Un dossier de fichiers STL déjà exportés » pour "
            "travailler sans CATIA, ou installez pywin32 :\n"
            "    pip install pywin32"
        ) from exc
    return pythoncom, win32com.client


def _run_macro(win32, name: str, code: str):
    """Écrit une macro VBScript dans le dossier temporaire et la fait exécuter."""
    temp_dir = os.environ.get("TEMP") or os.environ.get("TMP") or r"C:\Temp"
    os.makedirs(temp_dir, exist_ok=True)

    macro_path = os.path.join(temp_dir, name)
    with open(macro_path, "w", encoding="utf-8") as handle:
        handle.write(code)

    try:
        catia = win32.Dispatch("CATIA.Application")
    except Exception as exc:
        raise CatiaError(
            "Impossible de joindre CATIA.\n\n"
            "Vérifiez que CATIA est lancé et qu'un document est ouvert, "
            "puis relancez l'opération.\n\n"
            f"Détail : {exc}"
        ) from exc

    try:
        catia.SystemService.ExecuteScript(temp_dir, 1, name, "CATMain", [])
    except Exception as exc:
        # La macro laisse CATIA en mode non interactif pendant son exécution :
        # si elle échoue en cours de route, l'interface resterait figée.
        try:
            catia.Interactive = True
            catia.RefreshDisplay = True
            catia.DisplayFileAlerts = True
        except Exception:
            pass
        raise CatiaError(
            f"L'exécution de la macro CATIA a échoué.\n\nDétail : {exc}"
        ) from exc

    return catia


def build_export_macro(export_dir: str, exclude_str: str = "") -> str:
    """Construit la macro VBScript d'export.

    Fonction pure, isolée du dialogue COM pour être vérifiable sans Windows :
    c'est ici que se jouait un bogue silencieux. La f-string doit rester
    BRUTE (préfixe ``rf``), sinon Python consomme les antislashes et la ligne
    qui assainit les noms de pièces devient un ``Replace`` de chaîne vide. Une
    pièce dont le nom CATIA contient un antislash produisait alors un chemin
    vers un sous-dossier inexistant, et son export échouait sans bruit sous
    ``On Error Resume Next`` : la pièce manquait dans la maquette sans que
    rien ne le signale.
    """
    return rf"""
Const EXPORT_FOLDER = "{export_dir}"
Const EXCLUDE_FILTER = "{exclude_str}"
Function IsExcluded(itemName)
    IsExcluded = False
    If Trim(EXCLUDE_FILTER) = "" Then Exit Function
    Dim filters, i, regEx, pat
    filters = Split(EXCLUDE_FILTER, ",")
    Set regEx = New RegExp
    regEx.IgnoreCase = True
    For i = 0 To UBound(filters)
        pat = Trim(filters(i))
        If pat <> "" Then
            pat = Replace(pat, "\", "\\"): pat = Replace(pat, ".", "\.")
            pat = Replace(pat, "(", "\("): pat = Replace(pat, ")", "\)")
            pat = Replace(pat, "[", "\["): pat = Replace(pat, "]", "\]")
            pat = Replace(pat, "*", ".*"): pat = Replace(pat, "?", ".")
            pat = "^" & pat & "$"
            regEx.Pattern = pat
            If regEx.Test(itemName) Then IsExcluded = True: Exit Function
        End If
    Next
End Function

Sub CATMain()
    CATIA.DisplayFileAlerts = False: CATIA.RefreshDisplay = False: CATIA.Interactive = False
    Dim productDocument1
    On Error Resume Next: Set productDocument1 = CATIA.ActiveDocument: On Error GoTo 0
    If productDocument1 Is Nothing Then CATIA.Interactive = True: Exit Sub

    Dim rootProduct: Set rootProduct = productDocument1.Product
    Dim leafList: Set leafList = CreateObject("Scripting.Dictionary")
    Call GetLeafProducts(rootProduct, leafList)
    If leafList.Count = 0 Then CATIA.Interactive = True: Exit Sub

    Dim dmoOffsets: Set dmoOffsets = rootProduct.GetTechnologicalObject("Offsets")
    Dim fso: Set fso = CreateObject("Scripting.FileSystemObject")
    Dim key, leafProd, groups, group, offsetDoc, arr(0): arr(0) = 0.0

    For Each key In leafList.Keys
        Set leafProd = leafList(key)
        Set groups = rootProduct.GetTechnologicalObject("Groups")
        Set group = groups.Add(): group.AddExplicit leafProd
        Set offsetDoc = dmoOffsets.ComputeAnOffset(group, 0.0, 0, arr)

        If Not offsetDoc Is Nothing Then
            Dim baseName: baseName = leafProd.Name
            baseName = Replace(baseName, "/", "_"): baseName = Replace(baseName, "\", "_"): baseName = Replace(baseName, ":", "_")
            Dim outputPath: outputPath = EXPORT_FOLDER & baseName & ".stl"
            Dim counter: counter = 1
            While fso.FileExists(outputPath)
                outputPath = EXPORT_FOLDER & baseName & "_" & counter & ".stl"
                counter = counter + 1
            Wend
            On Error Resume Next
            offsetDoc.ExportData outputPath, "stl": offsetDoc.Close
            On Error GoTo 0
        End If
        If group.CountExplicit > 0 Then group.RemoveExplicit 1
        groups.Remove group
    Next
    CATIA.Interactive = True: CATIA.RefreshDisplay = True
End Sub

Sub GetLeafProducts(prod, dict)
    Dim children: Set children = prod.Products
    If children.Count = 0 Then
        If Not IsExcluded(prod.Name) Then dict.Add prod.Name & "_" & dict.Count, prod
    Else
        Dim i: For i = 1 To children.Count: Call GetLeafProducts(children.Item(i), dict): Next
    End If
End Sub
"""


def build_import_macro(abs_path: str) -> str:
    """Construit la macro VBScript d'insertion d'un STL dans le produit actif.

    Volontairement sans ``MsgBox`` : une boîte de dialogue ouverte par la macro
    attend un clic dans CATIA et bloquerait l'application sans rien afficher de
    son côté. La macro sort, et c'est Python qui rend compte.
    """
    return rf"""
Sub CATMain()
    Dim doc
    On Error Resume Next
    Set doc = CATIA.ActiveDocument
    On Error GoTo 0
    If doc Is Nothing Then Exit Sub

    Dim rootProduct: Set rootProduct = doc.Product
    Dim components: Set components = rootProduct.Products
    Dim filesToInsert(0): filesToInsert(0) = "{abs_path}"
    components.AddComponentsFromFiles filesToInsert, "All"
End Sub
"""


def run_catia_export_via_vba(exclude_str: str = "") -> Path:
    """Exporte chaque pièce du produit actif en STL, dans le cache de travail.

    Args:
        exclude_str: motifs séparés par des virgules (``U258*, *-DUMMY``)
            désignant les pièces à ne pas exporter.

    Returns:
        Le dossier contenant les STL produits.

    Raises:
        CatiaError: pywin32 absent, CATIA injoignable, ou macro en échec.
            L'erreur remonte volontairement jusqu'à l'interface : la version
            précédente se contentait de l'afficher en console, et l'utilisateur
            voyait ensuite « aucun fichier .stl trouvé » sans savoir pourquoi.
    """
    pythoncom, win32 = _com()
    pythoncom.CoInitialize()
    try:
        ensure_cache_folders()
        os.makedirs(STL_FOLDER, exist_ok=True)

        print("🧹 Nettoyage des anciens fichiers STL dans le cache...")
        for filepath in glob.glob(os.path.join(str(STL_FOLDER), "*.stl")):
            try:
                os.remove(filepath)
            except OSError as exc:
                print(f"⚠️ Impossible de supprimer l'ancien fichier {filepath} : {exc}")

        # Chemin terminé par un séparateur : la macro y concatène le nom de
        # fichier directement.
        export_dir = os.path.join(str(STL_FOLDER), "")

        vba_code = build_export_macro(export_dir, exclude_str)

        _run_macro(win32, "HarnessOpt_Export.catvbs", vba_code)

        produced = glob.glob(os.path.join(str(STL_FOLDER), "*.stl"))
        if not produced:
            raise CatiaError(
                "La macro s'est exécutée mais n'a produit aucun fichier STL.\n\n"
                "Vérifiez qu'un Product est bien actif dans CATIA et que le "
                "filtre d'exclusion n'écarte pas toutes les pièces."
            )

        print(f"✅ {len(produced)} pièce(s) exportée(s) vers {STL_FOLDER}")
        return STL_FOLDER

    finally:
        pythoncom.CoUninitialize()


def load_path_in_catia(stl_path: str | Path) -> None:
    """Insère un STL de faisceau dans le document CATIA actif.

    Appel bloquant : à lancer depuis un fil d'exécution séparé si l'interface
    doit rester réactive. La version précédente créait elle-même son fil, ce
    qui empêchait l'appelant de savoir si l'import avait réussi.

    Raises:
        CatiaError: pywin32 absent, CATIA injoignable, ou macro en échec.
    """
    pythoncom, win32 = _com()
    pythoncom.CoInitialize()
    try:
        abs_path = os.path.abspath(str(stl_path))
        if not os.path.exists(abs_path):
            raise CatiaError(f"Fichier introuvable : {abs_path}")

        print(f"⏳ Importation de {os.path.basename(abs_path)} dans CATIA...")

        vba_code = build_import_macro(abs_path)

        _run_macro(win32, "HarnessOpt_Import.catvbs", vba_code)
        print("✅ Faisceau importé dans le document CATIA actif.")

    finally:
        pythoncom.CoUninitialize()
