from pathlib import Path
import pythoncom
import os
import win32com.client

BASE_CACHE = Path(r"C:\Temp\HarnessOpt_cache")
STL_FOLDER = BASE_CACHE / "stl"
COLOR_DIR = BASE_CACHE / "color"
FUSION_DIR = BASE_CACHE / "fusion"
COLOR_PATH = COLOR_DIR / "dmu_color_parts.xlsx"

def run_catia_export_via_vba(exclude_str=""):
    import os
    import glob
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()

    if os.path.exists(STL_FOLDER):
        print("🧹 Nettoyage des anciens fichiers STL dans le cache...")
        for filepath in glob.glob(os.path.join(STL_FOLDER, "*.stl")):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"⚠️ Impossible de supprimer l'ancien fichier {filepath}: {e}")
    else:
        os.makedirs(STL_FOLDER)
    # -----------------------------------------------

    vba_code = f"""
Const EXPORT_FOLDER = "{STL_FOLDER}\\"
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
    try:
        temp_dir = os.environ.get("TEMP", r"C:\Temp")
        macro_path = os.path.join(temp_dir, "HarnessOpt_Export.catvbs")
        with open(macro_path, "w", encoding="utf-8") as f:
            f.write(vba_code)
        catia = win32com.client.Dispatch("CATIA.Application")
        catia.SystemService.ExecuteScript(temp_dir, 1, "HarnessOpt_Export.catvbs", "CATMain", [])
    except Exception as e:
        try:
            catia = win32com.client.GetActiveObject("CATIA.Application")
            catia.Interactive = True
        except:
            pass
        print(f"Erreur VBA : {e}")
    finally:
        pythoncom.CoUninitialize()

def load_path_in_catia(self, stl_path):
    """Insère le STL généré directement dans le document CATIA actif."""
    import threading

    def run_import():
        import win32com.client
        import pythoncom
        import os

        pythoncom.CoInitialize()
        try:
            print(f"⏳ Importation du STL {os.path.basename(stl_path)} dans CATIA...")
            abs_path = os.path.abspath(stl_path)

            catia = win32com.client.Dispatch("CATIA.Application")

            # Le script VBS est très robuste pour forcer l'ajout dans le produit actif
            vba_code = f"""
            Sub CATMain()
                On Error Resume Next
                Dim doc
                Set doc = CATIA.ActiveDocument
                If doc Is Nothing Then
                    MsgBox "Aucun document actif dans CATIA. Veuillez ouvrir un Product.", vbCritical
                    Exit Sub
                End If

                Dim rootProduct
                Set rootProduct = doc.Product

                Dim components
                Set components = rootProduct.Products

                Dim filesToInsert(0)
                filesToInsert(0) = "{abs_path}"

                components.AddComponentsFromFiles filesToInsert, "All"
            End Sub
            """
            temp_dir = os.environ.get("TEMP", r"C:\Temp")
            macro_path = os.path.join(temp_dir, "HarnessOpt_Import.catvbs")
            with open(macro_path, "w", encoding="utf-8") as f:
                f.write(vba_code)

            # Exécution du script
            catia.SystemService.ExecuteScript(temp_dir, 1, "HarnessOpt_Import.catvbs", "CATMain", [])
            print("✅ Faisceau importé dans le document CATIA actif avec succès !")

        except Exception as e:
            print(f"❌ Erreur d'import CATIA : {e}")
        finally:
            pythoncom.CoUninitialize()

    # On lance dans un thread pour ne pas geler Tkinter
    threading.Thread(target=run_import, daemon=True).start()
