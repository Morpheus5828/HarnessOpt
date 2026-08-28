import os
from pathlib import Path
import pandas as pd

class MeshModel:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.stl_folder = self.cache_dir / "stl"
        self.color_path = self.cache_dir / "color" / "dmu_color_parts.xlsx"
        self.fusion_dir = self.cache_dir / "fusion"

    # =========================================================
    # 🎨 ANALYSE OFFICIELLE GESTION COULEURS DMU H160
    # =========================================================
    def run_color_analysis(self, target_dir, progress_callback=None):
        stl_stems = [p for p in os.listdir(target_dir) if p.lower().endswith('.stl')]
        total = len(stl_stems)
        if total == 0:
            raise ValueError("Aucun fichier STL trouvé dans le dossier cible.")

        all_parts, all_color, all_presence = [], [], []

        for i, stl in enumerate(stl_stems):
            part_num = stl.split(".")[0].split("_")[0]
            current_color = "standard"

            if len(part_num) > 1:
                c2 = part_num[1]

                if c2 == "1":
                    current_color = "equipement"
                elif c2 == "2" and len(part_num) > 2:
                    c3 = part_num[2]
                    if c3 == "1" and len(part_num) > 3:
                        c4 = part_num[3]
                        if c4 == "1": current_color = "ecs_air_circuit"
                        elif c4 == "4": current_color = "p3_circuit"
                        elif c4 == "5": current_color = "ecs_cold_circuit"
                        elif c4 == "6": current_color = "equipement"
                    elif c3 == "8":
                        current_color = "fuel"
                    elif c3 == "9" and len(part_num) > 3:
                        c4 = part_num[3]
                        if c4 == "1": current_color = "high_pressure_system"
                        elif c4 == "2": current_color = "return_system"
                        elif c4 == "3": current_color = "air_azote"
                        elif c4 == "4": current_color = "suction"
                elif c2 == "5":
                    current_color = "structure"
                elif c2 == "6" and len(part_num) > 2:
                    c3 = part_num[2]
                    if c3 == "7":
                        current_color = "flight_control_system-fcs"

            part_num_lower = part_num.lower()
            if "insonorisation" in part_num_lower:
                current_color = "insonorisation"
            elif "copper" in part_num_lower or "foil" in part_num_lower:
                current_color = "copper_foils"
            elif "mecanical" in part_num_lower:
                current_color = "mecanical_installation"

            all_parts.append(stl)
            all_color.append(current_color)
            all_presence.append(True)

            if progress_callback and i % 20 == 0:
                prog_pct = (i / total) * 0.4
                progress_callback(i, total, f"🎨 Analyse des couleurs: {i}/{total}", prog_pct)

        df = pd.DataFrame({"Part Number": all_parts, "Color": all_color, "Presence": all_presence})
        self.color_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(str(self.color_path), index=False)
        return df
