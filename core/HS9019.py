
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import config


class Rules:
    def __init__(self):
        self.MAX_DISTANCE = getattr(config, "MAX_DISTANCE", None)

        self.DISTANCE_WITH_STRUCTURE = getattr(config, "DISTANCE_WITH_STRUCTURE", 10)
        self.DISTANCE_WITH_VENTILATION_REFRIGERANT = getattr(
            config, "DISTANCE_WITH_VENTILATION_REFRIGERANT", 10
        )
        self.DISTANCE_HOT_AIR_LINES = getattr(config, "DISTANCE_HOT_AIR_LINES", 20)
        self.DISTANCE_WITH_HIGH_PRESSURE_HYDRAULIC_LINE = getattr(
            config, "DISTANCE_WITH_HIGH_PRESSURE_HYDRAULIC_LINE", 70
        )

        # Conception
        self.HARNESS_DIAMETER = getattr(config, "HARNESS_DIAMETER", 40)



