

from core.mesh_processor import MeshModel
from ui.main_window import MainWindow
from controllers.controller import AppController



if __name__ == "__main__":
    model = MeshModel(cache_dir="C:/Temp/HarnessOpt_cache")
    view = MainWindow()
    controller = AppController(model, view)

    view.set_controller(controller)

    view.mainloop()
