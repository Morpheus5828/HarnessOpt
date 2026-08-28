"""Point d'entrée de HarnessOpt.

Lancement :

    python main.py

Le dossier de travail est choisi automatiquement selon la plateforme (voir
:mod:`core.paths`) ; la variable d'environnement ``HARNESSOPT_CACHE`` permet
de le forcer.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from controller.app_controller import AppController
        from ui.app_window import AppWindow
    except ImportError as exc:
        print(
            "❌ Impossible de démarrer l'interface.\n"
            f"   Dépendance manquante : {exc}\n\n"
            "   Installez les dépendances avec :\n"
            "       pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        return 1

    window = AppWindow()
    window.set_controller(AppController(window))
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
