"""Traductions de l'interface.

Les textes étaient auparavant dispersés dans des dictionnaires en ligne au
milieu du code des écrans, ce qui rendait toute relecture pénible et laissait
des libellés non traduits. Ils sont désormais regroupés ici.

Le français est la langue de référence : chaque clé y est obligatoirement
présente. Les autres langues peuvent être partielles, la traduction manquante
retombe alors sur l'anglais puis sur le français, ce qui vaut toujours mieux
qu'un libellé vide à l'écran.
"""

from __future__ import annotations

__all__ = ["LANGUAGES", "Translator", "t"]

#: Langues proposées, avec leur libellé de menu.
LANGUAGES: dict[str, str] = {
    "FR": "🇫🇷 Français",
    "EN": "🇬🇧 English",
    "DE": "🇩🇪 Deutsch",
    "ES": "🇪🇸 Español",
}


FR: dict[str, str] = {
    # -- application --------------------------------------------------
    "app.title": "HarnessOpt — Cheminement de harnais",
    "app.ready": "Prêt",
    "app.quit": "Quitter",
    "app.quit.confirm": "Voulez-vous quitter l'application ?",
    "app.quit.cache": "Vider le cache en quittant pour libérer de l'espace disque",
    # -- menus --------------------------------------------------------
    "menu.file": "Fichier",
    "menu.edit": "Édition",
    "menu.view": "Affichage",
    "menu.help": "Aide",
    "menu.language": "Langue",
    "menu.theme.dark": "Mode sombre",
    "menu.theme.light": "Mode clair",
    "menu.colors.default": "Couleurs standard",
    "menu.colors.colorblind": "Couleurs adaptées daltonisme",
    "menu.cache.clear": "Vider le cache",
    "menu.cache.open": "Ouvrir le dossier de travail",
    "menu.help.about": "À propos",
    # -- étapes -------------------------------------------------------
    "stepper.step": "Étape",
    "step.project": "Projet",
    "step.project.sub": "Charger la maquette",
    "step.rules": "Règles",
    "step.rules.sub": "Distances et fixations",
    "step.routing": "Cheminement",
    "step.routing.sub": "Lancer les agents",
    "step.report": "Rapport",
    "step.report.sub": "Vérifier et exporter",
    "step.locked": "Terminez l'étape précédente pour accéder à celle-ci.",
    "step.locked.project": "Chargez d'abord la maquette à l'étape 1.",
    "step.locked.routing": "Lancez d'abord un cheminement à l'étape 3.",
    "rules.continue": "Continuer vers le cheminement",
    "rules.ready": "Règles enregistrées.",
    # -- étape 1 : projet ---------------------------------------------
    "project.title": "1. Charger la maquette numérique",
    "project.intro": (
        "Indiquez où se trouvent les pièces de l'hélicoptère. "
        "Elles seront assemblées en une seule maquette servant d'environnement aux agents."
    ),
    "project.source": "D'où viennent les pièces ?",
    "project.source.folder": "Un dossier de fichiers STL déjà exportés",
    "project.source.catia": "À exporter depuis CATIA maintenant",
    "project.folder": "Dossier des pièces (STL)",
    "project.folder.help": "Le dossier contenant les fichiers .stl de la zone à équiper.",
    "project.browse": "Parcourir…",
    "project.exclude": "Pièces à ignorer",
    "project.exclude.help": "Motif facultatif, par exemple U258* pour écarter toute une série.",
    "project.clamps": "Dossier des fixations de référence",
    "project.clamps.help": (
        "Les modèles de crabes et de peignes recherchés dans la maquette "
        "avant de commencer le cheminement."
    ),
    "project.run": "Charger la maquette",
    "project.rerun": "Recharger la maquette",
    "project.cancel": "Interrompre",
    "project.running": "Chargement en cours…",
    "project.done": "Maquette prête",
    "project.stat.parts": "Pièces",
    "project.stat.faces": "Facettes",
    "project.stat.size": "Encombrement",
    "project.stat.families": "Familles",
    "project.families.title": "Répartition des pièces par famille",
    "project.bbox": "Zone couverte par la maquette",
    "project.preview": "Aperçu de la maquette",
    "project.empty": "Aucune maquette chargée pour l'instant.",
    "project.error.folder": "Ce dossier n'existe pas.",
    "project.error.empty": "Aucun fichier .stl trouvé dans ce dossier.",
    # -- étape 2 : règles ---------------------------------------------
    "rules.title": "2. Régler les contraintes d'intégration",
    "rules.intro": (
        "Ces valeurs définissent ce qu'est un cheminement acceptable. "
        "Elles serviront à la fois à guider les agents et à contrôler le résultat."
    ),
    "rules.active.title": "Règles appliquées",
    "rules.active.help": "Décochez une règle pour la retirer du problème : elle cesse d'être évaluée, de compter dans le classement et de guider les agents.",
    "rules.harness": "Le harnais",
    "rules.diameter": "Diamètre du toron",
    "rules.diameter.help": "Diamètre extérieur du faisceau une fois assemblé.",
    "rules.bend_factor": "Rayon de cintrage minimal",
    "rules.bend_factor.help": (
        "Exprimé en multiples du diamètre. 6 fois le diamètre est la valeur usuelle : "
        "en deçà, le câble marque une cassure."
    ),
    "rules.bend_result": "Soit un rayon minimal de",
    "rules.clearance": "Distances à la structure",
    "rules.clearance.min": "Distance minimale",
    "rules.clearance.min.help": "Le câble ne doit jamais s'approcher plus près d'une pièce.",
    "rules.clearance.max": "Distance maximale",
    "rules.clearance.max.help": (
        "Au-delà, le câble ne longe plus rien et devient impossible à fixer."
    ),
    "rules.families.title": "Distances renforcées par famille de pièces",
    "rules.families.help": (
        "Certaines pièces imposent un écart plus important que la distance minimale générale. "
        "Ces valeurs s'appliquent automatiquement dès que la maquette est classée par couleur."
    ),
    "rules.families.detected": "Familles présentes dans la maquette",
    "rules.families.none": (
        "La maquette chargée ne fournit pas de classification par couleur : "
        "la distance minimale générale s'appliquera partout."
    ),
    "rules.fixations": "Fixations",
    "rules.edge": "Distance minimale au bord de tôle",
    "rules.edge.help": "Le câble ne longe pas un chant de tôle : il y userait sa "
                       "gaine, et surtout aucune fixation ne peut y être posée.",
    "rules.pitch": "Pas maximal entre fixations",
    "rules.pitch.help": (
        "Au-delà de cette distance sans point de fixation, l'agent doit poser un crabe."
    ),
    "rules.parallel": "Écart de pose toléré",
    "rules.parallel.help": (
        "Écart angulaire admis entre l'embase du crabe et la structure : au-delà, "
        "le crabe ne repose pas à plat."
    ),
    "rules.clamps_folder": "Dossier des fixations existantes",
    "rules.clamps_folder.help": "Modèles STL des fixations déjà montées. L'application les recale sur la maquette et en déduit les passages imposés (p_in / p_out) des peignes. Laisser vide pour ignorer les fixations existantes.",
    "rules.clamp_model": "Modèle de crabe à poser",
    "rules.clamp_model.help": "Fichier STL du crabe utilisé pour vérifier l'encombrement.",
    "rules.reset": "Revenir aux valeurs standard",
    "rules.summary": "Résumé des règles retenues",
    # -- étape 3 : cheminement ----------------------------------------
    "routing.title": "3. Faire cheminer le harnais",
    "routing.intro": (
        "Indiquez le départ et l'arrivée, choisissez une façon de travailler, puis lancez. "
        "Les agents cherchent ensemble et la meilleure route est retenue."
    ),
    "routing.endpoints": "Départ et arrivée",
    "routing.source": "Point de départ",
    "routing.target": "Point d'arrivée",
    "routing.coords.help": "Coordonnées dans le repère avion, en millimètres.",
    "routing.pick": "Choisir dans la vue 3D",
    "routing.strategy": "Façon de travailler",
    "routing.team": "Équipe d'agents",
    "routing.team.help": "Chaque agent se consacre à un aspect différent du cheminement.",
    "routing.start_path": "Chemin de départ",
    "routing.start_path.help": (
        "Comment relier les deux points avant que les agents ne l'améliorent. "
        "La recherche dans l'espace libre traverse les ouvertures et part déjà "
        "à bonne distance de la structure."
    ),
    "routing.start_path.geodesic": "Le long de la surface (géodésique)",
    "routing.start_path.geodesic.help": (
        "Suit le maillage. Inopérant sur une maquette faite de pièces disjointes, "
        "où il se réduit à une ligne droite collée à la structure."
    ),
    "routing.use_fixations": "Emprunter les fixations existantes",
    "routing.use_fixations.help": "Le chemin de départ traverse chaque encoche reconnue, par son entrée puis sa sortie. Décochez pour laisser les agents choisir librement leur passage.",
    "routing.explore": "Exploration ↔ Exploitation",
    "routing.explore.help": (
        "À gauche, les agents peaufinent la meilleure route trouvée. "
        "À droite, ils cherchent large, quitte à repartir de zéro. "
        "Le réglage se modifie en cours de calcul."
    ),
    "routing.explore.explore": "Exploration totale",
    "routing.explore.balanced": "Équilibré",
    "routing.explore.exploit": "Exploitation totale",
    "routing.explore.left": "Peaufiner",
    "routing.explore.right": "Chercher large",
    "routing.start": "Lancer le cheminement",
    "routing.pause": "Mettre en pause",
    "routing.resume": "Reprendre",
    "routing.reset": "Recommencer",
    "routing.scanning": "Recherche des fixations existantes…",
    "routing.phase": "Étape en cours",
    "routing.best": "Meilleur agent",
    "routing.iteration": "Itération",
    "routing.compliance": "Conformité du meilleur tracé",
    "routing.agents": "Les agents au travail",
    "routing.view": "Vue 3D",
    "routing.view.mesh": "Maquette",
    "routing.view.edges": "Arêtes",
    "routing.view.bbox": "Encombrement",
    "routing.view.clamps": "Fixations",
    "routing.view.detach": "Ouvrir la vue 3D",
    "routing.view.attach": "Fermer la vue 3D",
    "routing.view.starting": "Préparation de la vue 3D…",
    "routing.view.ready": "Vue 3D prête.",
    "routing.view.closed": "La vue 3D s'ouvre dans sa propre fenêtre.",
    "routing.view.opened": "Vue 3D ouverte. Glisser pour tourner, molette pour zoomer.",
    "routing.view.unavailable": "Vue 3D indisponible sur ce poste.",
    "routing.view.still_running": "Le calcul du cheminement, lui, continue normalement.",
    "routing.view.none": "Aucune vue 3D à ouvrir pour l'instant.",
    "routing.view.edit": "Édition manuelle (BETA)",
    "routing.view.edit.on": "Poignées posées sur le tracé : glissez-en une pour imposer un point de passage.",
    "routing.view.edit.off": "Édition manuelle arrêtée. Les points imposés sont conservés.",
    "routing.view.edit.none": "Aucun tracé à modifier pour l'instant.",
    "routing.view.edit.pinned": "Point imposé : les agents le replacent à chaque itération.",
    "routing.view.edit.clear": "Libérer les points imposés",
    "routing.prepare.scan": "Analyse des fixations existantes…",
    "routing.prepare.mesh": "Préparation des maillages de travail…",
    "routing.prepare.path": "Recherche du chemin de départ…",
    "routing.advice": "Conseils",
    "advice.applied": "Réglage appliqué à l'étape « Règles ». Relancez le cheminement pour qu'il prenne effet.",
    "advice.not_settable": "Ce conseil ne correspond à aucun réglage : suivez l'indication donnée.",
    "routing.charts": "Progression",
    "routing.charts.window": "Courbes en plein écran",
    "routing.charts.window.close": "Fermer les courbes",
    "routing.charts.window.none": "Fenêtre des courbes indisponible :",
    "routing.charts.expand": "Agrandir les courbes",
    "routing.charts.shrink": "Réduire les courbes",
    "routing.advanced": "Réglages avancés",
    "routing.calibrate": "Calibrer pour ce faisceau",
    "routing.calibrate.help": (
        "Redéduit le nombre de points, le plafond et le pas de la longueur du "
        "faisceau et du rayon de cintrage du toron."
    ),
    "routing.calibrate.done": "Calibré : {points} points, un tous les {spacing:.0f} mm.",
    "routing.calibrate.none": "Renseignez d'abord le départ et l'arrivée.",
    "routing.advanced.warn": (
        "Ces réglages n'ont normalement pas besoin d'être touchés : "
        "la façon de travailler choisie plus haut les ajuste déjà."
    ),
    # -- KPI ----------------------------------------------------------
    "kpi.length": "Longueur",
    "kpi.clashes": "Interférences",
    "kpi.bend": "Cintrage le plus serré",
    "kpi.straight": "Parcours rectiligne",
    "kpi.clamps": "Fixations posées",
    "kpi.distance": "Distance moyenne",
    "kpi.bends": "Coudes",
    "kpi.points": "Points",
    # -- étape 4 : rapport --------------------------------------------
    "report.title": "4. Contrôler et exporter",
    "report.intro": "Vérifiez le verdict, puis exportez le cheminement retenu.",
    "report.verdict.ok": "Cheminement conforme",
    "report.verdict.deliverable": "Livrable avec réserves",
    "report.verdict.ko": "Cheminement non conforme",
    "report.verdict.none": "Aucun cheminement calculé pour l'instant.",
    "report.detail": "Détail des règles",
    "report.exports": "Exports",
    "report.export.stl": "Exporter en STL",
    "report.export.csv": "Exporter les points (CSV)",
    "report.export.report": "Enregistrer le rapport",
    "report.export.catia": "Réinsérer dans CATIA",
    "report.catia.sending": "Envoi vers CATIA…",
    "report.catia.done": "Faisceau inséré dans le document CATIA actif",
    "report.exported": "Export terminé",
    "report.choose_agent": "Tracé à exporter",
    # -- messages généraux --------------------------------------------
    "common.yes": "Oui",
    "common.no": "Non",
    "common.cancel": "Annuler",
    "common.close": "Fermer",
    "common.ok": "D'accord",
    "common.mm": "mm",
    "common.deg": "°",
    "common.none": "—",
    "common.elapsed": "Temps écoulé",
    "common.eta": "Temps restant estimé",
    "common.limit": "limite",
    "common.measured": "mesuré",
    "cache.title": "Session précédente détectée",
    "cache.question": (
        "Des données d'une session précédente ont été trouvées.\n\n"
        "Voulez-vous les conserver pour reprendre où vous en étiez ?"
    ),
    "cache.cleared": "Cache vidé.",
    "cache.size": "Taille du cache",
}


EN: dict[str, str] = {
    "app.title": "HarnessOpt — Harness routing",
    "app.ready": "Ready",
    "app.quit": "Quit",
    "app.quit.confirm": "Do you want to quit the application?",
    "app.quit.cache": "Clear the cache on exit to free disk space",
    "menu.file": "File",
    "menu.edit": "Edit",
    "menu.view": "View",
    "menu.help": "Help",
    "menu.language": "Language",
    "menu.theme.dark": "Dark mode",
    "menu.theme.light": "Light mode",
    "menu.colors.default": "Standard colours",
    "menu.colors.colorblind": "Colour-blind friendly",
    "menu.cache.clear": "Clear cache",
    "menu.cache.open": "Open working folder",
    "menu.help.about": "About",
    "stepper.step": "Step",
    "step.project": "Project",
    "step.project.sub": "Load the mock-up",
    "step.rules": "Rules",
    "step.rules.sub": "Clearances and fixations",
    "step.routing": "Routing",
    "step.routing.sub": "Run the agents",
    "step.report": "Report",
    "step.report.sub": "Check and export",
    "step.locked": "Complete the previous step to unlock this one.",
    "step.locked.project": "Load the mock-up first, at step 1.",
    "step.locked.routing": "Run a routing first, at step 3.",
    "rules.continue": "Continue to routing",
    "rules.ready": "Rules saved.",
    "project.title": "1. Load the digital mock-up",
    "project.intro": (
        "Tell the application where the helicopter parts are. They will be merged into "
        "a single mock-up used as the agents' environment."
    ),
    "project.source": "Where do the parts come from?",
    "project.source.folder": "A folder of already exported STL files",
    "project.source.catia": "To be exported from CATIA now",
    "project.folder": "Parts folder (STL)",
    "project.folder.help": "The folder holding the .stl files of the area to equip.",
    "project.browse": "Browse…",
    "project.exclude": "Parts to skip",
    "project.exclude.help": "Optional pattern, e.g. U258* to skip a whole series.",
    "project.clamps": "Reference fixations folder",
    "project.clamps.help": (
        "The clamp and comb models searched for in the mock-up before routing starts."
    ),
    "project.run": "Load the mock-up",
    "project.rerun": "Reload the mock-up",
    "project.cancel": "Stop",
    "project.running": "Loading…",
    "project.done": "Mock-up ready",
    "project.stat.parts": "Parts",
    "project.stat.faces": "Facets",
    "project.stat.size": "Extent",
    "project.stat.families": "Families",
    "project.families.title": "Parts per family",
    "project.bbox": "Area covered by the mock-up",
    "project.preview": "Mock-up preview",
    "project.empty": "No mock-up loaded yet.",
    "project.error.folder": "This folder does not exist.",
    "project.error.empty": "No .stl file found in this folder.",
    "rules.title": "2. Set the integration constraints",
    "rules.intro": (
        "These values define what an acceptable route is. They guide the agents "
        "and are used to check the result."
    ),
    "rules.active.title": "Applied rules",
    "rules.active.help": "Untick a rule to take it out of the problem: it stops being evaluated, stops counting in the ranking and stops guiding the agents.",
    "rules.harness": "The harness",
    "rules.diameter": "Bundle diameter",
    "rules.diameter.help": "Outer diameter of the assembled bundle.",
    "rules.bend_factor": "Minimum bend radius",
    "rules.bend_factor.help": (
        "Expressed in multiples of the diameter. Six times the diameter is the usual "
        "value: below it, the cable kinks."
    ),
    "rules.bend_result": "That is a minimum radius of",
    "rules.clearance": "Clearances to structure",
    "rules.clearance.min": "Minimum clearance",
    "rules.clearance.min.help": "The cable must never come closer to any part.",
    "rules.clearance.max": "Maximum clearance",
    "rules.clearance.max.help": (
        "Beyond this the cable no longer follows anything and cannot be clamped."
    ),
    "rules.families.title": "Reinforced clearances per part family",
    "rules.families.help": (
        "Some parts require a larger gap than the general minimum. These values apply "
        "automatically as soon as the mock-up is classified by colour."
    ),
    "rules.families.detected": "Families present in the mock-up",
    "rules.families.none": (
        "The loaded mock-up provides no colour classification: the general minimum "
        "clearance will apply everywhere."
    ),
    "rules.fixations": "Fixations",
    "rules.edge": "Minimum clearance to sheet edge",
    "rules.edge.help": "The cable does not run along a sheet edge: it would chafe "
                       "its sleeve, and no fixation can be seated there.",
    "rules.pitch": "Maximum spacing between fixations",
    "rules.pitch.help": "Past this distance without a fixation, the agent must place a clamp.",
    "rules.parallel": "Allowed seating deviation",
    "rules.parallel.help": (
        "Angular deviation allowed between the clamp base and the structure: beyond it, "
        "the clamp does not sit flat."
    ),
    "rules.clamps_folder": "Existing fixations folder",
    "rules.clamps_folder.help": "STL models of the fixations already installed. They are registered onto the mock-up and the imposed comb passages (p_in / p_out) are derived from them. Leave empty to ignore existing fixations.",
    "rules.clamp_model": "Clamp model to place",
    "rules.clamp_model.help": "STL file of the clamp used to check for interference.",
    "rules.reset": "Back to standard values",
    "rules.summary": "Summary of the selected rules",
    "routing.title": "3. Route the harness",
    "routing.intro": (
        "Set the start and end points, choose how to work, then run. The agents search "
        "together and the best route is kept."
    ),
    "routing.endpoints": "Start and end",
    "routing.source": "Start point",
    "routing.target": "End point",
    "routing.coords.help": "Aircraft-frame coordinates, in millimetres.",
    "routing.pick": "Pick in the 3D view",
    "routing.strategy": "How to work",
    "routing.team": "Agent team",
    "routing.team.help": "Each agent focuses on a different aspect of the route.",
    "routing.start_path": "Starting path",
    "routing.start_path.help": (
        "How the two points are linked before the agents improve it. Free-space "
        "search goes through openings and starts at a sensible distance from "
        "the structure."
    ),
    "routing.start_path.geodesic": "Along the surface (geodesic)",
    "routing.start_path.geodesic.help": (
        "Follows the mesh. Ineffective on a mock-up made of disjoint parts, where "
        "it degrades to a straight line stuck to the structure."
    ),
    "routing.use_fixations": "Route through existing fixations",
    "routing.use_fixations.help": "The starting path goes through each recognised notch, in then out. Untick to let the agents choose their own way.",
    "routing.explore": "Exploration ↔ Exploitation",
    "routing.explore.help": (
        "On the left, agents refine the best route found. On the right, they search wide, "
        "even starting over. The setting can be changed while running."
    ),
    "routing.explore.explore": "Full exploration",
    "routing.explore.balanced": "Balanced",
    "routing.explore.exploit": "Full exploitation",
    "routing.explore.left": "Refine",
    "routing.explore.right": "Search wide",
    "routing.start": "Start routing",
    "routing.pause": "Pause",
    "routing.resume": "Resume",
    "routing.reset": "Start over",
    "routing.scanning": "Looking for existing fixations…",
    "routing.phase": "Current stage",
    "routing.best": "Best agent",
    "routing.iteration": "Iteration",
    "routing.compliance": "Compliance of the best route",
    "routing.agents": "Agents at work",
    "routing.view": "3D view",
    "routing.view.mesh": "Mock-up",
    "routing.view.edges": "Edges",
    "routing.view.bbox": "Extent",
    "routing.view.clamps": "Fixations",
    "routing.view.detach": "Open 3D view",
    "routing.view.attach": "Close 3D view",
    "routing.view.starting": "Preparing the 3D view…",
    "routing.view.ready": "3D view ready.",
    "routing.view.closed": "The 3D view opens in its own window.",
    "routing.view.opened": "3D view open. Drag to rotate, scroll to zoom.",
    "routing.view.unavailable": "3D view unavailable on this workstation.",
    "routing.view.still_running": "Route computation carries on regardless.",
    "routing.view.none": "No 3D view to open yet.",
    "routing.view.edit": "Manual editing (BETA)",
    "routing.view.edit.on": "Handles placed on the route: drag one to impose a waypoint.",
    "routing.view.edit.off": "Manual editing stopped. Imposed points are kept.",
    "routing.view.edit.none": "No route to edit yet.",
    "routing.view.edit.pinned": "Point imposed: the agents put it back at every iteration.",
    "routing.view.edit.clear": "Release imposed points",
    "routing.prepare.scan": "Scanning existing fixations…",
    "routing.prepare.mesh": "Preparing working meshes…",
    "routing.prepare.path": "Searching for the initial path…",
    "routing.advice": "Advice",
    "advice.applied": "Setting applied in the Rules step. Restart the routing for it to take effect.",
    "advice.not_settable": "This advice maps to no setting: follow the indication given.",
    "routing.charts": "Progress",
    "routing.charts.window": "Charts full screen",
    "routing.charts.window.close": "Close charts",
    "routing.charts.window.none": "Charts window unavailable:",
    "routing.charts.expand": "Enlarge charts",
    "routing.charts.shrink": "Shrink charts",
    "routing.advanced": "Advanced settings",
    "routing.calibrate": "Calibrate for this harness",
    "routing.calibrate.help": (
        "Re-derives the point count, the ceiling and the step from the harness "
        "length and the bundle bend radius."
    ),
    "routing.calibrate.done": "Calibrated: {points} points, one every {spacing:.0f} mm.",
    "routing.calibrate.none": "Set the start and end points first.",
    "routing.advanced.warn": (
        "These settings normally need no adjustment: the way of working chosen above "
        "already tunes them."
    ),
    "kpi.length": "Length",
    "kpi.clashes": "Interferences",
    "kpi.bend": "Tightest bend",
    "kpi.straight": "Straight run",
    "kpi.clamps": "Fixations placed",
    "kpi.distance": "Mean clearance",
    "kpi.bends": "Bends",
    "kpi.points": "Points",
    "report.title": "4. Check and export",
    "report.intro": "Review the verdict, then export the selected route.",
    "report.verdict.ok": "Compliant route",
    "report.verdict.deliverable": "Deliverable with reservations",
    "report.verdict.ko": "Non-compliant route",
    "report.verdict.none": "No route computed yet.",
    "report.detail": "Rule by rule",
    "report.exports": "Exports",
    "report.export.stl": "Export to STL",
    "report.export.csv": "Export points (CSV)",
    "report.export.report": "Save the report",
    "report.export.catia": "Send back to CATIA",
    "report.catia.sending": "Sending to CATIA…",
    "report.catia.done": "Harness inserted into the active CATIA document",
    "report.exported": "Export complete",
    "report.choose_agent": "Route to export",
    "common.yes": "Yes",
    "common.no": "No",
    "common.cancel": "Cancel",
    "common.close": "Close",
    "common.ok": "OK",
    "common.mm": "mm",
    "common.deg": "°",
    "common.none": "—",
    "common.elapsed": "Elapsed",
    "common.eta": "Estimated remaining",
    "common.limit": "limit",
    "common.measured": "measured",
    "cache.title": "Previous session detected",
    "cache.question": (
        "Data from a previous session was found.\n\n"
        "Do you want to keep it and resume where you left off?"
    ),
    "cache.cleared": "Cache cleared.",
    "cache.size": "Cache size",
}


#: Traductions partielles : seules les entrées les plus visibles sont
#: fournies, le reste retombe sur l'anglais.
DE: dict[str, str] = {
    "app.ready": "Bereit",
    "app.quit": "Beenden",
    "menu.file": "Datei",
    "menu.edit": "Bearbeiten",
    "menu.view": "Ansicht",
    "menu.help": "Hilfe",
    "menu.language": "Sprache",
    "menu.theme.dark": "Dunkelmodus",
    "menu.theme.light": "Hellmodus",
    "menu.cache.clear": "Cache leeren",
    "step.project": "Projekt",
    "step.rules": "Regeln",
    "step.routing": "Verlegung",
    "step.report": "Bericht",
    "common.yes": "Ja",
    "common.no": "Nein",
    "common.cancel": "Abbrechen",
    "common.close": "Schließen",
}

ES: dict[str, str] = {
    "app.ready": "Listo",
    "app.quit": "Salir",
    "menu.file": "Archivo",
    "menu.edit": "Edición",
    "menu.view": "Ver",
    "menu.help": "Ayuda",
    "menu.language": "Idioma",
    "menu.theme.dark": "Modo oscuro",
    "menu.theme.light": "Modo claro",
    "menu.cache.clear": "Vaciar caché",
    "step.project": "Proyecto",
    "step.rules": "Reglas",
    "step.routing": "Trazado",
    "step.report": "Informe",
    "common.yes": "Sí",
    "common.no": "No",
    "common.cancel": "Cancelar",
    "common.close": "Cerrar",
}


_CATALOGS: dict[str, dict[str, str]] = {"FR": FR, "EN": EN, "DE": DE, "ES": ES}


class Translator:
    """Fournit les libellés dans la langue courante."""

    def __init__(self, lang: str = "FR"):
        self.lang = lang if lang in _CATALOGS else "FR"

    def set_language(self, lang: str) -> str:
        """Change la langue courante et renvoie le code effectivement retenu."""
        self.lang = lang if lang in _CATALOGS else "FR"
        return self.lang

    def __call__(self, key: str, **kwargs) -> str:
        """Traduit une clé, avec repli en cascade puis sur la clé elle-même."""
        for code in (self.lang, "EN", "FR"):
            value = _CATALOGS.get(code, {}).get(key)
            if value:
                return value.format(**kwargs) if kwargs else value
        return key

    @property
    def is_english(self) -> bool:
        return self.lang == "EN"


#: Traducteur partagé par l'application.
t = Translator()


def missing_keys(lang: str) -> list[str]:
    """Clés absentes d'un catalogue, par rapport au français (outil de relecture)."""
    return sorted(set(FR) - set(_CATALOGS.get(lang, {})))
