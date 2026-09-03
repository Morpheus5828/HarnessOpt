"""Balayage de réglages, sans interface : quels paramètres sur *votre* DMU.

La calibration (:mod:`core.calibration`) donne un point de départ cohérent.
Elle ne peut pas savoir si votre cellule est encombrée, si vos passages sont
étroits, ni combien d'itérations votre machine vous laisse. Ce script le
mesure : il lance le cheminement sur plusieurs jeux de réglages et les classe
sur ce qui compte — la trajectoire est-elle admissible, et à quel prix.

Chaque essai tourne le même budget d'itérations, sur le même DMU, avec les
mêmes extrémités : c'est la seule façon de comparer. Le classement retenu est
celui de l'application elle-même (``RouteReport.score``), de sorte que le
gagnant du balayage est aussi celui que l'application choisirait.

Usage::

    python tools/sweep.py --mesh maquette.stl --from 0,0,0 --to 2015,0,0 \\
        --diameter 20 --seconds 60 --out balayage.csv

Sans ``--mesh``, le script explique ce qu'il attend et s'arrête : un balayage
sur une maquette inventée ne dirait rien de votre cellule.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from core.calibration import calibrate  # noqa: E402

#: Facteurs appliqués autour de la valeur calibrée. Trois valeurs par axe : la
#: calibration, plus grossier, plus fin. Au-delà, le nombre d'essais explose
#: sans rien apprendre — chaque essai coûte une minute de calcul.
FACTORS = (0.7, 1.0, 1.4)


def build_grid(base, axes):
    """Jeux de réglages à essayer, autour de la calibration."""
    combos = []
    for values in itertools.product(*(FACTORS for _ in axes)):
        trial = dict(base)
        for axis, factor in zip(axes, values):
            if axis == "initial_points":
                trial[axis] = max(8, int(round(base[axis] * factor)))
                trial["max_points"] = int(trial[axis] * 2.5)
            else:
                trial[axis] = round(base[axis] * factor, 2)
        if trial not in combos:
            combos.append(trial)
    return combos


def run_one(mesh, start, goal, rules, config, seconds):
    """Un essai : renvoie le rapport final, ou ``None`` s'il n'a rien produit."""
    from controller.app_controller import AppController
    from core.agent_team import build_benchmark_algos
    from core.agent_worker import algo_worker
    from core.orchestrator import Orchestrator
    from core.path_planner import PlannerSettings, plan_route

    plan = plan_route(mesh, start, goal, rules.clearance.default_min_mm,
                      rules.clearance.max_mm, PlannerSettings(),
                      num_points=config["initial_points"])
    if not plan.success or len(plan.points) < 3:
        return None
    route = plan.points

    orchestrator = Orchestrator("balanced", temperature=0.5)
    algos = build_benchmark_algos(orchestrator.team, max_points=config["max_points"])
    name = list(algos)[0]
    spec = {s.name: s for s in orchestrator.team}.get(name)

    shared = {"is_playing": True, "is_running": True, "config": dict(config),
              "algos": {name: AppController._blank_agent_state(np.asarray(route, np.float32))}}
    locks = (threading.Lock(), threading.Lock())
    threading.Thread(
        target=lambda: (time.sleep(seconds), shared.__setitem__("is_running", False)),
        daemon=True,
    ).start()
    algo_worker(name, algos, np.asarray(route, np.float32), mesh, shared,
                locks[0], locks[1], np.asarray(start), np.asarray(goal),
                rules=rules, spec=spec)
    return shared["algos"][name]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh", help="maquette (STL, PLY, OBJ…)")
    parser.add_argument("--from", dest="start", help="départ, x,y,z en mm")
    parser.add_argument("--to", dest="goal", help="arrivée, x,y,z en mm")
    parser.add_argument("--diameter", type=float, default=20.0, help="diamètre du toron")
    parser.add_argument("--seconds", type=float, default=60.0, help="durée par essai")
    parser.add_argument("--axes", default="initial_points,max_step_mm,min_margin",
                        help="réglages à faire varier, séparés par des virgules")
    parser.add_argument("--out", default="balayage.csv")
    args = parser.parse_args(argv)

    if not args.mesh or not args.start or not args.goal:
        parser.print_help()
        print("\nIl faut au minimum --mesh, --from et --to : un balayage sur une "
              "maquette inventée ne dirait rien de votre cellule.")
        return 2

    import trimesh

    from core.routing_rules import ClearanceModel, HarnessSpec, RoutingRules

    mesh = trimesh.load_mesh(args.mesh, force="mesh")
    start = np.array([float(v) for v in args.start.split(",")])
    goal = np.array([float(v) for v in args.goal.split(",")])

    base = calibrate(float(np.linalg.norm(goal - start)), diameter_mm=args.diameter)
    print(base.report())
    print()

    from core.agent.config import CONFIG

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    grid = build_grid(base.as_config(), axes)
    print(f"{len(grid)} essai(s) de {args.seconds:.0f} s "
          f"≈ {len(grid) * args.seconds / 60:.0f} min au total.\n")

    lignes = []
    classement = []
    for index, trial in enumerate(grid, start=1):
        config = dict(CONFIG)
        config.update(trial)
        config.setdefault("crabe_stl_path", "")
        config.setdefault("existing_crabes", [])
        config.setdefault("mandatory_combs", [])
        config.setdefault("pinned_points", [])
        config["harness_diameter"] = args.diameter
        rules = RoutingRules(
            harness=HarnessSpec(diameter_mm=args.diameter),
            clearance=ClearanceModel(default_min_mm=config["min_margin"],
                                     max_mm=config["max_margin"]),
        )
        config["min_bend_radius"] = rules.harness.min_bend_radius_mm

        print(f"[{index}/{len(grid)}] " + ", ".join(f"{a}={trial[a]}" for a in axes),
              end=" ... ", flush=True)
        state = run_one(mesh, start, goal, rules, config, args.seconds)
        if state is None or state.get("report") is None:
            print("aucun tracé")
            continue

        report = state["report"]
        kpis = report.kpis
        # Le score est un tuple lexicographique — plus petit, meilleur. On le
        # garde tel quel pour trier, et on ne le recopie pas dans le CSV : les
        # colonnes qui suivent en disent la même chose, en lisible.
        classement.append(tuple(state.get("score") or ()))
        lignes.append({
            **{a: trial[a] for a in axes},
            "admissible": int(bool(report.is_deliverable)),
            "conforme": int(bool(report.is_compliant)),
            "iterations": state.get("iteration", 0),
            "longueur_mm": round(kpis.get("length_mm", 0.0), 1),
            "rayon_min_mm": round(kpis.get("min_bend_radius_mm", 0.0), 1),
            "marge_min_mm": round(kpis.get("min_distance_mm", 0.0), 1),
            "clashs": kpis.get("n_clashes", 0),
            "zigzags": kpis.get("n_zigzags", 0),
        })
        print(f"{'admissible' if report.is_deliverable else 'NON admissible'}, "
              f"{kpis.get('length_mm', 0):.0f} mm")

    if not lignes:
        print("\nAucun essai n'a produit de tracé : vérifiez les extrémités.")
        return 1

    # Le classement de l'application, à l'identique : c'est tout l'intérêt du
    # balayage que son gagnant soit aussi celui que l'application choisirait.
    # Le score hiérarchise déjà les clashs avant la longueur ; le trier « à la
    # main » sur la longueur aurait désigné un autre gagnant que l'application.
    ordre = sorted(range(len(lignes)), key=lambda i: classement[i])
    lignes = [lignes[i] for i in ordre]
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lignes[0]))
        writer.writeheader()
        writer.writerows(lignes)

    print(f"\n{len(lignes)} essai(s) écrits dans {args.out}")

    if not any(ligne["admissible"] for ligne in lignes):
        # Un classement entre trajectoires toutes inadmissibles ne désigne pas
        # de bon réglage : il désigne la moins mauvaise, ce qui n'est pas la
        # même chose. Le dire, plutôt que laisser lire un gagnant.
        print(f"\nAucun essai n'est admissible en {args.seconds:.0f} s. Le classement "
              "ci-dessous ne compare que des trajectoires à corriger : relancez "
              "avec --seconds plus large (300 s est un ordre de grandeur "
              "raisonnable sur un faisceau de 2 m) avant d'en tirer un réglage.")

    print("\nMeilleur jeu de réglages :")
    for cle, valeur in lignes[0].items():
        print(f"  {cle:16s} {valeur}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
