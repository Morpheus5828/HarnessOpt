import networkx as nx
import numpy as np
from tqdm import tqdm
import pyvista as pv
import os

import networkx as nx
import numpy as np
from tqdm import tqdm
import pyvista as pv
import os


def generator(graph, source, target, N, sigma, strict_clamps=True, branch=None, clamp_segments=None, tolerance=50.0):
    all_variants = []
    all_clamp_stats = []

    target_unique_paths = branch if branch else N
    max_attempts = target_unique_paths * 5
    attempts = 0
    num_edges = graph.number_of_edges()

    pbar = tqdm(
        total=target_unique_paths,
        desc="🎲 Génération des variantes uniques",
        unit="chemin"
    )

    while len(all_variants) < target_unique_paths and attempts < max_attempts:
        attempts += 1

        # 1. Application du bruit (Noise) sur le graphe pour cette variante
        noises = np.random.normal(1.0, sigma, num_edges) if sigma > 0 else np.ones(num_edges)
        perturbed_weights = {}
        for idx, (u, v, data) in enumerate(graph.edges(data=True)):
            orig_weight = data.get("weight", 1.0)
            new_weight = max(0.001, orig_weight * noises[idx])
            perturbed_weights[(u, v)] = new_weight
            if not graph.is_directed():
                perturbed_weights[(v, u)] = new_weight

        def custom_weight(u, v, d):
            return perturbed_weights.get((u, v), 1.0)

        # 2. Initialisation du cheminement pour cette tentative
        current = source
        full_path = [source]
        remaining_clamps = list(clamp_segments) if clamp_segments else []
        visited_clamps_centers = []

        nbr_clamps_visites = 0

        if strict_clamps:
            while True:
                try:
                    # Calcul des distances One-Shot depuis la position actuelle et vers la destination
                    dist_from_current = nx.single_source_dijkstra_path_length(graph, current, weight=custom_weight)
                    dist_to_target = nx.single_source_dijkstra_path_length(graph, target, weight=custom_weight)
                except nx.NetworkXNoPath:
                    break

                if current not in dist_from_current or target not in dist_to_target:
                    break

                # Calcul du chemin de source (current) à destination sur le graphe
                dist_source_dest = dist_from_current[target]

                # Recherche du clamp le plus proche qui respecte tes critères
                valid_clamps = []
                for pair in remaining_clamps:
                    c_in, c_out = pair
                    if c_in in dist_from_current and c_out in dist_to_target:

                        # Calcul du centre physique du collier actuel pour le test des 110mm
                        c_center = (np.array(c_in) + np.array(c_out)) / 2.0

                        # Condition d'exclusion : pas à moins de 110mm des colliers déjà visités
                        too_close_to_old_clamp = False
                        for old_center in visited_clamps_centers:
                            if np.linalg.norm(c_center - old_center) < 110.0:
                                too_close_to_old_clamp = True
                                break

                        if too_close_to_old_clamp:
                            continue

                        # Vérification de la progression topologique vers la cible
                        if current != source:
                            if dist_to_target[c_out] >= dist_to_target[current]:
                                continue

                        # --- MODIFICATION CRITIQUE : DISTANCE EUCLIDIENNE ---
                        # Au lieu de prendre la distance du graphe, on calcule la vraie distance géométrique 3D
                        dist_euclienne_c1 = np.linalg.norm(np.array(current) - np.array(c_in))

                        # On garde quand même la distance du graphe pour le test de la tolérance d'arêtes
                        dist_graphe_c1 = dist_from_current[c_in]

                        # On stocke la distance euclidienne en premier élément pour le tri futur
                        valid_clamps.append((dist_euclienne_c1, dist_graphe_c1, c_in, c_out, c_center, pair))

                # Si aucun collier n'est disponible ou valide, on termine la boucle vers la destination
                if not valid_clamps:
                    break

                # --- TRI PAR PROXIMITÉ PHYSIQUE STRICTE ---
                # Le lambda x: x[0] cible le premier paramètre : dist_euclienne_c1
                valid_clamps.sort(key=lambda x: x[0])
                dist_eucl_c1, dist_graph_c1, c1_in, c1_out, c1_center, chosen_pair = valid_clamps[0]

                # On s'assure que le chemin sur le graphe reste sous le seuil de tolérance défini
                if dist_graph_c1 < dist_source_dest + tolerance:

                    path_to_c1 = nx.dijkstra_path(graph, current, c1_in, weight=custom_weight)
                    full_path.extend(path_to_c1[1:])
                    full_path.append(c1_out)

                    nbr_clamps_visites += 1

                    current = c1_out
                    visited_clamps_centers.append(c1_center)
                    remaining_clamps.remove(chosen_pair)
                else:
                    # Si le chemin du graphe pour atteindre ce collier physique est trop tortueux, on rompt la chaîne
                    break

        # Raccordement final vers la destination
        if current != target:
            try:
                path_to_dest = nx.dijkstra_path(graph, current, target, weight=custom_weight)
                full_path.extend(path_to_dest[1:])
            except nx.NetworkXNoPath:
                continue

        # Sauvegarde du chemin s'il est unique
        if full_path not in all_variants:
            all_variants.append(full_path)

            all_clamp_stats.append({
                "passe_par_clamps": nbr_clamps_visites > 0,
                "nombre_de_clamps": nbr_clamps_visites
            })

            pbar.update(1)

        pbar.set_postfix(essais=f"{attempts}/{max_attempts}", trouves=len(all_variants))

    pbar.close()

    return all_variants, all_clamp_stats
    """for idx, var in enumerate(all_variants):
        if isinstance(var, (list, np.ndarray)) and len(var) > 1:
            line_gen = pv.MultipleLines(points=np.array(var))
            tube_gen = line_gen.tube(radius=5)
            var_gen_path = os.path.join(f"variant_{idx:03d}_generator_raw.stl")
            tube_gen.save(var_gen_path)"""

