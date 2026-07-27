import numpy as np
import pyvista as pv
import vtk


def densify_points(pts, step_mm):
    """Densifie les segments droits pour avoir des points de test tous les X mm."""
    dense_pts = [np.array(pts[0])]
    for i in range(len(pts) - 1):
        p1, p2 = np.array(pts[i]), np.array(pts[i + 1])
        dist = np.linalg.norm(p2 - p1)
        num_steps = max(1, int(dist / step_mm))
        for j in range(1, num_steps + 1):
            dense_pts.append(p1 + (p2 - p1) * (j / num_steps))
    return np.array(dense_pts)



def magnetize_path_to_walls(path, mesh_global, rayon_harnais, distance_capture=15.0):
    """Plaque le chemin contre les cloisons si elles sont à portée (Capture native PyVista)."""
    optimized_path = []

    if mesh_global is None or len(path) == 0:
        return path

    # Calcul des normales si elles ne sont pas déjà en cache
    if 'Normals' not in mesh_global.cell_data:
        mesh_global.compute_normals(cell_normals=True, point_normals=False, inplace=True)

    normals = mesh_global.cell_normals

    for i, pt in enumerate(path):
        # 🎯 PYVISTA NATIF : Trouve l'ID de la cellule la plus proche ET le point d'impact 3D d'un coup
        c_id, closest_p = mesh_global.find_closest_cell(pt, return_closest_point=True)

        if c_id != -1:
            # Calcul de la distance linéaire réelle
            dist = np.linalg.norm(np.array(pt) - np.array(closest_p))

            if dist <= distance_capture:
                normale_face = normals[c_id]
                pt_impact = np.array(closest_p)

                if i < len(path) - 1:
                    tangent = np.array(path[i + 1]) - np.array(pt)
                else:
                    tangent = np.array(pt) - np.array(path[i - 1])

                norm_tangent = np.linalg.norm(tangent)
                if norm_tangent > 1e-6:
                    tangent /= norm_tangent
                    dot_product = np.abs(np.dot(tangent, normale_face))

                    # Si le câble avance parallèlement à la tôle, on l'y plaque à distance réglementaire
                    if dot_product <= 0.4:
                        point_ideal = pt_impact + normale_face * (rayon_harnais + 3.0)
                        optimized_path.append(point_ideal)
                        continue

        optimized_path.append(pt)

    return optimized_path
