import pickle
import numpy as np
import pyvista as pv
import networkx as nx
import argparse
from pathlib import Path
from scipy.spatial import cKDTree


def visualize(graph_path, stl_path=None, paths_path=None, clamp_dir=None, valid_points_path=None, source=None,
              destination=None, branche=None, clamp_pts=None):
    pl = pv.Plotter(title="HarnessOPT - Viewer 3D (Light Mode)")
    pl.set_background("white")

    print(f"-> Chargement du graphe: {graph_path}")
    if not Path(graph_path).exists():
        print(f"Erreur: Fichier {graph_path} introuvable.")
        return

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    nodes = list(G.nodes(data=True))

    pl.add_text(f"🕸️ Graphe : {len(nodes):,} noeuds", position='upper_right', color='black', font_size=12)

    path_status_msg = ""
    if source is not None and destination is not None and nodes:
        print("\n--- 🔍 Vérification de la connectivité ---")
        nodes_coords_only = np.array([n[0] for n in nodes])
        tree = cKDTree(nodes_coords_only)

        _, src_idx = tree.query(source)
        _, dst_idx = tree.query(destination)

        closest_src = tuple(nodes_coords_only[src_idx])
        closest_dst = tuple(nodes_coords_only[dst_idx])

        try:
            if nx.has_path(G, closest_src, closest_dst):
                print(f"✅ OUI : Un chemin existe entre la Source et la Target !")
                path_status_msg = "Chemin Possible : OUI"
                text_color = "green"
            else:
                print(f"❌ NON : Le réseau est coupé entre la Source et la Target.")
                path_status_msg = "Chemin Possible : NON (Reseau coupe)"
                text_color = "red"
        except nx.NodeNotFound:
            path_status_msg = "Erreur : Noeuds introuvables"
            text_color = "red"

        pl.add_text(path_status_msg, position='upper_left', color=text_color, font_size=14)

    graph_actor = None
    if nodes:
        node_coords = np.array([n[0] for n in nodes], dtype=float)
        node_colors = [n[1].get('color', 'black') for n in nodes]
        node_colors = ['gray' if c.lower() == 'white' else c for c in node_colors]

        node_to_idx = {n[0]: i for i, n in enumerate(nodes)}

        lines = []
        for u, v in G.edges():
            lines.append([2, node_to_idx[u], node_to_idx[v]])

        node_mesh = pv.PolyData(node_coords)
        if lines:
            node_mesh.lines = np.hstack(lines)

        graph_actor = pl.add_mesh(
            node_mesh, scalars=node_colors, rgba=False, preference='point',
            point_size=3, render_points_as_spheres=True, opacity=0.15, label="Graphe (Points & Edges)"
        )

    if stl_path and Path(stl_path).exists():
        struct_mesh = pv.read(stl_path)
        pl.add_mesh(struct_mesh, opacity=0.1, color="black", style="wireframe", label="Structure")

    if paths_path and Path(paths_path).exists():
        try:
            path_geometry = pv.read(paths_path)
            pl.add_mesh(path_geometry, color="blue", opacity=1.0, label="Harnais Final", smooth_shading=True)
        except Exception:
            pass

    print(f"Dossier Clamps CAO (STL) : {clamp_dir}")
    if clamp_dir:
        clamp_path = Path(clamp_dir)
        if clamp_path.exists() and clamp_path.is_dir():
            clamp_files = list(clamp_path.glob("*.stl")) + list(clamp_path.glob("*.STL"))
            if clamp_files:
                added_legend = False
                for cf in clamp_files:
                    try:
                        clamp_mesh = pv.read(cf)
                        label_str = "Clamps" if not added_legend else None
                        pl.add_mesh(clamp_mesh, color="orange", opacity=0.9, smooth_shading=True, label=label_str)
                        added_legend = True
                    except:
                        pass

    # Chargement de la grille de Voxels verte
    if valid_points_path:
        v_path = Path(valid_points_path)
        if v_path.exists():
            try:
                data = np.load(v_path, allow_pickle=True)
                points = data["valid_points"]
                if points.ndim == 2 and points.shape[1] == 3:
                    cloud = pv.PolyData(points)
                    sphere_geom = pv.Sphere(radius=5.0)
                    spheres = cloud.glyph(geom=sphere_geom, scale=False)
                    pl.add_mesh(spheres, color="green", opacity=0.6, smooth_shading=True, label="Valid Points (5mm)")
                    pl.add_text(f"🧊 Cache (Voxel) : {len(points):,} pts", position='lower_right', color='green', font_size=12)
            except:
                pass

    # =========================================================================
    # 🟢 SÉCURITÉ ET AUTO-CHARGEMENT DES POINTS DE CLAMPS (BLEUS)
    # =========================================================================
    if (clamp_pts is None or len(clamp_pts) == 0) and valid_points_path:
        v_path = Path(valid_points_path)
        if v_path.exists():
            try:
                data = np.load(v_path, allow_pickle=True)
                if "clamp_points" in data:
                    clamp_data = data["clamp_points"]
                    if clamp_data.size > 0:
                        segments = clamp_data.tolist()
                        clamp_pts = []
                        for seg in segments:
                            if isinstance(seg, dict) and "in" in seg and "out" in seg:
                                clamp_pts.append(seg["in"])
                                clamp_pts.append(seg["out"])
                        print(f"🛰️ [AUTO-LOAD] {len(clamp_pts) // 2} paires de terminaux bleus chargées depuis le cache.")
            except Exception as e:
                print(f"⚠️ Échec de l'auto-chargement des points de clamps : {e}")

    if clamp_pts is not None and len(clamp_pts) > 0:
        pts_arr = np.asarray(clamp_pts, dtype=float)

        colors_arr = np.full((len(pts_arr), 3), [0, 0, 255], dtype=np.uint8)
        colors_arr[1::2] = [255, 0, 0]

        pl.add_points(
            pts_arr,
            color="blue",
            scalars=colors_arr,
            rgb=True,
            point_size=15,
            render_points_as_spheres=True,
            label="Points de Clamps (In/Out)"
        )

    pts_spec = {"SOURCE": source, "DESTINATION": destination, "BRANCHE": branche}
    for nom, coord in pts_spec.items():
        if coord is not None:
            pt = np.array([coord], dtype=float)
            pl.add_points(pt, color="magenta", point_size=15, render_points_as_spheres=True)
            pl.add_point_labels(pt, [nom], font_size=12, text_color="red")

    pl.add_legend(bcolor='lightgray')

    def toggle_graph():
        if graph_actor:
            graph_actor.SetVisibility(not graph_actor.GetVisibility())
            pl.render()

    pl.add_key_event("g", toggle_graph)
    pl.add_axes(line_width=2, color="black")
    pl.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--stl_path", type=str, default=None)
    parser.add_argument("--paths_path", type=str, default=None)
    parser.add_argument("--clamp", type=str, default=None)
    parser.add_argument("--valid_points", type=str, default=None)

    def coord(s):
        return [float(item) for item in s.split(',')]

    parser.add_argument("--source", type=coord, default=None)
    parser.add_argument("--destination", type=coord, default=None)
    parser.add_argument("--branche", type=coord, default=None)

    args = parser.parse_args()

    visualize(
        graph_path=args.graph_path,
        stl_path=args.stl_path,
        paths_path=args.paths_path,
        clamp_dir=args.clamp,
        valid_points_path=args.valid_points,
        source=args.source,
        destination=args.destination,
        branche=args.branche
    )
