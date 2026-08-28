"""
def process_global_chunk(
        indices_chunk,
        current_locator,
        base_clearance_sq,
        max_ray_dist
):
    local_closest_p = [0.0] * 3
    local_cell_id, local_sub_id = vtk.reference(0), vtk.reference(0)
    local_d2 = vtk.reference(0.0)

    # Variables VTK pour le Raycasting
    t_int = vtk.reference(0.0)
    x_int = [0.0, 0.0, 0.0]
    pcoords = [0.0, 0.0, 0.0]
    sub_id_int = vtk.reference(0)

    chunk_results = np.ones(len(indices_chunk), dtype=bool)
    chunk_tc = np.zeros(len(indices_chunk), dtype=bool)
    chunk_tf = np.zeros(len(indices_chunk), dtype=bool)

    # Les 6 directions cardinales (X, Y, Z)
    directions = [
        np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    ]

    for i, idx in enumerate(indices_chunk):
        pt = points_in_bbox[idx]

        # A) Filtre strict : Trop près (collision avec structure)
        current_locator.FindClosestPoint(pt, local_closest_p, local_cell_id, local_sub_id, local_d2)
        if float(local_d2) < base_clearance_sq:
            chunk_results[i] = False
            chunk_tc[i] = True
            continue

        # B) Filtre Raycast : Flotte dans le vide
        is_supported = False
        pt_v = (float(pt[0]), float(pt[1]), float(pt[2]))

        for dir_vec in directions:
            end_pt = pt + dir_vec * max_ray_dist
            end_v = (float(end_pt[0]), float(end_pt[1]), float(end_pt[2]))

            # On tire un laser : si on touche qqchose != 0, le point est soutenu !
            if current_locator.IntersectWithLine(pt_v, end_v, 0.001, t_int, x_int, pcoords, sub_id_int) != 0:
                is_supported = True
                break

        if not is_supported:
            chunk_results[i] = False
            chunk_tf[i] = True

    return indices_chunk, chunk_results, chunk_tc, chunk_tf


def process_color_chunk(
        indices_chunk,
        current_locator,
        color_clearance_sq
):
    local_closest_p = [0.0] * 3
    local_cell_id, local_sub_id = vtk.reference(0), vtk.reference(0)
    local_d2 = vtk.reference(0.0)

    chunk_results = np.ones(len(indices_chunk), dtype=bool)
    chunk_cc = np.zeros(len(indices_chunk), dtype=bool)

    for i, idx in enumerate(indices_chunk):
        pt = points_in_bbox[idx]
        current_locator.FindClosestPoint(pt, local_closest_p, local_cell_id, local_sub_id, local_d2)
        if float(local_d2) <= color_clearance_sq:
            chunk_results[i] = False
            chunk_cc[i] = True

    return indices_chunk, chunk_results, chunk_cc
"""

import os
import time
import json
import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_global_chunk(indices_chunk, pts_subset, current_locator, global_mesh, base_clearance_sq, max_ray_dist):
    """
    Exécute les vérifications de collision et de support sur un lot de points.
    """
    local_closest_p = [0.0] * 3
    local_cell_id, local_sub_id = vtk.reference(0), vtk.reference(0)
    local_d2 = vtk.reference(0.0)

    # Variables VTK pour le Raycasting
    t_int = vtk.reference(0.0)
    x_int = [0.0, 0.0, 0.0]
    pcoords = [0.0, 0.0, 0.0]
    sub_id_int = vtk.reference(0)

    chunk_results = np.ones(len(indices_chunk), dtype=bool)
    chunk_tc = np.zeros(len(indices_chunk), dtype=bool)
    chunk_tf = np.zeros(len(indices_chunk), dtype=bool)

    # Directions de support (6 directions cardinales)
    directions = [
        np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    ]

    # Extraction locale des coordonnées pour ce chunk afin d'éviter les verrous de thread
    chunk_points = pts_subset[indices_chunk]

    # --- SÉCURITÉ COLLISION INTERNE (SOLID MESH) ---
    # On élimine d'emblée les points piégés à l'intérieur d'un volume fermé
    poly_pts = pv.PolyData(chunk_points)
    interior_mask = poly_pts.select_interior_points(global_mesh, check_surface=False)["selected_points"].astype(bool)

    for i, pt in enumerate(chunk_points):
        # Si le point est physiquement confiné dans la matière CAO -> Élimination directe
        if interior_mask[i]:
            chunk_results[i] = False
            chunk_tc[i] = True
            continue

        # A) Filtre de proximité de surface (Rayon de clearance)
        current_locator.FindClosestPoint(pt, local_closest_p, local_cell_id, local_sub_id, local_d2)
        if float(local_d2) < base_clearance_sq:
            chunk_results[i] = False
            chunk_tc[i] = True
            continue

        # B) Filtre Raycast : Détection du support pour éviter la flottaison
        is_supported = False
        pt_v = (float(pt[0]), float(pt[1]), float(pt[2]))

        for dir_vec in directions:
            end_pt = pt + dir_vec * max_ray_dist
            end_v = (float(end_pt[0]), float(end_pt[1]), float(end_pt[2]))

            if current_locator.IntersectWithLine(pt_v, end_v, 0.001, t_int, x_int, pcoords, sub_id_int) != 0:
                is_supported = True
                break

        if not is_supported:
            chunk_results[i] = False
            chunk_tf[i] = True

    return indices_chunk, chunk_results, chunk_tc, chunk_tf


def process_color_chunk(indices_chunk, pts_subset, current_locator, color_mesh, color_clearance_sq):
    """
    Exécute les règles métier ATA (distances minimales spécifiques par couleur).
    """
    local_closest_p = [0.0] * 3
    local_cell_id, local_sub_id = vtk.reference(0), vtk.reference(0)
    local_d2 = vtk.reference(0.0)

    chunk_results = np.ones(len(indices_chunk), dtype=bool)
    chunk_cc = np.zeros(len(indices_chunk), dtype=bool)

    chunk_points = pts_subset[indices_chunk]

    # Sécurité d'inclusion pour la couleur spécifique
    poly_pts = pv.PolyData(chunk_points)
    interior_mask = poly_pts.select_interior_points(color_mesh, check_surface=False)["selected_points"].astype(bool)

    for i, pt in enumerate(chunk_points):
        if interior_mask[i]:
            chunk_results[i] = False
            chunk_cc[i] = True
            continue

        current_locator.FindClosestPoint(pt, local_closest_p, local_cell_id, local_sub_id, local_d2)
        if float(local_d2) <= color_clearance_sq:
            chunk_results[i] = False
            chunk_cc[i] = True

    return indices_chunk, chunk_results, chunk_cc
