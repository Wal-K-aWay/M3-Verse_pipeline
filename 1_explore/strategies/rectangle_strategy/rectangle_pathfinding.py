"""Path finding and trajectory generation utilities.

This module contains functions for calculating paths between rectangles,
generating trajectories, and optimizing routes.
"""

import math
from typing import List, Dict, Tuple, Set
from core.config import Rectangle
from ..utils import PathFinder, calculate_path_length, manhattan_distance
from .rectangle_distance import calculate_rectangle_distance_with_points

from .rectangle_visualization import visualize_inter_rectangle_paths


def _find_nearest_rectangles_with_closest_points(rect_idx: int, rectangles: List[Rectangle], 
                                               max_neighbors: int = 4) -> List[Tuple[int, Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """Find nearest rectangles to the given rectangle and return closest point pairs."""
    distances_with_points = []
    
    for i, other_rect in enumerate(rectangles):
        if i != rect_idx:
            distance, point_pair = calculate_rectangle_distance_with_points(rectangles[rect_idx], other_rect)
            distances_with_points.append((distance, i, point_pair))
    
    distances_with_points.sort(key=lambda x: x[0])
    
    if len(distances_with_points) == 0:
        return []
    
    threshold_index = min(max_neighbors - 1, len(distances_with_points) - 1)
    threshold = distances_with_points[threshold_index][0]
    
    return [(idx, point_pair) for distance, idx, point_pair in distances_with_points if distance <= threshold]
    threshold = distances_with_points[max_neighbors - 1][0]
    return [(idx, point_pair) for distance, idx, point_pair in distances_with_points if distance <= threshold]


def _calculate_shortest_path_between_rectangles(rect1_idx: int, rect2_idx: int, 
                                               rectangles: List[Rectangle],
                                               grid_size: float,
                                               pathfinder: PathFinder,
                                               closest_point_pair: Tuple[Tuple[float, float], Tuple[float, float]] = None) -> Tuple[List[Tuple[float, float]], Tuple[Tuple[float, float], Tuple[float, float]], float]:
    """Calculate shortest path between two rectangles using hint points and nearest contour points."""
    rect1 = rectangles[rect1_idx]
    rect2 = rectangles[rect2_idx]
    
    if closest_point_pair is None:
        return [], (None, None), float('inf')
    
    hint_point1, hint_point2 = closest_point_pair
    
    rect1_contour = rect1.get_boundary_points(grid_size)
    rect2_contour = rect2.get_boundary_points(grid_size)
    
    if not rect1_contour or not rect2_contour:
        return [], (None, None), float('inf')
        
    rect1_distances = []
    for p in rect1_contour:
        if p != hint_point1:
            manhattan_dist = manhattan_distance(p, hint_point1)
            rect1_distances.append((manhattan_dist, p))
    rect1_distances.sort(key=lambda x: x[0])
    selected_rect1_points = [hint_point1] + [p for _, p in rect1_distances[:4]]
    
    rect2_distances = []
    for p in rect2_contour:
        if p != hint_point2:
            manhattan_dist = manhattan_distance(p, hint_point2)
            rect2_distances.append((manhattan_dist, p))
    rect2_distances.sort(key=lambda x: x[0])
    selected_rect2_points = [hint_point2] + [p for _, p in rect2_distances[:4]]
    
    shortest_path = []
    best_endpoints = (None, None)
    min_distance = float('inf')
    
    for start_point in selected_rect1_points:
        for end_point in selected_rect2_points:
            try:
                path = pathfinder.find_path(start_point, end_point)
                if path:
                    distance = calculate_path_length(path)
                    if distance < min_distance:
                        min_distance = distance
                        shortest_path = path
                        best_endpoints = (start_point, end_point)
            except Exception:
                continue
    
    return shortest_path, best_endpoints, min_distance



def _optimize_rectangle_graph(inter_paths: Dict[Tuple[int, int], List[Tuple[float, float]]], 
                             best_endpoints: Dict[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]],
                             rectangles: List[Rectangle]) -> Tuple[Dict[Tuple[int, int], List[Tuple[float, float]]], Dict[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """Optimize rectangle graph using shortest path algorithms."""
    def _build_distance_matrix() -> List[List[float]]:
        n = len(rectangles)
        matrix = [[float('inf')] * n for _ in range(n)]
        
        for i in range(n):
            matrix[i][i] = 0.0
        
        for (i, j), path in inter_paths.items():
            if path:
                distance = calculate_path_length(path)
                matrix[i][j] = matrix[j][i] = distance
        
        return matrix

    def _floyd_warshall(matrix: List[List[float]]) -> List[List[float]]:
        """Apply Floyd-Warshall algorithm for shortest paths."""
        n = len(matrix)
        dist = [row[:] for row in matrix]
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        return dist
    
    def _identify_redundant_edges(floyd_matrix: List[List[float]]) -> Set[Tuple[int, int]]:
        redundant_edges = set()
        tolerance = 0.01
        
        for (i, j), path in inter_paths.items():
            if path:
                direct_distance = calculate_path_length(path)
                shortest_distance = floyd_matrix[i][j]
                
                if direct_distance > shortest_distance + tolerance:
                    redundant_edges.add((i, j))
        
        return redundant_edges

    distance_matrix = _build_distance_matrix()
    optimized_matrix = _floyd_warshall(distance_matrix)
    redundant_edges = _identify_redundant_edges(optimized_matrix)
    
    optimized_paths = {k: v for k, v in inter_paths.items() if k not in redundant_edges}
    optimized_endpoints = {k: v for k, v in best_endpoints.items() if k not in redundant_edges}
    
    # For now, return original paths (can be enhanced with actual optimization)
    return optimized_paths, optimized_endpoints


def get_inter_rectangle_paths(rectangles: List[Rectangle], 
                              filtered_reachable_coords: Set[Tuple[float, float]],
                              grid_size: float,
                              pathfinder: PathFinder,
                              max_neighbors: int = 4) -> Tuple[Dict[Tuple[int, int], List[Tuple[float, float]]], Dict[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """Get paths between rectangles."""
    if len(rectangles) < 2:
        print("Less than 2 rectangles, no need to calculate inter-rectangle paths")
        return {}, {}
    
    inter_paths = {}
    best_endpoints = {}
    calculated_pairs = set()
    
    for rect_idx in range(len(rectangles)):
        nearest_neighbors_with_points = _find_nearest_rectangles_with_closest_points(rect_idx, rectangles, max_neighbors)
        
        for neighbor_idx, closest_point_pair in nearest_neighbors_with_points:
            pair_key = (min(rect_idx, neighbor_idx), max(rect_idx, neighbor_idx))
            
            if pair_key not in calculated_pairs:
                calculated_pairs.add(pair_key)

                path, endpoints, distance = _calculate_shortest_path_between_rectangles(
                    rect_idx, neighbor_idx, rectangles, grid_size, pathfinder, closest_point_pair)
                
                if path and distance < float('inf'):
                    inter_paths[pair_key] = path
                    best_endpoints[pair_key] = endpoints
    

    ####
    # paths_list = list(inter_paths.values()) if inter_paths else []
    # visualize_inter_rectangle_paths(
    #     reachable_coords=filtered_reachable_coords,
    #     rectangles=rectangles,
    #     inter_rectangle_paths=paths_list,
    #     title="Inter-Rectangle Paths Visualization",
    #     save_path="inter_rectangle_paths_visualization.png"
    # )
    # import pdb;pdb.set_trace()
    ####

    optimized_paths, optimized_endpoints = _optimize_rectangle_graph(inter_paths, best_endpoints, rectangles)

    ####
    # paths_list = list(optimized_paths.values()) if optimized_paths else []
    # visualize_inter_rectangle_paths(
    #     reachable_coords=filtered_reachable_coords,
    #     rectangles=rectangles,
    #     inter_rectangle_paths=paths_list,
    #     title="Inter-Rectangle Paths Visualization",
    #     save_path="inter_rectangle_paths_visualization_optimized.png"
    # )
    # import pdb;pdb.set_trace()
    ####
    
    return optimized_paths, optimized_endpoints