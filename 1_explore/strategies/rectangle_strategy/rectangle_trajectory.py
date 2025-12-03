"""Generates a trajectory for exploration based on a list of rectangles."""

import math
from typing import List, Tuple, Set, Dict, Optional
from collections import defaultdict

from ..utils import manhattan_distance, calculate_path_length, PathFinder
from core.config import ExplorationConfig, Rectangle
from .rectangle_pathfinding import get_inter_rectangle_paths

from .rectangle_visualization import visualize_sample_points, visualize_inter_rectangle_paths


def generate_trajectory_from_rectangles(config: ExplorationConfig, filtered_reachable_coords: Set[Tuple[float, float]], rectangles: List[Rectangle], start_position: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Generate a trajectory based on the rectangles."""
    
    if not rectangles:
        return [start_position]

    pathfinder = PathFinder(filtered_reachable_coords, config.gridSize)
    sample_points = get_sample_points(filtered_reachable_coords, rectangles, config.gridSize, pathfinder)
    # visualize_sample_points(filtered_reachable_coords=filtered_reachable_coords, intersection_points=sample_points, save_path='sample_points.png')
    # import pdb;pdb.set_trace()

    all_points = [start_position] + sample_points
    graph = build_point_graph(all_points, pathfinder)
    optimal_order = solve_tsp_for_points(graph, all_points)
    complete_trajectory = generate_complete_trajectory(all_points, optimal_order, start_position, pathfinder)
    
    return complete_trajectory

def get_sample_points(filtered_reachable_coords: Set[Tuple[float, float]], rectangles: List[Rectangle], grid_size: float, pathfinder: PathFinder) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    inter_rectangle_paths, inter_rectangle_path_endpoints = get_inter_rectangle_paths(rectangles, filtered_reachable_coords, grid_size, pathfinder)
    # import pdb;pdb.set_trace()
    # visualize_inter_rectangle_paths(filtered_reachable_coords, rectangles, inter_rectangle_paths)
    
    intersection_points = _extract_intersection_points(filtered_reachable_coords, inter_rectangle_paths, inter_rectangle_path_endpoints)
    corner_points = _extract_rectangle_corner_points(filtered_reachable_coords, rectangles)
    contour_points, path_points = _extract_contour_and_path_points(filtered_reachable_coords, rectangles, inter_rectangle_paths, grid_size, 3)
    # visualize_sample_points(filtered_reachable_coords, rectangles, intersection_points, corner_points, contour_points, path_points)
    
    all_sample_points = intersection_points + corner_points + contour_points + path_points
    
    unique_points = list(set(all_sample_points))
    
    # filtered_points = _filter_dense_points(unique_points, grid_size, 2) 
    filtered_points = _filter_dense_points(unique_points, grid_size, 2, 
                                         intersection_points, corner_points, 
                                         contour_points, path_points)
    # visualize_sample_points(filtered_reachable_coords=filtered_reachable_coords, intersection_points=filtered_points)
    # import pdb;pdb.set_trace()
    
    return filtered_points

def _extract_intersection_points(filtered_reachable_coords: Set[Tuple[float, float]], inter_rectangle_paths: Dict[Tuple[int, int], List[Tuple[float, float]]], inter_rectangle_path_endpoints: Dict[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """Extract intersection points between rectangles."""
    intersection_points = set()
    
    if not inter_rectangle_paths or not inter_rectangle_path_endpoints:
        return []
    
    for (rect_i, rect_j), (start_point, end_point) in inter_rectangle_path_endpoints.items():
        if start_point in filtered_reachable_coords:
            intersection_points.add(start_point)
        if end_point in filtered_reachable_coords:
            intersection_points.add(end_point)
    
    result = list(intersection_points)
    return result

def _extract_rectangle_corner_points(filtered_reachable_coords: Set[Tuple[float, float]], rectangles: List[Rectangle]) -> List[Tuple[float, float]]:
    corner_points = set()
    
    for rect in rectangles:
        corners = rect.get_corners()
        
        for corner in corners:
            if corner in filtered_reachable_coords:
                corner_points.add(corner)
    
    result = list(corner_points)
    return result

def _extract_contour_and_path_points(filtered_reachable_coords: Set[Tuple[float, float]], rectangles: List[Rectangle], inter_rectangle_paths: Dict[Tuple[int, int], List[Tuple[float, float]]], grid_size: float, point_spaceing: int = 3) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:

    contour_points = _extract_contour_points_from_rectangles(filtered_reachable_coords, rectangles, grid_size, point_spaceing)
    path_points_list = _extract_contour_points_from_inter_paths(filtered_reachable_coords, inter_rectangle_paths, grid_size, point_spaceing)
    
    return contour_points, path_points_list

def _filter_dense_points(all_points: List[Tuple[float, float]], grid_size: float, min_grid_spacing: Optional[int] = 1,
                        intersection_points: List[Tuple[float, float]] = None,
                        corner_points: List[Tuple[float, float]] = None,
                        contour_points: List[Tuple[float, float]] = None,
                        path_points: List[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
    """Filter dense points based on grid spacing with priority-based removal.
    """
    if not all_points:
        return []
    
    if intersection_points is None or corner_points is None or contour_points is None or path_points is None:
        min_distance = min_grid_spacing * grid_size
        filtered_points = []
        for point in all_points:
            too_close = False
            for selected_point in filtered_points:
                distance = math.sqrt((point[0] - selected_point[0])**2 + (point[1] - selected_point[1])**2)
                if distance < min_distance:
                    too_close = True
                    break
            if not too_close:
                filtered_points.append(point)
        return filtered_points
    
    min_distance = min_grid_spacing * grid_size
    
    intersection_set = set(intersection_points)
    corner_set = set(corner_points)
    contour_set = set(contour_points)
    path_set = set(path_points)
    
    def get_priority(point):
        if point in intersection_set:
            return 4
        elif point in corner_set:
            return 3
        elif point in path_set and point not in intersection_set:
            return 2
        elif point in contour_set and point not in corner_set:
            return 1
        else:
            return 0
    
    sorted_points = sorted(all_points, key=get_priority, reverse=True)
    
    filtered_points = []
    
    for point in sorted_points:
        too_close = False
        for selected_point in filtered_points:
            distance = math.sqrt((point[0] - selected_point[0])**2 + (point[1] - selected_point[1])**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            filtered_points.append(point)
    
    return filtered_points


def build_point_graph(points: List[Tuple[float, float]], pathfinder: PathFinder) -> Dict[int, Dict[int, float]]:
    """Build graph based on points with pathfinding distances."""

    graph = defaultdict(dict)
    n = len(points)
    
    for i in range(n):
        for j in range(i + 1, n):
            path = pathfinder.find_path(points[i], points[j])
            if path:
                distance = calculate_path_length(path)
                graph[i][j] = distance
                graph[j][i] = distance
            else:
                graph[i][j] = float('inf')
                graph[j][i] = float('inf')
    
    return graph

def solve_tsp_for_points(graph: Dict[int, Dict[int, float]], points: List[Tuple[float, float]]) -> List[int]:
    """Solve TSP problem for the points using greedy approach."""
    n = len(points)
    if n <= 1:
        return list(range(n))
    
    visited = [False] * n
    current = 0  # 从起始位置开始
    visited[0] = True
    tour = [0]
    
    for _ in range(n - 1):
        min_dist = float('inf')
        next_node = -1
        
        for j in range(n):
            if not visited[j] and graph[current].get(j, float('inf')) < min_dist:
                min_dist = graph[current].get(j, float('inf'))
                next_node = j
        
        if next_node != -1:
            visited[next_node] = True
            tour.append(next_node)
            current = next_node
    
    return tour

def generate_complete_trajectory(points: List[Tuple[float, float]], optimal_order: List[int], start_position: Tuple[float, float], pathfinder: PathFinder) -> List[Tuple[float, float]]:
    """Generate complete trajectory by connecting points with pathfinding."""
    complete_trajectory = []
    current_pos = start_position
    
    for point_idx in optimal_order:
        target_pos = points[point_idx]
        
        if abs(current_pos[0] - target_pos[0]) < 1e-6 and abs(current_pos[1] - target_pos[1]) < 1e-6:
            if not complete_trajectory or complete_trajectory[-1] != target_pos:
                complete_trajectory.append(target_pos)
            current_pos = target_pos
            continue
        
        path = pathfinder.find_path(current_pos, target_pos)
        
        if path:
            for i, point in enumerate(path):
                if i == 0 and complete_trajectory and complete_trajectory[-1] == point:
                    continue  # 跳过重复的起始点
                complete_trajectory.append(point)
            current_pos = target_pos
        else:
            print(f"Warning: Cannot find path from {current_pos} to {target_pos}")
            complete_trajectory.append(target_pos)
            current_pos = target_pos
    
    return complete_trajectory


def _extract_contour_points_from_rectangles(filtered_reachable_coords: Set[Tuple[float, float]], rectangles: List[Rectangle], grid_size: float, point_spacing: Optional[int] = None) -> List[Tuple[float, float]]:
    """Extract points from rectangle contours by clipping based on length."""
    def _get_contour_points_form_single_rectangle(filtered_reachable_coords: Set[Tuple[float, float]], rect: Rectangle, point_spacing: float) -> List[Tuple[float, float]]:
        """Get contour points for a single rectangle by clipping based on length."""
        contour_points = []
        
        width = rect.width()
        height = rect.height()
        
        if width < 1e-6 and height < 1e-6:
            # 点矩形
            contour_points.append((rect.x_min, rect.z_min))
        elif width < 1e-6 or height < 1e-6:
            # 线段矩形
            if width < 1e-6:  # 垂直线
                # 先添加两个端点
                start_point = (round(rect.x_min, 2), round(rect.z_min, 2))
                end_point = (round(rect.x_min, 2), round(rect.z_max, 2))
                if start_point in filtered_reachable_coords:
                    contour_points.append(start_point)
                if end_point in filtered_reachable_coords and end_point != start_point:
                    contour_points.append(end_point)
                
                # 再添加中间的点
                z = rect.z_min + point_spacing
                while z < rect.z_max - 1e-6:
                    point = (round(rect.x_min, 2), round(z, 2))
                    if point in filtered_reachable_coords:
                        contour_points.append(point)
                    z = round(z + point_spacing, 2)
            else:  # 水平线
                # 先添加两个端点
                start_point = (round(rect.x_min, 2), round(rect.z_min, 2))
                end_point = (round(rect.x_max, 2), round(rect.z_min, 2))
                if start_point in filtered_reachable_coords:
                    contour_points.append(start_point)
                if end_point in filtered_reachable_coords and end_point != start_point:
                    contour_points.append(end_point)
                
                # 再添加中间的点
                x = rect.x_min + point_spacing
                while x < rect.x_max - 1e-6:
                    point = (round(x, 2), round(rect.z_min, 2))
                    if point in filtered_reachable_coords:
                        contour_points.append(point)
                    x = round(x + point_spacing, 2)
        else:
            corners = [
                (round(rect.x_min, 2), round(rect.z_max, 2)),  # 左上角
                (round(rect.x_max, 2), round(rect.z_max, 2)),  # 右上角
                (round(rect.x_max, 2), round(rect.z_min, 2)),  # 右下角
                (round(rect.x_min, 2), round(rect.z_min, 2))   # 左下角
            ]
            
            for corner in corners:
                if corner in filtered_reachable_coords:
                    contour_points.append(corner)

            x = rect.x_min + point_spacing
            while x < rect.x_max - 1e-6:
                point = (round(x, 2), round(rect.z_max, 2))
                if point in filtered_reachable_coords:
                    contour_points.append(point)
                x = round(x + point_spacing, 2)
            
            z = rect.z_max - point_spacing
            while z > rect.z_min + 1e-6:
                point = (round(rect.x_max, 2), round(z, 2))
                if point in filtered_reachable_coords:
                    contour_points.append(point)
                z = round(z - point_spacing, 2)
            
            x = rect.x_max - point_spacing
            while x > rect.x_min + 1e-6:
                point = (round(x, 2), round(rect.z_min, 2))
                if point in filtered_reachable_coords:
                    contour_points.append(point)
                x = round(x - point_spacing, 2)
            
            z = rect.z_min + point_spacing
            while z < rect.z_max - 1e-6:
                point = (round(rect.x_min, 2), round(z, 2))
                if point in filtered_reachable_coords:
                    contour_points.append(point)
                z = round(z + point_spacing, 2)
        
        return contour_points

    if point_spacing is None:
        point_spacing = 1
    
    actual_spacing = grid_size * point_spacing
    
    all_contour_points = []
    
    for rect in rectangles:
        contour_points = _get_contour_points_form_single_rectangle(filtered_reachable_coords, rect, actual_spacing)
        all_contour_points.extend(contour_points)
    
    seen = set()
    unique_points = []
    for point in all_contour_points:
        rounded_point = (round(point[0], 2), round(point[1], 2))
        if rounded_point not in seen:
            seen.add(rounded_point)
            unique_points.append(rounded_point)
        
    return unique_points

def _extract_contour_points_from_inter_paths(filtered_reachable_coords: Set[Tuple[float, float]], inter_rectangle_paths: Dict[Tuple[int, int], List[Tuple[float, float]]], grid_size: float, point_spacing: int = 3) -> List[Tuple[float, float]]:
    """Extract points from inter-rectangle paths by sampling based on spacing."""
    def _get_contour_points_from_single_path(filtered_reachable_coords: Set[Tuple[float, float]], path: List[Tuple[float, float]], point_spacing: float) -> List[Tuple[float, float]]:
        """Get contour points from a single path by sampling based on spacing.
        """
        if len(path) <= 2:
            return []
        
        path_points = []
        
        # Add start and end points if they are reachable
        start_point = (round(path[0][0], 2), round(path[0][1], 2))
        end_point = (round(path[-1][0], 2), round(path[-1][1], 2))
        
        if start_point in filtered_reachable_coords:
            path_points.append(start_point)
        
        if end_point in filtered_reachable_coords and end_point != start_point:
            path_points.append(end_point)
        
        # Sample intermediate points based on spacing
        current_distance = 0.0
        next_sample_distance = point_spacing
        
        for i in range(len(path) - 1):
            current_point = path[i]
            next_point = path[i + 1]
            
            # Calculate segment length
            segment_length = manhattan_distance(current_point, next_point)
            
            # Check if we need to sample points in this segment
            while next_sample_distance <= current_distance + segment_length:
                # Calculate the position of the sample point
                t = (next_sample_distance - current_distance) / segment_length if segment_length > 0 else 0
                sample_x = current_point[0] + t * (next_point[0] - current_point[0])
                sample_z = current_point[1] + t * (next_point[1] - current_point[1])
                sample_point = (round(sample_x, 2), round(sample_z, 2))
                
                if sample_point in filtered_reachable_coords:
                    path_points.append(sample_point)
                
                next_sample_distance += point_spacing
            
            current_distance += segment_length
        
        return path_points

    if not inter_rectangle_paths:
        return []
    
    actual_spacing = grid_size * point_spacing
    all_path_points = []
    
    for (rect_i, rect_j), path in inter_rectangle_paths.items():
        if path and len(path) > 2:
            path_points = _get_contour_points_from_single_path(filtered_reachable_coords, path, actual_spacing)
            all_path_points.extend(path_points)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_points = []
    for point in all_path_points:
        rounded_point = (round(point[0], 2), round(point[1], 2))
        if rounded_point not in seen:
            seen.add(rounded_point)
            unique_points.append(rounded_point)
    
    return unique_points
