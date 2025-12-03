"""Rectangle generation and processing utilities.

This module contains functions for generating, filtering, and processing rectangles
from reachable coordinates.
"""

import math
from typing import List, Dict, Tuple, Optional, Set
from core.config import ExplorationConfig, Rectangle


def _build_coordinate_grid(coords: Set[Tuple[float, float]], 
                          x_coords: List[float], z_coords: List[float]) -> Dict[Tuple[int, int], bool]:
    """Build coordinate grid for efficient rectangle detection."""
    x_to_idx = {x: i for i, x in enumerate(x_coords)}
    z_to_idx = {z: i for i, z in enumerate(z_coords)}
    
    return {(x_to_idx[x], z_to_idx[z]): True for x, z in coords}


def _get_high_density_points(coords: Set[Tuple[float, float]], step: float) -> List[Tuple[float, float]]:
    """Get high density points for rectangle generation."""
    density_scores = []
    
    for x, z in coords:
        neighbors = sum(1 for dx in [-step, 0, step] for dz in [-step, 0, step]
                      if (round(x + dx, 2), round(z + dz, 2)) in coords)
        density_scores.append((neighbors, x, z))
    
    density_scores.sort(reverse=True)
    num_candidates = max(100, len(density_scores) // 2)
    return [(x, z) for _, x, z in density_scores[:num_candidates]]


def _is_rectangle_valid(start_x_idx: int, end_x_idx: int, 
                            start_z_idx: int, end_z_idx: int,
                            coord_grid: Dict[Tuple[int, int], bool]) -> bool:
    """Fast rectangle validity check."""
    # Check boundaries
    for x_idx in range(start_x_idx, end_x_idx + 1):
        if (not coord_grid.get((x_idx, start_z_idx), False) or 
            not coord_grid.get((x_idx, end_z_idx), False)):
            return False
    
    for z_idx in range(start_z_idx, end_z_idx + 1):
        if (not coord_grid.get((start_x_idx, z_idx), False) or 
            not coord_grid.get((end_x_idx, z_idx), False)):
            return False
    
    return True


def _expand_rectangle_from_point(start_x: float, start_z: float,
                                x_coords: List[float], z_coords: List[float],
                                x_to_idx: Dict[float, int], z_to_idx: Dict[float, int],
                                coord_grid: Dict[Tuple[int, int], bool]) -> Optional[Rectangle]:
    """Expand rectangle from a starting point using dynamic programming."""
    start_x_idx = x_to_idx[start_x]
    start_z_idx = z_to_idx[start_z]
    best_rect, max_area = None, 0
    
    # Dynamic programming expansion
    for end_x_idx in range(start_x_idx, len(x_coords)):
        max_valid_z_idx = start_z_idx
        
        for end_z_idx in range(start_z_idx, len(z_coords)):
            if not _is_rectangle_valid(start_x_idx, end_x_idx, start_z_idx, end_z_idx, coord_grid):
                break
            max_valid_z_idx = end_z_idx
        
        if max_valid_z_idx > start_z_idx:
            x_min, x_max = x_coords[start_x_idx], x_coords[end_x_idx]
            z_min, z_max = z_coords[start_z_idx], z_coords[max_valid_z_idx]
            rect = Rectangle(x_min, x_max, z_min, z_max)
            
            if rect.area() > max_area:
                max_area = rect.area()
                best_rect = rect
    
    return best_rect


def _find_largest_rectangle(coords: Set[Tuple[float, float]],
                                     x_coords: List[float], z_coords: List[float],
                                     x_to_idx: Dict[float, int], z_to_idx: Dict[float, int],
                                     coord_grid: Dict[Tuple[int, int], bool], step: float) -> Optional[Rectangle]:
    """Find the largest rectangle in the given coordinates."""
    if not coords:
        return None
    
    best_rect, max_area = None, 0
    candidate_points = _get_high_density_points(coords, step)
    
    for start_x, start_z in candidate_points:
        rect = _expand_rectangle_from_point(
            start_x, start_z, x_coords, z_coords, x_to_idx, z_to_idx, coord_grid
        )
        if rect and rect.area() > max_area:
            max_area = rect.area()
            best_rect = rect
    
    return best_rect


def _update_coordinate_grid(grid: Dict[Tuple[int, int], bool], 
                           removed_points: Set[Tuple[float, float]],
                           x_to_idx: Dict[float, int], z_to_idx: Dict[float, int]) -> None:
    """Update coordinate grid by removing points."""
    for x, z in removed_points:
        x_idx = x_to_idx[x]
        z_idx = z_to_idx[z]
        grid.pop((x_idx, z_idx), None)


def _filter_rectangles(rectangles: List[Rectangle], config: ExplorationConfig) -> List[Rectangle]:
    """Filter rectangles based on size criteria."""
    half_fov_radians = math.radians(config.fieldOfView / 2)
    threshold_length = int(math.sin(half_fov_radians) * config.visibilityDistance * 0.5 / config.gridSize) * config.gridSize
    return [rect for rect in rectangles if rect.width() + rect.height() >= 2 * threshold_length]


def _is_point_reachable_approximate(x: float, z: float, 
                                   filtered_reachable_coords: Set[Tuple[float, float]],
                                   grid_size: float) -> bool:
    """Check if a point is approximately reachable."""
    if (round(x, 2), round(z, 2)) in filtered_reachable_coords:
        return True
    
    tolerance = grid_size * 0.1
    for coord_x, coord_z in filtered_reachable_coords:
        if abs(coord_x - x) < tolerance and abs(coord_z - z) < tolerance:
            return True
    return False


def _is_shrunk_rectangle_valid(rect: Rectangle, config: ExplorationConfig, 
                              filtered_reachable_coords: Set[Tuple[float, float]]) -> bool:
    """Check if a shrunk rectangle is valid."""
    step = config.gridSize
    
    # Check top and bottom edges
    x = rect.x_min
    while x <= rect.x_max:
        if (not _is_point_reachable_approximate(x, rect.z_min, filtered_reachable_coords, config.gridSize) or 
            not _is_point_reachable_approximate(x, rect.z_max, filtered_reachable_coords, config.gridSize)):
            return False
        x = round(x + step, 2)
    
    # Check left and right edges
    z = rect.z_min
    while z <= rect.z_max:
        if (not _is_point_reachable_approximate(rect.x_min, z, filtered_reachable_coords, config.gridSize) or 
            not _is_point_reachable_approximate(rect.x_max, z, filtered_reachable_coords, config.gridSize)):
            return False
        z = round(z + step, 2)
    
    return True


def _find_closest_reachable_point(target_x: float, target_z: float,
                                  filtered_reachable_coords: Set[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Find the closest reachable point to the target."""
    min_distance = float('inf')
    closest_point = None
    
    for x, z in filtered_reachable_coords:
        distance = abs(x - target_x) + abs(z - target_z)
        if distance < min_distance:
            min_distance = distance
            closest_point = (x, z)
    
    return closest_point


def _shrink_single_rectangle(rect: Rectangle, initial_margin: int, config: ExplorationConfig,
                            filtered_reachable_coords: Set[Tuple[float, float]]) -> Optional[Rectangle]:
    """Shrink a single rectangle by the given margin."""
    margin = initial_margin
    
    while margin >= 0:
        margin_distance = margin * config.gridSize
        new_x_min = rect.x_min + margin_distance
        new_x_max = rect.x_max - margin_distance
        new_z_min = rect.z_min + margin_distance
        new_z_max = rect.z_max - margin_distance
        
        new_width = new_x_max - new_x_min
        new_height = new_z_max - new_z_min
        
        # Point case
        if new_width <= 0 and new_height <= 0:
            closest_point = _find_closest_reachable_point(rect.center()[0], rect.center()[1], filtered_reachable_coords)
            if closest_point:
                return Rectangle(closest_point[0], closest_point[0], closest_point[1], closest_point[1])
            margin -= 1
            continue
        
        # Line segment cases
        elif new_width <= 0 and new_height > 0:
            closest_center = _find_closest_reachable_point(rect.center()[0], (new_z_min + new_z_max) / 2, filtered_reachable_coords)
            if closest_center:
                return Rectangle(closest_center[0], closest_center[0], new_z_min, new_z_max)
            margin -= 1
            continue
        
        elif new_height <= 0 and new_width > 0:
            closest_center = _find_closest_reachable_point((new_x_min + new_x_max) / 2, rect.center()[1], filtered_reachable_coords)
            if closest_center:
                return Rectangle(new_x_min, new_x_max, closest_center[1], closest_center[1])
            margin -= 1
            continue
        
        # Smaller rectangle case
        else:
            shrunk_rect = Rectangle(new_x_min, new_x_max, new_z_min, new_z_max)
            if _is_shrunk_rectangle_valid(shrunk_rect, config, filtered_reachable_coords):
                return shrunk_rect
            margin -= 1
    
    return None


def _shrink_rectangles(rectangles: List[Rectangle], config: ExplorationConfig,
                      filtered_reachable_coords: Set[Tuple[float, float]]) -> List[Rectangle]:
    """Shrink rectangles based on configuration."""
    half_fov_radians = math.radians(config.fieldOfView / 2)
    margin = int(math.sin(half_fov_radians) * config.visibilityDistance * 0.3 / config.gridSize)
    
    shrunken_rectangles = []
    for rect in rectangles:
        # Only shrink rectangles with width or height of 1 or 2 grid sizes
        if (abs(rect.width() - config.gridSize) < 1e-6 or 
            abs(rect.height() - config.gridSize) < 1e-6 or 
            abs(rect.width() - 2 * config.gridSize) < 1e-6 or 
            abs(rect.height() - 2 * config.gridSize) < 1e-6):
            shrunk_rect = _shrink_single_rectangle(rect, margin, config, filtered_reachable_coords)
            if shrunk_rect:
                shrunken_rectangles.append(shrunk_rect)
        else:
            # Keep other rectangles unchanged
            shrunken_rectangles.append(rect)
    
    return shrunken_rectangles


# def generate_rectangles(filtered_reachable_coords: Set[Tuple[float, float]], # very slow but best
#                        config: ExplorationConfig) -> List[Rectangle]:
#     """Generate rectangles from filtered reachable coordinates.
    
#     New algorithm: Generate maximum-sized rectangles where only the perimeter
#     needs to be reachable, allowing unreachable points inside the rectangle.
#     Ensures no overlap between generated rectangles.
    
#     Args:
#         filtered_reachable_coords: Set of filtered reachable coordinates
#         config: Exploration configuration
        
#     Returns:
#         List of generated rectangles
#     """
#     if not filtered_reachable_coords:
#         return []
    
#     coords = list(filtered_reachable_coords)
#     rectangles = []
#     used_points = set()
    
#     # Sort points by x, then by z for systematic processing
#     coords.sort(key=lambda p: (p[0], p[1]))
    
#     def _is_perimeter_reachable(x_min: float, x_max: float, z_min: float, z_max: float) -> bool:
#         """Check if all perimeter points of a rectangle are reachable."""
#         step = config.gridSize
        
#         # Check top and bottom edges
#         x = x_min
#         while x <= x_max + 1e-6:  # Add small epsilon for floating point comparison
#             top_point = (round(x, 2), round(z_max, 2))
#             bottom_point = (round(x, 2), round(z_min, 2))
            
#             if top_point not in filtered_reachable_coords or bottom_point not in filtered_reachable_coords:
#                 return False
#             x = round(x + step, 2)
        
#         # Check left and right edges (excluding corners already checked)
#         z = z_min + step
#         while z < z_max:
#             left_point = (round(x_min, 2), round(z, 2))
#             right_point = (round(x_max, 2), round(z, 2))
            
#             if left_point not in filtered_reachable_coords or right_point not in filtered_reachable_coords:
#                 return False
#             z = round(z + step, 2)
        
#         return True
    
#     def _rectangle_overlaps_used_area(x_min: float, x_max: float, z_min: float, z_max: float) -> bool:
#         """Check if a rectangle overlaps with already used area."""
#         step = config.gridSize
        
#         # Check all points in the rectangle area
#         x = x_min
#         while x <= x_max + 1e-6:
#             z = z_min
#             while z <= z_max + 1e-6:
#                 point = (round(x, 2), round(z, 2))
#                 if point in used_points:
#                     return True
#                 z = round(z + step, 2)
#             x = round(x + step, 2)
        
#         return False
    
#     def _find_max_rectangle_from_point(start_x: float, start_z: float) -> Optional[Rectangle]:
#         """Find the maximum rectangle starting from a given point."""
#         if (start_x, start_z) in used_points:
#             return None
            
#         max_rect = None
#         max_area = 0
        
#         # Try all possible rectangles with this point as bottom-left corner
#         for end_x, end_z in coords:
#             if end_x < start_x or end_z < start_z:
#                 continue
            
#             # Check if this rectangle overlaps with used area
#             if _rectangle_overlaps_used_area(start_x, end_x, start_z, end_z):
#                 continue
                
#             # Check if this rectangle has reachable perimeter
#             if _is_perimeter_reachable(start_x, end_x, start_z, end_z):
#                 rect = Rectangle(start_x, end_x, start_z, end_z)
#                 area = rect.area()
                
#                 if area > max_area:
#                     max_area = area
#                     max_rect = rect
        
#         return max_rect
    
#     def _get_rectangle_points(rect: Rectangle) -> Set[Tuple[float, float]]:
#         """Get all points within a rectangle area (not just perimeter)."""
#         rectangle_points = set()
#         step = config.gridSize
        
#         # Get all points in the rectangle area
#         x = rect.x_min
#         while x <= rect.x_max + 1e-6:
#             z = rect.z_min
#             while z <= rect.z_max + 1e-6:
#                 rectangle_points.add((round(x, 2), round(z, 2)))
#                 z = round(z + step, 2)
#             x = round(x + step, 2)
        
#         return rectangle_points
    
#     # Main algorithm: greedily find maximum rectangles
#     while True:
#         best_rect = None
#         best_area = 0
        
#         # Try each unused point as a potential corner
#         for x, z in coords:
#             if (x, z) in used_points:
#                 continue
                
#             rect = _find_max_rectangle_from_point(x, z)
#             if rect and rect.area() > best_area:
#                 best_area = rect.area()
#                 best_rect = rect
        
#         # If no valid rectangle found, break
#         if not best_rect:
#             break
            
#         rectangles.append(best_rect)
        
#         # Mark all points in the rectangle area as used to prevent overlap
#         rectangle_points = _get_rectangle_points(best_rect)
#         used_points.update(rectangle_points)
    
#     # Filter and shrink rectangles
#     filtered_rectangles = _filter_rectangles(rectangles, config)
#     shrunken_rectangles = _shrink_rectangles(filtered_rectangles, config, filtered_reachable_coords)
    
#     return shrunken_rectangles

def generate_rectangles(filtered_reachable_coords: Set[Tuple[float, float]], 
                       config: ExplorationConfig) -> List[Rectangle]:
    """Generate rectangles from filtered reachable coordinates.
    
    Optimized algorithm: Generate maximum-sized rectangles by removing covered points
    after each rectangle generation to prevent overlap.
    
    Args:
        filtered_reachable_coords: Set of filtered reachable coordinates
        config: Exploration configuration
        
    Returns:
        List of generated rectangles
    """
    if not filtered_reachable_coords:
        return []
    
    # Work with a copy to avoid modifying the original set
    remaining_coords = set(filtered_reachable_coords)
    rectangles = []
    step = config.gridSize
    
    def _is_perimeter_reachable(x_min: float, x_max: float, z_min: float, z_max: float, 
                               coord_set: Set[Tuple[float, float]]) -> bool:
        """Check if rectangle perimeter is reachable."""
        # Check corners first (most likely to fail)
        corners = [
            (x_min, z_min), (x_min, z_max),
            (x_max, z_min), (x_max, z_max)
        ]
        for x, z in corners:
            if (round(x, 2), round(z, 2)) not in coord_set:
                return False
        
        # Check edges
        # Top and bottom edges
        x = x_min + step
        while x < x_max:
            if ((round(x, 2), round(z_min, 2)) not in coord_set or 
                (round(x, 2), round(z_max, 2)) not in coord_set):
                return False
            x = round(x + step, 2)
        
        # Left and right edges
        z = z_min + step
        while z < z_max:
            if ((round(x_min, 2), round(z, 2)) not in coord_set or 
                (round(x_max, 2), round(z, 2)) not in coord_set):
                return False
            z = round(z + step, 2)
        
        return True
    
    def _find_max_rectangle_from_point(start_x: float, start_z: float, 
                                      coord_set: Set[Tuple[float, float]]) -> Optional[Rectangle]:
        """Find maximum rectangle starting from given point."""
        if (round(start_x, 2), round(start_z, 2)) not in coord_set:
            return None
            
        max_rect = None
        max_area = 0
        
        # Get all valid end coordinates
        coords_list = list(coord_set)
        valid_end_coords = [(x, z) for x, z in coords_list 
                           if x >= start_x and z >= start_z]
        
        # Sort by potential area (descending) for better pruning
        valid_end_coords.sort(key=lambda p: (p[0] - start_x) * (p[1] - start_z), reverse=True)
        
        for end_x, end_z in valid_end_coords:
            potential_area = (end_x - start_x) * (end_z - start_z)
            
            # Pruning: skip if potential area is smaller than current best
            if potential_area <= max_area:
                continue
                
            # Check if perimeter is reachable
            if _is_perimeter_reachable(start_x, end_x, start_z, end_z, coord_set):
                rect = Rectangle(start_x, end_x, start_z, end_z)
                area = rect.area()
                
                if area > max_area:
                    max_area = area
                    max_rect = rect
        
        return max_rect
    
    def _get_rectangle_covered_points(rect: Rectangle) -> Set[Tuple[float, float]]:
        """Get all grid points covered by the rectangle."""
        covered_points = set()
        
        x = rect.x_min
        while x <= rect.x_max + 1e-6:
            z = rect.z_min
            while z <= rect.z_max + 1e-6:
                covered_points.add((round(x, 2), round(z, 2)))
                z = round(z + step, 2)
            x = round(x + step, 2)
        
        return covered_points
    
    # Main algorithm: iteratively find rectangles and remove covered points
    iteration_count = 0
    max_iterations = len(remaining_coords)  # Prevent infinite loops
    
    while remaining_coords and iteration_count < max_iterations:
        best_rect = None
        best_area = 0
        
        # Try each remaining point as a potential corner
        coords_list = list(remaining_coords)
        coords_list.sort(key=lambda p: (p[0], p[1]))  # Systematic processing
        
        for x, z in coords_list:
            rect = _find_max_rectangle_from_point(x, z, remaining_coords)
            if rect and rect.area() > best_area:
                best_area = rect.area()
                best_rect = rect
        
        # If no valid rectangle found or area is too small, break
        if not best_rect or best_area < step * step:  # Minimum meaningful size
            break
            
        rectangles.append(best_rect)
        
        # Remove all points covered by this rectangle from remaining coordinates
        covered_points = _get_rectangle_covered_points(best_rect)
        remaining_coords -= covered_points
        
        iteration_count += 1
    
    # Filter and shrink rectangles
    filtered_rectangles = _filter_rectangles(rectangles, config)
    shrunken_rectangles = _shrink_rectangles(filtered_rectangles, config, filtered_reachable_coords)
    
    return shrunken_rectangles