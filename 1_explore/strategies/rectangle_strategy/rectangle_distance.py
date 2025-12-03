"""Distance calculation utilities for rectangle-based exploration strategy.

This module contains functions for calculating distances between rectangles,
lines, and points in various geometric configurations.
"""

from typing import List, Tuple
from core.config import Rectangle
from ..utils import manhattan_distance


def _is_degenerate_point(rect: Rectangle) -> bool:
    return len(set(rect.get_corners())) == 1

def _is_degenerate_line(rect: Rectangle) -> bool:
    return len(set(rect.get_corners())) == 2


def _get_point_coords(rect: Rectangle) -> Tuple[float, float]:
    """Get coordinates of a point rectangle."""
    return (rect.x_min, rect.z_min)


def _get_line_endpoints(rect: Rectangle) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Get endpoints of a line segment rectangle."""
    if abs(rect.x_max - rect.x_min) < 1e-6:  # Vertical line
        return ((rect.x_min, rect.z_min), (rect.x_min, rect.z_max))
    else:  # Horizontal line
        return ((rect.x_min, rect.z_min), (rect.x_max, rect.z_min))


def _point_to_point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Manhattan distance between two points."""
    return manhattan_distance(p1, p2)


def _point_to_line_distance(point: Tuple[float, float], 
                           line_start: Tuple[float, float], 
                           line_end: Tuple[float, float]) -> float:
    """Calculate distance from point to line segment."""
    px, pz = point
    x1, z1 = line_start
    x2, z2 = line_end
    
    # Check if line is vertical
    if abs(x2 - x1) < 1e-6:
        # Vertical line: check if point is within z range
        if min(z1, z2) - 1e-6 <= pz <= max(z1, z2) + 1e-6:
            return abs(px - x1)  # Perpendicular distance
        else:
            # Distance to nearest endpoint
            return min(manhattan_distance(point, line_start),
                      manhattan_distance(point, line_end))
    
    # Check if line is horizontal
    if abs(z2 - z1) < 1e-6:
        # Horizontal line: check if point is within x range
        if min(x1, x2) <= px <= max(x1, x2):
            return abs(pz - z1)  # Perpendicular distance
        else:
            # Distance to nearest endpoint
            return min(manhattan_distance(point, line_start),
                      manhattan_distance(point, line_end))
    
    # For diagonal lines, use Manhattan distance to nearest endpoint
    return min(manhattan_distance(point, line_start),
              manhattan_distance(point, line_end))


def _point_to_rectangle_distance(point: Tuple[float, float], rect: Rectangle) -> float:
    """Calculate minimum distance from point to rectangle edges."""
    px, pz = point
    
    # If point is inside rectangle, distance is 0
    if rect.x_min <= px <= rect.x_max and rect.z_min <= pz <= rect.z_max:
        return 0.0
    
    # Calculate distances to all four edges
    distances = []
    
    # Top edge (z = rect.z_max)
    if rect.x_min <= px <= rect.x_max:
        distances.append(abs(pz - rect.z_max))
    else:
        distances.append(min(
            manhattan_distance(point, (rect.x_min, rect.z_max)),
            manhattan_distance(point, (rect.x_max, rect.z_max))
        ))
    
    # Bottom edge (z = rect.z_min)
    if rect.x_min <= px <= rect.x_max:
        distances.append(abs(pz - rect.z_min))
    else:
        distances.append(min(
            manhattan_distance(point, (rect.x_min, rect.z_min)),
            manhattan_distance(point, (rect.x_max, rect.z_min))
        ))
    
    # Left edge (x = rect.x_min)
    if rect.z_min <= pz <= rect.z_max:
        distances.append(abs(px - rect.x_min))
    else:
        distances.append(min(
            manhattan_distance(point, (rect.x_min, rect.z_min)),
            manhattan_distance(point, (rect.x_min, rect.z_max))
        ))
    
    # Right edge (x = rect.x_max)
    if rect.z_min <= pz <= rect.z_max:
        distances.append(abs(px - rect.x_max))
    else:
        distances.append(min(
            manhattan_distance(point, (rect.x_max, rect.z_min)),
            manhattan_distance(point, (rect.x_max, rect.z_max))
        ))
    
    return min(distances)


def _line_to_rectangle_distance(line_start: Tuple[float, float], 
                               line_end: Tuple[float, float], 
                               rect: Rectangle) -> float:
    """Calculate distance from line segment to rectangle as minimum distance to all rectangle edges."""
    # Get all four edges of the rectangle
    rect_edges = get_rectangle_edges(rect)
    
    min_distance = float('inf')
    
    # Calculate distance from line segment to each rectangle edge
    for edge_start, edge_end in rect_edges:
        distance = line_segment_distance(line_start, line_end, edge_start, edge_end)
        min_distance = min(min_distance, distance)
    
    return min_distance


def get_rectangle_edges(rect: Rectangle) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Get all four edges of a rectangle as line segments."""
    return [
        ((rect.x_min, rect.z_min), (rect.x_max, rect.z_min)),  # Bottom edge
        ((rect.x_max, rect.z_min), (rect.x_max, rect.z_max)),  # Right edge
        ((rect.x_max, rect.z_max), (rect.x_min, rect.z_max)),  # Top edge
        ((rect.x_min, rect.z_max), (rect.x_min, rect.z_min))   # Left edge
    ]


def line_segment_distance(seg1_start: Tuple[float, float], seg1_end: Tuple[float, float],
                         seg2_start: Tuple[float, float], seg2_end: Tuple[float, float]) -> float:
    """Calculate distance between two line segments.
    """
    # Determine if the two line segments are parallel
    # Check if the first segment is vertical
    seg1_is_vertical = abs(seg1_start[0] - seg1_end[0]) < 1e-6
    # Check if the first segment is horizontal
    seg1_is_horizontal = abs(seg1_start[1] - seg1_end[1]) < 1e-6
    
    # Check if the second segment is vertical
    seg2_is_vertical = abs(seg2_start[0] - seg2_end[0]) < 1e-6
    # Check if the second segment is horizontal
    seg2_is_horizontal = abs(seg2_start[1] - seg2_end[1]) < 1e-6
    
    # Determine if segments are parallel
    segments_are_parallel = (seg1_is_vertical and seg2_is_vertical) or (seg1_is_horizontal and seg2_is_horizontal)
    
    if segments_are_parallel:
        # Two segments are parallel
        if seg1_is_vertical and seg2_is_vertical:
            # Two vertical segments are parallel
            # Check z coordinate ranges for overlap
            z1_min, z1_max = min(seg1_start[1], seg1_end[1]), max(seg1_start[1], seg1_end[1])
            z2_min, z2_max = min(seg2_start[1], seg2_end[1]), max(seg2_start[1], seg2_end[1])
            
            # Check if z coordinates have overlap
            if z1_max >= z2_min and z2_max >= z1_min:  # There is z overlap
                # Return absolute difference in x coordinates
                return abs(seg1_start[0] - seg2_start[0])
            else:
                # No z overlap, return minimum Manhattan distance of four endpoints
                distances = [
                    manhattan_distance(seg1_start, seg2_start),
                    manhattan_distance(seg1_start, seg2_end),
                    manhattan_distance(seg1_end, seg2_start),
                    manhattan_distance(seg1_end, seg2_end)
                ]
                return min(distances)
                
        elif seg1_is_horizontal and seg2_is_horizontal:
            # Two horizontal segments are parallel
            # Check x coordinate ranges for overlap
            x1_min, x1_max = min(seg1_start[0], seg1_end[0]), max(seg1_start[0], seg1_end[0])
            x2_min, x2_max = min(seg2_start[0], seg2_end[0]), max(seg2_start[0], seg2_end[0])
            
            # Check if x coordinates have overlap
            if x1_max >= x2_min and x2_max >= x1_min:  # There is x overlap
                # Return absolute difference in z coordinates
                return abs(seg1_start[1] - seg2_start[1])
            else:
                # No x overlap, return minimum Manhattan distance of four endpoints
                distances = [
                    manhattan_distance(seg1_start, seg2_start),
                    manhattan_distance(seg1_start, seg2_end),
                    manhattan_distance(seg1_end, seg2_start),
                    manhattan_distance(seg1_end, seg2_end)
                ]
                return min(distances)
    
    # Two segments are perpendicular (not parallel)
    seg1_to_seg2_distances = [
        _point_to_line_distance(seg1_start, seg2_start, seg2_end),
        _point_to_line_distance(seg1_end, seg2_start, seg2_end)
    ]
    
    seg2_to_seg1_distances = [
        _point_to_line_distance(seg2_start, seg1_start, seg1_end),
        _point_to_line_distance(seg2_end, seg1_start, seg1_end)
    ]
    
    # Return minimum distance among all point-to-line distances
    return min(seg1_to_seg2_distances + seg2_to_seg1_distances)


def _get_closest_point_on_line(point: Tuple[float, float], 
                              line_start: Tuple[float, float], 
                              line_end: Tuple[float, float]) -> Tuple[float, float]:
    """Get the closest point on a line segment to a given point."""
    px, pz = point
    x1, z1 = line_start
    x2, z2 = line_end
    
    # Check if line is vertical
    if abs(x2 - x1) < 1e-6:
        # Vertical line: clamp z coordinate to line range
        z_clamped = max(min(z1, z2), min(pz, max(z1, z2)))
        return (x1, z_clamped)
    
    # Check if line is horizontal
    if abs(z2 - z1) < 1e-6:
        # Horizontal line: clamp x coordinate to line range
        x_clamped = max(min(x1, x2), min(px, max(x1, x2)))
        return (x_clamped, z1)
    
    # For diagonal lines, return nearest endpoint
    dist_to_start = manhattan_distance(point, line_start)
    dist_to_end = manhattan_distance(point, line_end)
    return line_start if dist_to_start <= dist_to_end else line_end


def _get_closest_point_on_rectangle(point: Tuple[float, float], rect: Rectangle) -> Tuple[float, float]:
    """Get the closest point on a rectangle to a given point."""
    px, pz = point
    
    # If point is inside rectangle, find closest edge
    if rect.x_min <= px <= rect.x_max and rect.z_min <= pz <= rect.z_max:
        # Find distances to all edges
        dist_to_left = px - rect.x_min
        dist_to_right = rect.x_max - px
        dist_to_bottom = pz - rect.z_min
        dist_to_top = rect.z_max - pz
        
        min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
        
        if min_dist == dist_to_left:
            return (rect.x_min, pz)
        elif min_dist == dist_to_right:
            return (rect.x_max, pz)
        elif min_dist == dist_to_bottom:
            return (px, rect.z_min)
        else:
            return (px, rect.z_max)
    
    # Point is outside rectangle, clamp to rectangle bounds
    x_clamped = max(rect.x_min, min(px, rect.x_max))
    z_clamped = max(rect.z_min, min(pz, rect.z_max))
    return (x_clamped, z_clamped)


def _line_to_rectangle_distance_with_points(line_start: Tuple[float, float], 
                                           line_end: Tuple[float, float], 
                                           rect: Rectangle) -> Tuple[float, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Calculate distance from line segment to rectangle and return closest points."""
    rect_edges = get_rectangle_edges(rect)
    
    min_distance = float('inf')
    best_point_pair = None
    
    # Calculate distance from line segment to each rectangle edge
    for edge_start, edge_end in rect_edges:
        distance, point_pair = _line_to_line_distance_with_points(line_start, line_end, edge_start, edge_end)
        if distance < min_distance:
            min_distance = distance
            best_point_pair = point_pair
    
    return min_distance, best_point_pair


def _line_to_line_distance_with_points(seg1_start: Tuple[float, float], seg1_end: Tuple[float, float],
                                      seg2_start: Tuple[float, float], seg2_end: Tuple[float, float]) -> Tuple[float, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Calculate distance between two line segments and return closest points."""
    # For simplicity, we'll check all endpoint combinations and return the closest pair
    min_distance = float('inf')
    best_point_pair = None
    
    # Check all combinations of endpoints
    point_combinations = [
        (seg1_start, seg2_start),
        (seg1_start, seg2_end),
        (seg1_end, seg2_start),
        (seg1_end, seg2_end)
    ]
    
    for p1, p2 in point_combinations:
        distance = manhattan_distance(p1, p2)
        if distance < min_distance:
            min_distance = distance
            best_point_pair = (p1, p2)
    
    # Also check closest points on line segments
    closest_on_seg2_to_seg1_start = _get_closest_point_on_line(seg1_start, seg2_start, seg2_end)
    closest_on_seg2_to_seg1_end = _get_closest_point_on_line(seg1_end, seg2_start, seg2_end)
    closest_on_seg1_to_seg2_start = _get_closest_point_on_line(seg2_start, seg1_start, seg1_end)
    closest_on_seg1_to_seg2_end = _get_closest_point_on_line(seg2_end, seg1_start, seg1_end)
    
    additional_combinations = [
        (seg1_start, closest_on_seg2_to_seg1_start),
        (seg1_end, closest_on_seg2_to_seg1_end),
        (closest_on_seg1_to_seg2_start, seg2_start),
        (closest_on_seg1_to_seg2_end, seg2_end)
    ]
    
    for p1, p2 in additional_combinations:
        distance = manhattan_distance(p1, p2)
        if distance < min_distance:
            min_distance = distance
            best_point_pair = (p1, p2)
    
    return min_distance, best_point_pair


def _parallel_edges_distance_with_points(rect1: Rectangle, rect2: Rectangle) -> Tuple[float, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Calculate distance between two rectangles and return closest points."""
    edges1 = get_rectangle_edges(rect1)
    edges2 = get_rectangle_edges(rect2)
    
    min_distance = float('inf')
    best_point_pair = None
    
    # Calculate distance between all combinations of edges (16 total)
    for edge1 in edges1:
        for edge2 in edges2:
            distance, point_pair = _line_to_line_distance_with_points(edge1[0], edge1[1], edge2[0], edge2[1])
            if distance < min_distance:
                min_distance = distance
                best_point_pair = point_pair
    
    return min_distance, best_point_pair


def _parallel_edges_distance(rect1: Rectangle, rect2: Rectangle) -> float:
    """Calculate distance between two rectangles as minimum distance between all edge combinations."""
    edges1 = get_rectangle_edges(rect1)
    edges2 = get_rectangle_edges(rect2)
    
    min_distance = float('inf')
    
    # Calculate distance between all combinations of edges (16 total)
    for edge1 in edges1:
        for edge2 in edges2:
            distance = line_segment_distance(edge1[0], edge1[1], edge2[0], edge2[1])
            min_distance = min(min_distance, distance)
    
    return min_distance


def _line_to_line_distance(rect1: Rectangle, rect2: Rectangle) -> float:
    """Calculate distance between two line segments."""
    line1_start, line1_end = _get_line_endpoints(rect1)
    line2_start, line2_end = _get_line_endpoints(rect2)
    
    return line_segment_distance(line1_start, line1_end, line2_start, line2_end)


def calculate_rectangle_distance_with_points(rect1: Rectangle, rect2: Rectangle) -> Tuple[float, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Calculate distance between two rectangles and return the closest point pair.
    
    Args:
        rect1: First rectangle
        rect2: Second rectangle
        
    Returns:
        Tuple of (distance, (point_on_rect1, point_on_rect2))
    """
    # Check if rectangles are degenerate
    rect1_is_point = _is_degenerate_point(rect1)
    rect2_is_point = _is_degenerate_point(rect2)
    rect1_is_line = _is_degenerate_line(rect1)
    rect2_is_line = _is_degenerate_line(rect2)
    
    # Case 1: Both are points
    if rect1_is_point and rect2_is_point:
        p1 = _get_point_coords(rect1)
        p2 = _get_point_coords(rect2)
        return _point_to_point_distance(p1, p2), (p1, p2)
    
    # Case 2: One point, one line
    elif rect1_is_point and rect2_is_line:
        point = _get_point_coords(rect1)
        line_start, line_end = _get_line_endpoints(rect2)
        distance = _point_to_line_distance(point, line_start, line_end)
        closest_on_line = _get_closest_point_on_line(point, line_start, line_end)
        return distance, (point, closest_on_line)
    
    elif rect2_is_point and rect1_is_line:
        point = _get_point_coords(rect2)
        line_start, line_end = _get_line_endpoints(rect1)
        distance = _point_to_line_distance(point, line_start, line_end)
        closest_on_line = _get_closest_point_on_line(point, line_start, line_end)
        return distance, (closest_on_line, point)
    
    # Case 3: One point, one rectangle
    elif rect1_is_point and not rect2_is_line:
        point = _get_point_coords(rect1)
        distance = _point_to_rectangle_distance(point, rect2)
        closest_on_rect = _get_closest_point_on_rectangle(point, rect2)
        return distance, (point, closest_on_rect)
    
    elif rect2_is_point and not rect1_is_line:
        point = _get_point_coords(rect2)
        distance = _point_to_rectangle_distance(point, rect1)
        closest_on_rect = _get_closest_point_on_rectangle(point, rect1)
        return distance, (closest_on_rect, point)
    
    # Case 4: One line, one rectangle
    elif rect1_is_line and not rect2_is_point and not rect2_is_line:
        line_start, line_end = _get_line_endpoints(rect1)
        distance, point_pair = _line_to_rectangle_distance_with_points(line_start, line_end, rect2)
        return distance, point_pair
    
    elif rect2_is_line and not rect1_is_point and not rect1_is_line:
        line_start, line_end = _get_line_endpoints(rect2)
        distance, point_pair = _line_to_rectangle_distance_with_points(line_start, line_end, rect1)
        return distance, (point_pair[1], point_pair[0])  # Swap order
    
    # Case 5: Both are lines
    elif rect1_is_line and rect2_is_line:
        line1_start, line1_end = _get_line_endpoints(rect1)
        line2_start, line2_end = _get_line_endpoints(rect2)
        distance, point_pair = _line_to_line_distance_with_points(line1_start, line1_end, line2_start, line2_end)
        return distance, point_pair
    
    # Case 6: Both are normal rectangles
    else:
        distance, point_pair = _parallel_edges_distance_with_points(rect1, rect2)
        return distance, point_pair
