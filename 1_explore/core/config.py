"""Configuration module for AI2Thor exploration.

Contains data structures and configuration classes for the exploration system.
"""

import math
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional


@dataclass
class ExplorationConfig:
    """Configuration for exploration parameters."""
    smooth_rotation_degrees: int = 30
    gridSize: float = 0.25
    fieldOfView: int = 120
    visibilityDistance: float = 1.5
    fps_trajectory: int = 15
    fps_first_person: int = 10
    fps_top_down: int = 10
    fps_depth: int = 10
    fps_segmentation: int = 10
    frame_size: Tuple[int, int] = (640, 640)
    base_output_dir: str = 'output'
    asset_output_dir: str = 'output/assets'


@dataclass
class AgentState:
    """State information for the exploration agent."""
    position: Tuple[float, float]
    rotation: float
    visited_positions: Set[Tuple[float, float]]
    trajectory: List[Tuple[float, float]]
    step_count: int = 0


@dataclass
class Rectangle:
    """Rectangle data structure for exploration areas."""
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    
    def area(self) -> float:
        """Calculate the area of the rectangle."""
        return (self.x_max - self.x_min) * (self.z_max - self.z_min)
    
    def contains_point(self, x: float, z: float) -> bool:
        """Check if a point is contained within the rectangle."""
        return self.x_min <= x <= self.x_max and self.z_min <= z <= self.z_max
    
    def width(self) -> float:
        """Get the width of the rectangle."""
        return self.x_max - self.x_min
    
    def height(self) -> float:
        """Get the height of the rectangle."""
        return self.z_max - self.z_min
    
    def center(self) -> Tuple[float, float]:
        """Get the center point of the rectangle."""
        return ((self.x_min + self.x_max) / 2, (self.z_min + self.z_max) / 2)
    
    def is_degenerate_line(self, grid_size: float) -> bool:
        """Check if rectangle is degenerate (line-like)."""
        return self.width() <= grid_size or self.height() <= grid_size
    
    def get_boundary_points(self, grid_size: float) -> List[Tuple[float, float]]:
        """Get boundary points of the rectangle."""
        points = []
        
        # Top and bottom edges
        x = self.x_min
        while x <= self.x_max:
            points.append((round(x, 2), round(self.z_min, 2)))
            points.append((round(x, 2), round(self.z_max, 2)))
            x += grid_size
        
        # Left and right edges
        z = self.z_min + grid_size
        while z < self.z_max:
            points.append((round(self.x_min, 2), round(z, 2)))
            points.append((round(self.x_max, 2), round(z, 2)))
            z += grid_size
        
        return list(set(points))  # Remove duplicates
    
    def is_point_on_boundary(self, pos: Tuple[float, float], tolerance: float) -> bool:
        """Check if a point is on the rectangle boundary."""
        x, z = pos
        
        on_left = abs(x - self.x_min) < tolerance and self.z_min <= z <= self.z_max
        on_right = abs(x - self.x_max) < tolerance and self.z_min <= z <= self.z_max
        on_bottom = abs(z - self.z_min) < tolerance and self.x_min <= x <= self.x_max
        on_top = abs(z - self.z_max) < tolerance and self.x_min <= x <= self.x_max
        
        return on_left or on_right or on_bottom or on_top
    
    def get_line_endpoints(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get endpoints for degenerate line rectangles."""
        if self.width() <= self.height():  # 垂直线段
            return (self.x_min, self.z_min), (self.x_min, self.z_max)
        else:  # 水平线段
            return (self.x_min, self.z_min), (self.x_max, self.z_min)
    
    def find_best_entry_point(self, current_pos: Tuple[float, float], 
                             grid_size: float) -> Tuple[float, float]:
        """Find the best entry point on rectangle boundary."""
        boundary_points = self.get_boundary_points(grid_size)
        if not boundary_points:
            return self.center()
        
        min_dist = float('inf')
        best_entry = boundary_points[0]
        
        for point in boundary_points:
            dist = math.sqrt((current_pos[0] - point[0])**2 + (current_pos[1] - point[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_entry = point
        
        return best_entry
    
    def order_boundary_points_clockwise(self, points: List[Tuple[float, float]], 
                                       clockwise: bool = True) -> List[Tuple[float, float]]:
        """Order boundary points in clockwise or counterclockwise direction."""
        if not points:
            return []
        
        center_x, center_z = self.center()
        
        def angle_from_center(point):
            dx = point[0] - center_x
            dz = point[1] - center_z
            return math.atan2(dz, dx)
        
        sorted_points = sorted(points, key=angle_from_center, reverse=clockwise)
        return sorted_points
    
    def is_valid_with_coords(self, coords: Set[Tuple[float, float]], grid_size: float) -> bool:
        """Check if rectangle is valid given coordinate constraints."""
        step = grid_size
        
        # Check top and bottom edges
        x = self.x_min
        while x <= self.x_max:
            if (round(x, 2), round(self.z_min, 2)) not in coords or \
               (round(x, 2), round(self.z_max, 2)) not in coords:
                return False
            x = round(x + step, 2)
        
        # Check left and right edges
        z = self.z_min
        while z <= self.z_max:
            if (round(self.x_min, 2), round(z, 2)) not in coords or \
               (round(self.x_max, 2), round(z, 2)) not in coords:
                return False
            z = round(z + step, 2)
        
        return True
    
    def get_covered_points(self, coords: Set[Tuple[float, float]]) -> Set[Tuple[float, float]]:
        """Get all coordinate points covered by this rectangle."""
        covered = set()
        
        for x, z in coords:
            if self.contains_point(x, z):
                covered.add((x, z))
        
        return covered

    def get_corners(self) -> List[Tuple[float, float]]:
        """Get the corners of the rectangle."""
        return set([
            (self.x_min, self.z_min),  # 左下角
            (self.x_min, self.z_max),  # 左上角
            (self.x_max, self.z_max),  # 右上角
            (self.x_max, self.z_min)   # 右下角
        ])