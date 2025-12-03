import math
import heapq
from typing import List, Dict, Tuple, Optional, Set, Any

def euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    x1, z1 = pos1
    x2, z2 = pos2
    return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)

def manhattan_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    x1, z1 = pos1
    x2, z2 = pos2
    return abs(x1 - x2) + abs(z1 - z2)

def calculate_path_length(path: List[Tuple[float, float]]) -> float:
    if not path or len(path) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(path) - 1):
        # distance = manhattan_distance(path[i], path[i + 1])
        x1, z1 = path[i]
        x2, z2 = path[i + 1]
        distance =  abs(x1 - x2) + abs(z1 - z2)
        total_length += distance
    
    return total_length


class PathFinder:
    """A* pathfinding implementation for navigation."""
    
    def __init__(self, reachable_coords: List[Tuple[float, float]], grid_size: float = 0.25):
        self.reachable_coords = set(reachable_coords)
        self.grid_size = grid_size
    
    def find_path(self, 
                  start: Tuple[float, float], 
                  goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        if start not in self.reachable_coords or goal not in self.reachable_coords:
            return None
        
        if start == goal:
            return [start]
        
        # A* algorithm implementation with turning penalty
        open_set = [(0, start)]
        came_from: Dict[Tuple[float, float], Tuple[float, float]] = {}
        g_score: Dict[Tuple[float, float], float] = {start: 0}
        f_score: Dict[Tuple[float, float], float] = {start: self._heuristic(start, goal)}
        closed_set: Set[Tuple[float, float]] = set()
        direction_from: Dict[Tuple[float, float], Optional[Tuple[float, float]]] = {start: None}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set or neighbor not in self.reachable_coords:
                    continue
                
                base_cost = 1.0
                
                turning_penalty = self._calculate_turning_penalty(current, neighbor, direction_from.get(current))
                
                tentative_g_score = g_score[current] + base_cost + turning_penalty
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                    direction_from[neighbor] = self._get_direction(current, neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def _get_neighbors(self, pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        x, z = pos
        neighbors = []
        
        directions = [
            (self.grid_size, 0), (-self.grid_size, 0),
            (0, self.grid_size), (0, -self.grid_size),
        ]
        
        for dx, dz in directions:
            neighbor = (round(x + dx, 2), round(z + dz, 2))
            neighbors.append(neighbor)
        
        return neighbors
    
    def _heuristic(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return manhattan_distance(pos1, pos2)
    
    def _get_direction(self, from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> Tuple[float, float]:
        dx = to_pos[0] - from_pos[0]
        dz = to_pos[1] - from_pos[1]
        # 归一化方向向量
        if abs(dx) > abs(dz):
            return (1.0 if dx > 0 else -1.0, 0.0)
        else:
            return (0.0, 1.0 if dz > 0 else -1.0)
    
    def _calculate_turning_penalty(self, current: Tuple[float, float], 
                                 next_pos: Tuple[float, float], 
                                 prev_direction: Optional[Tuple[float, float]]) -> float:
        if prev_direction is None:
            return 0.0
        
        current_direction = self._get_direction(current, next_pos)
        
        # 计算两个方向向量的点积
        dot_product = prev_direction[0] * current_direction[0] + prev_direction[1] * current_direction[1]
        
        # 根据点积判断转向角度并应用惩罚
        if abs(dot_product - 1.0) < 1e-6:  # 同方向，点积为1
            return 0.0  # 直行，无惩罚
        elif abs(dot_product + 1.0) < 1e-6:  # 反方向，点积为-1
            return 6.0  # 180度转向，高惩罚
        else:  # 垂直方向，点积为0
            return 3.0  # 90度转向，惩罚为2
    
    def _reconstruct_path(self, 
                         came_from: Dict[Tuple[float, float], Tuple[float, float]], 
                         current: Tuple[float, float]) -> List[Tuple[float, float]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path