"""Rectangle-based exploration strategy for AI2Thor.

Implements an exploration strategy based on identifying and traversing rectangular areas.
"""

from typing import List, Tuple, Optional

from ..utils import PathFinder
from core.config import ExplorationConfig, Rectangle, AgentState

from .rectangle_filter_edge_points import filter_edge_points
from .rectangle_generation import generate_rectangles
from .rectangle_trajectory import generate_trajectory_from_rectangles

from .rectangle_visualization import create_trajectory_gif, visualize_rectangles_with_coords 


class RectangleBasedStrategy:
    def __init__(self, 
                 config: ExplorationConfig,
                 reachable_coords: List[Tuple[float, float]],):
        """Initialize rectangle-based exploration strategy.
        """
        self.config = config
        grid_size = config.gridSize
        self.grid_size = grid_size
        self.primitive_reachable_coords = set(reachable_coords)
        # self.filtered_reachable_coords = set(filter_edge_points(reachable_coords, grid_size))
        # self.pathfinder = PathFinder(self.filtered_reachable_coords, grid_size)
        
        # Strategy state
        self.rectangles: List[Rectangle] = []
        self.current_rectangle_index = 0
        self.current_trajectory: List[Tuple[float, float]] = []
        self.trajectory_index = 0
        self.strategy_initialized = False
        self.current_target_position: Optional[Tuple[float, float]] = None
        self.target_rotation = 0.0
    
    def initialize_strategy(self, agent_state: AgentState) -> None:
        """Initialize the exploration strategy.
        """
        print("\nInitializing rectangle exploration strategy...")
        self.filtered_reachable_coords = set(filter_edge_points(self.primitive_reachable_coords, self.grid_size, agent_state.position))
        self.pathfinder = PathFinder(self.filtered_reachable_coords, self.grid_size)
        import time
        t1 = time.time()
        self.rectangles = self._generate_rectangles()
        t2 = time.time()
        print(f'\nrectangles generating takes {t2-t1}s')
        # visualize_rectangles_with_coords(self.primitive_reachable_coords, self.filtered_reachable_coords, self.rectangles, )
        # import pdb;pdb.set_trace()
         
        print(f"Generated {len(self.rectangles)} rectangles")

        self.current_trajectory = self._generate_trajectory(agent_state)

        self.strategy_initialized = True

    def get_actions(self, agent_state: AgentState) -> Optional[str]:
        """Get the next action for the agent."""
        def positions_equal(pos1: Tuple[float, float], pos2: Tuple[float, float], tolerance: float = 0.05) -> bool:
            return abs(pos1[0] - pos2[0]) < tolerance and abs(pos1[1] - pos2[1]) < tolerance

        def normalize_rotation(rotation: float) -> float:
            normalized = rotation % 360
            if abs(normalized) < 0.01:
                return 0.0
            elif abs(normalized - 360) < 0.01:
                return 0.0
            elif abs(normalized - 90) < 0.01:
                return 90.0
            elif abs(normalized - 180) < 0.01:
                return 180.0
            elif abs(normalized - 270) < 0.01:
                return 270.0
            return normalized

        def _get_actions_to_target(current_pos: Tuple[float, float], 
                                    target_pos: Tuple[float, float], current_rotation: float) -> List[str]:
            actions = []
            
            dx = target_pos[0] - current_pos[0]
            dz = target_pos[1] - current_pos[1]
            
            if abs(dx) < 0.15 and abs(dz) < 0.15:
                return []
            
            if abs(dx) > abs(dz):
                target_rotation = 90.0 if dx > 0 else 270.0  # 向右或向左
            else:
                target_rotation = 0.0 if dz > 0 else 180.0   # 向前或向后
            
            current_rotation = normalize_rotation(current_rotation)
            
            angle_diff = (target_rotation - current_rotation) % 360
            
            if angle_diff > 180:
                angle_diff -= 360
            
            # 只有当角度差大于阈值时才旋转
            if abs(angle_diff) > 5.0:  # 增加角度阈值
                rotation_count = abs(angle_diff) // 90
                rotation_direction = 'RotateRight' if angle_diff > 0 else 'RotateLeft'
                
                for _ in range(int(rotation_count)):
                    actions.append(rotation_direction)
            
            actions.append('MoveAhead')
            
            return actions

        if not self.strategy_initialized:
            self.initialize_strategy(agent_state)
           
        if self.trajectory_index >= len(self.current_trajectory):
            return None
        
        target_position = self.current_trajectory[self.trajectory_index]
        current_position = agent_state.position
        
        if positions_equal(current_position, target_position, tolerance=0.01):
            self.trajectory_index += 1
            if self.trajectory_index >= len(self.current_trajectory):
                return None
            target_position = self.current_trajectory[self.trajectory_index]
        
        return _get_actions_to_target(current_position, target_position, agent_state.rotation), target_position

    def _generate_rectangles(self) -> List[Rectangle]:
        return generate_rectangles(self.filtered_reachable_coords, self.config)

    def _generate_trajectory(self, agent_state: AgentState) -> List[Tuple[float, float]]:
        """Generate a trajectory based on the rectangles."""
        trajectory = generate_trajectory_from_rectangles(
            self.config,
            self.filtered_reachable_coords, 
            self.rectangles, 
            agent_state.position
        )
        
        return trajectory
    