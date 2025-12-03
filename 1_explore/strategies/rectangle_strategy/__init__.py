"""Rectangle-based exploration strategy package.

This package contains all components for the rectangle-based exploration strategy,
including rectangle generation, pathfinding, trajectory planning, and visualization.
"""

from .rectangle_based_strategy import RectangleBasedStrategy
from .rectangle_pathfinding import PathFinder
from .rectangle_visualization import (
    visualize_rectangle_strategy_result,
    create_trajectory_gif,
    save_visualization,
    visualize_sample_points
)

__all__ = [
    'RectangleBasedStrategy',
    'generate_rectangles', 
    'get_inter_rectangle_paths',
    'generate_trajectory_from_rectangles',
    'visualize_rectangle_strategy_result',
    'create_trajectory_gif',
    'save_visualization',
    'visualize_sample_points',
    '_calculate_rectangle_distance',
    'filter_edge_points'
]