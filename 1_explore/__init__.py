"""AI2Thor Exploration Package

A modular package for AI2Thor environment exploration with trajectory visualization.
Reorganized for better code structure and maintainability.
"""

__version__ = "1.0.0"
__author__ = "Kewei Wei"

# Core modules
from .core.config import ExplorationConfig
from .core.engine import ExplorationEngine
from .core.scene_manager import SceneManager

# Strategy modules
from .strategies import RectangleBasedStrategy

# Manipulation modules
from .manipulation.object_manipulator import ObjectManipulator

# Analysis modules
from .analysis.room_analyzer import RoomAnalyzer

# Vision modules
from .vision.vision import VisionManager, CameraManager, VideoRecorder, TrajectoryVisualizer

# Utility modules
from .utils.utils import (
    read_json_dict, 
    get_moveable_objects, 
    get_receptacle_objects, 
    DataDictManager, 
    get_global_data_manager, 
    analyze_scene_objects
)

__all__ = [
    # Core
    'ExplorationConfig',
    'ExplorationEngine',
    'SceneManager',
    
    # Strategies
    'RectangleBasedStrategy',
    
    # Manipulation
    'ObjectManipulator',
    
    # Analysis
    'RoomAnalyzer',
    
    # Vision
    'VisionManager',
    'CameraManager', 
    'VideoRecorder',
    'TrajectoryVisualizer',
    
    # Utils
    'read_json_dict',
    'get_moveable_objects',
    'get_receptacle_objects',
    'DataDictManager',
    'get_global_data_manager',
    'analyze_scene_objects'
]