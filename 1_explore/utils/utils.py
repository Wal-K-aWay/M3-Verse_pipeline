"""Utility functions for AI2Thor exploration.

Contains helper functions for data processing and object management.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

__all__ = [
    'read_json_dict',
    'get_moveable_objects', 
    'get_receptacle_objects',
    'format_position',
    'get_scene_bounds',
    'validate_position',
    'round_position',
    'filter_objects_by_type',
    'get_object_positions',
    'create_exploration_summary',
    'DataDictManager',
    'get_global_data_manager',
    'load_global_data_dicts'
]


def read_json_dict(json_file_path: str) -> Optional[Dict[str, Any]]:
    """Read JSON file and return as dictionary.
    
    Args:
        json_file_path: Path to JSON file
        
    Returns:
        Dictionary from JSON file or None if error
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"读取JSON文件失败 {json_file_path}: {e}")
        return None


def get_moveable_objects(controller) -> List[Dict[str, Any]]:
    """Get list of moveable objects in the scene.
    
    Args:
        controller: AI2Thor controller instance
        
    Returns:
        List of moveable object dictionaries
    """
    try:
        event = controller.step(action="GetObjects")
        if event.metadata['lastActionSuccess']:
            objects = event.metadata['objects']
            moveable_objects = [
                obj for obj in objects 
                if obj.get('moveable', False) and obj.get('visible', False)
            ]
            return moveable_objects
        else:
            print("获取对象列表失败")
            return []
    except Exception as e:
        print(f"获取可移动对象时出错: {e}")
        return []


def get_receptacle_objects(controller) -> List[Dict[str, Any]]:
    """Get list of receptacle objects in the scene.
    
    Args:
        controller: AI2Thor controller instance
        
    Returns:
        List of receptacle object dictionaries
    """
    try:
        event = controller.step(action="GetObjects")
        if event.metadata['lastActionSuccess']:
            objects = event.metadata['objects']
            receptacle_objects = [
                obj for obj in objects 
                if obj.get('receptacle', False) and obj.get('visible', False)
            ]
            return receptacle_objects
        else:
            print("获取对象列表失败")
            return []
    except Exception as e:
        print(f"获取容器对象时出错: {e}")
        return []


def format_position(position: Dict[str, float], precision: int = 2) -> str:
    """Format position dictionary as string.
    
    Args:
        position: Position dictionary with x, y, z keys
        precision: Number of decimal places
        
    Returns:
        Formatted position string
    """
    return f"({position['x']:.{precision}f}, {position['y']:.{precision}f}, {position['z']:.{precision}f})"


def get_scene_bounds(controller) -> Optional[Dict[str, Any]]:
    """Get scene boundary information.
    
    Args:
        controller: AI2Thor controller instance
        
    Returns:
        Scene bounds dictionary or None if failed
    """
    try:
        event = controller.step(action="GetMapViewCameraProperties")
        if event.metadata['lastActionSuccess']:
            return event.metadata.get('sceneBounds')
        else:
            print("Failed to get scene bounds")
            return None
    except Exception as e:
        print(f"Error while getting scene bounds: {e}")
        return None


def validate_position(position: Dict[str, float]) -> bool:
    """Validate if position dictionary has required keys.
    
    Args:
        position: Position dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['x', 'y', 'z']
    return all(key in position for key in required_keys)


def round_position(position: Dict[str, float], precision: int = 2) -> Dict[str, float]:
    """Round position values to specified precision.
    
    Args:
        position: Position dictionary
        precision: Number of decimal places
        
    Returns:
        Position dictionary with rounded values
    """
    return {
        'x': round(position['x'], precision),
        'y': round(position['y'], precision),
        'z': round(position['z'], precision)
    }


def filter_objects_by_type(objects: List[Dict[str, Any]], object_type: str) -> List[Dict[str, Any]]:
    """Filter objects by their type.
    
    Args:
        objects: List of object dictionaries
        object_type: Type of objects to filter for
        
    Returns:
        Filtered list of objects
    """
    return [obj for obj in objects if obj.get('objectType', '').lower() == object_type.lower()]


def get_object_positions(objects: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Extract positions from object list.
    
    Args:
        objects: List of object dictionaries
        
    Returns:
        List of position dictionaries
    """
    positions = []
    for obj in objects:
        if 'position' in obj and validate_position(obj['position']):
            positions.append(obj['position'])
    return positions


# def create_exploration_summary(stats: Dict[str, Any]) -> str:
#     """Create a formatted summary of exploration statistics.
    
#     Args:
#         stats: Exploration statistics dictionary
        
#     Returns:
#         Formatted summary string
#     """
#     summary = f"""
# 探索总结:
# ========
# 总步数: {stats.get('total_steps', 0)}
# 可达位置总数: {stats.get('total_reachable_positions', 0)}
# 已访问位置: {stats.get('visited_positions', 0)}
# 覆盖率: {stats.get('coverage_percentage', 0):.2f}%
# 轨迹长度: {stats.get('trajectory_length', 0)}
# """
#     return summary


class DataDictManager:
    """Manages AI2Thor object data dictionaries."""
    
    def __init__(self, data_dir: str = None):
        """Initialize data dictionary manager.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir or self._get_default_data_dir()
        self.pickupable_dict = None
        self.receptacle_dict = None
        self.actionable_properties_dict = None
        self._loaded = False
    
    def _get_default_data_dir(self) -> str:
        """Get default data directory path."""
        # Try to find data directory relative to current file
        current_dir = Path(__file__).parent.parent
        data_dir = current_dir / "data"
        if data_dir.exists():
            return str(data_dir)
        
        # Fallback to absolute path
        return "/data/kww/ai2-thor/data"
    
    def load_all_dicts(self) -> bool:
        """Load all data dictionaries.
        
        Returns:
            True if all dictionaries loaded successfully
        """
        success = True
        
        # Load pickupable objects dictionary
        pickupable_file = os.path.join(self.data_dir, "pickupable_dict.json")
        self.pickupable_dict = read_json_dict(pickupable_file)
        if self.pickupable_dict is None:
            print(f"Warning: Failed to load pickupable dictionary from {pickupable_file}")
            success = False
        else:
            print(f"Loaded pickupable dictionary with {len(self.pickupable_dict)} entries")
        
        # Load receptacle objects dictionary
        receptacle_file = os.path.join(self.data_dir, "receptacle_dict.json")
        self.receptacle_dict = read_json_dict(receptacle_file)
        if self.receptacle_dict is None:
            print(f"Warning: Failed to load receptacle dictionary from {receptacle_file}")
            success = False
        else:
            print(f"Loaded receptacle dictionary with {len(self.receptacle_dict)} entries")
        
        # Load actionable properties dictionary
        actionable_file = os.path.join(self.data_dir, "actionable_properties_dict.json")
        self.actionable_properties_dict = read_json_dict(actionable_file)
        if self.actionable_properties_dict is None:
            print(f"Warning: Failed to load actionable properties dictionary from {actionable_file}")
            success = False
        else:
            print(f"Loaded actionable properties dictionary with {len(self.actionable_properties_dict)} entries")
        
        self._loaded = success
        return success
    
    def get_pickupable_objects(self) -> Dict[str, Any]:
        """Get pickupable objects dictionary.
        
        Returns:
            Pickupable objects dictionary
        """
        if not self._loaded:
            self.load_all_dicts()
        return self.pickupable_dict or {}
    
    def get_receptacle_objects(self) -> Dict[str, Any]:
        if not self._loaded:
            self.load_all_dicts()
        return self.receptacle_dict or {}
    
    def get_actionable_properties(self) -> Dict[str, Any]:
        if not self._loaded:
            self.load_all_dicts()
        return self.actionable_properties_dict or {}
    
    def is_sliceable(self, object_type: str) -> bool:
        actionable_dict = self.get_actionable_properties()
        return object_type in actionable_dict.get('sliceable', [])
    
    def is_pickupable(self, object_type: str) -> bool:
        pickupable_dict = self.get_pickupable_objects()
        return object_type in pickupable_dict
    
    def is_receptacle(self, object_type: str) -> bool:
        receptacle_dict = self.get_receptacle_objects()
        return object_type in receptacle_dict
    
    def is_openable(self, object_type: str) -> bool:
        actionable_dict = self.get_actionable_properties()
        return object_type in actionable_dict.get('openable', [])

    def is_toggleable(self, object_type: str) -> bool:
        actionable_dict = self.get_actionable_properties()
        return object_type in actionable_dict.get('toggleable', [])

    def is_fillable(self, object_type: str) -> bool:    
        actionable_dict = self.get_actionable_properties()
        return object_type in actionable_dict.get('fillable', [])

    def is_breakable(self, object_type: str) -> bool:
        actionable_dict = self.get_actionable_properties()
        return object_type in actionable_dict.get('breakable', [])

    def is_dirtyable(self, object_type: str) -> bool:
        actionable_dict = self.get_actionable_properties()
        return object_type in actionable_dict.get('dirtyable', [])

    def is_movable(self, object_type: str) -> bool:
        actionable_dict = self.get_actionable_properties()
        return object_type in actionable_dict.get('moveable', [])
    
    def is_cookable(self, object_type: str) -> bool:
        """Check if object type is cookable."""
        actionable_dict = self.get_actionable_properties()
        return object_type in actionable_dict.get('cookable', [])
    
    def is_usedup(self, object_type: str) -> bool:
        """Check if object type can be used up."""
        actionable_dict = self.get_actionable_properties()
        return object_type in actionable_dict.get('usedup', [])



# Global data dictionary manager instance
_global_data_manager = None


def get_global_data_manager() -> DataDictManager:
    """Get global data dictionary manager instance.
    
    Returns:
        Global DataDictManager instance
    """
    global _global_data_manager
    if _global_data_manager is None:
        _global_data_manager = DataDictManager()
    return _global_data_manager


def load_global_data_dicts() -> bool:
    """Load global data dictionaries.
    
    Returns:
        True if all dictionaries loaded successfully
    """
    manager = get_global_data_manager()
    return manager.load_all_dicts()

def analyze_scene_objects(controller, data_manager: DataDictManager):
    """Analyze objects in the current scene using data dictionaries.
    
    Args:
        controller: AI2Thor controller
        data_manager: Data dictionary manager
    """
    print("\n   Analyzing scene objects...")
    
    try:
        event = controller.last_event
        if event and hasattr(event, 'metadata'):
            objects = event.metadata.get('objects', [])
            
            if objects:
                pickupable_count = 0
                receptacle_count = 0
                other_count = 0
                object_types = set()
                
                for obj in objects:
                    obj_type = obj.get('objectType', 'Unknown')
                    object_types.add(obj_type)
                    
                    if data_manager.is_pickupable(obj_type):
                        pickupable_count += 1
                    elif data_manager.is_receptacle(obj_type):
                        receptacle_count += 1
                    else:
                        other_count += 1
                
                print(f"   Total objects: {len(objects)}")
                print(f"   Unique object types: {len(object_types)}")
                print(f"   Pickupable objects: {pickupable_count}")
                print(f"   Receptacle objects: {receptacle_count}")
                print(f"   Other objects: {other_count}")
                
                # Show some example object types
                if len(object_types) > 0:
                    example_types = list(object_types)[:5]
                    print(f"   Example types: {', '.join(example_types)}")
            else:
                print("   No objects found in scene metadata")
        else:
            print("   No scene metadata available")
            
    except Exception as e:
        print(f"   Error analyzing objects: {str(e)}")