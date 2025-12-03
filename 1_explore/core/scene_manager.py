"""Scene management for AI2Thor exploration.

This module provides functionality to load and manage scenes from ProcTHOR-10k dataset
and other scene sources for batch exploration.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class SceneManager:
    """Manages scene loading and selection for exploration."""
    
    def __init__(self, data_dir: str = None):
        """Initialize scene manager.
        
        Args:
            data_dir: Directory containing scene data files
        """
        self.data_dir = data_dir or self._get_default_data_dir()
        self.scenes = []
        self.current_scene_index = 0
    
    def _get_default_data_dir(self) -> str:
        """Get default data directory path."""
        # Get project root directory (parent of parent of explore directory)
        project_root = Path(__file__).parent.parent.parent
        # Use ai2-thor_data subdirectory within project
        data_dir = project_root / "0_ai2-thor_data"
        return str(data_dir)
    
    def load_procthor_scenes(self, split: str = "val", max_scenes: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load scenes from ProcTHOR-10k dataset.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            max_scenes: Maximum number of scenes to load (None for all)
            
        Returns:
            List of scene dictionaries
        """
        scene_file = os.path.join(self.data_dir, "procthor-10k", f"{split}.jsonl")
        
        if not os.path.exists(scene_file):
            raise FileNotFoundError(f"Scene file not found: {scene_file}")
        
        scenes = []
        with open(scene_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if max_scenes and line_num >= max_scenes:
                    break
                try:
                    scene = json.loads(line.strip())
                    scenes.append(scene)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num + 1}: {e}")
                    continue
        
        self.scenes = scenes
        print(f"Loaded {len(scenes)} scenes from {split} split")
        return scenes
    
    def load_custom_scenes(self, scene_file: str) -> List[Dict[str, Any]]:
        """Load scenes from custom file.
        
        Args:
            scene_file: Path to scene file (JSON or JSONL)
            
        Returns:
            List of scene dictionaries
        """
        if not os.path.exists(scene_file):
            raise FileNotFoundError(f"Scene file not found: {scene_file}")
        
        scenes = []
        
        if scene_file.endswith('.jsonl'):
            # JSONL format
            with open(scene_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        scene = json.loads(line.strip())
                        scenes.append(scene)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num + 1}: {e}")
                        continue
        else:
            # JSON format
            with open(scene_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    scenes = data
                else:
                    scenes = [data]
        
        self.scenes = scenes
        print(f"Loaded {len(scenes)} scenes from {scene_file}")
        return scenes
    
    def get_scene(self, index: int) -> Optional[Dict[str, Any]]:
        """Get scene by index.
        
        Args:
            index: Scene index
            
        Returns:
            Scene dictionary or None if index is invalid
        """
        if 0 <= index < len(self.scenes):
            return self.scenes[index]
        return None
    
    def get_current_scene(self) -> Optional[Dict[str, Any]]:
        """Get current scene.
        
        Returns:
            Current scene dictionary or None
        """
        return self.get_scene(self.current_scene_index)
    
    def next_scene(self) -> Optional[Dict[str, Any]]:
        """Move to next scene.
        
        Returns:
            Next scene dictionary or None if at end
        """
        if self.current_scene_index < len(self.scenes) - 1:
            self.current_scene_index += 1
            return self.get_current_scene()
        return None
    
    def reset_scene_index(self):
        """Reset scene index to beginning."""
        self.current_scene_index = 0
    
    def get_scene_count(self) -> int:
        """Get total number of loaded scenes.
        
        Returns:
            Number of scenes
        """
        return len(self.scenes)
    
    def get_scene_info(self, index: int = None) -> Dict[str, Any]:
        """Get scene information.
        
        Args:
            index: Scene index (None for current scene)
            
        Returns:
            Dictionary with scene information
        """
        if index is None:
            index = self.current_scene_index
        
        scene = self.get_scene(index)
        if scene is None:
            return {}
        
        info = {
            'index': index,
            'total_scenes': len(self.scenes),
            'scene_id': scene.get('id', f'scene_{index}'),
        }
        
        # Add additional scene metadata if available
        if 'metadata' in scene:
            info.update(scene['metadata'])
        
        return info