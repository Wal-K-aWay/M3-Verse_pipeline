import json
import os
from typing import Dict, Any, Optional

class ObjectAttributesLoader:
    """Object attributes loader"""
    
    def __init__(self, attributes_file_path: str):
        self.attributes_file_path = attributes_file_path
        self.attributes_data = self._load_attributes()
    
    def _load_attributes(self) -> Dict[str, Any]:
        """Load object attributes data"""
        if not os.path.exists(self.attributes_file_path):
            print(f"Warning: Attributes file not found at {self.attributes_file_path}")
            return {}
        
        try:
            with open(self.attributes_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading attributes file: {e}")
            return {}
    
    def get_attributes_by_asset_id(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Get object attributes by asset_id"""
        if asset_id in self.attributes_data:
            # Return the first attribute object (usually each asset_id has only one attribute object)
            return self.attributes_data[asset_id] if self.attributes_data[asset_id] else None
        return None
    
    def create_object_id_to_attributes_mapping(self, object_asset_mapping: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Create mapping from object_id to attributes"""
        mapping = {}
        
        for object_id, asset_id in object_asset_mapping.items():
            attributes = self.get_attributes_by_asset_id(asset_id)
            if attributes:
                mapping[object_id] = attributes
        
        return mapping
    
    def enrich_objects_with_attributes(self, objects_state: Dict[str, Any], object_asset_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Enrich object state data with attribute information"""
        # Create mapping from object_id to attributes
        id_to_attributes = self.create_object_id_to_attributes_mapping(object_asset_mapping)
        
        # Add attribute information for each object
        objs_state = objects_state.get('objects_state', [])
        for object_id in objs_state.keys():
            if object_id in id_to_attributes:
                objs_state[object_id]['attributes'] = id_to_attributes[object_id]

        enriched_state = objects_state.copy()
        enriched_state['objects_state'] = objs_state
        
        return enriched_state