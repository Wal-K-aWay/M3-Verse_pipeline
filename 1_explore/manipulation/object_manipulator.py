"""Object manipulation module for AI2Thor exploration.

This module provides functionality for manipulating objects in AI2Thor scenes,
including changing properties of pickupable, moveable and actionable objects.
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple
from ai2thor.controller import Controller
from core.config import ExplorationConfig
from utils.utils import DataDictManager, get_global_data_manager


class ObjectManipulator:
    """Manages object manipulation operations in AI2Thor scenes."""
    
    def __init__(self, controller: Controller, data_manager: DataDictManager = None, config: ExplorationConfig = None, scene_id: str = None, room_analyzer=None):
        """Initialize object manipulator.
        
        Args:
            controller: AI2Thor controller
            data_manager: Data dictionary manager for object information (optional)
            room_analyzer: Room analyzer instance for room information (optional)
        """
        self.controller = controller
        self.data_manager = data_manager or self._get_default_data_manager()
        self.manipulation_log = []
        self.room_analyzer = room_analyzer
        self.executed_operations = []  # Track executed operations

        self.base_output_dir = config.base_output_dir if config else None
        self.scene_id = scene_id
        
    def _get_default_data_manager(self) -> DataDictManager:
        """Get default data manager instance."""
        return get_global_data_manager()
    
    def _is_receptacle_with_objects(self, container_obj: Dict[str, Any], all_objects: List[Dict[str, Any]]) -> bool:
        """Check if a container contains other objects.
        
        Args:
            container_obj: The container object to check
            all_objects: List of all objects in the scene
            
        Returns:
            True if the container contains other objects, False otherwise
        """
        if not container_obj.get('receptacle', False):
            return False
        
        container_id = container_obj['objectId']
        
        for obj in all_objects:
            if obj['objectId'] != container_id:
                parent_receptacles = obj.get('parentReceptacles', [])
                if parent_receptacles and container_id in parent_receptacles:
                    return True
        
        return False
    
    def get_all_possible_operations(self) -> Dict[str, Any]:
        """Get all possible operations that can be performed in the current scene.
        
        Returns:
            Dictionary containing all possible operations organized by type
        """
        objects = self.controller.last_event.metadata.get('objects', [])
        
        operations = {
            'teleport_operations': self._get_teleport_operations(objects),
            'state_change_operations': self._get_state_change_operations(objects),
        }
        
        return operations
    
    def _get_teleport_operations(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get all possible teleport operations.
        
        Args:
            objects: List of all objects in the scene
            
        Returns:
            List of teleport operation possibilities
        """
        teleport_ops = []
        
        # Separate pickupable and moveable objects
        pickupable_objects = [obj for obj in objects if obj.get('pickupable', False)]
        moveable_only_objects = [obj for obj in objects if obj.get('moveable', False) and not obj.get('pickupable', False)]
        
        # Get receptacle objects
        receptacle_objects = [obj for obj in objects if obj.get('receptacle', False)]
        
        # Load pickupable and receptacle dictionaries
        pickupable_dict = self.data_manager.get_pickupable_objects()
        receptacle_dict = self.data_manager.get_receptacle_objects()
        
        # Process pickupable objects (original logic)
        for pickupable_obj in pickupable_objects:
            obj_type = pickupable_obj['objectType']

            if self._is_receptacle_with_objects(pickupable_obj, objects):
                continue

            if pickupable_obj['parentReceptacles'] is None:
                current_parent = None
            else:
                current_parent = pickupable_obj['parentReceptacles'][0]
            
            # Get compatible receptacles for this object type from pickupable_dict
            compatible_receptacles = pickupable_dict.get(obj_type, [])
            
            for receptacle_obj in receptacle_objects:
                receptacle_type = receptacle_obj['objectType']

                receptacle_compatible_objects = receptacle_dict.get(receptacle_type, [])

                if (receptacle_type in compatible_receptacles and 
                    obj_type in receptacle_compatible_objects):
                    teleport_ops.append({
                        'action': 'PlaceObjectAtPoint',
                        'object_id': pickupable_obj['objectId'],
                        'object_type': obj_type,
                        'old_pos': pickupable_obj['position'],
                        'old_receptacle': current_parent,
                        'new_receptacle': receptacle_obj['objectId'],
                        'description': f"Move {pickupable_obj['objectId']} from {current_parent} to {receptacle_obj['objectId']}" if current_parent is not None else f"Place {pickupable_obj['objectId']} at {receptacle_obj['objectId']}",
                    })
        
        # Process moveable-only objects (restricted logic)
        for moveable_obj in moveable_only_objects:
            obj_type = moveable_obj['objectType']

            if self._is_receptacle_with_objects(moveable_obj, objects):
                continue

            if moveable_obj['parentReceptacles'] is None:
                current_parent = None
            else:
                current_parent = moveable_obj['parentReceptacles'][0]
            
            if current_parent is not None:
                target_receptacles = [current_parent]
            else:
                target_receptacles = []

            # if obj_type == 'GarbageBag':
            #     garbage_cans = [obj['objectId'] for obj in receptacle_objects if obj['objectType'] == 'GarbageCan']
            #     target_receptacles.extend(garbage_cans)

            for receptacle_obj in target_receptacles:
                teleport_ops.append({
                    'action': 'PlaceObjectAtPoint',
                    'object_id': moveable_obj['objectId'],
                    'object_type': obj_type,
                    'old_pos': moveable_obj['position'],
                    'old_receptacle': current_parent,
                    'new_receptacle': receptacle_obj,
                    'description': f"Move {moveable_obj['objectId']} from {current_parent} to {receptacle_obj}" if current_parent is not None else f"Place {moveable_obj['objectId']} at {receptacle_obj}",
                })
        
        return teleport_ops
    
    def _get_state_change_operations(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get all possible state change operations.
        
        Args:
            objects: List of all objects in the scene
            
        Returns:
            List of state change operation possibilities
        """
        state_ops = []
        
        for obj in objects:
            obj_id = obj['objectId']
            obj_type = obj['objectType']
            
            # Open/Close operations
            if obj.get('openable', False):
                openness = obj['openness']
                def openness_desc(openness):
                    if openness == 0:
                        desc = 'totally closed'
                    elif openness == 1:
                        desc = 'totally open'
                    else:
                        desc = f'partially open with {openness:.2f} openness'
                    return desc

                if openness != 0: # open(partly, fully)
                    state_ops.append({
                        'action': 'CloseObject',
                        'object_id': obj_id,
                        'object_type': obj_type,
                        'description': f"Close {obj_id} from {openness_desc(openness)} to {openness_desc(0)} openness",
                    })
                if openness != 1:
                    # Randomly select one target openness value instead of iterating through all
                    target_openness = random.choice([0.75, 0.5, 0.25])
                    state_ops.append({
                        'action': 'OpenObject',
                        'object_id': obj_id,
                        'object_type': obj_type,
                        'openness': target_openness,
                        'description': f"Close {obj_id} from {openness_desc(openness)} to {openness_desc(0)}" if target_openness < openness 
                                        else f"Open {obj_id} from {openness_desc(openness)} to {openness_desc(target_openness)}",
                    })
            
            # Slicing operations
            if obj.get('sliceable', False) and not obj.get('isSliced', False):
                state_ops.append({
                    'action': 'SliceObject',
                    'object_id': obj_id,
                    'object_type': obj_type,
                    'description': f"Slice {obj_id} into pieces",
                    
                })
            
            # Breaking operations
            if obj.get('breakable', False) and not obj.get('isBroken', False):
                state_ops.append({
                    'action': 'BreakObject',
                    'object_id': obj_id,
                    'object_type': obj_type,
                    'description': f"Break {obj_id}",
                })
            
            # Toggle operations
            if obj.get('toggleable', False):
                if obj.get('isToggled', False):
                    state_ops.append({
                        'action': 'ToggleObjectOff',
                        'object_id': obj_id,
                        'object_type': obj_type,
                        'description': f"Turn off {obj_id}",
                    })
                else:
                    state_ops.append({
                        'action': 'ToggleObjectOn',
                        'object_id': obj_id,
                        'object_type': obj_type,
                        'description': f"Turn on {obj_id}",
                        
                    })
            
            # Cleanliness state changes
            if obj.get('dirtyable', False):
                if obj.get('isDirty', False):
                    state_ops.append({
                        'action': 'CleanObject',
                        'object_id': obj_id,
                        'object_type': obj_type,
                        'description': f"Clean {obj_id}",
                    })
                else:
                    state_ops.append({
                        'action': 'DirtyObject',
                        'object_id': obj_id,
                        'object_type': obj_type,
                        'description': f"Make {obj_id} dirty",
                    })

            # Cook state changes
            if obj.get('cookable', False):
                if not obj.get('isCooked', False):
                    state_ops.append({
                        'action': 'CookObject',
                        'object_id': obj_id,
                        'object_type': obj_type,
                        'description': f"Cook {obj_id}",
                    })
            
            # Fill/Empty operations
            if obj.get('canFillWithLiquid', False):
                if obj.get('isFilledWithLiquid', False):
                    fillLiquid = obj['fillLiquid']
                    state_ops.append({
                        'action': 'EmptyLiquidFromObject',
                        'object_id': obj_id,
                        'object_type': obj_type,
                        'liquid_type':fillLiquid,
                        'description': f"Empty {obj_id} that was originally filled with {fillLiquid}" if fillLiquid else f"Empty {obj_id}",
                    })
                elif not obj.get('isBroken', False):
                    for liquid in ['water', 'coffee', 'wine']:
                        state_ops.append({
                            'action': 'FillObjectWithLiquid',
                            'object_id': obj_id,
                            'object_type': obj_type,
                            'liquid_type': liquid,
                            'description': f"Fill empty {obj_id} with {liquid}",
                        })

            # Use up operations
            # if obj_type == 'ToiletPaper' or obj_type == 'PaperTowelRoll' or obj_type == 'SoapBottle' or obj_type == 'TissueBox':
            #     import pdb;pdb.set_trace()
            if obj.get('canBeUsedUp', False) and not obj.get('isUsedUp', False):
                state_ops.append({
                    'action': 'UseUpObject',
                    'object_id': obj_id,
                    'object_type': obj_type,
                    'description': f"Use up {obj_id}",
                })
        
        return state_ops
    
    def filter_operations(self, all_operations, operated_objects):
        num_ops = []
        available_operations = {}
        for operation_type, operations in all_operations.items():
            available_ops = []
            for op in operations:
                if op.get('object_id') not in operated_objects:
                    available_ops.append(op)
            available_operations[operation_type] = available_ops
            num_ops.append(len(available_ops))
        
        filtered_ops = []
        min_ops_count = min(num_ops)*2
        if min_ops_count > 0:
            for k, v in available_operations.items():
                filtered_ops += random.sample(v, min(min_ops_count, len(v)))
        random.shuffle(filtered_ops)

        return filtered_ops
    
    def execute_random_operations(self) -> List[Dict[str, Any]]:
        """Execute random operations from all possible operations.
        
        Args:
            max_operations: Maximum number of operations to execute
            
        Returns:
            List of execution results
        """
        results = []
        operated_objects = set()  # Track objects that have been successfully operated on
        all_operations = self.get_all_possible_operations()
        flat_operations = self.filter_operations(all_operations, operated_objects)
        
        max_operations = max(1, min(len(flat_operations) // 10, 15))

        success_num = 0
        while success_num < max_operations:
            
            selected_operation = random.choice(flat_operations)
            
            result = self.execute_operation(selected_operation)
            
            print(f'{result["operation"]["description"]}, {result["success"]}')
            results.append(result)
            
            if result['success']:
                # Add the object to operated objects set
                object_id = selected_operation.get('object_id')
                if object_id:
                    operated_objects.add(object_id)
                    
                    # If it's a SliceObject operation, also add the new sliced objects
                    if selected_operation.get('action') == 'SliceObject':
                        # Get all objects after slicing
                        current_objects = self.controller.last_event.metadata.get('objects', [])
                        # Find new sliced objects that contain the original object ID
                        for obj in current_objects:
                            obj_id = obj['objectId']
                            # Sliced objects typically have 'Sliced' in their ID and contain the original object ID
                            if 'Sliced' in obj_id and object_id in obj_id:
                                operated_objects.add(obj_id)
                
                all_operations = self.get_all_possible_operations()
                flat_operations = self.filter_operations(all_operations, operated_objects)

                success_num += 1
        
        return results
    
    def _create_success_result(self, operation: Dict[str, Any], event) -> Dict[str, Any]:
        result = {
            'operation': operation,
            'success': event.metadata.get('lastActionSuccess', False),
        }
        self.manipulation_log.append(result)
        return result
    
    def _ensure_receptacle_open(self, receptacle_id: str) -> bool:
        objects = self.controller.last_event.metadata.get('objects', [])
        receptacle_obj = next((obj for obj in objects if obj['objectId'] == receptacle_id), None)
        
        if receptacle_obj and receptacle_obj.get('openable', False) and receptacle_obj.get('openness', 0) == 0:
            # Create a complete operation for opening the receptacle
            operation = {
                'action': 'OpenObject',
                'object_id': receptacle_id,
                'object_type': receptacle_obj.get('objectType', 'Unknown'),
                'openness': 1.0,
                'description': f"Open {receptacle_id}"
            }
            
            open_event = self._execute_open_object(operation)
            success = open_event.metadata.get('lastActionSuccess', False)
            
            # Record this operation if successful
            if success:
                operation_record = {
                    'action': 'OpenObject',
                    'object_id': receptacle_id,
                    'description': operation['description'],
                    'execution_order': len(self.executed_operations) + 1
                }
                self.executed_operations.append(operation_record)
            
            return success
        return True
    
    def _get_spawn_positions(self, receptacle_id: str) -> List[Dict[str, float]]:
        pos_event = self.controller.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=receptacle_id,
            anywhere=True,
        )
        if pos_event.metadata.get('lastActionSuccess') and pos_event.metadata.get('actionReturn'):
            return pos_event.metadata['actionReturn']
        return []
    
    def _try_place_at_positions(self, action: str, object_id: str, positions: List[Dict[str, float]]) -> Tuple[bool, str, Any]:
        random.shuffle(positions)
        for position in positions:
            event = self.controller.step(
                action=action,
                objectId=object_id,
                position=position,
            )
            
            if event.metadata.get('lastActionSuccess', False):
                return True, position, event
            
        return False, event.metadata.get('errorMessage', ''), event
    
    def _execute_place_object_at_point(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        object_id = operation.get('object_id')
        receptacle_id = operation.get('new_receptacle')
        
        if not receptacle_id:
            return {'operation': operation, 'success': False, 'error_message': 'Can\'t get available receptacle'}
        
        if not self._ensure_receptacle_open(receptacle_id):
            return {'operation': operation, 'success': False, 'error_message': 'Failed to open receptacle'}
        
        positions = self._get_spawn_positions(receptacle_id)
        if not positions:
            return {'operation': operation, 'success': False, 'error_message': 'Can\'t get spawn coordinates above receptacle'}
        
        res = self._try_place_at_positions('PlaceObjectAtPoint', object_id, positions)
        success = res[0]
        event = res[2]
        if not success:
            error_msg = res[1]
            return {'operation': operation, 'success': False, 'error_message': f'Failed to place object at any position. Last error: {error_msg}'}
        
        operation['new_pos'] = res[1]
        return self._create_success_result(operation, event)
    
    def _execute_open_object(self, operation: Dict[str, Any]) -> Any:
        object_id = operation.get('object_id')
        openness = operation.get('openness', 1)
        return self.controller.step(
            action='OpenObject',
            objectId=object_id,
            openness=openness,
            forceAction=True
        )
    
    def _execute_simple_action(self, action: str, object_id: str) -> Any:
        return self.controller.step(
            action=action,
            objectId=object_id,
            forceAction=True
        )
    
    def _execute_fill_object_with_liquid(self, operation: Dict[str, Any]) -> Any:
        object_id = operation.get('object_id')
        liquid_type = operation['liquid_type']
        return self.controller.step(
            action='FillObjectWithLiquid',
            objectId=object_id,
            fillLiquid=liquid_type,
            forceAction=True
        )
    
    def execute_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific operation with forceAction=True.
        
        Args:
            operation: Operation dictionary from get_all_possible_operations
            
        Returns:
            Result dictionary with execution details
        """
        
        try:
            action = operation['action']
            object_id = operation.get('object_id')
            
            if action == 'PlaceObjectAtPoint':
                result = self._execute_place_object_at_point(operation)
            
            elif action == 'OpenObject':
                event = self._execute_open_object(operation)
                if event.metadata.get('lastActionSuccess', False):
                    result = self._create_success_result(operation, event)
                else:
                    result = {'operation': operation, 'success': False, 'error_message': event.metadata.get('errorMessage', '')}
            
            elif action in ['SliceObject', 'BreakObject', 'CloseObject', 'ToggleObjectOff', 
                           'ToggleObjectOn', 'CleanObject', 'DirtyObject', 'CookObject', 
                           'EmptyLiquidFromObject', 'UseUpObject']:
                # if action == 'SliceObject':
                #     import pdb;pdb.set_trace()
                event = self._execute_simple_action(action, object_id)
                if event.metadata.get('lastActionSuccess', False):
                    result = self._create_success_result(operation, event)
                else:
                    result = {'operation': operation, 'success': False, 'error_message': event.metadata.get('errorMessage', '')}
            
            elif action == 'FillObjectWithLiquid':
                event = self._execute_fill_object_with_liquid(operation)
                if event.metadata.get('lastActionSuccess', False):
                    result = self._create_success_result(operation, event)
                else:
                    result = {'operation': operation, 'success': False, 'error_message': event.metadata.get('errorMessage', '')}
            
            else:
                result = {'operation': operation, 'success': False, 'error_message': f'Unsupported action: {action}'}
            
            # Record successful operations
            if result.get('success', False):
                # operation_record = {
                #     'action': action,
                #     'object_id': object_id,
                #     'description': operation.get('description', ''),
                #     'execution_order': len(self.executed_operations) + 1
                # }
                # self.executed_operations.append(operation_record)
                self.executed_operations.append(operation)
            
            return result
            
        except Exception as e:
            return {'operation': operation, 'success': False, 'error_message': str(e)}

    def set_output_config(self, base_output_dir: str, split: str, scene_id: int):
        self.base_output_dir = base_output_dir
        self.split = split
        self.scene_id = scene_id
        self.executed_operations = []  # Reset operations log for new scene
    
    def get_objects_state(self) -> Dict[str, Any]:
        objects = self.controller.last_event.metadata.get('objects', [])
        objects_state = {}
        
        for obj in objects:
            obj_id = obj['objectId']
            
            # Get room information for this object
            room_id = -1
            room_name = "Unknown"
            if self.room_analyzer and obj.get('position'):
                obj_position = (obj['position']['x'], obj['position']['z'])
                room_id = self.room_analyzer.get_agent_current_room(obj_position)
                if room_id >= 0 and room_id < len(self.room_analyzer.room_names):
                    room_name = self.room_analyzer.room_names[room_id]
            
            state_info = {
                'objectType': obj.get('objectType'),
                'position': obj.get('position'),
                'rotation': obj.get('rotation'),
                'room_id': room_id,
                'room_name': room_name,
                'parentReceptacles': obj.get('parentReceptacles'),
                'receptacleObjectIds': obj.get('receptacleObjectIds'),
                'openness': obj.get('openness'),
                'isOpen': obj.get('isOpen'),
                'isSliced': obj.get('isSliced'),
                'isBroken': obj.get('isBroken'),
                'isToggled': obj.get('isToggled'),
                'isDirty': obj.get('isDirty'),
                'isCooked': obj.get('isCooked'),
                'isFilledWithLiquid': obj.get('isFilledWithLiquid'),
                'fillLiquid': obj.get('fillLiquid'),
                'isUsedUp': obj.get('isUsedUp'),
                'pickupable': obj.get('pickupable'),
                'moveable': obj.get('moveable'),
                'receptacle': obj.get('receptacle'),
                'openable': obj.get('openable'),
                'sliceable': obj.get('sliceable'),
                'breakable': obj.get('breakable'),
                'toggleable': obj.get('toggleable'),
                'dirtyable': obj.get('dirtyable'),
                'cookable': obj.get('cookable'),
                'canFillWithLiquid': obj.get('canFillWithLiquid'),
                'canBeUsedUp': obj.get('canBeUsedUp'),
                'axisAlignedBoundingBox': obj.get('axisAlignedBoundingBox'),
                'objectOrientedBoundingBox': obj.get('objectOrientedBoundingBox'),
            }
            objects_state[obj_id] = state_info
        
        return objects_state
    
    def save_objects_state(self, state_data: Dict[str, Any], filename: str, manipulation_path: str) -> str:
        if not self.base_output_dir or self.scene_id is None:
            raise ValueError("Output configuration not set. Call set_output_config() first.")
        
        output_dir = os.path.join(self.base_output_dir, 'data', f'scene_{self.scene_id}')
        manipulation_dir = os.path.join(output_dir, manipulation_path)
        os.makedirs(manipulation_dir, exist_ok=True)
        
        file_path = os.path.join(manipulation_dir, filename)
        
        save_data = {
            'scene_info': {
                'split': 'train',
                'scene_id': self.scene_id,
                'manipulation_path': manipulation_path
            },
            'objects_count': len(state_data),
            'objects_state': state_data
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Objects state saved to: {file_path}")
        return file_path

    def save_scene_state(self, manipulation_path: str = "manipulation"):
        print("\nSaving objects state before manipulation...")
        self.save_objects_state(self.get_objects_state(), "objects_state.json", manipulation_path)
    
    def save_operation_log(self, filename: str = "operations_log.json", manipulation_path: str = "manipulation") -> str:
        """Save the log of executed operations.
        
        Args:
            filename: Name of the file to save
            manipulation_path: Path within the scene directory
            
        Returns:
            Path to the saved file
        """
        if not self.base_output_dir or self.scene_id is None:
            raise ValueError("Output configuration not set. Call set_output_config() first.")
        
        # Build path structure consistent with VideoRecorder
        output_dir = os.path.join(self.base_output_dir, 'data', f'scene_{self.scene_id}')
        manipulation_dir = os.path.join(output_dir, manipulation_path)
        os.makedirs(manipulation_dir, exist_ok=True)
        
        file_path = os.path.join(manipulation_dir, filename)
        
        # Sort operations by execution order to ensure proper sequence
        sorted_operations = sorted(self.executed_operations, key=lambda x: x.get('execution_order', 0))
        
        save_data = {
            'scene_info': {
                'split': 'train',
                'scene_id': self.scene_id,
                'manipulation_path': manipulation_path
            },
            'operations_count': len(sorted_operations),
            'operations': sorted_operations
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Operations log saved to: {file_path}")
        return file_path
    