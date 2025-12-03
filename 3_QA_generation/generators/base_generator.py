import os
import json
import math
import random
import pickle

import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from itertools import combinations

from utils.object_attributes_loader import ObjectAttributesLoader
from utils.geometry_utils import analyze_polygon

current_dir = os.path.dirname(os.path.abspath(__file__)) # generators
parent_dir = os.path.dirname(current_dir) # 3_QA_generation
root_dir = os.path.dirname(parent_dir) # M^3-Verse_pipeline

class BaseGenerator(ABC):    
    def __init__(self, scene_path: str):
        self.scene_path = scene_path
        self.attributes_loader = ObjectAttributesLoader(attributes_file_path = os.path.join(root_dir,"2_object_descriptions/descriptions/object_attributes.json"))

        self.load_scene_data()
        self._get_state_room_objects_dict()
        self._get_all_room_and_object_names()
        self._get_all_objects_dict()
        self._get_all_rooms_dict()
        self._load_general_data()

        self.no_valid_option = 'No correct option is listed.'


    def _load_general_data(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        general_data_path = os.path.join(current_path, "../../M^3-Verse/general_data")
        
        all_object_types_path = os.path.join(general_data_path, "all_object_types.json")
        if os.path.exists(all_object_types_path):
            with open(all_object_types_path, 'r', encoding='utf-8') as f:
                self.all_object_types = json.load(f)
        
        # 加载房间类型和功能映射
        room_types_path = os.path.join(general_data_path, "room_types.json")
        if os.path.exists(room_types_path):
            with open(room_types_path, 'r', encoding='utf-8') as f:
                self.room_types = json.load(f)
        
        # 加载房间形状选项
        room_shapes_path = os.path.join(general_data_path, "room_shapes.json")
        if os.path.exists(room_shapes_path):
            with open(room_shapes_path, 'r', encoding='utf-8') as f:
                self.room_shape_options = json.load(f)
        else:
            self.room_shape_options = []
        
        # 加载物体属性选项
        object_colors_path = os.path.join(general_data_path, "object_colors.json")
        if os.path.exists(object_colors_path):
            with open(object_colors_path, 'r', encoding='utf-8') as f:
                self.color_options = json.load(f)
        else:
            self.color_options = []
        
        object_functions_path = os.path.join(general_data_path, "object_functions.json")
        if os.path.exists(object_functions_path):
            with open(object_functions_path, 'r', encoding='utf-8') as f:
                self.function_options = json.load(f)
        else:
            self.function_options = []
        
        object_shapes_path = os.path.join(general_data_path, "object_shapes.json")
        if os.path.exists(object_shapes_path):
            with open(object_shapes_path, 'r', encoding='utf-8') as f:
                self.shape_options = json.load(f)
        else:
            self.shape_options = []


    def load_state_data(self, state_path: str) -> Dict[str, Any]:
        state_data = {}
        
        trajectory_path = os.path.join(state_path, "agent_trajectory.jsonl")
        from collections import defaultdict
        if os.path.exists(trajectory_path):
            trajectory = []
            object_seen_counts = defaultdict(int)
            with open(trajectory_path, 'r', encoding='utf-8') as f:
                for line in f:
                    frame_data = json.loads(line.strip())
                    trajectory.append(frame_data)
                    for obj_id in frame_data['visible_objects']:
                        object_seen_counts[obj_id] += 1
            state_data['agent_trajectory'] = trajectory
        
        objects_state_path = os.path.join(state_path, "objects_state.json")
        if os.path.exists(objects_state_path):
            with open(objects_state_path, 'r', encoding='utf-8') as f:
                objects_state = json.load(f)
        
        filtered_objects_state = {}
        for k, v in objects_state['objects_state'].items():
            # if k in object_seen_counts.keys() and object_seen_counts[k] >= 5:
            if k in object_seen_counts.keys():
                filtered_objects_state[k] = v
        state_data['objects_state'] = {'scene_info': objects_state['scene_info'],
                                           'objects_count': len(filtered_objects_state), 
                                           'objects_state': filtered_objects_state}

        operations_path = os.path.join(state_path, "operations_log.json")
        if os.path.exists(operations_path):
            with open(operations_path, 'r', encoding='utf-8') as f:
                state_data['operations_log'] = json.load(f)
        
        segmentation_path = os.path.join(state_path, "segmentation")
        if os.path.exists(segmentation_path):
            state_data['segmentation_dir'] = segmentation_path
        
        return state_data


    def load_scene_data(self):
        data = {}
        
        object_mapping_path = os.path.join(self.scene_path, "object_asset_mapping.json")
        if os.path.exists(object_mapping_path):
            with open(object_mapping_path, 'r', encoding='utf-8') as f:
                data['object_mapping'] = json.load(f)
    
        room_static_path = os.path.join(self.scene_path, "room_static_analysis.json")
        if os.path.exists(room_static_path):
            with open(room_static_path, 'r', encoding='utf-8') as f:
                data['room_static'] = json.load(f)
        
        for item in os.listdir(self.scene_path):
            if item.startswith("state_"):
                state_path = os.path.join(self.scene_path, item)
                if os.path.isdir(state_path):
                    data[item] = self.load_state_data(state_path)
        
        self.scene_data = data

        self._enrich_all_objects_with_attributes()
        self._analyze_room_shapes()

    def _analyze_room_shapes(self):
        room_static_data = self.scene_data['room_static']
        
        for room_detail in room_static_data['room_static_details']:
            polygon_coords = room_detail['polygon_coordinates']

            shape_name = analyze_polygon(polygon_coords)

            room_detail['shape'] = shape_name
    
    def _enrich_all_objects_with_attributes(self):
        """为所有状态的物体数据添加属性信息"""
        if 'object_mapping' not in self.scene_data:
            return
    
        # 遍历所有状态数据
        for state_key, state_data in self.scene_data.items():
            if state_key in ['room_static', 'object_mapping']:
                continue
            # 使用属性加载器丰富物体信息
            enriched_state = self.attributes_loader.enrich_objects_with_attributes(
                state_data['objects_state'], self.scene_data['object_mapping']
            )
            self.scene_data[state_key]['objects_state'] = enriched_state
    

    def _get_state_room_objects_dict(self) -> List[Dict[str, Any]]:
        self.room_objects = {}
        for state_key in self.scene_data.keys():
            if state_key in ['room_static', 'object_mapping']:
                continue
                
            state_data = self.scene_data[state_key]
            if 'objects_state' not in state_data:
                continue

            objects_dict = state_data['objects_state']
            
            if isinstance(objects_dict, dict):
                if 'objects_state' in objects_dict:
                    actual_objects_state = objects_dict['objects_state']
                else:
                    actual_objects_state = objects_dict
                
                objects = []
                for obj_id, obj_data in actual_objects_state.items():
                    if isinstance(obj_data, dict):
                        obj_copy = obj_data.copy()
                        obj_copy['objectId'] = obj_id
                        objects.append(obj_copy)
            else:
                objects = objects_dict
            
            self.room_objects[state_key] = self._get_single_state_room_objects_dict(objects)

    def _get_single_state_room_objects_dict(self, objects: List[Dict]) -> Dict[str, List[Dict]]:
        # 按房间分组对象
        room_objects = {}
        for obj in objects:
            room_name = obj.get('room_name')
            if room_name is not None:
                if room_name not in room_objects:
                    room_objects[room_name] = []
                room_objects[room_name].append(obj)
        
        return room_objects

    def _get_all_objects_dict(self) -> Dict[str, Dict]:
        all_objects = {}
        for state_key, room_objects in self.room_objects.items():
            for room_name, objects in room_objects.items():
                for obj in objects:
                    all_objects[obj['objectId']] = obj
        
        self.all_objects = all_objects

    def _get_all_rooms_dict(self) -> Dict[str, Dict]:
        all_rooms = {}
        rooms = self.scene_data['room_static']['room_static_details']
        for room in rooms:
            all_rooms[room['room_name']] = room
        
        self.all_rooms = all_rooms


    def calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        return math.sqrt(
            (pos1['x'] - pos2['x'])**2 + 
            (pos1['z'] - pos2['z'])**2
        )
    
    def calculate_3d_box_volume(self, corner_points: List[List[float]]) -> float:
        """
        Calculate the volume of a 3D bounding box from 8 corner points.
        
        Args:
            corner_points: List of 8 points, each point is [x, y, z]
            
        Returns:
            float: Volume of the 3D box
        """
        if len(corner_points) != 8:
            raise ValueError("Expected exactly 8 corner points for a 3D box")

        points = np.array(corner_points)
        # Select the first point as the reference corner
        p0 = points[0]
        # Create vectors from the reference corner to all other points
        vectors = points[1:] - p0
        
        # Calculate dot products for all combinations of 3 vectors
        best_combination = None
        min_dot_product_sum = float('inf')
        
        for v_indices in combinations(range(len(vectors)), 3):
            v1, v2, v3 = vectors[v_indices[0]], vectors[v_indices[1]], vectors[v_indices[2]]
            dot_product_sum = sum([
                abs(np.dot(v1, v2)),
                abs(np.dot(v1, v3)),
                abs(np.dot(v2, v3))
            ])
            
            if dot_product_sum < min_dot_product_sum:
                min_dot_product_sum = dot_product_sum
                best_combination = (v1, v2, v3)
        
        if best_combination is None:
            raise ValueError("Could not find suitable vector combination for volume calculation.")
        
        # Use the most orthogonal combination to calculate volume
        v1, v2, v3 = best_combination
        edge_lengths = [np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)]
        volume = edge_lengths[0] * edge_lengths[1] * edge_lengths[2]
        
        return volume
    

    def get_room_by_id(self, room_id: int) -> Optional[Dict]:
        if 'room_static' in self.scene_data and 'room_static_details' in self.scene_data['room_static']:
            for room in self.scene_data['room_static']['room_static_details']:
                if room['room_id'] == room_id:
                    return room
        return None
    
    def get_objects_in_room(self, room_id: int, state_key: str = 'original') -> List[Dict]:
        objects = []
        if state_key in self.scene_data and 'objects_state' in self.scene_data[state_key]:
            for obj in self.scene_data[state_key]['objects_state']:
                if obj.get('room_id') == room_id:
                    objects.append(obj)
        return objects
    

    def create_range_choices(self, values: List[float], correct_value: float, unit: str, num_choices: int = 4, if_int: bool = False) -> List[str]:
        if not values:
            return False, -1

        min_val = min(values)
        max_val = max(values)

        if correct_value < min_val:
            min_val = correct_value
        if correct_value > max_val:
            max_val = correct_value

        if min_val == max_val:
            if if_int:
                val = int(round(min_val))
                return [f"{val}{unit} - {val}{unit}"], 0
            else:
                return [f"{min_val:.2f}{unit} - {max_val:.2f}{unit}"], 0

        choices = []
        correct_choice = -1

        if if_int:
            min_val = math.floor(min_val)
            max_val = math.ceil(max_val)
            correct_value = int(round(correct_value))

            if max_val - min_val < num_choices - 1:
                max_val = min_val + num_choices - 1

            step = (max_val - min_val) / num_choices
            
            last_end = -1
            for i in range(num_choices):
                if i == 0:
                    start = min_val
                else:
                    start = last_end + 1
                
                if i == num_choices - 1:
                    end = max_val
                else:
                    end = min_val + (i + 1) * step
                
                start = int(round(start))
                end = int(round(end))

                if i < num_choices -1 and start > end:
                    end = start

                if start <= correct_value <= end:
                    correct_choice = i
                
                choices.append(f"{start}{unit} - {end}{unit}")
                last_end = end
        else:
            step = (max_val - min_val) / num_choices
            last_end = -1.0
            for i in range(num_choices):
                if i == 0:
                    start = min_val
                else:
                    start = last_end + 0.01

                if i == num_choices - 1:
                    end = max_val
                else:
                    end = min_val + (i + 1) * step

                if start <= correct_value <= end:
                    correct_choice = i
                
                choices.append(f"{start:.2f}{unit} - {end:.2f}{unit}")
                last_end = end

        # Fallback if correct choice not found
        if correct_choice == -1:
            for i, choice_str in enumerate(choices):
                parts = choice_str.replace(unit, '').split(' - ')
                s = float(parts[0])
                e = float(parts[1])
                if s <= correct_value <= e:
                    correct_choice = i
                    break

        if correct_choice == -1:
            return False, -1
        return choices, correct_choice


    def _count_turns(self, trajectory: List[Dict]) -> Tuple[int, int]:
        start_idx = trajectory[0]['step_count']
        left_turns = []
        right_turns = []
        last_angle = None
        turn_start_idx = 0

        for i, point in enumerate(trajectory):
            rotation = point['rotation']
            if rotation % 90 != 0:
                continue

            if last_angle is None:
                last_angle = rotation
                continue
            
            if rotation == last_angle:
                turn_start_idx = i
                continue
            
            diff = rotation - last_angle
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360

            if diff > 0:
                right_turns.append(list(range(start_idx + turn_start_idx, start_idx + i + 1)))
            elif diff < 0:
                left_turns.append(list(range(start_idx + turn_start_idx, start_idx + i + 1)))
                        
            last_angle = rotation
            turn_start_idx = i
                
        return left_turns, right_turns


    def _get_ordered_states(self):
        states = sorted(
            [key for key in self.scene_data.keys() if key not in ['room_static', 'object_mapping']]
        )
        return states



    def _get_specific_room(self, room: Dict, state_key: str, level: int = 0, another_state: str=None) -> Dict[str, Optional[str]]:
        result = {
            'type': None,
            'biggest_room': None,
            'smallest_room': None,
            'biggest_type': None,
            'smallest_type': None,
            'unique_object': None,
            'shape': None
        }
        
        room_type = room['room_type']
        room_area = room['area']
        room_areas = sorted([r['area'] for r in self.scene_data['room_static']['room_static_details']])
        
        # Check if room type is unique in the scene
        if self.scene_data['room_static']['room_type_distribution'][room_type] == 1:
            result['type'] = room_type
        
        # Check if it's the smallest room in the scene
        if room_area == room_areas[0]:
            if len(room_areas) > 1 and (room_areas[1] - room_areas[0]) / room_areas[1] >= 0.1:
                result['smallest_room'] = 'the smallest room'
        
        # Check if it's the largest room in the scene
        if room_area == room_areas[-1]:
            if len(room_areas) > 1 and (room_areas[-1] - room_areas[-2]) / room_areas[-1] >= 0.1:
                result['biggest_room'] = 'the largest room'
        
        # Filter all rooms with the same room_type as the current room
        same_type_rooms = [r for r in self.scene_data['room_static']['room_static_details'] if r['room_type'] == room_type]
        same_type_room_areas = sorted([r['area'] for r in same_type_rooms])
        
        # Check if it's the smallest room of the same type
        if room_area == same_type_room_areas[0]:
            if len(same_type_room_areas) > 1 and (same_type_room_areas[1] - same_type_room_areas[0]) / same_type_room_areas[1] >= 0.1:
                result['smallest_type'] = f'the smallest {room_type}'
        
        # Check if it's the largest room of the same type
        if room_area == same_type_room_areas[-1]:
            if len(same_type_room_areas) > 1 and (same_type_room_areas[-1] - same_type_room_areas[-2]) / same_type_room_areas[-1] >= 0.1:
                result['biggest_type'] = f'the largest {room_type}'
        
        # Check if room shape is unique in the scene
        current_room_shape = None
        shape_counts = {}
        
        for room_data in self.scene_data['room_static']['room_static_details']:
            room_shape = room_data['shape']
            if room_shape in shape_counts:
                shape_counts[room_shape] += 1
            else:
                shape_counts[room_shape] = 1
            
            if room_data['room_name'] == room['room_name']:
                current_room_shape = room_shape
        
        # Only set shape if it's unique in the scene
        if current_room_shape and shape_counts[current_room_shape] == 1:
            result['shape'] = f'the {current_room_shape} room'

        # Check if the room contains objects that can be uniquely referenced without room context
        if (level <= 1 
            and state_key 
            and another_state
            and state_key in self.room_objects 
            and room['room_name'] in self.room_objects[state_key] 
            and room['room_name'] in self.room_objects[another_state]
            and level == 0
            and another_state is not None):
            # Get all objects in this room
            room_objects = self.room_objects[state_key][room['room_name']]
            another_state_room_objects_ = self.room_objects[another_state][room['room_name']] ###
            another_state_room_objects = [obj['objectId'] for obj in another_state_room_objects_]
            
            # Find objects that can be uniquely referenced without using room context
            unique_objects = {}
            for obj in room_objects:
                obj_id = obj['objectId']
                if obj_id not in another_state_room_objects:
                    continue
                
                # Get object reference without providing room_name to avoid room context
                obj_references = self._get_specific_object(obj, state_key, level=level + 1)
                
                # Check if we can get a unique reference that doesn't involve room context
                obj_reference = None
                for ref_type, ref_value in obj_references.items():
                    if ref_value is not None and ref_type != 'type_in_room' and ref_type != 'position':  # Exclude room-based references
                        obj_reference = ref_value
                        break
                
                if obj_reference:
                    unique_objects[obj_id] = f'the room with {obj_reference} in it'
            
            # Store all unique object references
            if unique_objects:
                result['unique_object'] = unique_objects
        
        return result

    def _get_specific_object(self, object: Dict, state_key: str, level: int = 0) -> Dict[str, Optional[str]]:
        # Check if attributes exist for this object
        object_state = self.scene_data[state_key]['objects_state']['objects_state'][object['objectId']]
        attributes = object_state.get('attributes', {}) if object_state else {}
        
        result = {
            'type': None,
            'type_in_room': None,
            'in_receptacle': None,
            'with_contents': None,
            'size': None,
            'position': None,
            'attribute': None,
            # 'description': None
        }
            
        all_other_objects = []
        receptacle = None
        contents = []
        for objects_in_room in self.room_objects[state_key].values():
            all_other_objects.extend(objects_in_room)
            for obj in objects_in_room:
                if obj['receptacleObjectIds'] and object['objectId'] in obj['receptacleObjectIds']:
                    receptacle = obj
                if object['receptacleObjectIds'] and obj['objectId'] in object['receptacleObjectIds']:
                    contents.append(obj)

        # Check if object type is unique in the scene
        object_type_distribute = {}
        for obj in all_other_objects:
            obj_type = obj['objectType']
            if obj_type in object_type_distribute.keys():
                object_type_distribute[obj_type] += 1
            else:
                object_type_distribute[obj_type] = 1
        
        object_type = object['objectType']
        if object_type_distribute[object_type] == 1:
            result['type'] = object_type
        else:
            # If there are multiple objects of the same type globally, try to distinguish by size or attributes first
            same_type_objects_global = [obj for obj in all_other_objects if obj['objectType'] == object_type]
            
            # First try to distinguish by size using objectOrientedBoundingBox
            size_distinction_success = False
            try:
                # Calculate volume for each object of the same type globally
                object_sizes = []
                for obj in same_type_objects_global:
                    obj_state = self.scene_data[state_key]['objects_state']['objects_state'][obj['objectId']]
                    if obj_state and obj_state['objectOrientedBoundingBox'] is not None:
                        bbox = obj_state['objectOrientedBoundingBox']
                        volume = self.calculate_3d_box_volume(bbox['cornerPoints'])
                        object_sizes.append((obj['objectId'], volume))
                
                # Sort by volume to determine size ranking
                if len(object_sizes) >= 2:
                    object_sizes.sort(key=lambda x: x[1])  # Sort by volume
                    current_obj_id = object['objectId']
                    
                    # Find current object's position in size ranking
                    for i, (obj_id, volume) in enumerate(object_sizes):
                        if obj_id == current_obj_id:
                            if i == 0:  # Smallest
                                # Check if smallest is at least 10% smaller than second smallest
                                if len(object_sizes) > 1:
                                    smallest_volume = object_sizes[0][1]
                                    second_smallest_volume = object_sizes[1][1]
                                    if second_smallest_volume > 0 and (second_smallest_volume - smallest_volume) / second_smallest_volume >= 0.1:
                                        if len(object_sizes) > 2:
                                            result['size'] = f"the smallest {object_type}"
                                        else:
                                            result['type'] = f"the smaller {object_type}"
                                        size_distinction_success = True
                            elif i == len(object_sizes) - 1:  # Largest
                                # Check if largest is at least 10% larger than second largest
                                if len(object_sizes) > 1:
                                    largest_volume = object_sizes[-1][1]
                                    second_largest_volume = object_sizes[-2][1]
                                    if second_largest_volume > 0 and (largest_volume - second_largest_volume) / second_largest_volume >= 0.1:
                                        if len(object_sizes) > 2:
                                            result['size'] = f"the largest {object_type}"
                                        else:
                                            result['size'] = f"the larger {object_type}"
                                        size_distinction_success = True
                            break
            except (KeyError, TypeError): # If size comparison fails, continue to attribute-based distinction
                pass
            
            pos_distinction_success = False
            # If size-based distinction didn't work, try height-based distinction for related objects
            if not size_distinction_success and '___' in object['objectId']:
                # Get the prefix of current object
                current_obj_id = object['objectId']
                current_prefix = current_obj_id.split('___')[0]
                
                # Find all objects with the same prefix
                current_group = []
                for obj in same_type_objects_global:
                    obj_id = obj['objectId']
                    if '___' in obj_id and obj_id.split('___')[0] == current_prefix:
                         current_group.append(obj)
                 
                # If current object is in a group with multiple related objects, try height distinction
                if current_group and len(current_group) >= 2:
                    try:
                        # Calculate height (Y coordinate) for each object in the group
                        object_heights = []
                        for obj in current_group:
                            obj_state = self.scene_data[state_key]['objects_state']['objects_state'][obj['objectId']]
                            if obj_state and 'position' in obj_state:
                                height = obj_state['position']['y']
                                object_heights.append((obj['objectId'], height))

                        # Sort by height to determine ranking
                        if len(object_heights) >= 2:
                            object_heights.sort(key=lambda x: x[1])  # Sort by height
                            
                            # Check if there are objects with same height (within 0.05 units)
                            heights_only = [h[1] for h in object_heights]
                            has_same_height = False
                            for i in range(len(heights_only)):
                                for j in range(i + 1, len(heights_only)):
                                    if abs(heights_only[i] - heights_only[j]) < 0.05:
                                        has_same_height = True
                                        break
                                if has_same_height:
                                    break
                            
                            # If there are objects with same height, skip height distinction
                            if has_same_height:
                                pass  # Cannot use height distinction
                            else:
                                # Find current object's position in height ranking
                                for i, (obj_id, height) in enumerate(object_heights):
                                    if obj_id == current_obj_id:
                                        total_objects = len(object_heights)
                                        
                                        # Check height differences with adjacent objects
                                        height_diff_valid = True
                                        current_height = object_heights[i][1]
                                        
                                        # Check difference with lower neighbor
                                        if i > 0:
                                            lower_height = object_heights[i-1][1]
                                            if current_height - lower_height < 0.1:
                                                height_diff_valid = False
                                        
                                        # Check difference with higher neighbor
                                        if i < total_objects - 1:
                                            higher_height = object_heights[i+1][1]
                                            if higher_height - current_height < 0.1:
                                                height_diff_valid = False
                                        
                                        if height_diff_valid:
                                            if i == 0:  # Lowest
                                                if total_objects > 2:
                                                    result['position'] = f"the lowest {object_type}"
                                                else:
                                                    result['position'] = f"the lower {object_type}"
                                                pos_distinction_success = True
                                            elif i == 1:  # Second lowest
                                                if total_objects > 3:
                                                    result['position'] = f"the second lowest {object_type}"
                                                elif total_objects == 3:
                                                    result['position'] = f"the middle {object_type}"
                                                pos_distinction_success = True
                                            elif i == total_objects - 2:  # Second highest
                                                if total_objects > 3:
                                                    result['position'] = f"the second highest {object_type}"
                                                elif total_objects == 3:
                                                    result['position'] = f"the middle {object_type}"
                                                pos_distinction_success = True
                                            elif i == total_objects - 1:  # Highest
                                                if total_objects > 2:
                                                    result['position'] = f"the highest {object_type}"
                                                else:
                                                    result['position'] = f"the higher {object_type}"
                                                pos_distinction_success = True
                                        break
                    except (KeyError, TypeError):
                        pass
            
            # If size-based distinction didn't work, try attributes for distinguishing globally
            if not pos_distinction_success:
                # Check each attribute individually first
                shape_unique = False
                # material_unique = False
                color_unique = False
                
                # Check shape attribute
                if 'shape' in attributes and attributes['shape'] is not None:
                    shape_unique = True
                    for other_obj in same_type_objects_global:
                        if other_obj['objectId'] != object['objectId']:
                            other_obj_state = self.scene_data[state_key]['objects_state']['objects_state'][other_obj['objectId']]
                            other_attrs = other_obj_state.get('attributes', {}) if other_obj_state else {}
                            if 'shape' in other_attrs and other_attrs['shape'] is not None:
                                # Check if one shape contains the other or vice versa
                                current_shape = str(attributes['shape']).lower()
                                other_shape = str(other_attrs['shape']).lower()
                                # if current_shape in other_shape or other_shape in current_shape:
                                if current_shape == other_shape:
                                    shape_unique = False
                                    break
                
                # Check color attribute
                if 'color' in attributes and attributes['color'] is not None:
                    color_unique = True
                    for other_obj in same_type_objects_global:
                        if other_obj['objectId'] != object['objectId']:
                            other_obj_state = self.scene_data[state_key]['objects_state']['objects_state'][other_obj['objectId']]
                            other_attrs = other_obj_state.get('attributes', {}) if other_obj_state else {}
                            if 'color' in other_attrs and other_attrs['color'] is not None:
                                # Check if one color contains the other or vice versa
                                current_color = str(attributes['color']).lower()
                                other_color = str(other_attrs['color']).lower()
                                # if current_color in other_color or other_color in current_color:
                                if current_color == other_color:
                                    color_unique = False
                                    break
                
                # Determine the best attribute combination
                distinguishing_attrs = []
                
                # Priority: single attribute first
                if color_unique:
                    distinguishing_attrs = [attributes['color']]
                elif shape_unique:
                    distinguishing_attrs = [attributes['shape']]
                # elif material_unique:
                #     distinguishing_attrs = [attributes['material']]
                # If no single attribute works, try shape + color combination
                elif ('shape' in attributes and attributes['shape'] is not None and 
                      'color' in attributes and attributes['color'] is not None):
                    # Check if shape + color combination is unique
                    shape_color_unique = True
                    for other_obj in same_type_objects_global:
                        if other_obj['objectId'] != object['objectId']:
                            other_obj_state = self.scene_data[state_key]['objects_state']['objects_state'][other_obj['objectId']]
                            other_attrs = other_obj_state.get('attributes', {}) if other_obj_state else {}
                            if ('shape' in other_attrs and other_attrs['shape'] is not None and
                                'color' in other_attrs and other_attrs['color'] is not None):
                                current_shape = str(attributes['shape']).lower()
                                other_shape = str(other_attrs['shape']).lower()
                                current_color = str(attributes['color']).lower()
                                other_color = str(other_attrs['color']).lower()
                                if ((current_shape in other_shape or other_shape in current_shape) and
                                    (current_color in other_color or other_color in current_color)):
                                    shape_color_unique = False
                                    break
                    if shape_color_unique:
                        distinguishing_attrs = [attributes['shape'], attributes['color']]
                
                # Build description with distinguishing attributes and store in result['attribute']
                if distinguishing_attrs:
                    attr_description = " ".join(distinguishing_attrs)
                    result['attribute'] = f"the {attr_description} {object_type}"

        # Check if object type is unique in the room (fallback if global distinction failed)
        room_name = None
        if level == 0:  # Only get room reference at level 0 to prevent recursion
            current_room = None
            for room in self.scene_data['room_static']['room_static_details']:
                if room['room_name'] == object['room_name']:
                    current_room = room
                    break
            if current_room is not None:
                room_references = self._get_specific_room(current_room, state_key=state_key, level=level + 1)
                # Use the first available room reference
                for ref_type, ref_value in room_references.items():
                    if ref_value is not None:
                        room_name = ref_value
                        break
        
        if level == 0 and room_name is not None:
            other_objects_in_same_room = [obj for obj in self.room_objects[state_key][object['room_name']]]
            object_type_distribute_in_room = {}
            for obj in other_objects_in_same_room:
                obj_type = obj['objectType']
                if obj_type in object_type_distribute_in_room.keys():
                    object_type_distribute_in_room[obj_type] += 1
                else:
                    object_type_distribute_in_room[obj_type] = 1

            if object_type_distribute_in_room[object_type] == 1:
                result['type_in_room'] = f"the {object_type} in {room_name}"
            # ###
            else:
                # If there are multiple objects of the same type in the room, first try to distinguish by size
                same_type_objects_in_room = [obj for obj in other_objects_in_same_room if obj['objectType'] == object_type]
                if len(same_type_objects_in_room) > 1:
                    # First try to distinguish by size using axisAlignedBoundingBox
                    try:
                        # Calculate volume for each object of the same type
                        object_sizes = []
                        for obj in same_type_objects_in_room:
                            obj_state = self.scene_data[state_key]['objects_state']['objects_state'][obj['objectId']]
                            if obj_state and obj_state['objectOrientedBoundingBox'] is not None:
                                bbox = obj_state['objectOrientedBoundingBox']
                                volume = self.calculate_3d_box_volume(bbox['cornerPoints'])
                                object_sizes.append((obj['objectId'], volume))
                        
                        # Sort by volume to determine size ranking
                        if len(object_sizes) >= 2:
                            object_sizes.sort(key=lambda x: x[1])  # Sort by volume
                            current_obj_id = object['objectId']
                            
                            # Find current object's position in size ranking
                            for i, (obj_id, volume) in enumerate(object_sizes):
                                if obj_id == current_obj_id:
                                    if i == 0:  # Smallest
                                        # Check if smallest is at least 10% smaller than second smallest
                                        if len(object_sizes) > 1:
                                            smallest_volume = object_sizes[0][1]
                                            second_smallest_volume = object_sizes[1][1]
                                            if second_smallest_volume > 0 and (second_smallest_volume - smallest_volume) / second_smallest_volume >= 0.1:
                                                if len(object_sizes) > 2:
                                                    result['size'] = f"the smallest {object_type} in {room_name}"
                                                else:
                                                    result['size'] = f"the smaller {object_type} in {room_name}"
                                    elif i == len(object_sizes) - 1:  # Largest
                                        # Check if largest is at least 10% larger than second largest
                                        if len(object_sizes) > 1:
                                            largest_volume = object_sizes[-1][1]
                                            second_largest_volume = object_sizes[-2][1]
                                            if second_largest_volume > 0 and (largest_volume - second_largest_volume) / second_largest_volume >= 0.1:
                                                if len(object_sizes) > 2:
                                                    result['size'] = f"the largest {object_type} in {room_name}"
                                                else:
                                                    result['size'] = f"the larger {object_type} in {room_name}"
                                    # Could add more size descriptions like "medium-sized" if needed
                                    break
                    except (KeyError, TypeError):
                        # If size comparison fails, fall back to attribute-based distinction
                        pass


        # Check if object is in a receptacle (only allow one level of recursion)
        if level == 0 and receptacle:
            # Check if there are other objects of the same type in the same receptacle
            same_type_in_receptacle = False
            if receptacle['receptacleObjectIds']:
                for other_obj_id in receptacle['receptacleObjectIds']:
                    if other_obj_id != object['objectId']:
                        # Find the object by ID
                        for room_objs in self.room_objects[state_key].values():
                            for other_obj in room_objs:
                                if other_obj['objectId'] == other_obj_id and other_obj['objectType'] == object_type:
                                    same_type_in_receptacle = True
                                    break
                            if same_type_in_receptacle:
                                break
                    if same_type_in_receptacle:
                        break
            
            # Only use receptacle reference if no same type objects in the same receptacle
            if not same_type_in_receptacle:
                receptacle_references = self._get_specific_object(receptacle, state_key, level=level + 1)
                # Use the first available receptacle reference
                receptacle_name = None
                for ref_type, ref_value in receptacle_references.items():
                    if ref_value is not None and ref_type != 'with_contents':
                        receptacle_name = ref_value
                        break
                
                if receptacle_name is not None:
                    result['in_receptacle'] = f"the {object_type} in/on {receptacle_name}"
        
        # Check if object has contents (only for level 0 to avoid recursion)
        if contents and level == 0:
            types = {}
            for content in contents:
                content_type = content['objectType']
                if content_type in types.keys():
                    types[content_type] += 1
                else:
                    types[content_type] = 1
            
            receptacle_objects = ""
            for idx, (content_type, num) in enumerate(types.items()):
                if len(types.keys()) == 1:
                    receptacle_objects = f"a/an {content_type}"
                elif idx == len(types.keys()) - 1:
                    receptacle_objects = receptacle_objects[:-2]
                    if num == 1:
                        receptacle_objects += f" and a/an {content_type}"
                    else:
                        receptacle_objects += f" and {num} {content_type}s"
                else:
                    if num == 1:
                        receptacle_objects += f"a/an {content_type}, "
                    else:
                        receptacle_objects += f"{num} {content_type}s, "
            
            result['with_contents'] = f"the {object_type} with {receptacle_objects} in/on it"
        
        return result

    def _get_all_room_names(self, state_key: str, another_state: str=None) -> Dict[str, Dict[str, Optional[str]]]:
        all_room_names = {}
        
        if 'room_static' in self.scene_data and 'room_static_details' in self.scene_data['room_static']:
            for room in self.scene_data['room_static']['room_static_details']:
                room_id = room['room_name']
                room_names = self._get_specific_room(room, state_key=state_key, another_state=another_state)
                all_room_names[room_id] = room_names
                
        return all_room_names
    
    def _get_all_object_names(self, state_key: str): 
        object_names = {}
        
        for room_name, objects in self.room_objects[state_key].items():
            for obj in objects:
                object_id = obj['objectId']
                obj_names = self._get_specific_object(obj, state_key)
                object_names[object_id] = obj_names
                
        return object_names
    
    def _get_all_room_and_object_names(self):
        self.room_names = {}
        self.object_names = {}

        states = []
        for state_key in self.scene_data.keys():
            if state_key not in ['room_static', 'object_mapping']:
                states.append(state_key)
        states = sorted(states)
        for i, state_key in enumerate(states):
            self.room_names[state_key] = self._get_all_room_names(state_key, another_state=states[i-1])
            self.object_names[state_key] = self._get_all_object_names(state_key)

    def _select_name(self, name_dict: Dict[str, Any], available_keys: List[str], first_mode: bool = True) -> Tuple[str, str]:
        if first_mode:
            for key in available_keys:
                if key in name_dict and name_dict[key] is not None:
                    if key != 'unique_object':
                        return name_dict[key], key
                    else:
                        candidate_names = list(name_dict[key].values())
                        if candidate_names:
                            return random.choice(candidate_names), key
        else:
            candidate_data = []
            for key in available_keys:
                if key in name_dict and name_dict[key] is not None:
                    if key != 'unique_object':
                        candidate_data.append((name_dict[key], key))
                    else:
                        unique_objects = list(name_dict[key].values())
                        for obj_name in unique_objects:
                            candidate_data.append((obj_name, key))
            if candidate_data:
                return random.choice(candidate_data)
        return None, None


    def get_visible_objs(self, state_key: str, room_name: str) -> List[Dict]:
        trajectory = self.scene_data[state_key]['agent_trajectory']
        visible_obj_ids = set()
        for frame in trajectory:
            if frame['room_name'] == room_name:
                visible_obj_ids.update(frame['visible_objects'])

        visible_objs = []
        for obj_id, obj in self.scene_data[state_key]['objects_state']['objects_state'].items():
            if obj_id in visible_obj_ids and obj['room_name'] == room_name:
                tmp_obj = obj.copy()
                tmp_obj['objectId'] = obj_id
                visible_objs.append(tmp_obj)
        return visible_objs


    def get_room_visit_frames(self, state_key: str, room_name: str) -> List[int]:
        """
        Get all frame indices where the agent is in the specified room
        
        Args:
            state_key: State key
            room_name: Room name to filter results for
            
        Returns:
            List of integers, each representing a frame index where the agent is in the room
            e.g., [0, 1, 2, 9, 10, 11] means agent was in the room during frames 0, 1, 2, 9, 10, and 11
        """
        if state_key not in self.scene_data or 'agent_trajectory' not in self.scene_data[state_key]:
            return []
        
        trajectory = self.scene_data[state_key]['agent_trajectory']
        
        # Find all frames where agent is in the specified room
        room_frames = []
        for frame_idx, frame_data in enumerate(trajectory):
            current_room_name = frame_data.get('room_name')
            if current_room_name == room_name:
                room_frames.append(frame_idx)
        
        return room_frames
      
    def get_object_visible_frames(self, state_key: str, obj_id: str) -> List[int]:
        """
        Get all frame indices where the specified object is visible
        
        Args:
            state_key: State key
            obj_id: Object ID to filter results for
            
        Returns:
            List of integers, each representing a frame index where the object is visible
            e.g., [0, 1, 2, 9, 10, 11] means object was visible during frames 0, 1, 2, 9, 10, and 11
        """
        if state_key not in self.scene_data or 'agent_trajectory' not in self.scene_data[state_key]:
            return []
        
        trajectory = self.scene_data[state_key]['agent_trajectory']
        
        # Find all frames where object is visible
        object_frames = []
        for frame_idx, frame_data in enumerate(trajectory):
            visible_objects = frame_data.get('visible_objects', [])
            if obj_id in visible_objects:
                object_frames.append(frame_idx)
        
        return object_frames
    
    def select_proper_room_frames(self, state_key: str, room: Dict[str, Any]) -> List[int]:
        """
        Select proper room frames based on clip length and boundary frame exclusion.

        Args:
            state_key: State key
            room: Room dictionary containing room information

        Returns:
            List of selected room frame indices.
        """
        room_frames = self.get_room_visit_frames(state_key, room['room_name'])
        room_clips = self.find_continous_clips(room_frames)
        
        if not room_clips:
            return []
        
        best_clip = None
        preferred_clips = []
        fallback_clips = []
        
        for room_clip in room_clips:
            # Check if clip contains start frame (0)
            contains_boundary = 0 in room_clip
            clip_length = len(room_clip)
            
            clip_info = {
                'clip': room_clip,
                'length': clip_length,
                'contains_boundary': contains_boundary
            }
            
            if contains_boundary:
                fallback_clips.append(clip_info)
            else:
                preferred_clips.append(clip_info)
        
        # First try to select from preferred clips (not containing boundary frames)
        candidate_clips = preferred_clips if preferred_clips else fallback_clips
        
        # Select the clip with maximum length
        if candidate_clips:
            best_clip_info = max(candidate_clips, key=lambda x: x['length'])
            best_clip = best_clip_info['clip']
        
        return best_clip if best_clip is not None else []
    
    def select_proper_connected_room_frames(self, state_key: str, room_tuple: Tuple[Dict[str, Any], Dict[str, Any]]) -> List[int]:
        """
        Select proper connected room frames that show continuous visit from room1 to room2 or room2 to room1.
        
        Args:
            state_key: State key
            room_tuple: Tuple containing two room names (room1, room2)
            
        Returns:
            List of selected frame indices showing the longest continuous transition between rooms.
        """
        room1_name = room_tuple[0]['room_name']
        room2_name = room_tuple[1]['room_name']
        
        # Get visit frames for each room using the same approach as select_proper_room_frames
        room1_frames = self.get_room_visit_frames(state_key, room1_name)
        room2_frames = self.get_room_visit_frames(state_key, room2_name)
        
        if not room1_frames or not room2_frames:
            return []

        # Combine and sort all frames from both rooms
        all_frames = sorted(set(room1_frames + room2_frames))
        
        # Find continuous clips that contain frames from both rooms
        all_clips = self.find_continous_clips(all_frames)
        connected_clips = []
        
        for clip in all_clips:
            # Check if this clip contains frames from both rooms
            clip_has_room1 = any(frame in room1_frames for frame in clip)
            clip_has_room2 = any(frame in room2_frames for frame in clip)
            
            if clip_has_room1 and clip_has_room2:
                connected_clips.append(clip)
        
        if not connected_clips:
            return []
        
        # Select the longest connected clip
        best_clip = max(connected_clips, key=len)
        
        return best_clip
    
    def select_proper_room_frames_with_objects(self, state_key: str, room: Dict[str, Any]) -> List[int]:
        """
        Select proper room frames based on object frames.

        Args:
            room_clips: List of room frame indices.
            obj_frames: List of object frame indices.

        Returns:
            Tuple of list of selected room frame indices and selected objects.
        """
        # num_frames = len(os.listdir(self.scene_data[state_key]['segmentation_dir']))
        room_frames = self.get_room_visit_frames(state_key, room['room_name'])
        room_clips = self.find_continous_clips(room_frames)

        room_visible_objs = self.get_visible_objs(state_key, room['room_name'])
        selected_objs = self.get_different_height_objs(room_visible_objs)
        obj_frames = []
        for obj in selected_objs:
            obj_frames.append(self.get_object_visible_frames(state_key, obj[0]))
        
        obj_frames_unions = self.find_frame_unions(obj_frames)
        
        best_clip = None
        preferred_clips = []
        fallback_clips = []
        
        for room_clip in room_clips:
            # Check if clip contains start frame (0) or end frame (num_frames-1)
            contains_boundary = 0 in room_clip
            obj_visible_frames = self.find_frame_intersections([room_clip, obj_frames_unions])
            visible_ratio = len(obj_visible_frames) / len(room_clip) if len(room_clip) > 0 else 0
            
            clip_info = {
                'clip': room_clip,
                'visible_ratio': visible_ratio,
                'contains_boundary': contains_boundary
            }
            
            if contains_boundary:
                fallback_clips.append(clip_info)
            else:
                preferred_clips.append(clip_info)
        
        # First try to select from preferred clips (not containing boundary frames)
        candidate_clips = preferred_clips if preferred_clips else fallback_clips
        
        # Select the clip with highest visible_ratio
        if candidate_clips:
            best_clip_info = max(candidate_clips, key=lambda x: x['visible_ratio'])
            best_clip = best_clip_info['clip']
        
        return best_clip if best_clip is not None else [], selected_objs

    def select_proper_objects_in_frames(self, state_key: str, start_id: int, end_id: int) -> List[Tuple[str, float]]:
        """
        Select proper objects with different heights in the specified frame range.
        
        Args:
            state_key: State key
            start_id: Start frame index
            end_id: End frame index
            
        Returns:
            List of tuples containing object ID and height
        """
        # Get the room name for the specified frame range
        trajectory = self.scene_data[state_key]['agent_trajectory']
        if not trajectory or start_id >= len(trajectory) or end_id >= len(trajectory):
            return []
        
        # Verify all frames are in the same room
        room_name = trajectory[start_id]['room_name']
        for frame_idx in range(start_id + 1, end_id + 1):
            if trajectory[frame_idx]['room_name'] != room_name:
                return []
        
        # Get all visible objects in the room
        room_visible_objs = self.get_visible_objs(state_key, room_name)
        if not room_visible_objs:
            return []
        
        # Filter objects that are visible in the specified frame range
        visible_obj_ids = set()
        for frame_idx in range(start_id, end_id + 1):
            visible_obj_ids.update(trajectory[frame_idx]['visible_objects'])
        
        # Filter objects that are both in the room and visible in the frame range
        valid_objs = []
        for obj in room_visible_objs:
            if obj['objectId'] in visible_obj_ids:
                valid_objs.append(obj)
        
        if not valid_objs:
            return []
        
        # Select objects with different heights
        selected_objs = self.get_different_height_objs(valid_objs, num=3)
        
        return selected_objs

    def select_proper_room_frames_with_most_visible_objects(self, state_key: str, room: Dict[str, Any]) -> List[int]:
        """
        Select a continuous 5-frame video clip that can see the most objects in the room.
        
        Args:
            state_key: State key
            room: Room dictionary containing room information
            
        Returns:
            List of 5 consecutive frame indices with maximum visible objects
        """
        room_frames = self.get_room_visit_frames(state_key, room['room_name'])
        room_clips = self.find_continous_clips(room_frames)
        
        if not room_clips:
            return []
        
        # Get all objects in this room
        room_visible_objs = self.get_visible_objs(state_key, room['room_name'])
        if not room_visible_objs:
            return []
        
        # Get object IDs for visibility checking
        obj_ids = [obj['objectId'] for obj in room_visible_objs]
        
        best_clip = None
        max_visible_objects = 0
        preferred_clips = []
        fallback_clips = []
        
        # Check all possible 5-frame windows in each room clip
        for room_clip in room_clips:
            if len(room_clip) < 5:
                continue
                
            # Use sliding window to find 5-frame segments
            for i in range(len(room_clip) - 4):
                window = room_clip[i:i+5]
                
                # Check if window contains boundary frame (0)
                contains_boundary = 0 in window
                
                # Count unique visible objects across all frames in this window
                visible_objects_in_window = set()
                
                if state_key in self.scene_data and 'agent_trajectory' in self.scene_data[state_key]:
                    trajectory = self.scene_data[state_key]['agent_trajectory']
                    
                    for frame_idx in window:
                        if frame_idx < len(trajectory):
                            frame_data = trajectory[frame_idx]
                            visible_objects = frame_data.get('visible_objects', [])
                            # Only count objects that belong to this room
                            for obj_id in visible_objects:
                                if obj_id in obj_ids:
                                    visible_objects_in_window.add(obj_id)
                
                visible_count = len(visible_objects_in_window)
                
                clip_info = {
                    'clip': window,
                    'visible_count': visible_count,
                    'contains_boundary': contains_boundary
                }
                
                if contains_boundary:
                    fallback_clips.append(clip_info)
                else:
                    preferred_clips.append(clip_info)
        
        # First try to select from preferred clips (not containing boundary frames)
        candidate_clips = preferred_clips if preferred_clips else fallback_clips
        
        # Select the clip with maximum visible objects
        if candidate_clips:
            best_clip_info = max(candidate_clips, key=lambda x: x['visible_count'])
            best_clip = best_clip_info['clip']
        
        return best_clip if best_clip is not None else []

    def get_room_turn_frmaes(self, state_key: str) -> Dict[str, List[List[int]]]:
        
        trajectory = self.scene_data[state_key]['agent_trajectory']
        trajectory_len = len(trajectory)
        
        room_turns = {}
        current_room_trajectory = []
        last_room_id = None
        
        for frame_idx, point in enumerate(trajectory):
            room_id = point['room_name']

            # 如果进入新房间或第一个点
            if last_room_id != room_id or frame_idx == trajectory_len - 1:
                if frame_idx == trajectory_len - 1:
                    current_room_trajectory.append(point)
                # 处理前一个房间的轨迹段
                if len(current_room_trajectory) >= 4:
                    left_turns, right_turns = self._count_turns(current_room_trajectory)
                    if last_room_id not in room_turns:
                        room_turns[last_room_id] = [[], [], []]  # [total_turns, left_turns, right_turns]
                    total_turns = left_turns + right_turns
                    total_turns.sort()
                    room_turns[last_room_id][0].extend(total_turns)
                    room_turns[last_room_id][1].extend(left_turns)
                    room_turns[last_room_id][2].extend(right_turns)
                
                # 开始新房间的轨迹段
                last_room_id = room_id
                current_room_trajectory = [point]
            else:
                # 继续当前房间的轨迹
                current_room_trajectory.append(point)
        
        return room_turns
        
    def find_frame_intersections(self, frames_list: List[List[int]]) -> List[int]:
        """
        Find the intersection of frame numbers between multiple lists of frame numbers.
        
        Args:
            frames_list: List of frame number lists
            
        Returns:
            List of frame numbers that represent the intersection
        """
        if not frames_list:
            return []
        
        # Convert all frame lists to sets of individual frame numbers
        all_frame_sets = []
        for frames in frames_list:
            all_frame_sets.append(set(frames))
        
        # Find intersection of all frame sets
        intersection_frames = all_frame_sets[0]
        for frame_set in all_frame_sets[1:]:
            intersection_frames = intersection_frames.intersection(frame_set)
        
        if not intersection_frames:
            return []
        
        # Sort the intersection frames
        sorted_frames = sorted(list(intersection_frames))
        
        return sorted_frames

    def find_frame_unions(self, frames_list: List[List[int]]) -> List[int]:
        """
        Find the union of frame numbers between multiple lists of frame numbers.
        
        Args:
            frames_list: List of frame number lists
            
        Returns:
            List of frame numbers that represent the union
        """
        if not frames_list:
            return []
        
        # Convert all frame lists to sets of individual frame numbers
        all_frame_sets = []
        for frames in frames_list:
            all_frame_sets.append(set(frames))
        
        # Find union of all frame sets
        union_frames = all_frame_sets[0]
        for frame_set in all_frame_sets[1:]:
            union_frames = union_frames.union(frame_set)
        
        if not union_frames:
            return []
        
        # Sort the union frames
        sorted_frames = sorted(list(union_frames))
        
        return sorted_frames


    def find_continous_clips(self, frames: List[int]) -> List[List[int]]:
        """
        Find the continuous clips from the frame list.

        Args:
            frames: List of frame numbers

        Returns:
            List of continuous clips
        """
        if not frames:
            return []
        
        clips = []
        current_clip = [frames[0]]
        
        for i in range(1, len(frames)):
            if frames[i] == frames[i-1] + 1:
                current_clip.append(frames[i])
            else:
                clips.append(current_clip)
                current_clip = [frames[i]]
        
        clips.append(current_clip)
        
        return clips


    def get_object_bounding_box(self, instance_seg: np.ndarray, object_id_to_color: Dict[str, List[int]], obj_id: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Get object bounding box (x1,y1,x2,y2) in the specified frame through segmentation file
        
        Args:
            state_key: State key
            frame_idx: Frame index
            obj_id: Object ID
            
        Returns:
            Bounding box coordinates (x1, y1, x2, y2) or None if not found
        """
        target_color = object_id_to_color[obj_id]
        
        mask = np.all(instance_seg == target_color, axis=2)
        coords = np.where(mask)
        
        # Check if the object is visible in this frame
        if len(coords[0]) == 0 or len(coords[1]) == 0:
            return None
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))

    def get_one_frame_data(self, frame_id: int, state_key: str, objs: List[Tuple[str, float]] = None, obj_size: bool = False) -> Dict[str, Any]:
        """
        Get one frame data from scene data
        
        Returns:
            One frame data
        """
        frame_data = {
            'frame_id': frame_id,
            'room_id': None,
            'objects': None
            }

        # get room_type
        trajectory = self.scene_data[state_key]['agent_trajectory']
        frame_data['room_id'] = trajectory[frame_id]['room_name']
        

        if objs is not None:
            segmentation_dir = self.scene_data[state_key]['segmentation_dir']
            seg_file_path = os.path.join(segmentation_dir, f"frame_{frame_id}.pkl")
            with open(seg_file_path, 'rb') as f:
                seg_data = pickle.load(f)

            # Get instance segmentation and object mapping
            instance_seg = seg_data.get('instance_segmentation')  # [h, w, 3] numpy array
            object_id_to_color = seg_data.get('object_id_to_color', {})

            for obj_id, obj_height in objs: ###
                bbox = self.get_object_bounding_box(instance_seg, object_id_to_color, obj_id)
                if bbox is None:
                    continue
                if frame_data['objects'] is None:
                    frame_data['objects'] = []
                
                frame_data['objects'].append(
                    {
                        'obj_id': obj_id,
                        'type': self.all_objects[obj_id]['objectType'],
                        'bbox': bbox,
                        'height': obj_height if obj_size else None,
                    }
                )

        return frame_data

    def get_clip_frames_data(self, frame_ids: List[int], state_key: str, objs: List[Tuple[str, float]] = None, obj_size: bool = False) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Get clip frames data
        
        Args:
            frame_ids: List of frame ids
            state_key: State key
            objs: List of object ids
            
        Returns:
            Tuple of (clip frames data, whether all frames are in the same room)
        """
        frames_data = [self.get_one_frame_data(frame_id, state_key, objs, obj_size) for frame_id in frame_ids]
        
        # Check if all frames are in the same room
        room_order = []
        if frames_data:
            first_room_type = frames_data[0]['room_id']
            room_order.append(first_room_type)
            for frame_data in frames_data[1:]:
                if frame_data['room_id'] != room_order[-1]:
                    room_order.append(frame_data['room_id'])
        
        return frames_data, room_order

    def get_one_clip_data(self, frame_ids: List[int], state_key: str, objs: List[Tuple[str, float]] = None, obj_size: bool = False, rooms: bool = False, room_area: bool = False) -> Dict[str, Any]:
        """
        Get one clip data from scene data
        
        Returns:
            One clip data or None if obj_ids is not None but all frames have no objects
        # """
        frames, room_order = self.get_clip_frames_data(frame_ids, state_key, objs, obj_size)
        
        # Check if obj_ids is not None but all frames have no objects
        objects = None
        if objs is not None:
            all_objects_none = all(frame['objects'] is None for frame in frames)
            if all_objects_none:
                return None
            objects = [obj[0] for obj in objs]
        
        rooms_info = None
        if rooms:
            area = None
            if room_area:
                area = {}
                room_data = self.scene_data['room_static']['room_static_details']
                room_ids = list(set(room_order))
                for room in room_data:
                    if room['room_name'] in room_ids:
                        area[room['room_name']] = room['area']
            
            rooms_info = {
                'visit_order': room_order,
                'area': area,
            }
        
        clip_data = {
                'frames': frames,
                'objects': objects,
                'rooms': rooms_info,
            }

        return clip_data

    def get_obj_volumes(self, objs: List[Dict]) -> Dict[str, float]:
        """
        Get object volumes
        
        Args:
            objs: List of objects
            
        Returns:
            Dictionary of object volumes
        """
        obj_volumes = {}
        for obj in objs:
            obj_id = obj['objectId']
            rotation = obj['rotation']
            if round(rotation['x']) % 90 == 0 and round(rotation['y']) % 90 == 0 and round(rotation['z']) % 90 == 0:
                obj_volumes[obj_id] = obj['axisAlignedBoundingBox']['size']['x'] * obj['axisAlignedBoundingBox']['size']['y'] * obj['axisAlignedBoundingBox']['size']['z']
            # elif obj['objectOrientedBoundingBox'] is not None:
            #     volume = self.calculate_3d_box_volume(obj['objectOrientedBoundingBox']['cornerPoints'])
            #     obj_volumes[obj_id] = volume
        
        return obj_volumes

    def get_obj_heights(self, objs: List[Dict]) -> Dict[str, float]:
        """
        Get object heights
        
        Args:
            objs: List of objects
            
        Returns:
            Dictionary of object heights
        """
        obj_heights = {}
        for obj in objs:
            obj_id = obj['objectId']
            rotation = obj['rotation']
            if round(rotation['x']) % 90 == 0 and round(rotation['y']) % 90 == 0 and round(rotation['z']) % 90 == 0:
                obj_heights[obj_id] = obj['axisAlignedBoundingBox']['size']['y']
        
        return obj_heights
    
    def get_biggest_objs(self, objs: List[Dict], num: int = 2) -> List[Dict]:
        """
        Get the biggest num objects from a list of objects
        
        Args:
            objs: List of objects
            num: Number of objects to get
            
        Returns:
            List of biggest num objects
        """
        obj_volume = self.get_obj_volumes(objs)    
        sorted_objs = sorted(obj_volume.keys(), key=lambda x: obj_volume[x], reverse=True)
        return sorted_objs[:num]

    def get_different_size_objs(self, objs: List[Dict], num: int = 3) -> List[Dict]:
        """
        Get num objects with different sizes from a list of objects, evenly distributed.

        Args:
            objs: List of objects
            num: Number of objects to get

        Returns:
            List of num objects with different sizes
        """
        if not objs or num <= 0:
            return []

        obj_volume = self.get_obj_volumes(objs)
        # Sort objects by volume in ascending order
        sorted_objs_by_volume = sorted(obj_volume.keys(), key=lambda x: obj_volume[x])

        if num >= len(sorted_objs_by_volume):
            return sorted_objs_by_volume

        selected_objs = []
        if num == 1:
            # If only one object is requested, return the median size object
            selected_objs.append(sorted_objs_by_volume[len(sorted_objs_by_volume) // 2])
        elif num == 2:
            # If two objects are requested, return the smallest and largest
            selected_objs.append(sorted_objs_by_volume[0])
            selected_objs.append(sorted_objs_by_volume[-1])
        else:
            # For more than two objects, take the smallest, largest, and then distribute the rest
            selected_objs.append(sorted_objs_by_volume[0])
            selected_objs.append(sorted_objs_by_volume[-1])
            remaining_num = num - 2
            # Calculate step for the remaining objects to be evenly distributed in the middle
            if len(sorted_objs_by_volume) - 2 > 0:
                step = (len(sorted_objs_by_volume) - 2) / remaining_num
                for i in range(remaining_num):
                    index = 1 + int(i * step)  # Start from the second element
                    selected_objs.append(sorted_objs_by_volume[index])
            else:
                # This case should ideally not be reached if num > 2 and len(sorted_objs_by_volume) >= num
                # But as a fallback, if there are no 'middle' elements, just return what we have
                pass

        return selected_objs

    def get_different_height_objs(self, objs: List[Dict], num: int = 3) -> List[Dict]:
        """
        Get num objects with different heights from a list of objects, evenly distributed and spatially dispersed.

        Args:
            objs: List of objects
            num: Number of objects to get

        Returns:
            List of num objects with different heights and maximum spatial dispersion
        """
        if not objs or num <= 0:
            return []

        obj_height = self.get_obj_heights(objs) 
        
        # Filter objects with valid heights and positions
        valid_objs = []
        for obj in objs:
            obj_id = obj['objectId']
            if obj_id in obj_height and obj_height[obj_id] > 0.1 and 'position' in obj:
                valid_objs.append(obj)
        
        if not valid_objs:
            return []
            
        if num >= len(valid_objs):
            result = []
            for obj in valid_objs:
                obj_id = obj['objectId']
                result.append((obj_id, obj_height[obj_id]))
            return result

        # Sort objects by height in ascending order
        valid_objs.sort(key=lambda x: obj_height[x['objectId']])
        # import pdb;pdb.set_trace()

        selected_objs = []
        if num == 1:
            # If only one object is requested, return the median size object
            selected_objs.append(valid_objs[len(valid_objs) // 2])
        elif num == 2:
            # For two objects, select the ones with maximum distance among height extremes
            candidates = [valid_objs[0], valid_objs[-1]]  # smallest and largest height
            # Also consider some middle height objects for better spatial distribution
            if len(valid_objs) > 2:
                mid_idx = len(valid_objs) // 2
                candidates.extend([valid_objs[mid_idx-1], valid_objs[mid_idx], valid_objs[mid_idx+1]])
            
            # Find the pair with maximum distance
            max_distance = 0
            best_pair = [candidates[0], candidates[1]]
            for i in range(len(candidates)):
                for j in range(i+1, len(candidates)):
                    distance = self.calculate_distance(candidates[i]['position'], candidates[j]['position'])
                    if distance > max_distance:
                        max_distance = distance
                        best_pair = [candidates[i], candidates[j]]
            selected_objs = best_pair
        else:
            # For more than two objects, use greedy selection for maximum spatial dispersion
            # Start with the smallest and largest height objects
            selected_objs = [valid_objs[0], valid_objs[-1]]
            remaining_candidates = valid_objs[1:-1]  # Exclude already selected objects
            
            # Greedily select remaining objects to maximize minimum distance to already selected ones
            for _ in range(num - 2):
                if not remaining_candidates:
                    break
                    
                best_candidate = None
                max_min_distance = 0
                
                for candidate in remaining_candidates:
                    # Calculate minimum distance to all already selected objects
                    min_distance = float('inf')
                    for selected_obj in selected_objs:
                        distance = self.calculate_distance(candidate['position'], selected_obj['position'])
                        min_distance = min(min_distance, distance)
                    
                    # Select candidate with maximum minimum distance (most dispersed)
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_candidate = candidate
                
                if best_candidate:
                    selected_objs.append(best_candidate)
                    remaining_candidates.remove(best_candidate)
            
        # Convert selected objects to tuples of (objectId, height)
        result = []
        for obj in selected_objs:
            obj_id = obj['objectId']
            height = obj_height[obj_id]
            result.append((obj_id, height))
        
        return result
    
    def get_video_num(self, state: str) -> str:
        return 'first' if state.split('_')[-1] == '0' else 'second'


    @abstractmethod
    def generate_questions(self) -> List[Dict[str, Any]]:
        pass
