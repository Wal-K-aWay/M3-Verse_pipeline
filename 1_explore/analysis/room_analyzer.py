#!/usr/bin/env python3
"""
Room analysis module for AI2Thor exploration.

This module provides functionality to analyze room information including:
- Room areas and counts
- Room connectivity relationships
- Agent time spent in each room
- Object distribution across rooms
"""

import json
import os
import numpy as np
from typing import Dict, Tuple, Any
from shapely.geometry import Point, Polygon
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


class RoomAnalyzer:
    """Analyzes room information and agent behavior in AI2Thor scenes."""
    
    def __init__(self, controller, grid_size):
        """Initialize room analyzer.
        
        Args:
            controller: AI2Thor controller instance
        """
        self.controller = controller
        self.grid_size = grid_size
        self.rooms_data = []
        self.room_polygons = []
        self.room_areas = []
        self.room_connectivity = {}
        self.agent_room_time = defaultdict(int)  # Changed to int for frame counts
        self.object_room_mapping = {}
        self.room_names = []
        self.agent_trajectory_rooms = []
        self.last_room_id = None
        self.frame_count = 0
        
        self._initialize_rooms()
    
    def _initialize_rooms(self):
        """Initialize room data from scene metadata."""
        try:
            scene_data = self.controller.scene
            self.rooms_data = scene_data['rooms']
            
            # Process room data
            for i, room in enumerate(self.rooms_data):
                room_id = room['id']
                self.room_names.append(room_id)
                
                # Create polygon from floor polygon
                if 'floorPolygon' in room:
                    floor_points = [(p['x'], p['z']) for p in room['floorPolygon']]
                    if len(floor_points) >= 3:
                        polygon = Polygon(floor_points)
                        self.room_polygons.append(polygon)
                        self.room_areas.append(polygon.area)
                    else:
                        self.room_polygons.append(None)
                        self.room_areas.append(0.0)
                else:
                    self.room_polygons.append(None)
                    self.room_areas.append(0.0)
            
            print(f"Initialized {len(self.rooms_data)} rooms")
            self._analyze_room_connectivity(scene_data['doors'])
            self._map_objects_to_rooms(scene_data['objects'])
            
        except Exception as e:
            print(f"Error initializing rooms: {e}")
        
    def _analyze_room_connectivity(self, doors_data):
        """Analyze connectivity between rooms using doors data."""
        self.room_connectivity = {i: [] for i in range(len(self.room_polygons))}
        
        # Use doors_data to determine room connectivity
        for door in doors_data:
            openness = door.get('openness')
            room0 = door.get('room0')
            room1 = door.get('room1')

            # Check if door is open and connects two different rooms
            if ((openness is None or openness == 1) and 
                room0 is not None and 
                room1 is not None and 
                room0 != room1):
                room0_id = self.room_names.index(room0)
                room1_id = self.room_names.index(room1)
                
                # Add bidirectional connectivity
                if room1_id not in self.room_connectivity[room0_id]:
                    self.room_connectivity[room0_id].append(room1_id)
                if room0_id not in self.room_connectivity[room1_id]:
                    self.room_connectivity[room1_id].append(room0_id)
    
    def _map_objects_to_rooms(self, objects):
        """Map objects to their respective rooms."""
        try:
            for obj in objects:
                obj_id = obj.get('objectId', '')
                obj_pos = obj.get('position', {})
                
                if 'x' in obj_pos and 'z' in obj_pos:
                    point = Point(obj_pos['x'], obj_pos['z'])
                    
                    # Find which room contains this object
                    for room_idx, polygon in enumerate(self.room_polygons):
                        if polygon and polygon.contains(point):
                            self.object_room_mapping[obj_id] = room_idx
                            break
                    else:
                        # Object not in any room
                        self.object_room_mapping[obj_id] = -1
                        
        except Exception as e:
            print(f"Error mapping objects to rooms: {e}")
    
    def get_agent_current_room(self, position: Tuple[float, float]) -> int:
        """Get the room ID where the agent is currently located.
        
        Args:
            position: Agent position as (x, z) tuple
            
        Returns:
            Room ID (index) or -1 if not in any room
        """
        point = Point(position[0], position[1])
        
        for room_idx, polygon in enumerate(self.room_polygons):
            if polygon and polygon.contains(point):
                return room_idx
        
        return -1  # Not in any room
    
    def update_agent_room_time(self, position: Tuple[float, float]):
        """Update the frame count agent spends in each room.
        
        Args:
            position: Agent position as (x, z) tuple
        """
        current_room = self.get_agent_current_room(position)
        
        # Increment frame count for current room
        if current_room >= 0:  # Valid room
            self.agent_room_time[current_room] += 1
        
        # Update tracking variables
        self.last_room_id = current_room
        self.frame_count += 1
        
        # Record room in trajectory
        self.agent_trajectory_rooms.append(current_room)
    
    def get_room_statistics(self) -> Dict[str, Any]:
        """Get comprehensive room statistics.
        
        Returns:
            Dictionary containing room statistics
        """
        stats = {
            'total_rooms': len(self.rooms_data),
            'room_details': [],
            'connectivity_matrix': self.room_connectivity,
            'agent_room_frames': dict(self.agent_room_time),
            'object_distribution': {},
            'total_area': sum(self.room_areas),
            'average_room_area': np.mean(self.room_areas) if self.room_areas else 0,
            'room_types': []
        }
        
        # Room details
        for i, room in enumerate(self.rooms_data):
            room_detail = {
                'room_id': i,
                'room_name': self.room_names[i] if i < len(self.room_names) else f'room_{i}',
                'room_type': room.get('roomType', 'Unknown'),
                'area': self.room_areas[i] if i < len(self.room_areas) else 0,
                'connected_rooms': self.room_connectivity.get(i, []),
                'agent_frames_spent': self.agent_room_time.get(i, 0),
                'objects_count': 0
            }
            stats['room_details'].append(room_detail)
            stats['room_types'].append(room.get('roomType', 'Unknown'))
        
        # Object distribution
        room_object_counts = Counter(self.object_room_mapping.values())
        for room_id, count in room_object_counts.items():
            if room_id >= 0:  # Valid room
                room_name = self.room_names[room_id] if room_id < len(self.room_names) else f'room_{room_id}'
                stats['object_distribution'][room_name] = count
                # Update room details
                if room_id < len(stats['room_details']):
                    stats['room_details'][room_id]['objects_count'] = count
        
        # Room type distribution
        stats['room_type_distribution'] = Counter(stats['room_types'])
        
        return stats
    
    def _extract_polygon_coordinates(self, room_index: int) -> Dict[str, Any]:
        """Extract polygon coordinates for a room.
        
        Args:
            room_index: Index of the room to analyze
            
        Returns:
            Dictionary containing polygon coordinate information
        """
        if room_index >= len(self.room_polygons) or self.room_polygons[room_index] is None:
            return []
        
        polygon = self.room_polygons[room_index]
        
        # Extract exterior coordinates
        exterior_coords = []
        if hasattr(polygon, 'exterior') and polygon.exterior:
            # Convert to list of [x, z] coordinates (rounded to 3 decimal places)
            exterior_coords = [[round(float(x), 3), round(float(z), 3)] for x, z in polygon.exterior.coords[:-1]]
        
        return exterior_coords

    # def _calculate_room_shape_properties(self, room_index: int) -> Dict[str, Any]:
    #     """Calculate shape properties for a room polygon.
        
    #     Args:
    #         room_index: Index of the room to analyze
            
    #     Returns:
    #         Dictionary containing shape properties
    #     """
    #     if room_index >= len(self.room_polygons) or self.room_polygons[room_index] is None:
    #         return {
    #             'aspect_ratio': 1.0,
    #             'compactness': 0.0,
    #             'rectangularity': 0.0,
    #             'is_convex': False,
    #             'bounding_box_width': 0.0,
    #             'bounding_box_height': 0.0,
    #             'perimeter': 0.0,
    #             'shape_category': 'Unknown'
    #         }
        
    #     polygon = self.room_polygons[room_index]
        
    #     # Basic measurements
    #     area = polygon.area
    #     perimeter = polygon.length
        
    #     # Bounding box
    #     minx, miny, maxx, maxy = polygon.bounds
    #     bbox_width = maxx - minx
    #     bbox_height = maxy - miny
        
    #     # Aspect ratio (width/height)
    #     aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
        
    #     # Compactness (4π * area / perimeter²)
    #     compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        
    #     # Rectangularity (area / bounding box area)
    #     bbox_area = bbox_width * bbox_height
    #     rectangularity = area / bbox_area if bbox_area > 0 else 0.0
        
    #     # Convexity
    #     convex_hull = polygon.convex_hull
    #     is_convex = abs(polygon.area - convex_hull.area) < 0.01
        
    #     # Shape category based on properties
    #     shape_category = self._classify_room_shape(aspect_ratio, compactness, rectangularity, is_convex)
        
    #     return {
    #         'aspect_ratio': round(aspect_ratio, 3),
    #         'compactness': round(compactness, 3),
    #         'rectangularity': round(rectangularity, 3),
    #         'is_convex': is_convex,
    #         'bounding_box_width': round(bbox_width, 3),
    #         'bounding_box_height': round(bbox_height, 3),
    #         'perimeter': round(perimeter, 3),
    #         'shape_category': shape_category
    #     }
    
    # def _classify_room_shape(self, aspect_ratio: float, compactness: float, rectangularity: float, is_convex: bool) -> str:
    #     """Classify room shape based on geometric properties.
        
    #     Args:
    #         aspect_ratio: Width/height ratio
    #         compactness: Measure of circularity
    #         rectangularity: Ratio of area to bounding box area
    #         is_convex: Whether the shape is convex
            
    #     Returns:
    #         Shape category string
    #     """
    #     # Square-like (aspect ratio close to 1, high rectangularity)
    #     if 0.8 <= aspect_ratio <= 1.2 and rectangularity > 0.8:
    #         return 'Square'
        
    #     # Rectangular (high rectangularity, aspect ratio not close to 1)
    #     elif rectangularity > 0.8:
    #         if aspect_ratio > 1.5:
    #             return 'Horizontal Rectangle'
    #         elif aspect_ratio < 0.67:
    #             return 'Vertical Rectangle'
    #         else:
    #             return 'Rectangle'
        
    #     # Circular/round (high compactness)
    #     elif compactness > 0.8:
    #         return 'Circular'
        
    #     # L-shaped or complex (low rectangularity, convex)
    #     elif rectangularity < 0.6 and is_convex:
    #         return 'L-shaped'
        
    #     # Irregular complex shape
    #     elif not is_convex:
    #         return 'Complex Irregular'
        
    #     # Elongated shapes
    #     elif aspect_ratio > 2.0 or aspect_ratio < 0.5:
    #         return 'Elongated'
        
    #     # Default
    #     else:
    #         return 'Irregular'

    def get_static_room_data(self) -> Dict[str, Any]:
        """Get static room data that doesn't change between explorations.
        
        Returns:
            Dictionary containing static room properties including shape information
        """
        static_data = {
            'total_rooms': len(self.rooms_data),
            'total_area': sum(self.room_areas),
            'average_room_area': np.mean(self.room_areas) if self.room_areas else 0,
            'room_static_details': [],
            'room_types': [],
            'room_type_distribution': {},
            'room_connectivity': self.room_connectivity
        }
        
        # Static room details (with shape information and polygon coordinates)
        for i, room in enumerate(self.rooms_data):
            # Get shape properties for this room
            # shape_props = self._calculate_room_shape_properties(i)
            
            # Get polygon coordinates for this room
            polygon_coords = self._extract_polygon_coordinates(i)
            
            room_detail = {
                'room_id': i,
                'room_name': self.room_names[i] if i < len(self.room_names) else f'room_{i}',
                'room_type': room.get('roomType', 'Unknown'),
                'area': self.room_areas[i] if i < len(self.room_areas) else 0,
                'connected_rooms': self.room_connectivity.get(i, []),
                # 'shape_properties': shape_props,
                'polygon_coordinates': polygon_coords
            }
            static_data['room_static_details'].append(room_detail)
            static_data['room_types'].append(room.get('roomType', 'Unknown'))
        
        # Room type distribution
        static_data['room_type_distribution'] = Counter(static_data['room_types'])
        
        # Shape statistics
        # import pdb;pdb.set_trace()
        # shape_categories = [detail['shape_properties']['shape_category'] for detail in static_data['room_static_details']]
        # static_data['shape_distribution'] = Counter(shape_categories)
        
        return static_data
    
    def get_dynamic_room_data(self) -> Dict[str, Any]:
        """Get dynamic room data that changes with each exploration.
        
        Returns:
            Dictionary containing dynamic room properties
        """
        dynamic_data = {
            'agent_frames_spent': dict(self.agent_room_time),
            'agent_trajectory_rooms': getattr(self, 'agent_trajectory_rooms', []),
            'object_distribution': {},
            'room_dynamic_details': []
        }
        
        # Object distribution
        room_object_counts = Counter(self.object_room_mapping.values())
        for room_id, count in room_object_counts.items():
            if room_id >= 0:  # Valid room
                room_name = self.room_names[room_id] if room_id < len(self.room_names) else f'room_{room_id}'
                dynamic_data['object_distribution'][room_name] = count
        
        # Dynamic room details (only dynamic properties)
        for i in range(len(self.rooms_data)):
            room_name = self.room_names[i] if i < len(self.room_names) else f'room_{i}'
            room_detail = {
                'room_id': i,
                'room_name': room_name,
                'agent_frames_spent': self.agent_room_time.get(i, 0),
                'objects_count': room_object_counts.get(i, 0)
            }
            dynamic_data['room_dynamic_details'].append(room_detail)
        
        return dynamic_data
    
    def save_room_analysis(self, output_dir: str, filename: str = "room_analysis.json") -> str:
        """Save room analysis to file.
        
        Args:
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        
        stats = self.get_room_statistics()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        stats = recursive_convert(stats)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            print(f"Room analysis saved to: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error saving room analysis: {e}")
            return None
    
    def save_static_room_data(self, output_dir: str, filename: str = "room_static_analysis.json") -> str:
        """Save static room data to file.
        
        Args:
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        
        static_data = self.get_static_room_data()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        static_data = recursive_convert(static_data)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(static_data, f, indent=2, ensure_ascii=False)
            
            print(f"Static room data saved to: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error saving static room data: {e}")
            return None
    
    def save_dynamic_room_data(self, output_dir: str, filename: str = "room_dynamic_analysis.json") -> str:
        """Save dynamic room data to file.
        
        Args:
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        
        dynamic_data = self.get_dynamic_room_data()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        dynamic_data = recursive_convert(dynamic_data)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dynamic_data, f, indent=2, ensure_ascii=False)
            
            print(f"Dynamic room data saved to: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error saving dynamic room data: {e}")
            return None
    
    def visualize_rooms(self, output_dir: str, filename: str = "room_layout.png") -> str:
        """Create a visualization of room layout and connectivity.
        
        Args:
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved visualization
        """
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        
        try:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot 1: Room layout with areas
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.room_polygons)))
            
            for i, (polygon, color) in enumerate(zip(self.room_polygons, colors)):
                if polygon is not None:
                    x, y = polygon.exterior.xy
                    ax1.fill(x, y, alpha=0.6, color=color, edgecolor='black', linewidth=1)
                    
                    # Add room label
                    centroid = polygon.centroid
                    room_name = self.room_names[i] if i < len(self.room_names) else f'Room {i}'
                    area = self.room_areas[i] if i < len(self.room_areas) else 0
                    ax1.text(centroid.x, centroid.y, f'{room_name}\nArea: {area:.1f}', 
                            ha='center', va='center', fontsize=8, 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax1.set_title('Room Layout and Areas')
            ax1.set_xlabel('X coordinate')
            ax1.set_ylabel('Z coordinate')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            

            
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Room visualization saved to: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error creating room visualization: {e}")
            return None
    
    def get_room_connectivity_summary(self) -> str:
        """Get a text summary of room connectivity.
        
        Returns:
            Formatted connectivity summary
        """
        summary = "\n房间连通关系分析:\n"
        summary += "=" * 30 + "\n"
        
        for room_id, connected_rooms in self.room_connectivity.items():
            room_name = self.room_names[room_id] if room_id < len(self.room_names) else f'Room {room_id}'
            if connected_rooms:
                connected_names = []
                for conn_id in connected_rooms:
                    conn_name = self.room_names[conn_id] if conn_id < len(self.room_names) else f'Room {conn_id}'
                    connected_names.append(conn_name)
                summary += f"{room_name}: 连接到 {', '.join(connected_names)}\n"
            else:
                summary += f"{room_name}: 无连接房间\n"
        
        return summary
    
    def print_room_summary(self):
        """Print a comprehensive room analysis summary."""
        stats = self.get_room_statistics()
        
        print("\n" + "=" * 50)
        print("房间分析总结")
        print("=" * 50)
        
        print(f"总房间数: {stats['total_rooms']}")
        print(f"总面积: {stats['total_area']:.2f} 平方单位")
        print(f"平均房间面积: {stats['average_room_area']:.2f} 平方单位")
        
        print("\n房间类型分布:")
        for room_type, count in stats['room_type_distribution'].items():
            print(f"  {room_type}: {count} 个")
        
        print("\n各房间详细信息:")
        for room in stats['room_details']:
            print(f"  {room['room_name']}:")
            print(f"    面积: {room['area']:.2f} 平方单位")
            print(f"    连接房间数: {len(room['connected_rooms'])}")
            print(f"    物体数量: {room['objects_count']}")
            print(f"    Agent停留帧数: {room['agent_frames_spent']} 帧")
        
        print("\n物体分布:")
        for room_name, obj_count in stats['object_distribution'].items():
            print(f"  {room_name}: {obj_count} 个物体")
        
        print(self.get_room_connectivity_summary())