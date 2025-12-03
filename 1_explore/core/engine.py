"""Core exploration engine for AI2Thor.

Contains the main ExplorationEngine class that orchestrates the exploration process.
"""

import os
import copy
import json
import shutil
import pickle
import numpy as np
import gc
import psutil
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple
from .config import ExplorationConfig, AgentState
from vision.vision import VisionManager, CameraManager, VideoRecorder, TrajectoryVisualizer
from strategies import RectangleBasedStrategy
from analysis.room_analyzer import RoomAnalyzer


class ExplorationEngine:
    """Main engine for orchestrating AI2Thor exploration."""
    
    def __init__(self, controller, config: ExplorationConfig, scene_id: int):
        """Initialize exploration engine.
        
        Args:
            controller: AI2Thor controller instance
            config: Exploration configuration
        """
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(os.path.dirname(current_dir))

        self.controller = controller
        self.config = config
        self.scene_id = scene_id

        self.asset_path = config.asset_output_dir
        
        # Initialize managers
        self.vision_manager = VisionManager()
        self.video_recorder = VideoRecorder(config.base_output_dir, 'data', scene_id)
        self.output_base_dir = os.path.join(config.base_output_dir, 'data', f'scene_{scene_id}')
        os.makedirs(os.path.join(config.base_output_dir, 'data'), exist_ok=True)
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        # Initialize visualizers
        reachable_coords = self.get_reachable_coords()
        self.trajectory_visualizer = TrajectoryVisualizer(reachable_coords)
        
        # Initialize agent state
        current_pos = self.get_current_position()
        current_rot = self.get_current_rotation()
        self.agent_state = AgentState(
            position=current_pos,
            rotation=current_rot,
            visited_positions={current_pos},
            trajectory=[current_pos]
        )
        
        # Video recording lists
        self.trajectory_frames = []
        self.first_person_frames = []
        self.top_down_frames = []
        self.depth_frames = []
        self.segmentation_frames = []
        
        # Raw data storage for pkl files
        self.raw_depth_data = []
        self.raw_segmentation_data = []

        self.strategy = None

        self.agent_logs = []
        # self.visible_pixel_threshold = 40
        
        # Setup third-party camera
        # CameraManager.add_third_party_camera(controller)
        
        # Initialize room analyzer
        self.room_analyzer = RoomAnalyzer(controller, config.gridSize)
        
        # Load valid object types from ai2thor_object_types.json
        self._load_valid_object_types()
        
        print(f"Exploration Engine initialized, reachable positions: {len(reachable_coords)}")
        
        # Memory monitoring
        self._initial_memory = self._get_memory_usage()
        self._frame_batch_size = 50  # Process frames in batches to manage memory

        self.current_manipulation_path = None
    
    def _load_valid_object_types(self):
        """Load valid object types from ai2thor_object_types.json file."""
        try:
            json_path = os.path.join(self.project_root, '0_ai2-thor_data', 'ai2thor_object_types.json')
            
            with open(json_path, 'r', encoding='utf-8') as f:
                self.valid_object_types = set(json.load(f))
            
            print(f"Loaded {len(self.valid_object_types)} valid object types from {json_path}")
            
        except Exception as e:
            print(f"Warning: Could not load valid object types: {e}")
            self.valid_object_types = set()
    
    def clear(self):
        # Clear all frame data lists to free memory
        self.trajectory_frames.clear()
        self.first_person_frames.clear()
        self.top_down_frames.clear()
        self.depth_frames.clear()
        self.segmentation_frames.clear()
        
        # Clear raw data storage
        self.raw_depth_data.clear()
        self.raw_segmentation_data.clear()
        
        # Reset agent trajectory state for fresh recording
        current_pos = self.get_current_position()
        current_rot = self.get_current_rotation()
        self.agent_state.position = current_pos
        self.agent_state.rotation = current_rot
        self.agent_state.visited_positions = {current_pos}
        self.agent_state.trajectory = [current_pos]
        self.agent_state.step_count = 0

        self.agent_logs.clear()
        
        # Force garbage collection to free memory immediately
        gc.collect()
        
        # Log memory usage after cleanup
        current_memory = self._get_memory_usage()
        print(f"Memory after cleanup: {current_memory:.1f} MB")
        
    
    def get_current_position(self) -> Tuple[float, float]:
        pos = self.controller.last_event.metadata['agent']['position']
        return (round(pos['x'], 2), round(pos['z'], 2))
    
    def get_current_rotation(self) -> float:
        return round(self.controller.last_event.metadata['agent']['rotation']['y'], 2)
    
    def get_reachable_coords(self) -> List[Tuple[float, float]]:
        """Get all reachable positions in the environment.
        """
        event = self.controller.step(action='GetReachablePositions')
        if not event.metadata["lastActionSuccess"]:
            print(event.metadata["errorMessage"])
        reachable_positions = event.metadata["actionReturn"]
        reachable_coords = [
            (round(rp["x"], 2), round(rp["z"], 2)) 
            for rp in reachable_positions
        ]
        
        return reachable_coords

    def get_room_coords(self) -> List[Tuple[float, float]]:
        """Get all room positions in the environment.
        """
        rooms_data = self.controller.scene['rooms']
        room_coords = []
        for room in rooms_data:
            poly = room['floorPolygon']
            coords_1_room = []
            for p in poly:
                coords_1_room.append((p["x"], p["z"]))
            room_coords.append(coords_1_room)
        
        return room_coords

    def _get_visible_objects(self) -> List[str]:
        """Get list of visible object IDs in current view from segmentation data.
        
        Returns:
            List of object IDs that are currently visible in the agent's view
        """
        try:
            vision_data = self.vision_manager.get_vision(self.controller)
            
            segmentations = vision_data['segmentation']['instance_segmentation']
            color_to_object_id = vision_data['segmentation']['color_to_object_id']
            object_id_to_color = vision_data['segmentation']['object_id_to_color']
            reshaped = segmentations.reshape(-1, 3)
            unique_colors = np.unique(reshaped, axis=0)
            visible_colors = set([tuple(row) for row in unique_colors])

            all_colors = set(list(color_to_object_id.keys()))
            
            visible_object_ids = [color_to_object_id[color] for color in all_colors if color in visible_colors]
            
            if hasattr(self, 'valid_object_types'):
                filtered_object_ids = []
                for obj_id in visible_object_ids:
                    obj_type = obj_id.split('|')[0] if '|' in obj_id else obj_id
                    if obj_type in self.valid_object_types:
                        filtered_object_ids.append(obj_id)
                        # color = object_id_to_color[obj_id]
                        # pixel_count = np.sum(np.all(reshaped == color, axis=1))
                        # if pixel_count > self.visible_pixel_threshold:
                        #     filtered_object_ids.append(obj_id)
                return filtered_object_ids
            
            return visible_object_ids
            
        except Exception as e:
            print(f"Error getting visible objects from segmentation: {e}")
            return []
    
    def _log_agent_info(self):
        # Get current room information
        current_room_id = -1
        current_room_name = "Unknown"
        if hasattr(self, 'room_analyzer'):
            current_room_id = self.room_analyzer.get_agent_current_room(self.agent_state.position)
            if current_room_id >= 0 and current_room_id < len(self.room_analyzer.room_names):
                current_room_name = self.room_analyzer.room_names[current_room_id]
        
        agent_info = {
            'step_count': self.agent_state.step_count,
            'position': {
                'x': self.agent_state.position[0],
                'z': self.agent_state.position[1]
            },
            'rotation': self.agent_state.rotation,
            'room_id': current_room_id,
            'room_name': current_room_name,
            'visible_objects': self._get_visible_objects()
        }
        
        self.agent_logs.append(agent_info)
        
        # Update room analyzer with current position
        if hasattr(self, 'room_analyzer'):
            self.room_analyzer.update_agent_room_time(self.agent_state.position)
    
    def save_agent_logs(self, output_path: str, filename: str = "agent_trajectory.jsonl") -> str:
        os.makedirs(os.path.join(self.output_base_dir, output_path), exist_ok=True)
        file_path = os.path.join(self.output_base_dir, output_path, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for log_entry in self.agent_logs:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                    
            print(f"Agent logs saved to: {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving agent logs: {e}")
            return None

    def step(self, action: Optional[str] = None, target_pos: Optional[Tuple[int, int]] = None) -> bool:
        """Execute one step of exploration.
        """        
        # Execute action if provided
        if action == 'MoveAhead':
            event = self.controller.step(action=action, moveMagnitude=self.config.gridSize)
            
            success = event.metadata['lastActionSuccess']
            
            if success:
                self.agent_state.position = target_pos
                self.agent_state.visited_positions.add(target_pos)
                self.agent_state.trajectory.append(target_pos)

                self._log_agent_info()
                
                self._record_visual_data()
                
                self.agent_state.step_count = self.agent_state.step_count  + 1
            else:
                print(event.metadata["errorMessage"])
            return success
        
        elif action in ['RotateLeft', 'RotateRight']:
            success = self.smooth_rotate(action)
            return success
        
        else:
            event = self.controller.step(action=action)
            
            success = event.metadata['lastActionSuccess']
            if success:
                self._log_agent_info()
                
                self._record_visual_data()
                
                self.agent_state.step_count = self.agent_state.step_count  + 1
            return event.metadata['lastActionSuccess']
    
    def smooth_rotate(self, action: str) -> bool:
        assert action in ['RotateLeft', 'RotateRight']
        smooth_degree = self.config.smooth_rotation_degrees
        assert 90 % smooth_degree == 0
        
        rotated_degree = 0
        target_degree = 90
        
        while rotated_degree < target_degree:
            event = self.controller.step(action=action, degrees=smooth_degree)
            
            if not event.metadata['lastActionSuccess']:
                return False

            self._log_agent_info()
            self.agent_state.step_count = self.agent_state.step_count  + 1

            if action == 'RotateLeft':
                self.agent_state.rotation = (self.agent_state.rotation - smooth_degree + 360) % 360
            else:
                self.agent_state.rotation = (self.agent_state.rotation + smooth_degree) % 360
            
            self._record_visual_data()
            
            rotated_degree += smooth_degree

        return True
    
    def steps(self, actions: List[Optional[str]], target_pos: Tuple[int, int]) -> bool:
        """Execute one step of exploration with strategy visualization support.
        """
        success = True
        for action in actions:
            success_ = self.step(action, target_pos)
            if not success_:
                print(f"Fail to execute action {action}")
                print(f"Current position: {self.agent_state.position}")
                print(f"Current rotation: {self.agent_state.rotation}")
                success = False
                break
        return success
    
    def _record_visual_data(self) -> None:
        """Record visual data from all cameras."""
        # Get vision data
        vision_data = self.vision_manager.get_vision(self.controller)
        
        # Record first-person view (use copy to avoid reference issues)
        if vision_data['rgb'] is not None:
            self.first_person_frames.append(vision_data['rgb'].copy())
        
        # Record depth view and raw depth data
        if vision_data['depth'] is not None:
            # Save raw depth data for pkl file
            self.raw_depth_data.append(vision_data['depth'].copy())
            # # Save visualization for video/frames (commented out to save computation)
            # depth_vis = self.vision_manager.vis_depth(vision_data['depth'])
            # self.depth_frames.append(depth_vis)
        
        # Record segmentation view and raw segmentation data
        if vision_data['segmentation']['instance_segmentation'] is not None:
            # Save raw segmentation data for pkl file - use shallow copy for dictionaries to save memory
            segmentation_data = {
                'instance_segmentation': vision_data['segmentation']['instance_segmentation'].copy(),
                'color_to_object_id': vision_data['segmentation']['color_to_object_id'],
                'object_id_to_color': vision_data['segmentation']['object_id_to_color']
            }
            self.raw_segmentation_data.append(segmentation_data)
            
        # Periodic memory cleanup to prevent accumulation
        if len(self.first_person_frames) % self._frame_batch_size == 0:
            gc.collect()
            current_memory = self._get_memory_usage()
            if current_memory > self._initial_memory + 1000:  # If memory increased by more than 1GB
                print(f"Warning: High memory usage detected: {current_memory:.1f} MB")
                
        # # Save visualization for video/frames (commented out to save computation)
        # seg_vis = self.vision_manager.process_segmentation(
        #     vision_data['segmentation']['instance_segmentation']
        # )
        # if seg_vis is not None:
        #     self.segmentation_frames.append(seg_vis)
        
        # # Record top-down view (commented out to save computation)
        # top_down_event = self.controller.step(
        #     action="GetMapViewCameraProperties",
        #     raise_for_failure=True
        # )
        # if top_down_event.third_party_camera_frames:
        #     top_down_frame = top_down_event.third_party_camera_frames[-1][...,:3]
        #     self.top_down_frames.append(top_down_frame)
        
        # # Record trajectory plot (commented out to save computation)
        # rectangles = None
        # if self.strategy and hasattr(self.strategy, 'rectangles'):
        #     rectangles = self.strategy.rectangles
        # 
        # trajectory_frame = self.trajectory_visualizer.generate_trajectory_frame(
        #     self.agent_state.step_count, 
        #     self.agent_state, 
        #     rectangles
        # )
        # self.trajectory_frames.append(trajectory_frame)
    
    def save_vision_results(self, path) -> Dict[str, str]:
        """Save all recorded data and visualizations as individual frames.
        
        Returns:
            Dictionary containing paths to saved files
        """
        saved_files = {}
        
        # Save first-person RGB frames
        if self.first_person_frames:
            frame_dir = self.video_recorder.save_frames(
                self.first_person_frames,
                "RGB",
                path
            )
            if frame_dir:
                saved_files['first_person_frames'] = frame_dir
        
        # # Save top-down frames (commented out to save computation)
        # if self.top_down_frames:
        #     frame_dir = self.video_recorder.save_frames(
        #         self.top_down_frames,
        #         "top_down",
        #         path
        #     )
        #     if frame_dir:
        #         saved_files['top_down_frames'] = frame_dir
        
        # # Save depth frames (commented out to save computation)
        # if self.depth_frames:
        #     frame_dir = self.video_recorder.save_frames(
        #         self.depth_frames,
        #         "first_person_depth",
        #         path
        #     )
        #     if frame_dir:
        #         saved_files['depth_frames'] = frame_dir
        
        # # Save segmentation frames (commented out to save computation)
        # if self.segmentation_frames:
        #     frame_dir = self.video_recorder.save_frames(
        #         self.segmentation_frames,
        #         "first_person_segmentation",
        #         path
        #     )
        #     if frame_dir:
        #         saved_files['segmentation_frames'] = frame_dir
        
        # # Save trajectory frames (commented out to save computation)
        # if self.trajectory_frames:
        #     frame_dir = self.video_recorder.save_frames(
        #         self.trajectory_frames,
        #         "trajectory",
        #         path
        #     )
        #     if frame_dir:
        #         saved_files['trajectory_frames'] = frame_dir
        
        # Save raw depth data as pkl file
        if self.raw_depth_data:
            depth_pkl_path = self._save_raw_data_pkl(
                self.raw_depth_data,
                "depth",
                path
            )
            if depth_pkl_path:
                saved_files['raw_depth_data'] = depth_pkl_path
        
        # Save raw segmentation data as pkl file
        if self.raw_segmentation_data:
            seg_pkl_path = self._save_raw_data_pkl(
                self.raw_segmentation_data,
                "segmentation",
                path
            )
            if seg_pkl_path:
                saved_files['raw_segmentation_data'] = seg_pkl_path
        
        # Clear frame data after saving to free memory immediately
        if self.first_person_frames:
            self.first_person_frames.clear()
        if self.raw_depth_data:
            self.raw_depth_data.clear()
        if self.raw_segmentation_data:
            self.raw_segmentation_data.clear()
            
        # Force garbage collection after saving
        gc.collect()
        
        print(f"Memory after saving vision results: {self._get_memory_usage():.1f} MB")
        
        return saved_files
    
    def _save_raw_data_pkl(self, data: List, base_name: str, path: str) -> Optional[str]:
        """Save raw data as individual pickle files in a folder.
        
        Args:
            data: List of raw data to save
            filename: Base name for the pkl files (without extension)
            path: Output path
            
        Returns:
            Path to saved folder or None if failed
        """
        try:
            # Create output directory for individual frames
            output_dir = os.path.join(self.output_base_dir, path, base_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each frame as individual pkl file
            for i, frame_data in enumerate(tqdm(data, desc=f"Saving {base_name} frames")):
                frame_filename = f"frame_{i}.pkl"
                frame_path = os.path.join(output_dir, frame_filename)
                
                with open(frame_path, 'wb') as f:
                    pickle.dump(frame_data, f)
            
            print(f"Raw data frames saved to folder: {output_dir} ({len(data)} frames)")
            return output_dir
            
        except Exception as e:
            print(f"Error saving raw data frames to {base_name}: {e}")
            return None
    
    def save_videos(self, path) -> Dict[str, str]:
        """Save all recorded data and visualizations.
        
        Returns:
            Dictionary containing paths to saved files
        """
        saved_files = {}
        
        # Save videos
        if self.first_person_frames:
            video_path = self.video_recorder.save_video(
                self.config.frame_size,
                self.first_person_frames,
                "first_person_RGB.mp4",
                path,
                self.config.fps_first_person
            )
            if video_path:
                saved_files['first_person_video'] = video_path
        
        # # Save top-down video (commented out to save computation)
        # if self.top_down_frames:
        #     frame_size = self.top_down_frames[0].shape[1::-1]
        #     video_path = self.video_recorder.save_video(
        #         frame_size,
        #         self.top_down_frames,
        #         "top_down.mp4",
        #         path,
        #         self.config.fps_top_down
        #     )
        #     if video_path:
        #         saved_files['top_down_video'] = video_path
        
        # # Save depth video (commented out to save computation)
        # if self.depth_frames:
        #     video_path = self.video_recorder.save_video(
        #         self.config.frame_size,
        #         self.depth_frames,
        #         "first_person_depth.mp4",
        #         path,
        #         self.config.fps_depth
        #     )
        #     if video_path:
        #         saved_files['depth_video'] = video_path
        
        # # Save segmentation video (commented out to save computation)
        # if self.segmentation_frames:
        #     video_path = self.video_recorder.save_video(
        #         self.config.frame_size,
        #         self.segmentation_frames,
        #         "first_person_segmentation.mp4",
        #         path,
        #         self.config.fps_segmentation
        #     )
        #     if video_path:
        #         saved_files['segmentation_video'] = video_path
        
        # # Save trajectory video (commented out to save computation)
        # if self.trajectory_frames:
        #     frame_size = self.trajectory_frames[0].shape[1::-1]
        #     video_path = self.video_recorder.save_video(
        #         frame_size,
        #         self.trajectory_frames,
        #         "trajectory.mp4",
        #         path,
        #         self.config.fps_trajectory
        #     )
        #     if video_path:
        #         saved_files['trajectory_video'] = video_path
        
        return saved_files
    
    # def save_room_analysis(self, path: str, save_static: bool = False) -> Dict[str, str]:
    #     """Save room analysis results.
        
    #     Args:
    #         path: Output path for room analysis files
    #         save_static: Whether to save static room data (only needed once per scene)
            
    #     Returns:
    #         Dictionary containing paths to saved room analysis files
    #     """
    #     saved_files = {}
        
    #     if hasattr(self, 'room_analyzer'):
    #         # Save dynamic room data (always)
    #         dynamic_path = self.room_analyzer.save_dynamic_room_data(
    #             os.path.join(self.output_base_dir, path),
    #             "room_dynamic_analysis.json"
    #         )
    #         if dynamic_path:
    #             saved_files['room_dynamic_analysis'] = dynamic_path
            
    #         # Save static room data (only when requested)
    #         if save_static:
    #             static_path = self.room_analyzer.save_static_room_data(
    #                 os.path.join(self.output_base_dir, path),
    #                 "room_static_analysis.json"
    #             )
    #             if static_path:
    #                 saved_files['room_static_analysis'] = static_path
                
    #             # Save room visualization (only with static data)
    #             viz_path = self.room_analyzer.visualize_rooms(
    #                 os.path.join(self.output_base_dir, path),
    #                 "room_layout.png"
    #             )
    #             if viz_path:
    #                 saved_files['room_visualization'] = viz_path
            
    #         # Save complete room analysis for backward compatibility
    #         analysis_path = self.room_analyzer.save_room_analysis(
    #             os.path.join(self.output_base_dir, path),
    #             "room_analysis.json"
    #         )
    #         if analysis_path:
    #             saved_files['room_analysis'] = analysis_path
            
    #         # Save room frame analysis
    #         frame_analysis_path = self.save_room_frame_analysis(path)
    #         if frame_analysis_path:
    #             saved_files['room_frame_analysis'] = frame_analysis_path
        
    #     return saved_files
    
    def save_room_frame_analysis(self, path: str) -> str:
        """Save room frame analysis data.
        
        Args:
            path: Output path for room frame analysis file
            
        Returns:
            Path to saved room frame analysis file
        """
        if not hasattr(self, 'room_analyzer'):
            return None
            
        import json
        
        # Prepare room frame data (dynamic data only)
        room_frame_data = {
            'total_frames': self.agent_state.step_count,
            'room_frame_counts': dict(self.room_analyzer.agent_room_time),
            'room_trajectory': self.room_analyzer.agent_trajectory_rooms,
            'room_names': self.room_analyzer.room_names
        }
        
        # Save to file
        room_frame_file = os.path.join(self.output_base_dir, path, 'room_frame_analysis.json')
        os.makedirs(os.path.dirname(room_frame_file), exist_ok=True)
        
        try:
            with open(room_frame_file, 'w', encoding='utf-8') as f:
                json.dump(room_frame_data, f, indent=2, ensure_ascii=False)
            print(f"Room frame analysis saved to: {room_frame_file}")
            return room_frame_file
        except Exception as e:
            print(f"Error saving room frame analysis: {e}")
            return None
    
    def save_static_room_analysis(self, scene_path: str) -> Dict[str, str]:
        """Save static room analysis data at scene level.
        
        Args:
            scene_path: Scene-level output path
            
        Returns:
            Dictionary containing paths to saved static room analysis files
        """
        saved_files = {}
        
        if hasattr(self, 'room_analyzer'):
            # Save static room data
            static_path = self.room_analyzer.save_static_room_data(
                os.path.join(self.output_base_dir, scene_path),
                "room_static_analysis.json"
            )
            if static_path:
                saved_files['room_static_analysis'] = static_path
            
            # Save room visualization
            viz_path = self.room_analyzer.visualize_rooms(
                os.path.join(self.output_base_dir, scene_path),
                "room_layout.png"
            )
            if viz_path:
                saved_files['room_visualization'] = viz_path
        
        return saved_files
    
    def explore(self, strategy: RectangleBasedStrategy): # tmp path
        self.strategy = strategy
        
        max_steps = len(self.strategy.current_trajectory)
        print("\nStarting exploration...")
        print(f"Max steps: {max_steps}")
        print(f"Memory before exploration: {self._get_memory_usage():.1f} MB")
        
        with tqdm(total=max_steps, desc="Exploration Progress", unit="step") as pbar:
            for step_count in range(max_steps):
                result = strategy.get_actions(self.get_agent_state())
                
                if result is None:
                    # print("No actions available, exploration complete.")
                    break
                
                actions, target_position = result
                
                success = self.steps(actions, target_position)
                if not success:
                    tqdm.write(f"Step {step_count} failed, exploration terminated.")
                    return False

                pbar.update(1)
        
        # Final memory cleanup after exploration
        gc.collect()
        print(f"Memory after exploration: {self._get_memory_usage():.1f} MB")
        
        return True
    
    def get_agent_state(self) -> AgentState:
        """Get current agent state.
        
        Returns:
            Current agent state
        """
        # Use shallow copy instead of deep copy to reduce memory overhead
        # Only copy the essential fields that might be modified
        state_copy = AgentState(
            position=self.agent_state.position,
            rotation=self.agent_state.rotation,
            visited_positions=self.agent_state.visited_positions.copy(),
            trajectory=self.agent_state.trajectory.copy()
        )
        state_copy.step_count = self.agent_state.step_count
        return state_copy
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def delete_output_dir(self):
        """Delete output directory."""
        if self.output_base_dir:
            shutil.rmtree(self.output_base_dir)
            print(f"Deleted output directory: {self.output_base_dir}")
    
    def extract_object_asset_mapping(self, output_dir=None, thread_prefix=""):
        """Extract objectId to assetId mapping for all objects in the scene.
        
        Args:
            output_dir: Directory to save the mapping file, defaults to self.output_base_dir
            thread_prefix: Thread prefix for logging
            
        Returns:
            dict: Mapping of objectId to assetId
        """
        try:
            # Get all objects in the scene
            event = self.controller.last_event
            objects = event.metadata['objects']
            
            # Create simple mapping dictionary with only objectId and assetId
            # Only save objects that have assetId
            object_asset_mapping = {}
            for obj in objects:
                object_id = obj['objectId']
                asset_id = obj.get('assetId')
                if asset_id:  # Only save if assetId exists and is not None/empty
                    object_asset_mapping[object_id] = asset_id
            
            # Save mapping to JSON file if output_dir is provided
            if output_dir is None:
                output_dir = self.output_base_dir
                
            mapping_file = os.path.join(output_dir, 'object_asset_mapping.json')
            os.makedirs(output_dir, exist_ok=True)
            
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(object_asset_mapping, f, indent=2, ensure_ascii=False)
            
            # print(f"{thread_prefix}Saved object-asset mapping with {len(object_asset_mapping)} objects to {mapping_file}")
            
            return object_asset_mapping
            
        except Exception as e:
            print(f"{thread_prefix}Error extracting object-asset mapping: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def save_asset_data(self, path: str) -> Dict[str, str]:
        """Save data organized by assetId for each object.
        
        For each assetId, creates a folder containing pkl files with mask and image_path information.
        Only saves the top 5 highest quality masks per assetId based on area and center proximity.
        Uses RGB frames saved by save_vision_results instead of saving separate RGB files.
        
        Args:
            path: Output path for asset data
            
        Returns:
            Dictionary containing paths to saved asset folders
        """
        saved_folders = {}
        
        if not self.raw_segmentation_data or not self.first_person_frames:
            print("No segmentation data or RGB frames available for asset data saving")
            return saved_folders
        
        def calculate_mask_quality(mask):
            """Calculate mask quality based on area and center proximity.
            
            Args:
                mask: Binary mask array
                
            Returns:
                Quality score (higher is better)
            """
            # Calculate area (number of True pixels)
            area = np.sum(mask)
            if area == 0:
                return 0
            
            # Get mask center of mass
            y_coords, x_coords = np.where(mask)
            if len(y_coords) == 0:
                return 0
            
            mask_center_y = np.mean(y_coords)
            mask_center_x = np.mean(x_coords)
            
            # Get image center
            img_height, img_width = mask.shape
            img_center_y = img_height / 2
            img_center_x = img_width / 2
            
            # Calculate distance from image center (normalized)
            distance_from_center = np.sqrt(
                ((mask_center_x - img_center_x) / img_width) ** 2 + 
                ((mask_center_y - img_center_y) / img_height) ** 2
            )
            
            # Quality score: area weight + center proximity weight
            # Normalize area by image size
            normalized_area = area / (img_height * img_width)
            center_score = 1 - distance_from_center  # Closer to center = higher score
            
            # Combine scores (area has more weight)
            quality_score = normalized_area * 0.7 + center_score * 0.3
            
            return quality_score
        
        try:
            # Create base asset data directory
            os.makedirs(self.asset_path, exist_ok=True)
            
            # Get RGB frames directory from save_vision_results
            rgb_frames_dir = os.path.join(self.output_base_dir.split('/')[-1], path, "RGB")
            
            # Get object to asset mapping
            object_asset_mapping = self.extract_object_asset_mapping()
            
            # Group data by assetId with quality scores
            asset_data = {}  # assetId -> list of (frame_data, quality_score)
            
            print(f"Processing {len(self.raw_segmentation_data)} frames for asset data...")
            
            for frame_idx, seg_data in enumerate(self.raw_segmentation_data):
                instance_seg = seg_data['instance_segmentation']
                color_to_object_id = seg_data['color_to_object_id']
                
                # Get unique colors in this frame
                reshaped = instance_seg.reshape(-1, 3)
                unique_colors = np.unique(reshaped, axis=0)
                visible_colors = set([tuple(row) for row in unique_colors])
                
                # Find visible objects and their assets
                for color_tuple in visible_colors:
                    if color_tuple in color_to_object_id:
                        object_id = color_to_object_id[color_tuple]
                        
                        # Get assetId for this object
                        asset_id = object_asset_mapping.get(object_id)
                        if not asset_id:
                            continue
                        
                        # Filter by valid object types if available
                        if hasattr(self, 'valid_object_types'):
                            obj_type = object_id.split('|')[0] if '|' in object_id else object_id
                            if obj_type not in self.valid_object_types:
                                continue
                        
                        # Create mask for this object
                        color_array = np.array(color_tuple)
                        mask = np.all(instance_seg == color_array, axis=2)
                        
                        # Calculate mask quality
                        quality_score = calculate_mask_quality(mask)
                        if quality_score == 0:
                            continue
                        
                        # Use RGB frame path from save_vision_results
                        rgb_filename = f"frame_{frame_idx}.png"
                        rgb_path = os.path.join(rgb_frames_dir, rgb_filename)
                        
                        # Prepare data for pkl file
                        frame_data = {
                            'mask': mask,
                            'image_path': rgb_path,
                        }
                        
                        # Initialize asset data if not exists
                        if asset_id not in asset_data:
                            asset_data[asset_id] = []
                        
                        asset_data[asset_id].append((frame_data, quality_score))
            
            # Select top 5 masks for each asset and save
            print(f"Selecting top 5 masks and saving asset data for {len(asset_data)} assets...")
            
            for asset_id, frames_with_scores in tqdm(asset_data.items(), desc="Saving asset data"):
                # Sort by quality score (descending) and take top 5
                frames_with_scores.sort(key=lambda x: x[1], reverse=True)
                top_frames = frames_with_scores[:5]
                
                # Create asset directory
                asset_dir = os.path.join(self.asset_path, asset_id)
                os.makedirs(asset_dir, exist_ok=True)
                # import pdb;pdb.set_trace()
                
                # Save top 5 frames as separate pkl files
                for i, (frame_data, quality_score) in enumerate(top_frames):
                    pkl_filename = f"{rgb_path.split('.')[0].replace('/', '_')}.pkl"
                    pkl_path = os.path.join(asset_dir, pkl_filename)
                    
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(frame_data, f)
                
                saved_folders[asset_id] = asset_dir
                # print(f"Saved top {len(top_frames)} masks for asset {asset_id} in {asset_dir}")
            
            return saved_folders
            
        except Exception as e:
            print(f"Error saving asset data: {e}")
            import traceback
            traceback.print_exc()
            return saved_folders
    
    def cleanup_scene_memory(self):
        """Comprehensive memory cleanup between scenes."""
        # Clear all frame data
        self.clear()
        
        # Reset strategy if exists
        if hasattr(self, 'strategy') and self.strategy:
            self.strategy = None
        
        # Reset room analyzer state
        if hasattr(self, 'room_analyzer'):
            self.room_analyzer.reset_dynamic_data()
        
        # Force multiple garbage collections
        for _ in range(3):
            gc.collect()
        
        print(f"Scene memory cleanup completed. Current memory: {self._get_memory_usage():.1f} MB")

