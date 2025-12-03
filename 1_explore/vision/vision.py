"""Vision management module for AI2Thor exploration.

Contains classes for handling visual data processing and camera management.
"""

import os
import cv2
import copy
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, Optional, List, Tuple

from core.config import AgentState


class VideoRecorder:
    """Video recorder for exploration sessions."""
    
    def __init__(self, base_output_dir: str = 'output', split: str = 'val', scene_id: int = 0):
        """Initialize video recorder.
        
        Args:
            base_output_dir: Base directory for output files
        """
        self.base_output_dir = base_output_dir
        self.output_dir = None
        self.split = split
        self._create_output_dir(scene_id)
    
    def _create_output_dir(self, scene_id: int) -> None:
        """Create timestamped output directory."""
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)
        
        # Add scene_id to the output directory
        self.output_dir = os.path.join(self.base_output_dir, self.split, f'scene_{scene_id}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\nOutput directory created: {self.output_dir}")
    
    def save_video(self, 
                   frame_size: Tuple[int, int], 
                   frames_list: List, 
                   video_name: str, 
                   manipulation_path: str,
                   fps: int = 24,) -> Optional[str]:
        """Save video from frame list.
        
        Args:
            frame_size: Video frame size (width, height)
            frames_list: List of video frames
            video_name: Output video filename
            manipulation_path: Subdirectory path for organizing videos
            fps: Frames per second
            
        Returns:
            Path to saved video file or None if failed
        """
        if not frames_list:
            print(f"Warning: {video_name} no frame data, skip saving")
            return None
        
        # Create manipulation subdirectory if it doesn't exist
        manipulation_dir = os.path.join(self.output_dir, manipulation_path)
        os.makedirs(manipulation_dir, exist_ok=True)
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(manipulation_dir, video_name)
        writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        
        for frame in tqdm(frames_list, desc=f"Saving video {video_name}"):
            if frame is not None:
                writer.write(frame)
        
        writer.release()
        print(f"Video has been saved as: {video_path}")
        return video_path
    
    def save_frames(self, 
                    frames_list: List, 
                    frame_type: str, 
                    manipulation_path: str) -> Optional[str]:
        """Save individual frames as images.
        
        Args:
            frames_list: List of image frames
            frame_type: Type of frames (e.g., 'first_person_RGB', 'top_down')
            manipulation_path: Subdirectory path for organizing frames
            
        Returns:
            Path to saved frames directory or None if failed
        """
        if not frames_list:
            print(f"Warning: {frame_type} no frame data, skip saving")
            return None
        
        # Create manipulation subdirectory if it doesn't exist
        manipulation_dir = os.path.join(self.output_dir, manipulation_path)
        os.makedirs(manipulation_dir, exist_ok=True)
        frames_dir = os.path.join(manipulation_dir, frame_type)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save each frame as an image
        for i, frame in enumerate(tqdm(frames_list, desc=f"Saving {frame_type} frames")):
            if frame is not None:
                frame_filename = f"frame_{i}.png"
                frame_path = os.path.join(frames_dir, frame_filename)
                
                # Convert BGR to RGB if needed (OpenCV uses BGR by default)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(frame_path, frame)
        
        print(f"Frames have been saved to: {frames_dir}")
        return frames_dir

    def get_output_dir(self) -> str:
        """Get the output directory path.
        
        Returns:
            Output directory path
        """
        return self.output_dir


class VisionManager:
    """Manager for visual data processing."""
    
    @staticmethod
    def get_vision(controller) -> Dict[str, Any]:
        """Extract visual data from controller.
        
        Args:
            controller: AI2Thor controller instance
            
        Returns:
            Dictionary containing RGB, depth, and segmentation frames
        """
        frames = {}
        event = controller.last_event
        frames['rgb'] = event.frame
        frames['depth'] = event.depth_frame
        frames['segmentation'] = {
            'instance_segmentation': event.instance_segmentation_frame,
            'color_to_object_id': event.color_to_object_id,
            'object_id_to_color': event.object_id_to_color
        }
        return frames
    
    @staticmethod
    def vis_depth(depth_map: np.ndarray) -> np.ndarray:
        """Visualize depth map with color coding.
        
        Args:
            depth_map: Input depth map array
            
        Returns:
            Color-coded depth visualization
        """
        inf_mask = depth_map == np.inf
        depth_map = np.where(depth_map == np.inf, 0, depth_map)
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        
        if depth_max == depth_min:
            normalized_depth = np.zeros_like(depth_map)
        else:
            normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
            
        colored_depth = cv2.applyColorMap(
            (normalized_depth * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        colored_depth[inf_mask] = [255, 255, 255]
        return colored_depth
    
    @staticmethod
    def process_segmentation(segmentation_frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Process segmentation frame for visualization.
        
        Args:
            segmentation_frame: Input segmentation frame
            
        Returns:
            Processed segmentation frame or None
        """
        if segmentation_frame is None:
            return None
        
        # Convert segmentation to RGB format for video saving
        if len(segmentation_frame.shape) == 3 and segmentation_frame.shape[2] == 3:
            return segmentation_frame
        else:
            # Convert single channel to pseudo-color
            return cv2.applyColorMap(
                segmentation_frame.astype(np.uint8), 
                cv2.COLORMAP_HSV
            )


class CameraManager:
    """Manager for camera operations."""
    
    @staticmethod
    def add_third_party_camera(controller) -> None:
        """Add third-party camera for top-down view.
        
        Args:
            controller: AI2Thor controller instance
        """
        event = controller.step(
            action="GetMapViewCameraProperties", 
            raise_for_failure=True
        )
        pose = copy.deepcopy(event.metadata["actionReturn"])
        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])
        
        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]
        
        controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )


class TrajectoryVisualizer:
    """Visualizer for agent trajectories."""
    
    def __init__(self, reachable_coords: List[Tuple[float, float]]):
        """Initialize trajectory visualizer.
        
        Args:
            reachable_coords: List of reachable coordinate tuples
        """
        self.reachable_coords = reachable_coords
        self.xs = [coord[0] for coord in reachable_coords]
        self.zs = [coord[1] for coord in reachable_coords]
    
    def visualize_trajectory(self, 
                           agent_state: AgentState, 
                           output_path: Optional[str] = None) -> None:
        """Visualize the agent's trajectory.
        
        Args:
            agent_state: Current agent state
            output_path: Optional path to save the visualization
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot reachable positions
        ax.scatter(self.xs, self.zs, c='lightgray', s=10, alpha=0.5, label='Reachable')
        
        # Plot visited positions
        if agent_state.visited_positions:
            visited_xs = [pos[0] for pos in agent_state.visited_positions]
            visited_zs = [pos[1] for pos in agent_state.visited_positions]
            ax.scatter(visited_xs, visited_zs, c='blue', s=20, alpha=0.7, label='Visited')
        
        # Plot trajectory
        if len(agent_state.trajectory) > 1:
            traj_xs = [pos[0] for pos in agent_state.trajectory]
            traj_zs = [pos[1] for pos in agent_state.trajectory]
            ax.plot(traj_xs, traj_zs, 'r-', linewidth=2, alpha=0.8, label='Trajectory')
        
        # Plot current position
        ax.scatter([agent_state.position[0]], [agent_state.position[1]], 
                  c='red', s=100, marker='*', label='Current')
        
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_title(f"Agent Trajectory (Step: {agent_state.step_count})")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def generate_trajectory_frame(self, current_step: int, agent_state: AgentState, rectangles: Optional[List] = None) -> np.ndarray:
        """Generate trajectory plot frame for video recording.
        
        Args:
            current_step: Current step number
            agent_state: Current agent state
            rectangles: Optional list of rectangles to display
            
        Returns:
            BGR image array for video recording
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        reach_xs = [pos[0] for pos in self.reachable_coords]
        reach_zs = [pos[1] for pos in self.reachable_coords]
        ax.scatter(reach_xs, reach_zs, c='lightblue', s=30, alpha=0.6, label='Reachable Positions')
        
        if rectangles:
            colors = plt.cm.Set3(np.linspace(0, 1, len(rectangles)))
            for i, rect in enumerate(rectangles):
                width = rect.x_max - rect.x_min
                height = rect.z_max - rect.z_min
                
                if width == 0 and height == 0:
                    ax.scatter(rect.x_min, rect.z_min, color=colors[i], s=100, 
                              marker='*', alpha=0.7, zorder=5)
                else:
                    rectangle_patch = patches.Rectangle(
                        (rect.x_min, rect.z_min), width, height,
                        linewidth=2, edgecolor=colors[i], facecolor=colors[i], alpha=0.2
                    )
                    ax.add_patch(rectangle_patch)
                    
                    center_x = (rect.x_min + rect.x_max) / 2
                    center_z = (rect.z_min + rect.z_max) / 2
                    ax.text(center_x, center_z, f'R{i+1}', 
                           ha='center', va='center', fontsize=8, fontweight='bold')
        
        if len(agent_state.trajectory) > 1:
            traj_xs = [pos[0] for pos in agent_state.trajectory]
            traj_zs = [pos[1] for pos in agent_state.trajectory]
            ax.plot(traj_xs, traj_zs, 'r-', linewidth=2, alpha=0.7, label='Agent Trajectory')
        
        if agent_state.visited_positions:
            visited_xs = [pos[0] for pos in agent_state.visited_positions]
            visited_zs = [pos[1] for pos in agent_state.visited_positions]
            ax.scatter(visited_xs, visited_zs, c='red', s=50, alpha=0.8, label='Visited Positions')
        
        if agent_state.trajectory:
            ax.scatter(agent_state.trajectory[0][0], agent_state.trajectory[0][1], 
                      c='green', s=100, marker='s', label='Start')
        
        current_pos = agent_state.position
        current_rotation = agent_state.rotation
        
        arrow_length = 0.3
        angle_rad = math.radians(current_rotation)
        dx = arrow_length * math.sin(angle_rad)
        dz = arrow_length * math.cos(angle_rad)
        
        ax.annotate('', xy=(current_pos[0] + dx, current_pos[1] + dz), 
                   xytext=(current_pos[0], current_pos[1]),
                   arrowprops=dict(arrowstyle='->', color='orange', lw=5, alpha=0.9),
                   zorder=11)
        
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Z Coordinate', fontsize=12)
        coverage = len(agent_state.visited_positions) / len(self.reachable_coords) * 100
        ax.set_title(f'Agent Exploration - Step: {current_step}\n'
                    f'Reachable: {len(self.reachable_coords)} | Visited: {len(agent_state.visited_positions)} | '
                    f'Coverage: {coverage:.1f}%', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        return img_bgr
