"""Visualization utilities for rectangles and paths.

This module contains functions for visualizing rectangles, points, and paths.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Optional
from core.config import Rectangle


def _visualize_points_and_rectangles(rectangles: List[Rectangle], 
                                    points: List[Tuple[float, float]],
                                    inter_paths: Dict[Tuple[int, int], List[Tuple[float, float]]],
                                    best_endpoints: Dict[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]],
                                    filtered_reachable_coords: Set[Tuple[float, float]],
                                    title: str = "Rectangles and Points Visualization") -> None:
    """Visualize rectangles, points, and paths.
    
    Args:
        rectangles: List of rectangles to visualize
        points: List of points to visualize
        inter_paths: Dictionary of paths between rectangles
        best_endpoints: Dictionary of best endpoints for paths
        filtered_reachable_coords: Set of filtered reachable coordinates
        title: Title for the visualization
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot filtered reachable coordinates as background
    if filtered_reachable_coords:
        reachable_x = [coord[0] for coord in filtered_reachable_coords]
        reachable_z = [coord[1] for coord in filtered_reachable_coords]
        ax.scatter(reachable_x, reachable_z, c='lightgray', s=1, alpha=0.3, label='Reachable Area')
    
    # Plot rectangles
    for i, rect in enumerate(rectangles):
        width = rect.x_max - rect.x_min
        height = rect.z_max - rect.z_min
        
        # Use different colors for different rectangles
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color = colors[i % len(colors)]
        
        rectangle_patch = patches.Rectangle(
            (rect.x_min, rect.z_min), width, height,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
            label=f'Rectangle {i}'
        )
        ax.add_patch(rectangle_patch)
        
        # Add rectangle center point
        center = rect.center()
        ax.plot(center[0], center[1], 'o', color=color, markersize=8)
        ax.text(center[0], center[1], str(i), fontsize=10, ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot inter-rectangle paths
    for (i, j), path in inter_paths.items():
        if path:
            path_x = [point[0] for point in path]
            path_z = [point[1] for point in path]
            ax.plot(path_x, path_z, 'k-', linewidth=2, alpha=0.7, label=f'Path {i}-{j}' if (i, j) == list(inter_paths.keys())[0] else "")
    
    # Plot best endpoints
    for (i, j), (start_point, end_point) in best_endpoints.items():
        if start_point and end_point:
            ax.plot(start_point[0], start_point[1], 'ro', markersize=6, label='Start Points' if (i, j) == list(best_endpoints.keys())[0] else "")
            ax.plot(end_point[0], end_point[1], 'go', markersize=6, label='End Points' if (i, j) == list(best_endpoints.keys())[0] else "")
    
    # Plot additional points
    if points:
        points_x = [point[0] for point in points]
        points_z = [point[1] for point in points]
        ax.scatter(points_x, points_z, c='yellow', s=20, alpha=0.8, label='Additional Points', edgecolors='black')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


def visualize_rectangle_strategy_result(rectangles: List[Rectangle],
                                       points: List[Tuple[float, float]],
                                       inter_paths: Dict[Tuple[int, int], List[Tuple[float, float]]],
                                       best_endpoints: Dict[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]],
                                       filtered_reachable_coords: Set[Tuple[float, float]],
                                       final_trajectory: Optional[List[Tuple[float, float]]] = None) -> None:
    """Visualize the complete rectangle strategy result.
    
    Args:
        rectangles: List of rectangles
        points: List of points
        inter_paths: Dictionary of paths between rectangles
        best_endpoints: Dictionary of best endpoints
        filtered_reachable_coords: Set of filtered reachable coordinates
        final_trajectory: Optional final trajectory to visualize
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left plot: Rectangles and paths
    _plot_rectangles_and_paths(ax1, rectangles, inter_paths, best_endpoints, 
                              filtered_reachable_coords, "Rectangles and Inter-Rectangle Paths")
    
    # Right plot: Points and trajectory
    _plot_points_and_trajectory(ax2, points, final_trajectory, 
                               filtered_reachable_coords, "Points and Final Trajectory")
    
    plt.tight_layout()
    plt.show()


def _plot_rectangles_and_paths(ax, rectangles: List[Rectangle],
                              inter_paths: Dict[Tuple[int, int], List[Tuple[float, float]]],
                              best_endpoints: Dict[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]],
                              filtered_reachable_coords: Set[Tuple[float, float]],
                              title: str) -> None:
    """Plot rectangles and paths on given axes."""
    # Plot filtered reachable coordinates as background
    if filtered_reachable_coords:
        reachable_x = [coord[0] for coord in filtered_reachable_coords]
        reachable_z = [coord[1] for coord in filtered_reachable_coords]
        ax.scatter(reachable_x, reachable_z, c='lightgray', s=1, alpha=0.3, label='Reachable Area')
    
    # Plot rectangles
    for i, rect in enumerate(rectangles):
        width = rect.x_max - rect.x_min
        height = rect.z_max - rect.z_min
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color = colors[i % len(colors)]
        
        rectangle_patch = patches.Rectangle(
            (rect.x_min, rect.z_min), width, height,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
            label=f'Rectangle {i}'
        )
        ax.add_patch(rectangle_patch)
        
        center = rect.center()
        ax.plot(center[0], center[1], 'o', color=color, markersize=8)
        ax.text(center[0], center[1], str(i), fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot inter-rectangle paths
    for (i, j), path in inter_paths.items():
        if path:
            path_x = [point[0] for point in path]
            path_z = [point[1] for point in path]
            ax.plot(path_x, path_z, 'k-', linewidth=2, alpha=0.7)
    
    # Plot best endpoints
    for (i, j), (start_point, end_point) in best_endpoints.items():
        if start_point and end_point:
            ax.plot(start_point[0], start_point[1], 'ro', markersize=6)
            ax.plot(end_point[0], end_point[1], 'go', markersize=6)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def _plot_points_and_trajectory(ax, points: List[Tuple[float, float]],
                               final_trajectory: Optional[List[Tuple[float, float]]],
                               filtered_reachable_coords: Set[Tuple[float, float]],
                               title: str) -> None:
    """Plot points and trajectory on given axes."""
    # Plot filtered reachable coordinates as background
    if filtered_reachable_coords:
        reachable_x = [coord[0] for coord in filtered_reachable_coords]
        reachable_z = [coord[1] for coord in filtered_reachable_coords]
        ax.scatter(reachable_x, reachable_z, c='lightgray', s=1, alpha=0.3, label='Reachable Area')
    
    # Plot points
    if points:
        points_x = [point[0] for point in points]
        points_z = [point[1] for point in points]
        ax.scatter(points_x, points_z, c='blue', s=30, alpha=0.8, label='Points', edgecolors='black')
        
        # Number the points
        for i, (x, z) in enumerate(points):
            ax.text(x, z, str(i), fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Plot final trajectory
    if final_trajectory:
        traj_x = [point[0] for point in final_trajectory]
        traj_z = [point[1] for point in final_trajectory]
        ax.plot(traj_x, traj_z, 'r-', linewidth=3, alpha=0.8, label='Final Trajectory')
        
        # Mark start and end points
        if len(final_trajectory) > 0:
            ax.plot(final_trajectory[0][0], final_trajectory[0][1], 'go', markersize=10, label='Start')
            ax.plot(final_trajectory[-1][0], final_trajectory[-1][1], 'ro', markersize=10, label='End')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def save_visualization(rectangles: List[Rectangle],
                      points: List[Tuple[float, float]],
                      inter_paths: Dict[Tuple[int, int], List[Tuple[float, float]]],
                      best_endpoints: Dict[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]],
                      filtered_reachable_coords: Set[Tuple[float, float]],
                      filename: str,
                      final_trajectory: Optional[List[Tuple[float, float]]] = None) -> None:
    """Save visualization to file.
    
    Args:
        rectangles: List of rectangles
        points: List of points
        inter_paths: Dictionary of paths between rectangles
        best_endpoints: Dictionary of best endpoints
        filtered_reachable_coords: Set of filtered reachable coordinates
        filename: Filename to save the visualization
        final_trajectory: Optional final trajectory to visualize
    """
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot all elements
    if filtered_reachable_coords:
        reachable_x = [coord[0] for coord in filtered_reachable_coords]
        reachable_z = [coord[1] for coord in filtered_reachable_coords]
        ax.scatter(reachable_x, reachable_z, c='lightgray', s=1, alpha=0.3, label='Reachable Area')
    
    # Plot rectangles
    for i, rect in enumerate(rectangles):
        width = rect.x_max - rect.x_min
        height = rect.z_max - rect.z_min
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color = colors[i % len(colors)]
        
        rectangle_patch = patches.Rectangle(
            (rect.x_min, rect.z_min), width, height,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
            label=f'Rectangle {i}'
        )
        ax.add_patch(rectangle_patch)
        
        center = rect.center()
        ax.plot(center[0], center[1], 'o', color=color, markersize=8)
        ax.text(center[0], center[1], str(i), fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot inter-rectangle paths
    for (i, j), path in inter_paths.items():
        if path:
            path_x = [point[0] for point in path]
            path_z = [point[1] for point in path]
            ax.plot(path_x, path_z, 'k-', linewidth=2, alpha=0.7)
    
    # Plot points
    if points:
        points_x = [point[0] for point in points]
        points_z = [point[1] for point in points]
        ax.scatter(points_x, points_z, c='yellow', s=20, alpha=0.8, label='Points', edgecolors='black')
    
    # Plot final trajectory
    if final_trajectory:
        traj_x = [point[0] for point in final_trajectory]
        traj_z = [point[1] for point in final_trajectory]
        ax.plot(traj_x, traj_z, 'r-', linewidth=3, alpha=0.8, label='Final Trajectory')
        
        if len(final_trajectory) > 0:
            ax.plot(final_trajectory[0][0], final_trajectory[0][1], 'go', markersize=10, label='Start')
            ax.plot(final_trajectory[-1][0], final_trajectory[-1][1], 'ro', markersize=10, label='End')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_title('Rectangle Strategy Visualization')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {filename}")


def visualize_sample_points(filtered_reachable_coords: Set[Tuple[float, float]], 
                           rectangles: List[Rectangle] = None, 
                           intersection_points: List[Tuple[float, float]] = None, 
                           corner_points: List[Tuple[float, float]] = None, 
                           contour_points: List[Tuple[float, float]] = None, 
                           path_points: List[Tuple[float, float]] = None, 
                           start_position: Tuple[float, float] = None,
                           save_path: str = "sample_points_visualization.png") -> None:
    """Visualize all extracted sample points along with rectangles and reachable coordinates.
    
    Args:
        filtered_reachable_coords: Set of reachable coordinates
        rectangles: List of rectangles
        intersection_points: Intersection points between rectangles
        corner_points: Rectangle corner points
        contour_points: Rectangle contour points
        path_points: Inter-rectangle path points
        start_position: Starting position (optional)
        save_path: Path to save the visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    
    # Get coordinate ranges for plotting
    if filtered_reachable_coords:
        x_coords = [x for x, z in filtered_reachable_coords]
        z_coords = [z for x, z in filtered_reachable_coords]
        x_min, x_max = min(x_coords), max(x_coords)
        z_min, z_max = min(z_coords), max(z_coords)
    else:
        x_min = x_max = z_min = z_max = 0
    
    # Add margin
    margin = 1.0
    x_min -= margin
    x_max += margin
    z_min -= margin
    z_max += margin
    
    # Plot reachable coordinates as background
    if filtered_reachable_coords:
        reachable_x = [x for x, z in filtered_reachable_coords]
        reachable_z = [z for x, z in filtered_reachable_coords]
        ax.scatter(reachable_x, reachable_z, c='black', s=10, alpha=0.5, label=f'Reachable Points ({len(filtered_reachable_coords)})')
    
    # Plot rectangles
    if rectangles:
        for i, rect in enumerate(rectangles):
            width = rect.x_max - rect.x_min
            height = rect.z_max - rect.z_min
            
            # Different colors for different rectangle types
            if width < 1e-6 and height < 1e-6:
                # Point rectangle
                ax.plot(rect.x_min, rect.z_min, 'ko', markersize=15, label='Point Rectangle' if i == 0 else "")
            elif width < 1e-6 or height < 1e-6:
                # Line rectangle
                if width < 1e-6:  # Vertical line
                    ax.plot([rect.x_min, rect.x_min], [rect.z_min, rect.z_max], 'b-', linewidth=5, 
                        label='Line Rectangle' if i == 0 else "")
                else:  # Horizontal line
                    ax.plot([rect.x_min, rect.x_max], [rect.z_min, rect.z_min], 'b-', linewidth=5,
                        label='Line Rectangle' if i == 0 else "")
            else:
                # Normal rectangle
                rectangle_patch = patches.Rectangle(
                    (rect.x_min, rect.z_min), width, height,
                    linewidth=10, edgecolor='blue', facecolor='lightblue', alpha=0.3,
                    label='Rectangle' if i == 0 else ""
                )
                ax.add_patch(rectangle_patch)
                
                # Add rectangle index
                center_x = rect.x_min + width / 2
                center_z = rect.z_min + height / 2
                ax.text(center_x, center_z, str(i), fontsize=10, ha='center', va='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    else:
        rectangles=[]

    # Plot different types of sample points with different colors and markers
    if intersection_points:
        inter_x = [x for x, z in intersection_points]
        inter_z = [z for x, z in intersection_points]
        ax.scatter(inter_x, inter_z, c='red', s=100, marker='*', 
                  label=f'Intersection Points ({len(intersection_points)})', zorder=5)
    else:
        intersection_points=[]

    if corner_points:
        corner_x = [x for x, z in corner_points]
        corner_z = [z for x, z in corner_points]
        ax.scatter(corner_x, corner_z, c='orange', s=80, marker='s', 
                  label=f'Corner Points ({len(corner_points)})', zorder=4)
    else:
        corner_points = []

    if contour_points:
        contour_x = [x for x, z in contour_points]
        contour_z = [z for x, z in contour_points]
        ax.scatter(contour_x, contour_z, c='green', s=60, marker='o', 
                  label=f'Contour Points ({len(contour_points)})', zorder=3)
    else:
        contour_points = []

    if path_points:
        path_x = [x for x, z in path_points]
        path_z = [z for x, z in path_points]
        ax.scatter(path_x, path_z, c='purple', s=40, marker='^', 
                  label=f'Path Points ({len(path_points)})', zorder=2)
    else:
        path_points = []

    # Plot start position if provided
    if start_position:
        ax.scatter(start_position[0], start_position[1], c='black', s=150, marker='X', 
                  label='Start Position', zorder=6)
    
    # Set plot properties
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Z Coordinate', fontsize=12)
    ax.set_title('Sample Points Visualization', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    
    # Add statistics text
    total_points = len(intersection_points) + len(corner_points) + len(contour_points) + len(path_points)
    stats_text = f"Total Sample Points: {total_points}\n"
    stats_text += f"Rectangles: {len(rectangles)}\n"
    stats_text += f"Reachable Coords: {len(filtered_reachable_coords)}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sample points visualization saved to: {save_path}")


def create_trajectory_gif(trajectory: List[Tuple[float, float]],
                         rectangles: List[Rectangle],
                         filtered_reachable_coords: Set[Tuple[float, float]],
                         gif_path: str = "trajectory_animation.gif",
                         fps: int = 24,
                         show_trail: bool = True,
                         trail_length: int = 10) -> None:
    """Create an animated GIF showing the trajectory movement.
    
    Args:
        trajectory: List of trajectory points
        rectangles: List of rectangles to display as background
        filtered_reachable_coords: Set of reachable coordinates
        gif_path: Path to save the GIF file
        fps: Frames per second for the animation
        show_trail: Whether to show a trail behind the agent
        trail_length: Length of the trail (number of previous positions)
    """
    if not trajectory:
        print("No trajectory to animate")
        return
    
    import io
    from PIL import Image
    
    # Get coordinate ranges for plotting
    if filtered_reachable_coords:
        x_coords = [x for x, z in filtered_reachable_coords]
        z_coords = [z for x, z in filtered_reachable_coords]
        x_min, x_max = min(x_coords), max(x_coords)
        z_min, z_max = min(z_coords), max(z_coords)
    else:
        # Use trajectory bounds if no reachable coords
        traj_x = [x for x, z in trajectory]
        traj_z = [z for x, z in trajectory]
        x_min, x_max = min(traj_x), max(traj_x)
        z_min, z_max = min(traj_z), max(traj_z)
    
    # Add margin
    margin = 1.0
    x_min -= margin
    x_max += margin
    z_min -= margin
    z_max += margin
    
    # Function to create a single frame
    def create_frame(frame_idx):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot reachable coordinates as background
        if filtered_reachable_coords:
            reachable_x = [x for x, z in filtered_reachable_coords]
            reachable_z = [z for x, z in filtered_reachable_coords]
            ax.scatter(reachable_x, reachable_z, c='black', s=3, alpha=0.7, label='Reachable Area')
        
        # Plot rectangles
        for i, rect in enumerate(rectangles):
            width = rect.x_max - rect.x_min
            height = rect.z_max - rect.z_min
            
            # Different colors for different rectangle types
            if width < 1e-6 and height < 1e-6:
                # Point rectangle
                ax.plot(rect.x_min, rect.z_min, 'ko', markersize=8)
            elif width < 1e-6 or height < 1e-6:
                # Line rectangle
                if width < 1e-6:  # Vertical line
                    ax.plot([rect.x_min, rect.x_min], [rect.z_min, rect.z_max], 'b-', linewidth=3)
                else:  # Horizontal line
                    ax.plot([rect.x_min, rect.x_max], [rect.z_min, rect.z_min], 'b-', linewidth=3)
            else:
                # Normal rectangle
                rectangle_patch = patches.Rectangle(
                    (rect.x_min, rect.z_min), width, height,
                    linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.3
                )
                ax.add_patch(rectangle_patch)
                
                # Add rectangle index
                center_x = rect.x_min + width / 2
                center_z = rect.z_min + height / 2
                ax.text(center_x, center_z, str(i), fontsize=10, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        current_pos = trajectory[frame_idx]
        
        # Plot complete trajectory up to current frame
        if frame_idx > 0:
            # Plot the complete path traveled so far
            traveled_points = trajectory[:frame_idx + 1]
            traveled_x = [x for x, z in traveled_points]
            traveled_z = [z for x, z in traveled_points]
            
            # Plot the complete traveled path
            if len(traveled_points) > 1:
                ax.plot(traveled_x, traveled_z, 'r-', linewidth=2, alpha=0.6, label='Traveled Path')
            
            # Plot trail with gradient effect if enabled
            if show_trail:
                trail_start = max(0, frame_idx - trail_length)
                trail_points = trajectory[trail_start:frame_idx + 1]
                
                if len(trail_points) > 1:
                    trail_x = [x for x, z in trail_points]
                    trail_z = [z for x, z in trail_points]
                    
                    # Create gradient effect for recent trail
                    for i in range(len(trail_points) - 1):
                        alpha = (i + 1) / len(trail_points) * 0.9
                        ax.plot([trail_x[i], trail_x[i+1]], [trail_z[i], trail_z[i+1]], 
                               'orange', linewidth=4, alpha=alpha)
        
        # Plot current position
        ax.plot(current_pos[0], current_pos[1], 'ro', markersize=12, 
               markeredgecolor='black', markeredgewidth=2, label='Agent')
        
        # Plot start and end positions
        start_pos = trajectory[0]
        end_pos = trajectory[-1]
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, 
               markeredgecolor='black', markeredgewidth=2, label='Start')
        ax.plot(end_pos[0], end_pos[1], 'bo', markersize=10, 
               markeredgecolor='black', markeredgewidth=2, label='End')
        
        # Add progress information
        progress_text = f"Step: {frame_idx + 1}/{len(trajectory)}\nPosition: ({current_pos[0]:.2f}, {current_pos[1]:.2f})"
        ax.text(0.02, 0.98, progress_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Set plot properties
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Z Coordinate', fontsize=12)
        ax.set_title('Trajectory Animation', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return img
    
    # Generate all frames with progress bar
    print("\n" + "="*80)
    print(f"Generating trajectory animation frames...")
    print("="*80)
    
    frame_pbar = tqdm(total=len(trajectory), 
                      desc="Generating frames", 
                      unit="frame", 
                      ncols=100,
                      bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                      colour='green')
    
    frames = []
    for i in range(len(trajectory)):
        frame_img = create_frame(i)
        frames.append(frame_img)
        frame_pbar.update(1)
    
    frame_pbar.close()
    
    # Save frames as GIF
    print("\n" + "="*80)
    print(f"Saving GIF to: {gif_path}")
    print("="*80)
    
    save_pbar = tqdm(desc="Saving GIF file", 
                     unit="%", 
                     total=100,
                     ncols=100,
                     bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}] {postfix}",
                     colour='blue')
    
    try:
        # Calculate frame duration in milliseconds
        duration = int(1000 / fps)
        
        # Save as GIF
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=True
        )
        
        save_pbar.n = 100
        save_pbar.set_postfix_str("Done")
        save_pbar.refresh()
        save_pbar.close()
        
        # Success message
        success_pbar = tqdm(total=1, 
                           desc="Successfully saved GIF file", 
                           ncols=100,
                           bar_format="{desc}: {bar}| GIF has been saved to {postfix}",
                           colour='green')
        success_pbar.set_postfix_str(gif_path)
        success_pbar.update(1)
        success_pbar.close()
        
    except Exception as e:
        save_pbar.close()
        
        # Error message
        error_pbar = tqdm(total=1, 
                         desc="Fail to save GIF file", 
                         ncols=100,
                         bar_format="{desc}: {bar}| Please check if Pillow has been installed correctly",
                         colour='red')
        error_pbar.update(1)
        error_pbar.close()
        print(f"\n错误详情: {str(e)}")
    
    print("\n" + "="*80)


def visualize_rectangles_with_coords(primitive_reachable_coords: Set[Tuple[float, float]],
                                    filtered_reachable_coords: Set[Tuple[float, float]],
                                    rectangles: List[Rectangle],
                                    title: str = "Rectangles with Coordinate Visualization",
                                    save_path: Optional[str] = 'rectangles.jpg') -> None:
    """Visualize primitive coordinates, filtered coordinates, and rectangles together.
    
    Args:
        primitive_reachable_coords: Set of original reachable coordinates
        filtered_reachable_coords: Set of filtered reachable coordinates
        rectangles: List of rectangles (may include degenerate rectangles as lines or points)
        title: Title for the visualization
        save_path: Optional path to save the visualization
    """
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot primitive reachable coordinates as background
    if primitive_reachable_coords:
        primitive_x = [coord[0] for coord in primitive_reachable_coords]
        primitive_z = [coord[1] for coord in primitive_reachable_coords]
        ax.scatter(primitive_x, primitive_z, c='lightgray', s=5, alpha=0.6, 
                  label=f'Primitive Reachable ({len(primitive_reachable_coords)} points)')
    
    # Plot filtered reachable coordinates
    if filtered_reachable_coords:
        filtered_x = [coord[0] for coord in filtered_reachable_coords]
        filtered_z = [coord[1] for coord in filtered_reachable_coords]
        ax.scatter(filtered_x, filtered_z, c='darkgray', s=5, alpha=0.6, 
                  label=f'Filtered Reachable ({len(filtered_reachable_coords)} points)')
    
    # Define colors for rectangles
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 
              'olive', 'cyan', 'magenta', 'yellow', 'navy', 'lime', 'maroon']
    
    # Plot rectangles with different handling for degenerate cases
    for i, rect in enumerate(rectangles):
        width = rect.x_max - rect.x_min
        height = rect.z_max - rect.z_min
        color = colors[i % len(colors)]
        
        # Handle different rectangle types
        if width < 1e-6 and height < 1e-6:
            # Point rectangle (degenerate case)
            ax.plot(rect.x_min, rect.z_min, 'o', color=color, markersize=8, 
                   markeredgecolor='black', markeredgewidth=2, 
                   label=f'Rectangle {i} (Point)' if i < 5 else "")
            # Add number label
            ax.text(rect.x_min + 0.1, rect.z_min + 0.1, str(i), fontsize=8, 
                   fontweight='bold', ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.9))
                            
        elif width < 1e-6 or height < 1e-6:
            # Line rectangle (degenerate case)
            if width < 1e-6:  # Vertical line
                ax.plot([rect.x_min, rect.x_min], [rect.z_min, rect.z_max], 
                       color=color, linewidth=4, alpha=0.8,
                       label=f'Rectangle {i} (V-Line)' if i < 5 else "")
                # Add number label at midpoint
                mid_z = (rect.z_min + rect.z_max) / 2
                ax.text(rect.x_min + 0.1, mid_z, str(i), fontsize=8, 
                       fontweight='bold', ha='left', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=color, alpha=0.9))
            else:  # Horizontal line
                ax.plot([rect.x_min, rect.x_max], [rect.z_min, rect.z_min], 
                       color=color, linewidth=4, alpha=0.8,
                       label=f'Rectangle {i} (H-Line)' if i < 5 else "")
                # Add number label at midpoint
                mid_x = (rect.x_min + rect.x_max) / 2
                ax.text(mid_x, rect.z_min + 0.1, str(i), fontsize=8, 
                       fontweight='bold', ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=color, alpha=0.9))
        else:
            # Normal rectangle
            rectangle_patch = patches.Rectangle(
                (rect.x_min, rect.z_min), width, height,
                linewidth=3, edgecolor=color, facecolor=color, alpha=0.3,
                label=f'Rectangle {i}' if i < 5 else ""
            )
            ax.add_patch(rectangle_patch)
            
            # Add rectangle center point and number
            center_x = rect.x_min + width / 2
            center_z = rect.z_min + height / 2
            ax.plot(center_x, center_z, 'o', color=color, markersize=8, 
                   markeredgecolor='black', markeredgewidth=1)
            ax.text(center_x, center_z, str(i), fontsize=12, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.9))
    
    # Set plot properties
    ax.set_xlabel('X Coordinate', fontsize=14)
    ax.set_ylabel('Z Coordinate', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add legend with better positioning
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    legend.set_title('Legend', prop={'size': 12, 'weight': 'bold'})

    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")


def visualize_inter_rectangle_paths(reachable_coords: Set[Tuple[float, float]],
                                   rectangles: List[Rectangle],
                                   inter_rectangle_paths: List[List[Tuple[float, float]]],
                                   title: str = "Inter-Rectangle Paths Visualization",
                                   save_path: Optional[str] = None) -> None:
    """Visualize reachable coordinates, rectangles, and inter-rectangle paths together.
    
    Args:
        reachable_coords: Set of reachable coordinates
        rectangles: List of rectangles
        inter_rectangle_paths: List of paths between rectangles
        title: Title for the visualization
        save_path: Optional path to save the visualization
    """
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot reachable coordinates as background
    if reachable_coords:
        reachable_x = [coord[0] for coord in reachable_coords]
        reachable_z = [coord[1] for coord in reachable_coords]
        ax.scatter(reachable_x, reachable_z, c='lightgray', s=5, alpha=0.6, 
                  label=f'Reachable Coords ({len(reachable_coords)} points)')
    
    # Define colors for rectangles
    rect_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 
                   'olive', 'cyan', 'magenta', 'yellow', 'navy', 'lime', 'maroon']
    
    # Plot rectangles with different handling for degenerate cases
    for i, rect in enumerate(rectangles):
        width = rect.x_max - rect.x_min
        height = rect.z_max - rect.z_min
        color = rect_colors[i % len(rect_colors)]
        
        # Handle different rectangle types
        if width < 1e-6 and height < 1e-6:
            # Point rectangle (degenerate case)
            ax.plot(rect.x_min, rect.z_min, 'o', color=color, markersize=10, 
                   markeredgecolor='black', markeredgewidth=2, 
                   label=f'Rectangle {i} (Point)' if i < 5 else "")
            # Add number label
            ax.text(rect.x_min + 0.1, rect.z_min + 0.1, str(i), fontsize=10, 
                   fontweight='bold', ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.9))
                            
        elif width < 1e-6 or height < 1e-6:
            # Line rectangle (degenerate case)
            if width < 1e-6:  # Vertical line
                ax.plot([rect.x_min, rect.x_min], [rect.z_min, rect.z_max], 
                       color=color, linewidth=5, alpha=0.8,
                       label=f'Rectangle {i} (V-Line)' if i < 5 else "")
                # Add number label at midpoint
                mid_z = (rect.z_min + rect.z_max) / 2
                ax.text(rect.x_min + 0.1, mid_z, str(i), fontsize=10, 
                       fontweight='bold', ha='left', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=color, alpha=0.9))
            else:  # Horizontal line
                ax.plot([rect.x_min, rect.x_max], [rect.z_min, rect.z_min], 
                       color=color, linewidth=5, alpha=0.8,
                       label=f'Rectangle {i} (H-Line)' if i < 5 else "")
                # Add number label at midpoint
                mid_x = (rect.x_min + rect.x_max) / 2
                ax.text(mid_x, rect.z_min + 0.1, str(i), fontsize=10, 
                       fontweight='bold', ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=color, alpha=0.9))
        else:
            # Normal rectangle
            rectangle_patch = patches.Rectangle(
                (rect.x_min, rect.z_min), width, height,
                linewidth=3, edgecolor=color, facecolor=color, alpha=0.3,
                label=f'Rectangle {i}' if i < 5 else ""
            )
            ax.add_patch(rectangle_patch)
            
            # Add rectangle center point and number
            center_x = rect.x_min + width / 2
            center_z = rect.z_min + height / 2
            ax.plot(center_x, center_z, 'o', color=color, markersize=10, 
                   markeredgecolor='black', markeredgewidth=1)
            ax.text(center_x, center_z, str(i), fontsize=12, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.9))
    
    # Define colors for paths
    path_colors = ['darkred', 'darkblue', 'darkgreen', 'darkorange', 'darkviolet', 
                   'saddlebrown', 'deeppink', 'darkolivegreen', 'darkcyan', 'darkmagenta']
    
    # Plot inter-rectangle paths
    import pdb;pdb.set_trace()
    for i, path in enumerate(inter_rectangle_paths.values()):
        if len(path) < 2:
            continue
            
        path_color = path_colors[i % len(path_colors)]
        
        # Extract x and z coordinates from path
        path_x = [coord[0] for coord in path]
        path_z = [coord[1] for coord in path]
        
        # Plot path as connected line segments
        ax.plot(path_x, path_z, color=path_color, linewidth=3, alpha=0.8,
               label=f'Path {i}' if i < 5 else "", marker='o', markersize=4)
        
        # Highlight start and end points
        ax.plot(path_x[0], path_z[0], 'o', color=path_color, markersize=8, 
               markeredgecolor='black', markeredgewidth=2)
        ax.plot(path_x[-1], path_z[-1], 's', color=path_color, markersize=8, 
               markeredgecolor='black', markeredgewidth=2)
        
        # Add path number label at midpoint
        if len(path) > 1:
            mid_idx = len(path) // 2
            mid_x, mid_z = path[mid_idx]
            ax.text(mid_x + 0.1, mid_z + 0.1, f'P{i}', fontsize=10, 
                   fontweight='bold', ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                            edgecolor=path_color, alpha=0.9))
    
    # Set plot properties
    ax.set_xlabel('X Coordinate', fontsize=14)
    ax.set_ylabel('Z Coordinate', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add legend with better positioning
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    legend.set_title('Legend', prop={'size': 12, 'weight': 'bold'})

    plt.tight_layout()
    
    # Save if path provided
    if not save_path:
        save_path = 'inter_rectangle_paths.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Inter-rectangle paths visualization saved to: {save_path}")
