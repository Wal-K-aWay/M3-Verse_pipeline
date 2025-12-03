#!/usr/bin/env python3
"""Main entry point for AI2Thor exploration.

This script demonstrates how to use the modular AI2Thor exploration package.
"""

import os
import gc
import sys
import psutil
import argparse
import traceback

from datetime import datetime
from ai2thor.platform import CloudRendering
from ai2thor.controller import Controller

from core.config import ExplorationConfig
from core.engine import ExplorationEngine
from core.scene_manager import SceneManager
from strategies import RectangleBasedStrategy
from manipulation.object_manipulator import ObjectManipulator

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)


def process_one_scene(config, scene_id, scene):
    """Process a single scene: reset, explore original, manipulate objects, and explore again.
    
    Args:
        config: ExplorationConfig instance
        scene_id: Scene identifier
        scene: Scene data
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    def get_memory_usage():
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    initial_memory = get_memory_usage()
    print(f"Scene {scene_id} - Initial memory: {initial_memory:.1f} MB")
    
    try:
        controller = Controller(
            scene=scene,
            platform=CloudRendering,
            
            gridSize=config.gridSize,
            snapToGrid=True,
            rotateStepDegrees=90,

            renderDepthImage=True,
            renderInstanceSegmentation=True,

            width=config.frame_size[0],
            height=config.frame_size[1],
            fieldOfView=config.fieldOfView,
            visibilityDistance=config.visibilityDistance,
        )
        # controller.reset(scene=scene)
        explore_engine = ExplorationEngine(controller, config, scene_id)
        manipulator = ObjectManipulator(controller=controller, config=config, scene_id=scene_id, room_analyzer=explore_engine.room_analyzer)
        
        print("Saving static room analysis...")
        saved_static_room_analysis = explore_engine.save_static_room_analysis('')
            
        # Initialize exploration strategy
        reachable_coords = explore_engine.get_reachable_coords()
        strategy = RectangleBasedStrategy(
            config=config,
            reachable_coords=reachable_coords,
        )
        
        strategy.initialize_strategy(explore_engine.get_agent_state())

        print("\nExploring original scene...")
        save_path = "state_0"
        success = explore_engine.explore(strategy)
        if not success:
            print("Exploration failed.")
            controller.stop()
            explore_engine.delete_output_dir()
            return False

        # Save results
        print("\nSaving original scene results...")
        # saved_files_original = explore_engine.save_videos(save_path) ###
        saved_asset_data_original = explore_engine.save_asset_data(save_path)

        saved_trajectory_original = explore_engine.save_agent_logs(save_path)
        saved_files_original = explore_engine.save_vision_results(save_path)
        saved_scene_result = manipulator.save_scene_state(save_path)

        explore_engine.clear()
        
        # Memory cleanup after original exploration
        gc.collect()
        
        time = 1
        for manip_num in range(time):
            print("\nStarting objects manipulation phase...")
            manipulate_results = manipulator.execute_random_operations()

            explore_engine = ExplorationEngine(controller, config, scene_id)
            save_path = f"state_{manip_num+1}"
            
            reachable_coords = explore_engine.get_reachable_coords()

            strategy = RectangleBasedStrategy(
                config=config,
                reachable_coords=reachable_coords,
            )
            strategy.initialize_strategy(explore_engine.get_agent_state())

            print(f"\nExploring scene after {manip_num+1} objects manipulation...")
            success = explore_engine.explore(strategy)
            if not success:
                print("Exploration failed.")
                controller.stop()
                explore_engine.delete_output_dir()
                return False
            
            # Save results
            print("\nSaving results...")
            # saved_files_modified = explore_engine.save_videos(save_path) ###

            saved_asset_data_modified = explore_engine.save_asset_data(save_path)

            saved_trajectory_modified = explore_engine.save_agent_logs(save_path)
            saved_files_modified = explore_engine.save_vision_results(save_path)
            saved_scene_result = manipulator.save_scene_state(save_path)
            saved_operations_log = manipulator.save_operation_log('operations_log.json', save_path)

            explore_engine.clear()
            
            # Memory cleanup after manipulation exploration
            gc.collect()
            memory_after_manip = get_memory_usage()
            print(f"Memory after manipulation {manip_num+1}: {memory_after_manip:.1f} MB")
        
        # Final memory cleanup for this scene
        gc.collect()
        final_memory = get_memory_usage()
        print(f"Scene {scene_id} - Final memory: {final_memory:.1f} MB (Total increase: +{final_memory-initial_memory:.1f} MB)")
        controller.stop()
        return True
        
    except Exception as e:
        print(f"\nError processing scene {scene_id}: {e}")
        traceback.print_exc()
        
        # Clean up: delete scene folder if it exists
        try:
            if 'explore_engine' in locals():
                explore_engine.delete_output_dir()
                print(f"Deleted output directory for scene {scene_id} due to error")
        except Exception as cleanup_error:
            print(f"Failed to delete output directory for scene {scene_id}: {cleanup_error}")
        
        # Stop controller if it exists
        try:
            if 'controller' in locals():
                controller.stop()
        except Exception as controller_error:
            print(f"Failed to stop controller for scene {scene_id}: {controller_error}")
        
        return False


def main():
    print("=" * 60)
    print("AI2Thor Exploration with Scene Management")
    print("=" * 60)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="AI2Thor Exploration with Scene Management")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index of the scene to process")
    parser.add_argument("--end_idx", type=int, default=300, help="Ending index of the scene to process (exclusive)")
    args = parser.parse_args()

    # Initialize scene manager and data dictionaries
    print("\nInitializing scene manager and data dictionaries...")
    scene_manager = SceneManager()
    
    # Configuration
    print("\nConfiguring exploration...")
    config = ExplorationConfig(
        smooth_rotation_degrees=30,
        gridSize=0.25,
        fieldOfView=120,
        visibilityDistance=1.5,

        frame_size=(1024, 768), # (W, H), 
        base_output_dir=os.path.join(parent_dir, 'M^3-Verse'),
        asset_output_dir=os.path.join(parent_dir, '2_object_descriptions/assets'),
    )

    # Initialize AI2Thor controller
    print("\nInitializing AI2Thor controller...")
    
    try:
        split = 'train'
        scenes = scene_manager.load_procthor_scenes(split=split)
        start_idx = args.start_idx
        end_idx = args.end_idx
        for i, scene in enumerate(scenes[start_idx:end_idx]):
            scene_id = start_idx + i
            print(f"\n{'='*50}")
            print(f"Processing scene {scene_id}/{len(scenes[start_idx:end_idx])}")
            print(f"{'='*50}")

            # Check if scene data already exists
            scene_output_dir = os.path.join(config.base_output_dir, 'data', f'scene_{scene_id}')

            if os.path.exists(scene_output_dir) and os.listdir(scene_output_dir):
                print(f"Scene {scene_id} data already exists, skipping...")
                continue
            
            success = process_one_scene(
                config=config, 
                scene_id=scene_id, 
                scene=scene, 
            )
            if not success:
                print(f"Failed to process scene {scene_id}, continuing to next scene...")
                continue
            
            # Memory cleanup after each scene
            gc.collect()
            
            # Get memory usage after cleanup
            try:
                process = psutil.Process()
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"Scene {scene_id} completed. Current memory: {current_memory:.1f} MB")
            except Exception:
                print(f"Scene {scene_id} completed. Memory cleaned up.")
                    
    except KeyboardInterrupt:
        print("\nUser interrupted exploration")
    except Exception as e:
        print(f"\nError while running exploration: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import time
    t1=time.time()
    main()
    t2=time.time()
    print(f'take {t2-t1}s to finish')