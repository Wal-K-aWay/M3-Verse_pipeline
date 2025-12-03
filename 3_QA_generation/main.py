#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA Generation System Batch Scene Processing Script

This script is used to batch generate question-answer data for multiple AI2-THOR scenes in a specified directory.
You can specify one or more question generators, or use all available generators by default.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from generators.qa_generator import QAGenerator

current_dir = os.path.dirname(os.path.abspath(__file__)) # 3_QA_generation
parent_dir = os.path.dirname(current_dir) # M^3-Verse_pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import SINGLE_STATE_TYPES, MULTI_STATE_TYPES

def find_scene_directories(base_path: Path) -> List[Path]:
    """
    Find all scene directories under the given path
    
    Args:
        base_path: Base search path
        
    Returns:
        List of scene directory paths
    """
    scene_dirs = []
    
    # Find all directories starting with scene_
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith('scene_'):
            # Verify if it's a valid scene directory (contains necessary files)
            if is_valid_scene_directory(item):
                scene_dirs.append(item)
            else:
                print(f"Warning: Skipping invalid scene directory '{item.name}'")
    
    # Sort by scene number
    scene_dirs.sort(key=lambda x: extract_scene_number(x.name))
    return scene_dirs

def extract_scene_number(scene_name: str) -> int:
    """
    Extract numeric ID from scene name
    
    Args:
        scene_name: Scene directory name, e.g., 'scene_800'
        
    Returns:
        Scene number
    """
    try:
        return int(scene_name.split('_')[1])
    except (IndexError, ValueError):
        return 0

def is_valid_scene_directory(scene_path: Path) -> bool:
    """
    Check if it's a valid scene directory
    
    Args:
        scene_path: Scene directory path
        
    Returns:
        Whether it's a valid scene directory
    """
    # Check if necessary files or directories exist
    required_items = ['state_0', 'object_asset_mapping.json']
    
    for item in required_items:
        if not (scene_path / item).exists():
            return False
    
    return True

def process_single_scene(scene_path: Path, qa_generator: QAGenerator, 
                        single_state_types: List[str], 
                        multi_state_types: List[str]) -> Dict[str, Any]:
    """
    Process a single scene
    
    Args:
        scene_path: Scene path
        qa_generator: QA generator instance
        single_state_types: List of single-state generator types
        multi_state_types: List of multi-state generator types
        
    Returns:
        Scene QA data dictionary
    """
    try:
        print(f"Processing scene: {scene_path.name}")
        start_time = time.time()
        
        # Generate QA data for the scene
        qa_data = qa_generator.generate_qa_for_scene(
            scene_path=str(scene_path),
            single_state_types=single_state_types,
            multi_state_types=multi_state_types
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Add processing time information
        qa_data['processing_time'] = processing_time
        qa_data['success'] = True
        
        print(f"Scene {scene_path.name} processing completed, took {processing_time:.2f} seconds")
        print(f"Generated questions: {qa_data['statistics']['total_questions']}")
        
        return qa_data
        
    except Exception as e:
        print(f"Error occurred while processing scene {scene_path.name}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'scene_path': str(scene_path),
            'success': False,
            'error': str(e),
            'processing_time': 0,
            'statistics': {
                'total_questions': 0,
                'single_state_count': 0,
                'multi_state_count': 0,
                'question_types': {}
            }
        }

def calculate_overall_statistics(all_scenes_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate overall statistics for all scenes
    
    Args:
        all_scenes_data: List of QA data for all scenes
        
    Returns:
        Overall statistics dictionary
    """
    stats = {
        'total_scenes': len(all_scenes_data),
        'successful_scenes': 0,
        'failed_scenes': 0,
        'total_questions': 0,
        'total_single_state': 0,
        'total_multi_state': 0,
        'total_processing_time': 0,
        'question_types': {},
        'scene_statistics': []
    }
    
    for scene_data in all_scenes_data:
        if scene_data.get('success', False):
            stats['successful_scenes'] += 1
            scene_stats = scene_data['statistics']
            stats['total_questions'] += scene_stats['total_questions']
            stats['total_single_state'] += scene_stats['single_state_count']
            stats['total_multi_state'] += scene_stats['multi_state_count']
            
            # Accumulate question counts by type
            for q_type, count in scene_stats['question_types'].items():
                if q_type not in stats['question_types']:
                    stats['question_types'][q_type] = 0
                stats['question_types'][q_type] += count
        else:
            stats['failed_scenes'] += 1
        
        stats['total_processing_time'] += scene_data.get('processing_time', 0)
        
        # Record brief statistics for each scene
        stats['scene_statistics'].append({
            'scene_name': Path(scene_data['scene_path']).name,
            'success': scene_data.get('success', False),
            'questions_count': scene_data['statistics']['total_questions'],
            'processing_time': scene_data.get('processing_time', 0)
        })
    
    return stats

def main():
    """Main execution function"""
    # Get all available generator names
    all_single_state_generators = list(SINGLE_STATE_TYPES.keys())
    all_multi_state_generators = list(MULTI_STATE_TYPES.keys())
    all_generators = all_single_state_generators + all_multi_state_generators

    parser = argparse.ArgumentParser(
        description='Batch generate QA data for multiple AI2-THOR scenes.',
        formatter_class=argparse.RawTextHelpFormatter # Preserve help text format
    )
    parser.add_argument(
        '--scenes_directory',
        type=str,
        default=None,
        help='Directory path containing multiple scenes.'
    )
    parser.add_argument(
        '-g', '--generators',
        type=str,
        help=f"""Specify generators to use, separated by commas.
If not provided, all available generators will be used.
Available generators: {', '.join(all_generators)}"""
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default=os.path.join(parent_dir, 'M^3-Verse/QAs'),
        help='Directory to save generated QA files '
    )
    parser.add_argument(
        '--max_scenes',
        type=int,
        help='Limit the maximum number of scenes to process (for testing)'
    )
    parser.add_argument(
        '--start_from',
        type=str,
        help='Start processing from specified scene (scene name, e.g., scene_100)'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip scenes that already have output files'
    )

    args = parser.parse_args()

    # --- Validate input ---
    if args.scenes_directory:
        scenes_dir = Path(args.scenes_directory)
    else:
        scenes_dir = Path(os.path.join(parent_dir, 'M^3-Verse/data'))
        print(f"No scenes_directory provided, using default: {scenes_dir}")

    if not scenes_dir.is_dir():
        print(f"Error: Scene directory '{scenes_dir}' is not a valid directory.")
        sys.exit(1)

    # --- Find all scene directories ---
    print(f"Scanning directory: {scenes_dir}")
    scene_paths = find_scene_directories(scenes_dir)
    
    if not scene_paths:
        print(f"Error: No valid scene directories found in '{scenes_dir}'.")
        sys.exit(1)
    
    print(f"Found {len(scene_paths)} valid scene directories")
    
    # --- Apply filtering conditions ---
    if args.start_from:
        start_index = None
        for i, path in enumerate(scene_paths):
            if path.name == args.start_from:
                start_index = i
                break
        if start_index is not None:
            scene_paths = scene_paths[start_index:]
            print(f"Starting from scene {args.start_from}, remaining {len(scene_paths)} scenes")
        else:
            print(f"Warning: Starting scene {args.start_from} not found, will process all scenes")
    
    if args.max_scenes:
        scene_paths = scene_paths[:args.max_scenes]
        print(f"Limited processing to {len(scene_paths)} scenes")

    # --- Parse generators ---
    single_state_to_run = []
    multi_state_to_run = []

    if args.generators:
        requested_generators = [g.strip() for g in args.generators.split(',')]
        for gen_name in requested_generators:
            if gen_name in all_single_state_generators:
                single_state_to_run.append(gen_name)
            elif gen_name in all_multi_state_generators:
                multi_state_to_run.append(gen_name)
            else:
                print(f"Warning: Unknown generator '{gen_name}', will be ignored.")

        if not single_state_to_run and not multi_state_to_run:
            print("Error: All specified generators are invalid.")
            sys.exit(1)
    else:
        # If not specified, use all generators
        single_state_to_run = all_single_state_generators
        multi_state_to_run = all_multi_state_generators

    # --- Display processing information ---
    print(f"\n=== Batch QA Generation Started ===")
    print(f"Scene directory: {scenes_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Scenes to process: {len(scene_paths)}")
    print(f"Single-state generators used: {single_state_to_run or 'None'}")
    print(f"Multi-state generators used: {multi_state_to_run or 'None'}")
    print("-" * 50)

    # --- Initialize QA generator ---
    qa_generator = QAGenerator(output_dir=args.output_dir)
    
    # --- Batch process scenes ---
    all_scenes_data = []
    all_jsonl_qa_data = []  # Store jsonl format QA data for all scenes
    start_time = time.time()
    
    for i, scene_path in enumerate(scene_paths, 1):
        print(f"\n[{i}/{len(scene_paths)}] ", end="")
        
        # Check if existing files should be skipped
        if args.skip_existing:
            output_file = Path(args.output_dir) / f"{scene_path.name}_qa.json"
            if output_file.exists():
                print(f"Skipping existing scene: {scene_path.name}")
                continue
        
        scene_data = process_single_scene(
            scene_path, qa_generator, 
            single_state_to_run, multi_state_to_run
        )
        all_scenes_data.append(scene_data)
        
        # Convert to jsonl format and add to total list
        if scene_data.get('success', False):
            scene_jsonl_data = qa_generator.convert_to_jsonl_format(scene_data)
            all_jsonl_qa_data.extend(scene_jsonl_data)
    
    # --- Calculate overall statistics ---
    overall_stats = calculate_overall_statistics(all_scenes_data)
        
    # --- Save merged jsonl file for all scenes ---
    if all_jsonl_qa_data:
        all_jsonl_filename = "M^3-Verse_all.jsonl"
        all_jsonl_path = qa_generator.save_qa_data_as_jsonl(all_jsonl_qa_data, all_jsonl_filename)
    else:
        all_jsonl_path = None
    
    # --- Display final results ---
    total_time = time.time() - start_time
    print(f"\n=== Batch Processing Completed ===")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Successfully processed scenes: {overall_stats['successful_scenes']}/{overall_stats['total_scenes']}")
    print(f"Failed scenes: {overall_stats['failed_scenes']}")
    print(f"Total generated questions: {overall_stats['total_questions']}")
    if all_jsonl_path:
        print(f"All scenes JSONL data saved to: {all_jsonl_path}")
    
    if overall_stats['failed_scenes'] > 0:
        print(f"\nFailed scenes:")
        for scene_stat in overall_stats['scene_statistics']:
            if not scene_stat['success']:
                print(f"  - {scene_stat['scene_name']}")

if __name__ == '__main__':
    main()
