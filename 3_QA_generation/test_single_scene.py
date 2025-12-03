#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA Generation System Single Scene Test Script

This script is used to generate question-answer data for a specified single AI2-THOR scene.
You can specify one or more question generators, or use all available generators by default.

Usage:
  # Use all generators to generate QA data for a scene
  python test_single_scene.py /path/to/your/scene_0

  # Use only the 'scene_info' generator
  python test_single_scene.py /path/to/your/scene_0 --generators scene_info

  # Use 'object_info' and 'position_changes' generators
  python test_single_scene.py /path/to/your/scene_0 -g object_info,position_changes
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from generators.qa_generator import QAGenerator
from config import SINGLE_STATE_TYPES, MULTI_STATE_TYPES

def main():
    """Main execution function"""
    # Get all available generator names
    all_single_state_generators = list(SINGLE_STATE_TYPES.keys())
    all_multi_state_generators = list(MULTI_STATE_TYPES.keys())
    all_generators = all_single_state_generators + all_multi_state_generators

    parser = argparse.ArgumentParser(
        description='Generate QA data for a single AI2-THOR scene.',
        formatter_class=argparse.RawTextHelpFormatter # Preserve help text format
    )
    parser.add_argument('scene_path', type=str, help='Path to the scene for QA generation.')
    parser.add_argument('-g', '--generators', type=str, help=f"""Specify generators to use, separated by commas. If not provided, all available generators will be used. Available generators: {', '.join(all_generators)}""")
    parser.add_argument('-o', '--output_dir', type=str, default='qa_output', help='Directory to save generated QA files (default: qa_output)')

    args = parser.parse_args()

    # --- Validate input ---
    scene_path = Path(args.scene_path)
    if not scene_path.is_dir():
        print(f"Error: Scene path '{args.scene_path}' is not a valid directory.")
        sys.exit(1)

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

    # --- Execute generation --- 
    print(f"Processing scene: {scene_path.name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Single-state generators to use: {single_state_to_run or 'None'}")
    print(f"Multi-state generators to use: {multi_state_to_run or 'None'}")
    print("-" * 30)

    try:
        # Initialize QA generator
        qa_generator = QAGenerator(output_dir=args.output_dir)

        # Generate QA data for the scene
        qa_data = qa_generator.generate_qa_for_scene(
            scene_path=str(scene_path),
            single_state_types=single_state_to_run,
            multi_state_types=multi_state_to_run
        )

        # Convert to jsonl format
        scene_jsonl_data = qa_generator.convert_to_jsonl_format(qa_data)

        # Save jsonl file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_filename = f"{scene_path.name}_qa_{timestamp}.jsonl"
        jsonl_path = qa_generator.save_qa_data_as_jsonl(scene_jsonl_data, jsonl_filename)

        print("\n--- Generation Complete ---")
        print(f"QA data successfully saved to: {jsonl_path}")

        # Print statistics
        stats = qa_data.get('statistics', {})
        total_questions = stats.get('total_questions', 0)
        print(f"Total {total_questions} questions generated.")
        for q_type, count in stats.get('question_types', {}).items():
            print(f"  - {q_type}: {count} questions")

    except Exception as e:
        print(f"\nError occurred during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
