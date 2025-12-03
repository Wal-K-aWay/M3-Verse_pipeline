#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main QA Generation System Class

This file defines the QAGenerator class, responsible for coordinating different types of QA generators
and formatting the generated questions.
"""

import json
from typing import Dict, List, Any
from pathlib import Path

# Import generators
from .single_state.scene_info_generator import SceneInfoGenerator
from .single_state.object_info_generator import ObjectInfoGenerator
from .single_state.agent_explore_generator import AgentExploreGenerator

from .multi_state.object_changes_generator import ObjectChangesGenerator
from .multi_state.multi_states_agent_explore_generator import MultiStateAgentExploreGenerator


# Import formatting tools
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.qa_formatter import QAFormatter

class QAGenerator:
    """Main QA generation system class"""
    
    def __init__(self, output_dir: str = "qa_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize QA formatter
        self.qa_formatter = QAFormatter()
        
        # Single-state generators
        self.single_state_generators = {
            'scene_info': SceneInfoGenerator,
            'object_info': ObjectInfoGenerator,
            'agent_explore': AgentExploreGenerator,
        }
        
        # Multi-state generators
        self.multi_state_generators = {
            'object_changes': ObjectChangesGenerator,
            'multi_states_agent_explore': MultiStateAgentExploreGenerator,
        }
    
    def generate_qa_for_scene(self, scene_path: str, 
                             single_state_types: List[str] = None,
                             multi_state_types: List[str] = None) -> Dict[str, Any]:
        """Generate QA data for a single scene
        
        Args:
            scene_path: Path to scene data
            single_state_types: List of single-state question types to generate
            multi_state_types: List of multi-state question types to generate
            
        Returns:
            Dictionary containing all generated questions
        """
        if single_state_types is None:
            single_state_types = list(self.single_state_generators.keys())
        if multi_state_types is None:
            multi_state_types = list(self.multi_state_generators.keys())
            
        scene_qa = {
            'scene_path': scene_path,
            'single_state_questions': {},
            'multi_state_questions': {},
            'statistics': {
                'total_questions': 0,
                'single_state_count': 0,
                'multi_state_count': 0,
                'question_types': {}
            }
        }
        
        # Generate single-state questions
        for question_type in single_state_types:
            if question_type in self.single_state_generators:
                # try:
                generator_class = self.single_state_generators[question_type]
                generator = generator_class(scene_path)
                questions = generator.generate_questions()
                
                # Apply formatting
                formatted_questions = self.qa_formatter.format_questions(questions)
                
                scene_qa['single_state_questions'][question_type] = formatted_questions
                scene_qa['statistics']['single_state_count'] += len(formatted_questions)
                scene_qa['statistics']['question_types'][f'single_state_{question_type}'] = len(formatted_questions)
                
                print(f"Generated {len(formatted_questions)} {question_type} single-state questions")
                    
                # except Exception as e:
                #     print(f"Error generating {question_type} single-state questions: {e}")
                #     scene_qa['single_state_questions'][question_type] = []
        
        # Generate multi-state questions
        for question_type in multi_state_types:
            if question_type in self.multi_state_generators:
                # try:
                generator_class = self.multi_state_generators[question_type]
                generator = generator_class(scene_path)
                questions = generator.generate_questions()
                
                # Apply formatting
                formatted_questions = self.qa_formatter.format_questions(questions)
                
                scene_qa['multi_state_questions'][question_type] = formatted_questions
                scene_qa['statistics']['multi_state_count'] += len(formatted_questions)
                scene_qa['statistics']['question_types'][f'multi_state_{question_type}'] = len(formatted_questions)
                
                print(f"Generated {len(formatted_questions)} {question_type} multi-state questions")
                    
                # except Exception as e:
                #     print(f"Error generating {question_type} multi-state questions: {e}")
                #     scene_qa['multi_state_questions'][question_type] = []
        
        # Update total count
        scene_qa['statistics']['total_questions'] = (
            scene_qa['statistics']['single_state_count'] + 
            scene_qa['statistics']['multi_state_count']
        )
        
        return scene_qa

    def convert_to_jsonl_format(self, scene_qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert scene QA data to jsonl format QA list
        
        Args:
            scene_qa: Scene QA data dictionary
            
        Returns:
            QA list in jsonl format
        """
        jsonl_qa_list = []
        scene_path = Path(scene_qa['scene_path'])
        scene_id = scene_path.name
        
        question_counter = 1
        # Process single-state questions
        for question_type, questions in scene_qa.get('single_state_questions', {}).items():
            for question in questions:
                jsonl_qa = {
                    "scene_id": scene_id,
                    "category": question['category'],
                    "question_id": f"{scene_id}-{question_counter}",
                    "state_type": "intra-state",
                    "question": question['question'],
                    "options": question['choices'],
                    "answer": [question['correct_answer']] if not isinstance(question['correct_answer'], list) else question['correct_answer'],
                    # Preserve original fields
                    "question_type": question['question_type'],
                    "subcategory": question['subcategory'],
                    "generator_name": question_type,
                    "hallucination": question['hallucination']
                }
                jsonl_qa_list.append(jsonl_qa)
                question_counter += 1
        
        # Process multi-state questions
        for question_type, questions in scene_qa.get('multi_state_questions', {}).items():
            for question in questions:
                jsonl_qa = {
                    "scene_id": scene_id,
                    "category": question['category'],
                    "question_id": f"{scene_id}-{question_counter}",
                    "state_type": "inter-states",
                    "question": question['question'],
                    "options": question['choices'],
                    "answer": [question['correct_answer']] if not isinstance(question['correct_answer'], list) else question['correct_answer'],
                    # Preserve original fields
                    "question_type": question['question_type'],
                    "subcategory": question['subcategory'],
                    "generator_name": question_type,
                    "hallucination": question['hallucination']
                }
                jsonl_qa_list.append(jsonl_qa)
                question_counter += 1
        
        return jsonl_qa_list
    
    def save_qa_data_as_jsonl(self, qa_data_list: List[Dict[str, Any]], filename: str = "qa_data.jsonl"):
        """Save QA data as jsonl format file
        
        Args:
            qa_data_list: QA data list
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa_item in qa_data_list:
                json.dump(qa_item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"\nQA data saved as JSONL to: {output_path}")
        print(f"Total QA items: {len(qa_data_list)}")
        
        return str(output_path)

    def save_qa_data(self, qa_data: Dict[str, Any], filename: str = "qa.json"):
        """Save QA data to file"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nQA data saved to: {output_path}")
        
        # Print statistics
        if 'overall_statistics' in qa_data:
            stats = qa_data['overall_statistics']
            print(f"\nDataset Statistics:")
            print(f"  Total scenes: {stats['total_scenes']}")
            print(f"  Successful scenes: {stats['successful_scenes']}")
            print(f"  Failed scenes: {stats['failed_scenes']}")
            print(f"  Total questions: {stats['total_questions']}")
            print(f"  Single-state questions: {stats['total_single_state']}")
            print(f"  Multi-state questions: {stats['total_multi_state']}")
            
            print(f"\nQuestion types:")
            for q_type, count in stats['question_types'].items():
                print(f"  {q_type}: {count}")
        elif 'statistics' in qa_data:
            stats = qa_data['statistics']
            print(f"\nScene Statistics:")
            print(f"  Total questions: {stats['total_questions']}")
            print(f"  Single-state questions: {stats['single_state_count']}")
            print(f"  Multi-state questions: {stats['multi_state_count']}")
            
            print(f"\nQuestion types:")
            for q_type, count in stats['question_types'].items():
                print(f"  {q_type}: {count}")
        
        return str(output_path)
