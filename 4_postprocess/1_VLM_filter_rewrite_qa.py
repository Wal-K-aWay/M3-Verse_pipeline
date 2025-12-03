import os
import sys
import json
import time
import random
import argparse
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Optional, Callable

import dashscope
from dashscope import Generation

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

@dataclass
class QAPipelineConfig:
    qa_file_path: str
    output_file_path: str
    api_key: str
    filter_no_visual: bool = True
    filter_ambiguity: bool = True
    rewrite_qa: bool = True
    start_index: Optional[int] = None
    end_index: Optional[int] = None

@dataclass
class QAItem:
    scene_id: str
    category: str
    question_id: str
    state_type: str
    question: str
    options: List[str]
    answer: str
    question_type: str
    subcategory: str
    generator_name: str
    hallucination: bool

class QAPipeline:
    def __init__(self, config: QAPipelineConfig):
        self.config = config
        # Initialize DashScope with API key from config
        dashscope.api_key = config.api_key

    def _get_qa_context_description(self) -> str:
        context_parts = [
            "This QA question is part of a dataset designed to evaluate multimodal large models' ability to identify environmental changes.",
            "The scenarios involve an observer navigating through an indoor environment consisting of multiple rooms. During the exploration, the observer visits each room, initially walking along a specified trajectory and recording observations of the entire scene in its original state. Subsequently, environmental changes are introduced—such as repositioning objects or altering their states—and the observer then follows a different trajectory to revisit the rooms and re-record the scene, ultimately ensuring that the entire environment has been fully observed.",
            "A large number of QAs are generated, including single-state and multi-state QAs. These QAs systematically cover all aspects of environmental changes, evaluating multimodal large models' abilities in spatial understanding, temporal understanding, counting, and identifying environmental changes, etc.",
            "The provided QA is just one of these QAs."
        ]
        
        return " ".join(context_parts)

    def _generate_response(self, model: str, prompt: str, temperature: float = 0.5, top_p: float = 0.9, max_tokens: int = 512) -> str:
        """
        Generate a response using Qwen API.
        Returns:
            str: The generated response content.
        """
        response = Generation.call(
            model=model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        return response.output.text

    def _filter_no_visual(self, qa_item: QAItem) -> bool:
        question = qa_item.question
        options = qa_item.options
        answer = qa_item.answer
        qa_context_description = self._get_qa_context_description()
        prompt = f"{qa_context_description}\n Your task is to determine whether a question requires visual information to answer.\nQuestion: {question}\nOptions: {options}\nAnswer: {answer}\nIf the question can be answered correctly without visual information, respond with \"Filter out\". Otherwise, respond with \"Keep\". Your response must be either \"Filter out\" or \"Keep\", with no additional text or explanation."
        response = self._generate_response('qwen-flash', prompt)
        return "keep" == response.lower()

    def _rewrite_qa(self, qa_item: QAItem) -> QAItem:
        # Save original option formats and order
        original_options = qa_item.options
        option_formats = []
        stripped_options = []
        
        # Extract option formats and content
        for opt in original_options:
            # According to the user's instruction, the first three characters are always 'Letter. '
            # e.g., "A. Option text" -> prefix_format = "A. ", content = "Option text"
            prefix_format = opt[:3]
            content = opt[3:]
            option_formats.append(prefix_format)
            stripped_options.append(content)

        qa_context_description = self._get_qa_context_description()
        prompt = f"{qa_context_description}\nYou are a helpful assistant. Here you are provided with a question and its options generated with templates, which may contain unnatural phrasing. Your task is to rewrite the given question and options to make them sound more like natural language without changing their semantic meaning. Do not change the order of the options. Pay special attention to rewriting expressions like 'state_0' and 'state_1' to be more natural and closer to real language. Provide the rewritten question and options in JSON format with keys 'rewritten_question' and 'rewritten_options'.\nQuestion: {qa_item.question}\nOptions: {json.dumps(stripped_options)}\nExample Output:\n{{\"rewritten_question\": \"What is the capital of France?\", \"rewritten_options\": [\"Paris\", \"London\", \"Berlin\"]}}"
        response = self._generate_response('qwen-plus', prompt, temperature=0.95, top_p=0.9, max_tokens=1024)
        
        try:
            rewritten_data = json.loads(response)

            rewritten_question = rewritten_data.get('rewritten_question', qa_item.question)

            rewritten_options = rewritten_data.get('rewritten_options', stripped_options)
            # Recombine options with original formats
            formatted_options = []
            for i, (format_prefix, content) in enumerate(zip(option_formats, rewritten_options)):
                if format_prefix:
                    formatted_options.append(f"{format_prefix}{content}")
                else:
                    formatted_options.append(content)
                        
            # Replace original content if rewritten
            qa_item.question = rewritten_question
            qa_item.options = formatted_options
            
        except json.JSONDecodeError:
            print(f"Failed to parse Qwen API response as JSON for rewriting: {response}")
        
        return qa_item

    def _filter_ambiguity(self, qa_item: QAItem) -> bool:
        qa_context_description = self._get_qa_context_description()
        prompt = f"{qa_context_description}\nYou are a QA quality control assistant. Your task is to filter out QAs where the options are ambiguous. Your evaluation should solely focus on identifying if the options are unclear or ambiguous based on the provided text and options. Do not consider visual aspects, question wording, or correctness of the answer at this stage. Evaluate the following question, options, and correct answer.\nQuestion: {qa_item.question}\nOptions: {', '.join(qa_item.options)}\nCorrect Answer: {qa_item.answer}\nIf you identify any issues (e.g., ambiguous options), respond with \"Filter out\". Otherwise, respond with \"Keep\". Your response must be either \"Filter out\" or \"Keep\", with no additional text or explanation."
        # prompt = f"{qa_context_description}\nYou are a QA quality control assistant. Your task is to filter out QAs where the options are ambiguous. Your evaluation should solely focus on identifying if the options are unclear or ambiguous based on the provided text and options. Do not consider visual aspects, question wording, or correctness of the answer at this stage. Evaluate the following question, options, and correct answer.\nQuestion: {qa_item.question}\nOptions: {', '.join(qa_item.options)}\nCorrect Answer: {qa_item.answer}\nIf you identify any issues (e.g., ambiguous options), respond with \"Filter out\". Otherwise, respond with \"Keep\". Your response must be either \"Filter out\" or \"Keep\", and explain why."
        response = self._generate_response('qwen-plus', prompt)
        # print(response)
        # import pdb;pdb.set_trace()
        return "keep" == response.lower()

    def run_pipeline(self):
        processed_qa = []
        with open(self.config.qa_file_path, 'r', encoding='utf-8') as f_in:
            all_qa_items = [json.loads(line) for line in f_in]
            random.shuffle(all_qa_items)
            total_questions = len(all_qa_items)

            # Determine the actual range to process
            start = self.config.start_index if self.config.start_index is not None else 0
            end = self.config.end_index if self.config.end_index is not None else total_questions

            # Ensure indices are within bounds
            start = max(0, start)
            end = min(total_questions, end)

            print(f"Processing QAs from index {start} to {end-1} (total {end-start} QAs).")
            try:
                for i in tqdm(range(start, end), desc="Processing questions"):
                    qa_data = all_qa_items[i]

                    qa_item = QAItem(**qa_data)

                    start_time = time.time()
                    
                    keep = True
                    filtered_by = None # Initialize a variable to track which filter discarded the QA

                    if self.config.filter_no_visual:
                        if not self._filter_no_visual(qa_item):
                            keep = False
                            filtered_by = "_filter_no_visual"
                    
                    if keep and self.config.rewrite_qa:
                        qa_item = self._rewrite_qa(qa_item)

                    if keep and self.config.filter_ambiguity:
                        if not self._filter_ambiguity(qa_item):
                            keep = False
                            filtered_by = "_filter_ambiguity"

                    if keep:
                        processed_qa.append(qa_item)

                    end_time = time.time()
                    time_taken = end_time - start_time
                    # Include filtered_by information in the print statement
                    filter_info = f"Filtered by: {filtered_by}" if not keep else ""
                    tqdm.write(f"Processing {i+1}/{total_questions} - Original Question: {qa_item.question}\nKeep: {keep} {filter_info}\nTime Taken: {time_taken:.2f} seconds\n---")

                with open(self.config.output_file_path, 'w', encoding='utf-8') as f_out:
                    for item in processed_qa:
                        # Remove rewritten_question and rewritten_options fields before writing to file
                        if hasattr(item, 'rewritten_question'):
                            del item.rewritten_question
                        if hasattr(item, 'rewritten_options'):
                            del item.rewritten_options
                        f_out.write(json.dumps(item.__dict__, ensure_ascii=False) + '\n')

                print(f"Pipeline complete. Kept {len(processed_qa)} QA pairs and saved to {self.config.output_file_path}")

            except Exception as e:
                print(f"An error occurred during pipeline execution: {e}")
                print(f"Saving {len(processed_qa)} partially processed QA pairs to {self.config.output_file_path} before exiting.")
                with open(self.config.output_file_path, 'w', encoding='utf-8') as f_out:
                    for item in processed_qa:
                        if hasattr(item, 'rewritten_question'):
                            del item.rewritten_question
                        if hasattr(item, 'rewritten_options'):
                            del item.rewritten_options
                        f_out.write(json.dumps(item.__dict__, ensure_ascii=False) + '\n')

                print(f"Pipeline complete. Kept {len(processed_qa)} QA pairs and saved to {self.config.output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process QA pairs with filtering and rewriting.')
    parser.add_argument('--input', type=str, default=os.path.join(parent_dir, 'M^3-Verse/QAs/M^3-Verse_all.jsonl'),
                      help='Path to the input QA file')
    parser.add_argument('--output', type=str, default=os.path.join(parent_dir, 'M^3-Verse/QAs/M^3-Verse_VLM_filtered.jsonl'),
                      help='Path to save the processed QA file')
    parser.add_argument('--api_key', type=str, required=True,
                      help='DashScope API key')
    parser.add_argument('--start_index', type=int, default=None,
                      help='Start processing from this index (inclusive)')
    parser.add_argument('--end_index', type=int, default=None,
                      help='Process up to this index (exclusive)')
    parser.add_argument('--filter_no_visual', type=bool, default=True,
                      help='Whether to filter out QAs that do not require visual information')
    parser.add_argument('--filter_ambiguity', type=bool, default=True,
                      help='Whether to filter out QAs with ambiguous options')
    parser.add_argument('--rewrite_qa', type=bool, default=True,
                      help='Whether to rewrite QAs to make them more natural')

    args = parser.parse_args()

    config = QAPipelineConfig(
        qa_file_path=args.input,
        output_file_path=args.output,
        api_key=args.api_key,
        start_index=args.start_index,
        end_index=args.end_index,
        filter_no_visual=args.filter_no_visual,
        filter_ambiguity=args.filter_ambiguity,
        rewrite_qa=args.rewrite_qa,
    )
    
    pipeline = QAPipeline(config)
    start_time = time.time()
    pipeline.run_pipeline()
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Total Time Taken: {time_taken:.2f} seconds")
 