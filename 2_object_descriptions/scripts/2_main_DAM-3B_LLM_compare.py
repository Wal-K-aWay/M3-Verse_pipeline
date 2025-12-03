#!/usr/bin/env python3
import argparse
import os
import pickle
import json
import sys
import torch
import numpy as np
import re
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add describe-anything directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'describe-anything'))

from dam import DescribeAnythingModel, disable_torch_init


# Query text templates for unified management
COMPARATIVE_ANALYSIS_TEMPLATE = """You are given descriptions of {num_objects} different {object_category} objects. Please analyze and compare these objects to identify their distinguishing characteristics.

{objects_text}

For each object, provide a JSON response that highlights what makes it unique compared to the others in the same category. Focus on distinguishing features like size, color and shape.

For 'shape', and 'color', the value should be a short phrase that fits grammatically in the following sentences:
- An object which is [shape]. (e.g., "rectangular", "round")
- An object which is [color]. (e.g., "red", "brown", "green", "blue")

Please respond with a JSON object where each key is the object ID and the value contains the object's distinctive attributes:

{{
  "{first_asset_id}": {{
    "shape": "<1 word, adjective describing shape>",
    "color": "<1 word, color description>",
    "other_features": "<2-4 words describing other notable features>",
    "description": "<1-2 sentences describing this specific {object_category} and what makes it different from the others>"
  }},
  "{second_asset_id}": {{
    "shape": "<1 word, adjective describing shape>",
    "color": "<1 word, color description>",
    "other_features": "<2-4 words describing other notable features>",
    "description": "<1-2 sentences describing this specific {object_category} and what makes it different from the others>"
  }}
  // ... continue for all objects
}}

Instructions:
- Focus on what makes each object unique within this category
- Ensure the values for shape and color are concise and fit the grammatical examples provided
- If any attribute cannot be determined, set its value to null
- Ensure descriptions emphasize distinguishing characteristics
- Output only valid JSON"""

SUMMARIZE_DESCRIPTIONS_TEMPLATE = """You are given multiple descriptions of the same object: {object_category}.

Here are the descriptions:
{descriptions_text}

Please analyze these descriptions and provide a JSON response with these attributes. Use general and common terms for the attributes. For example, use 'metal' instead of a specific type of metal, and use common color and shape names.
For 'shape' and 'color', the value should be a short phrase that fits grammatically in the following sentences:
- An object which is [shape]. (e.g., "rectangular", "round")
- An object which is [color]. (e.g., "red", "brown", "green", "blue")

{{
  "shape": "<1 word, adjective describing shape>",
  "color": "<1 word, color description>",
  "other_features": "<2-4 words describing other notable features>",
  "description": "<1-2 sentences providing a comprehensive summary of all descriptions, without explicitly mentioning the object category>"
}}

Instructions:
- Synthesize information from all descriptions.
- Focus on the most consistent and frequently mentioned attributes.
- If any attribute cannot be determined from the descriptions, set its value to null.
- Ensure the values for shape and color are concise and fit the grammatical examples provided.
- Output only valid JSON."""

CONSOLIDATE_DESCRIPTIONS_TEMPLATE = """You are given multiple descriptions of the same object. Please consolidate these descriptions into a single, comprehensive, and coherent description that:
1. Integrates all consistent information from the multiple descriptions
2. Resolves any contradictions by choosing the most frequently mentioned or most detailed information
3. Fills in missing details by combining information from different descriptions
4. Produces a more complete and reliable description than any individual description

Here are the descriptions to consolidate:
{descriptions_text}

Please provide a single consolidated description that is comprehensive, coherent, and incorporates the best information from all descriptions. Focus on physical properties, appearance and distinguishing features.

Consolidated description:"""

CATEGORY_FUNCTION_TEMPLATE = """You are given an object category: {object_category}.

Please provide the primary function or purpose of this type of object. The function should be a short phrase (maximum 5 words) that fits grammatically in the sentence:
"An object which is [function]."

For example:
- For "knife": "used for cutting"
- For "chair": "used for sitting"
- For "lamp": "used for lighting"
- For "food items": "food"

Please respond with only the function phrase, nothing else.

Function for {object_category}:"""


def get_category_functions(model, tokenizer, categories, args):
    """
    Get the primary function for each object category using LLM.
    
    Args:
        model: The local LLM model.
        tokenizer: The local LLM tokenizer.
        categories: List of object categories.
        args: Command line arguments.
        
    Returns:
        dict: Dictionary mapping category to its function.
    """
    category_functions = {}
    
    for category in categories:
        print(f"Getting function for category: {category}")
        
        query_text = CATEGORY_FUNCTION_TEMPLATE.format(object_category=category)
        
        try:
            response_text = generate_with_local_llm(
                model, tokenizer, query_text,
                temperature=args.local_temperature,
                top_p=args.local_top_p,
                max_new_tokens=50,  # Short response expected
                enable_thinking=args.enable_thinking
            )
            
            # Clean up the response
            function = response_text.strip().lower()
            # Remove any extra text that might be added
            if '\n' in function:
                function = function.split('\n')[0]
            
            category_functions[category] = function
            print(f"  - Function: {function}")
            
        except Exception as e:
            print(f"  - Error getting function for {category}: {e}")
            # Fallback to a generic function based on category name
            if 'food' in category.lower() or category.lower() in ['egg', 'bread', 'apple']:
                category_functions[category] = "food"
            else:
                category_functions[category] = f"used as {category.lower()}"
            print(f"  - Using fallback function: {category_functions[category]}")
    
    return category_functions

def load_local_llm(model_path):
    """Load the local LLM model and tokenizer"""
    print(f"Loading local LLM from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype='auto',
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    ).eval()

    print("Local LLM loaded.")
    return model, tokenizer

def download_model(model_name, local_dir, hf_mirror):
    """
    Downloads a model from a Hugging Face mirror if it doesn't exist locally.
    
    Args:
        model_name (str): The name of the model on Hugging Face (e.g., 'nvidia/DAM-3B').
        local_dir (str): The local directory to save models.
        hf_mirror (str): The URL of the Hugging Face mirror.
        
    Returns:
        str: The local path to the model.
    """
    model_folder_name = model_name.split('/')[-1]
    local_model_path = os.path.join(local_dir, model_folder_name)
    
    if os.path.exists(local_model_path) and os.listdir(local_model_path):
        print(f"Model '{model_name}' found locally at '{local_model_path}'.")
        return local_model_path
        
    print(f"Model '{model_name}' not found locally. Downloading from {hf_mirror}...")
    os.makedirs(local_model_path, exist_ok=True)
    
    snapshot_download(
        repo_id=model_name,
        local_dir=local_model_path,
        endpoint=hf_mirror,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Model downloaded successfully to '{local_model_path}'.")
    return local_model_path

def initialize_dam_model(args):
    """Initialize the DAM model."""
    abs_local_model_dir = os.path.abspath(args.local_model_dir)
    
    # Check if local model directory already exists
    if os.path.exists(abs_local_model_dir) and os.listdir(abs_local_model_dir):
        print(f"Using existing local model at '{abs_local_model_dir}'.")
        local_model_path = abs_local_model_dir
    else:
        print(f"Local model not found, downloading...")
        model_name = 'nvidia/DAM-3B'
        local_model_path = download_model(
            model_name=model_name,
            local_dir=os.path.dirname(abs_local_model_dir),
            hf_mirror=args.hf_mirror
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disable_torch_init()

    print("Loading DAM model...")
    dam = DescribeAnythingModel(
        model_path=local_model_path,
        conv_mode=args.conv_mode,
        prompt_mode="full+focal_crop",
    ).to(device)
    print("DAM model loaded.")
    return dam

def get_object_category(asset_id):
    """Extract object category from asset ID."""
    parts = asset_id.split('_')
    cutoff_index = len(parts)  # Default to keeping all parts
    for j, part in enumerate(parts):
        if part.isdigit():
            cutoff_index = j
            break
    
    category_parts = parts[:cutoff_index]
    
    # If the name was purely numeric (e.g., '325_1'), fallback to using the full name.
    if not category_parts:
        return asset_id.replace('_', ' ')
    return ' '.join(category_parts)


def generate_dam_descriptions(dam, args, project_root):
    """
    Generate descriptions for all assets using the DAM model.
    
    Args:
        dam: The initialized DAM model.
        args: Command line arguments.
        project_root: The root directory of the project.
        
    Returns:
        dict: category_descriptions - Dictionary mapping object categories to asset descriptions
    """
    category_descriptions = {}

    # Get list of asset directories
    asset_ids = sorted([d for d in os.listdir(args.assets_dir) if os.path.isdir(os.path.join(args.assets_dir, d))])
    if args.limit_assets:
        asset_ids = asset_ids[:args.limit_assets]

    print(f"Processing {len(asset_ids)} assets...")
    print("=" * 60)

    # Process each asset
    for i, asset_id in enumerate(asset_ids, 1):
        asset_dir = os.path.join(args.assets_dir, asset_id)
        pkl_files = sorted([f for f in os.listdir(asset_dir) if f.endswith('.pkl')])
        
        if not pkl_files:
            continue

        object_category = get_object_category(asset_id)
        print(f"\n[{i}/{len(asset_ids)}] Processing asset: {asset_id} (as '{object_category}')")

        # Initialize category group if not exists
        if object_category not in category_descriptions:
            category_descriptions[object_category] = {}

        for pkl_file in pkl_files:
            pkl_path = os.path.join(asset_dir, pkl_file)
            
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            mask_np = data.get('mask')
            rgb_path_relative = data.get('image_path')

            if mask_np is None or rgb_path_relative is None:
                print(f"  - Skipping {pkl_file}: missing 'mask' or 'image_path'.")
                continue

            # Construct the absolute path from the project root and the relative path
            rgb_path = os.path.join(project_root, 'M^3-Verse/data', rgb_path_relative)

            if not os.path.exists(rgb_path):
                print(f"  - Warning: RGB image not found at {rgb_path}. Skipping.")
                continue

            img = Image.open(rgb_path).convert('RGB')
            mask = Image.fromarray((mask_np * 255).astype(np.uint8))

            query = (
                f"<image>\nThe masked object is a(n) {object_category}. Describe it in detail, focusing on its "
                "type, shape, color and physical properties."
            )
            if img.size != mask.size:
                import pdb;pdb.set_trace()
            
            description = dam.get_description(
                img,
                mask,
                query,
                temperature=args.dam_temperature,
                top_p=args.dam_top_p,
                num_beams=1,
                max_new_tokens=args.dam_max_tokens,
            )

            description_text = description.strip()
            if description_text:
                # Add to category group for comparative analysis
                if asset_id not in category_descriptions[object_category]:
                    category_descriptions[object_category][asset_id] = []
                if description_text not in category_descriptions[object_category][asset_id]:
                    category_descriptions[object_category][asset_id].append(description_text)
                
                print(f"    - Generated: {description_text[:80]}...")

    return category_descriptions

def generate_with_local_llm(model, tokenizer, prompt, temperature=0.7, top_p=0.9, max_new_tokens=1024, enable_thinking=False):
    """Generate text using local LLM"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Check if tokenizer supports apply_chat_template (for newer models like Qwen)
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate text
        generated_ids = model.generate(
            **model_inputs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # For Qwen models, find the end of thinking token (151668)
        if 151668 in output_ids:
            index = len(output_ids) - output_ids[::-1].index(151668)
        else:
            index = 0
            
        response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    else:
        # Fallback for older models
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text.replace(prompt, "").strip()
    
    return response

def comparative_analysis_with_local_llm(model, tokenizer, object_category: str, assets_data: dict, args):
    """
    Perform comparative analysis of multiple objects in the same category using local LLM.
    
    Args:
        model: The local LLM model.
        tokenizer: The local LLM tokenizer.
        object_category: The object category.
        assets_data: Dictionary with asset_id as key and description string as value.
        args: Command line arguments.
        
    Returns:
        A dictionary with asset_id as key and structured attributes as value.
    """
    if len(assets_data) < 2:
        return None
    
    # Build the comparative prompt
    objects_text = ""
    asset_ids = list(assets_data.keys())
    
    for i, (asset_id, description) in enumerate(assets_data.items(), 1):
        # Use the description directly as it's now a single string
        objects_text += f"\nObject {i} (ID: {asset_id}):\n  - {description}\n"
    
    query_text = COMPARATIVE_ANALYSIS_TEMPLATE.format(
        num_objects=len(assets_data),
        object_category=object_category,
        objects_text=objects_text,
        first_asset_id=asset_ids[0],
        second_asset_id=asset_ids[1] if len(asset_ids) > 1 else 'object_2'
    )
    
    # Call the local LLM
    response_text = generate_with_local_llm(
        model, tokenizer, query_text,
        temperature=args.local_temperature,
        top_p=args.local_top_p,
        max_new_tokens=args.local_max_tokens * 2,  # Allow more tokens for multiple objects
        enable_thinking=args.enable_thinking
    )
    
    # Parse the JSON response
    # If direct parsing fails, try to extract the JSON part
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = response_text[start_idx:end_idx+1]
        # Clean up the JSON string
        cleaned_json = re.sub(r'\n\s*', ' ', json_str.strip())
        comparative_results = json.loads(cleaned_json)
    else:
        raise json.JSONDecodeError("No valid JSON found", response_text, 0)
    
    # Format results to match expected structure
    formatted_results = {}
    required_keys = ['shape', 'color', 'other_features', 'description']
    
    for asset_id in asset_ids:
        if asset_id in comparative_results:
            asset_attributes = comparative_results[asset_id]
            
            # Ensure all required keys exist
            for key in required_keys:
                if key not in asset_attributes:
                    asset_attributes[key] = None
            
            asset_attributes['object_category'] = object_category
            formatted_results[asset_id] = asset_attributes
        else:
            print(f"    - Warning: No results found for asset {asset_id}")
    
    return formatted_results if formatted_results else None

def summarize_descriptions_with_local_llm(model, tokenizer, object_category: str, descriptions: list, args):
    """
    Summarize multiple descriptions and extract key information using the local LLM.
    
    Args:
        model: The local LLM model.
        tokenizer: The local LLM tokenizer.
        asset_id: The asset ID.
        object_category: The object category.
        descriptions: A list of descriptions.
        args: Command line arguments.
        
    Returns:
        A dictionary containing the summary information.
    """
    # Build the prompt text
    descriptions_text = "\n".join([f"Description {i+1}: {desc}" for i, desc in enumerate(descriptions)])
    
    query_text = SUMMARIZE_DESCRIPTIONS_TEMPLATE.format(
        object_category=object_category,
        descriptions_text=descriptions_text
    )
    
    # Call the local LLM
    response_text = generate_with_local_llm(
        model, tokenizer, query_text,
        temperature=args.local_temperature,
        top_p=args.local_top_p,
        max_new_tokens=args.local_max_tokens,
        enable_thinking=args.enable_thinking
    )
    
    # Parse the JSON response
    # If direct parsing fails, try to extract the JSON part
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = response_text[start_idx:end_idx+1]
        # Clean up the JSON string
        cleaned_json = re.sub(r'\n\s*', ' ', json_str.strip())
        structured_desc = json.loads(cleaned_json)
    else:
        raise json.JSONDecodeError("No valid JSON found", response_text, 0)
    
    # Check if required keys exist
    required_keys = ['shape', 'color', 'other_features', 'description']
    missing_keys = [key for key in required_keys if key not in structured_desc]
    
    if missing_keys:
        print(f"    - Warning: Local LLM response is missing required keys: {missing_keys}")
        
        # Ensure all required keys exist (set to None if missing)
        for key in required_keys:
            if key not in structured_desc:
                structured_desc[key] = None

    structured_desc['object_category'] = object_category
    return structured_desc

def consolidate_asset_descriptions(local_llm_model, local_llm_tokenizer, category_descriptions, args):
    """
    Consolidate multiple descriptions for each asset into a single comprehensive description.
    
    Args:
        local_llm_model: The local LLM model.
        local_llm_tokenizer: The local LLM tokenizer.
        category_descriptions: Dictionary mapping object categories to asset descriptions.
        args: Command line arguments.
        
    Returns:
        dict: A dictionary mapping object categories to consolidated asset descriptions.
              Format: {category: {asset_id: consolidated_description}}
    """
    consolidated_by_category = {}
    
    if not (local_llm_model and local_llm_tokenizer):
        print("\nSkipping description consolidation as local model is not enabled.")
        # Return original descriptions as fallback, maintaining category structure
        for category, assets in category_descriptions.items():
            consolidated_by_category[category] = {}
            for asset_id, descriptions in assets.items():
                if descriptions:
                    consolidated_by_category[category][asset_id] = descriptions[0]  # Use first description
        return consolidated_by_category
    
    for category, assets in category_descriptions.items():
        consolidated_by_category[category] = {}
        
        for asset_id, descriptions in assets.items():
            if len(descriptions) <= 1:
                # Single description, no need to consolidate
                consolidated_by_category[category][asset_id] = descriptions[0] if descriptions else ""
                print(f"\nAsset {asset_id}: Using single description (no consolidation needed)")
                continue
                
            print(f"\nConsolidating {len(descriptions)} descriptions for asset {asset_id}...")
            
            # Build consolidation prompt
            descriptions_text = "\n".join([f"Description {i+1}: {desc}" for i, desc in enumerate(descriptions)])
            
            query_text = CONSOLIDATE_DESCRIPTIONS_TEMPLATE.format(
                descriptions_text=descriptions_text
            )
            
            try:
                consolidated_desc = generate_with_local_llm(
                    local_llm_model, local_llm_tokenizer, query_text,
                    temperature=args.local_temperature,
                    top_p=args.local_top_p,
                    max_new_tokens=args.local_max_tokens,
                    enable_thinking=args.enable_thinking
                )
                
                consolidated_desc = consolidated_desc.strip()
                if consolidated_desc:
                    consolidated_by_category[category][asset_id] = consolidated_desc
                    print(f"  - ✓ Consolidated: {consolidated_desc[:80]}...")
                else:
                    # Fallback to first description if consolidation fails
                    consolidated_by_category[category][asset_id] = descriptions[0]
                    print(f"  - ✗ Consolidation failed, using first description")
                    
            except Exception as e:
                print(f"  - ✗ Error during consolidation for {asset_id}: {e}")
                # Fallback to first description
                consolidated_by_category[category][asset_id] = descriptions[0]
    
    return consolidated_by_category

def process_categories_with_llm(local_llm_model, local_llm_tokenizer, consolidated_by_category, args):
    """
    Process object categories with the local LLM for comparative analysis.
    
    Args:
        local_llm_model: The local LLM model.
        local_llm_tokenizer: The local LLM tokenizer.
        consolidated_by_category: Dictionary mapping object categories to consolidated asset descriptions.
                                 Format: {category: {asset_id: consolidated_description}}
        args: Command line arguments.
        
    Returns:
        dict: A dictionary of structured attributes for each asset.
    """
    
    structured_attributes = {}

    print(f"\n{'='*60}")
    print("Phase 3: Comparative analysis by object category with Local LLM")
    print(f"{'='*60}")
    
    # Calculate total work for progress tracking
    total_categories = len(consolidated_by_category)
    total_assets = sum(len(assets) for assets in consolidated_by_category.values())
    print(f"Total categories to process: {total_categories}")
    print(f"Total assets to process: {total_assets}")
    print()
    
    def create_balanced_batches(assets_data, max_batch_size=10):
        """Create balanced batches where each batch size is between 2 and max_batch_size,
        and the smallest batch size is maximized."""
        asset_items = list(assets_data.items())
        total_assets = len(asset_items)
        
        if total_assets <= max_batch_size:
            return [dict(asset_items)]
        
        # Find the optimal number of batches that maximizes the minimum batch size
        best_num_batches = None
        best_min_batch_size = 0
        
        # Try different numbers of batches
        for num_batches in range(2, total_assets + 1):
            # Check if this number of batches is feasible
            min_possible_batch_size = total_assets // num_batches
            max_possible_batch_size = (total_assets + num_batches - 1) // num_batches
            
            # Skip if any batch would be too large or too small
            if max_possible_batch_size > max_batch_size or min_possible_batch_size < 2:
                continue
            
            # This configuration is valid, check if it's better
            if min_possible_batch_size > best_min_batch_size:
                best_min_batch_size = min_possible_batch_size
                best_num_batches = num_batches
        
        # If no valid configuration found, fall back to simple division
        if best_num_batches is None:
            best_num_batches = (total_assets + max_batch_size - 1) // max_batch_size
            # Adjust to avoid single-item batches
            remainder = total_assets % max_batch_size
            if remainder == 1 and best_num_batches > 1:
                best_num_batches -= 1
        
        # Create batches with the optimal number
        base_batch_size = total_assets // best_num_batches
        extra_items = total_assets % best_num_batches
        
        batches = []
        start_idx = 0
        
        for i in range(best_num_batches):
            # Distribute extra items to first few batches
            current_batch_size = base_batch_size + (1 if i < extra_items else 0)
            
            end_idx = start_idx + current_batch_size
            batch_items = asset_items[start_idx:end_idx]
            batches.append(dict(batch_items))
            start_idx = end_idx
        
        return batches
    
    processed_assets = 0
    
    for category, assets_data in consolidated_by_category.items():
        
        # Comparative analysis for categories with multiple objects
        if len(assets_data) >= 2:            
            # Create balanced batches to manage memory usage
            batches = create_balanced_batches(assets_data)
            
            for batch_idx, batch_assets in enumerate(batches, 1):
                
                try:
                    comparative_result = comparative_analysis_with_local_llm(
                        local_llm_model, local_llm_tokenizer, category, batch_assets, args
                    )

                    if comparative_result:
                        success_count = 0
                        for asset_id, result_data in comparative_result.items():
                            structured_attributes[asset_id] = result_data
                            success_count += 1
                        
                        tqdm.write(f"    Batch {batch_idx}: {success_count}/{len(batch_assets)} assets processed")
                        processed_assets += success_count
                    else:
                        tqdm.write(f"    Batch {batch_idx}: Analysis failed")
                        
                except Exception as e:
                    tqdm.write(f"    Batch {batch_idx}: Error - {str(e)[:80]}...")
            
        else:
            # Individual summarization for single assets or when comparison is disabled
            if len(assets_data) == 1:
                tqdm.write(f"\nCategory '{category}': 1 asset → Standard summarization")
            else:
                tqdm.write(f"\nCategory '{category}': {len(assets_data)} assets → Individual summarization")
            
            for asset_id, description in assets_data.items():
                
                try:
                    summary_result = summarize_descriptions_with_local_llm(
                        local_llm_model, local_llm_tokenizer, category, [description], args
                    )
                    
                    if summary_result:
                        structured_attributes[asset_id] = summary_result
                        tqdm.write(f"    {asset_id}: Summarized")
                        processed_assets += 1
                    else:
                        tqdm.write(f"    {asset_id}: Summarization failed")
                        
                except Exception as e:
                    tqdm.write(f"    {asset_id}: Error - {str(e)[:80]}...")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"   Processing Summary:")
    print(f"   Total categories processed: {total_categories}")
    print(f"   Total assets processed: {processed_assets}/{total_assets}")
    print(f"   Success rate: {processed_assets/max(1, total_assets)*100:.1f}%")
    print(f"   Unique assets with results: {len(structured_attributes)}")
    print(f"{'='*60}")

    return structured_attributes


def setup_arg_parser():
    """
    Set up the argument parser.
    """
    parser = argparse.ArgumentParser(description="Generate object descriptions and structured attributes using DAM and local LLMs.")
    
    # Model and Path Arguments
    parser.add_argument('--local_model_dir', type=str, default='2_object_descriptions/DAM-3B', help='Local directory to save models.')
    parser.add_argument('--hf_mirror', type=str, default='https://hf-mirror.com', help='Hugging Face mirror URL.')
    parser.add_argument('--assets_dir', type=str, default='2_object_descriptions/assets', help='Directory containing asset pkl files.')
    parser.add_argument('--output_dir', type=str, default='2_object_descriptions/descriptions', help='Directory to save output files.')
    parser.add_argument('--local_llm_path', type=str, default=None, help='Path to the local LLM model for summarization.')

    # DAM Model Arguments
    parser.add_argument('--conv_mode', type=str, default='v1', help='Conversation mode for DAM.')
    parser.add_argument('--dam_temperature', type=float, default=0.1, help='Temperature for DAM model generation.')
    parser.add_argument('--dam_top_p', type=float, default=0.3, help='Top-p for DAM model generation.')
    parser.add_argument('--dam_max_tokens', type=int, default=512, help='Max new tokens for DAM model generation.')

    # Local LLM Arguments
    parser.add_argument('--local_temperature', type=float, default=0.1, help='Temperature for local LLM generation.')
    parser.add_argument('--local_top_p', type=float, default=0.3, help='Top-p for local LLM generation.')
    parser.add_argument('--local_max_tokens', type=int, default=1024, help='Max new tokens for local LLM generation.')
    parser.add_argument('--enable_thinking', action='store_true', help='Enable thinking mode for local LLM (if supported).')
    
    # Processing Arguments
    parser.add_argument('--limit_assets', type=int, default=None, help='Limit the number of assets to process.')

    return parser.parse_args()

def save_results(consolidated_by_category, structured_attributes, args):
    """
    Save the consolidated descriptions and structured attributes to files.

    Args:
        consolidated_by_category (dict): Consolidated descriptions grouped by category.
        structured_attributes (dict): All structured attributes.
        args: Command-line arguments.
    """
    # Flatten consolidated descriptions for saving
    consolidated_descriptions = {}
    for category, assets in consolidated_by_category.items():
        consolidated_descriptions.update(assets)
    
    # Save textual descriptions
    if args.text_output_path:
        os.makedirs(os.path.dirname(args.text_output_path), exist_ok=True)
        with open(args.text_output_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_descriptions, f, indent=2, ensure_ascii=False)
        print(f"\nConsolidated descriptions saved to {args.text_output_path}")

    # Save structured attributes
    if args.json_output_path and structured_attributes:
        os.makedirs(os.path.dirname(args.json_output_path), exist_ok=True)
        with open(args.json_output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_attributes, f, indent=2, ensure_ascii=False)
        print(f"Structured attributes saved to {args.json_output_path}")
        
        # Statistics
        total_assets = len(consolidated_descriptions)
        structured_assets = len(structured_attributes)
        print(f"\nSummary Statistics:")
        print(f"- Total assets processed: {total_assets}")
        print(f"- Assets with consolidated descriptions: {total_assets}")
        print(f"- Assets with structured attributes: {structured_assets}")
        if total_assets > 0:
            success_rate = (structured_assets / total_assets) * 100
            print(f"- Local LLM comparative analysis success rate: {success_rate:.1f}%")


def main():
    """Main function to run the description generation and summarization pipeline."""
    args = setup_arg_parser()

    # Add output paths to args for the save_results function.
    # These are used by save_results but not defined in the arg parser.
    args.text_output_path = os.path.join(args.output_dir, 'object_descriptions.json')
    args.json_output_path = os.path.join(args.output_dir, 'object_attributes.json')

    # Initialize DAM model
    dam_model = initialize_dam_model(args)

    # Load local LLM
    if not args.local_llm_path:
        print("Error: --local_llm_path is required.")
        return
    local_llm_model, local_llm_tokenizer = load_local_llm(args.local_llm_path)
    
    # The pkl files contain relative paths from the project root.
    # The 'generate_dam_descriptions' function needs this root path to find the images.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Phase 1: Generate DAM descriptions
    print(f"\n{'='*60}")
    print("Phase 1: Generating descriptions with DAM")
    print(f"{'='*60}")
    category_descriptions = generate_dam_descriptions(dam_model, args, project_root)

    # Phase 2: Consolidate multiple descriptions for each asset
    print(f"\n{'='*60}")
    print("Phase 2: Consolidating descriptions for each asset")
    print(f"{'='*60}")
    consolidated_by_category = consolidate_asset_descriptions(
        local_llm_model, local_llm_tokenizer, category_descriptions, args
    )

    # Phase 3: Get category functions first
    print(f"\n{'='*60}")
    print("Phase 3: Getting category functions")
    print(f"{'='*60}")
    categories = list(consolidated_by_category.keys())
    category_functions = get_category_functions(
        local_llm_model, local_llm_tokenizer, categories, args
    )
    
    # Phase 4: Process categories with LLM for comparative analysis
    print(f"\n{'='*60}")
    print("Phase 4: Processing categories with LLM")
    print(f"{'='*60}")
    structured_attributes = process_categories_with_llm(
        local_llm_model, local_llm_tokenizer, consolidated_by_category, args
    )
    
    # Phase 5: Add category functions to structured attributes
    print(f"\n{'='*60}")
    print("Phase 5: Adding category functions to results")
    print(f"{'='*60}")
    for asset_id, attributes in structured_attributes.items():
        category = get_object_category(asset_id)
        if category in category_functions:
            attributes['function'] = category_functions[category]
            print(f"Added function for {asset_id} ({category}): {category_functions[category]}")

    # Save results
    save_results(consolidated_by_category, structured_attributes, args)
    print("\nProcessing complete.")

if __name__ == '__main__':
    main()