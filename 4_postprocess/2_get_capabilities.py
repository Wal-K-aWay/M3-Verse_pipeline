import os
import json
import argparse
from openai import OpenAI
from dashscope import Generation
from http import HTTPStatus

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Configuration
INPUT_FILE = os.path.join(parent_dir, 'M^3-Verse/QAs/M^3-Verse_VLM_filtered.jsonl')
OUTPUT_FILE = os.path.join(parent_dir, 'M^3-Verse/QAs/M^3-Verse_VLM_filtered_capabilities.jsonl')

# Shared available capabilities
AVAILABLE_CAPABILITIES = [
    {"name": "temporal understanding", "description": "Involves understanding and reasoning about temporal sequences, durations, and changes across observations. This includes determining object appearance order (first/last seen), observation duration comparisons, temporal state changes (before/after comparisons), and understanding event sequences in multi-state scenarios."},
    {"name": "spatial understanding", "description": "Involves comprehending spatial relationships, positions, and arrangements in the environment. This includes object location determination (on/under surfaces), distance estimation between objects, spatial proximity comparisons, room layout understanding, and object containment relationships."},
    {"name": "attribute recognition", "description": "Involves identifying and comparing object properties and characteristics. This includes color identification, shape recognition, size comparisons, material properties, object states (open/closed, on/off, filled/empty), visual appearance attributes across different observations, and scenarios where options themselves describe attributes."},
    {"name": "reasoning", "description": "Involves logical deduction, quantitative analysis, and complex inference tasks. This includes counting objects, comparing quantities, performing multi-step logical reasoning, identifying patterns, making comparisons across different observations, synthesizing multiple information sources to draw conclusions, and situations requiring comparisons between different options."}
]

def classify_question_deepseek(question, options, api_key):
    if not api_key:
        raise ValueError("DeepSeek API_KEY not provided.")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    cap_names = [cap['name'] for cap in AVAILABLE_CAPABILITIES]

    prompt = f"""You are a capability classification assistant. Given a question and some options, you need to determine which capabilities of the model this question tests.
Please select one or more of the most relevant capabilities from the following list: {', '.join(cap_names)}.
Here are the descriptions for each capability to help you make a better judgment:
"""
    for cap in AVAILABLE_CAPABILITIES:
        prompt += f"- {cap['name']}: {cap['description']}\n"

    question_prompt = f"""
Question: {question}
Options: {", ".join(options)}

Please return the result in JSON format, for example: {{\"capabilities\": [\"capability1\", \"capability2\"]}}."""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question_prompt},
        ],
        stream=False
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print(f"DeepSeek JSON decode error: {response.choices[0].message.content}")
        return {"capabilities": []}

def classify_question_qwen(question, options, api_key):
    if not api_key:
        raise ValueError("Qwen API_KEY not provided.")

    cap_names = [cap['name'] for cap in AVAILABLE_CAPABILITIES]
    prompt = f"""You are a capability classification assistant. Given a question and some options, you need to determine which capabilities of the model this question tests.
Please select one or more of the most relevant capabilities from the following list: {', '.join(cap_names)}.
Here are the descriptions for each capability to help you make a better judgment:
"""
    for cap in AVAILABLE_CAPABILITIES:
        prompt += f"- {cap['name']}: {cap['description']}\n"
    prompt += f"""
Question: {question}
Options: {", ".join(options)}

Please return the result in JSON format, for example: {{\"capabilities\": [\"capability1\", \"capability2\"]}}."""

    response = Generation.call(
        model='qwen-plus',
        prompt=prompt,
        api_key=api_key,
    )

    if response.status_code == HTTPStatus.OK:
        try:
            return json.loads(response.output.text)
        except json.JSONDecodeError:
            print(f"Qwen JSON decode error: {response.output.text}")
            return {"capabilities": []}
    else:
        print(f"Qwen request failed: {response.status_code}, {response.message}")
        return {"capabilities": []}

def merge_capabilities(deepseek_results, qwen_results):
    merged_caps = set()

    # Flatten the list of lists into a single list of capabilities for each model
    deepseek_all_caps = [cap for sublist in deepseek_results for cap in sublist]
    qwen_all_caps = [cap for sublist in qwen_results for cap in sublist]

    deepseek_caps_set = set(deepseek_all_caps)
    qwen_caps_set = set(qwen_all_caps)

    # Common capabilities
    common_caps = deepseek_caps_set.intersection(qwen_caps_set)
    merged_caps.update(common_caps)

    # Specific rules
    if "temporal understanding" in deepseek_caps_set and "temporal understanding" not in qwen_caps_set:
        merged_caps.add("temporal understanding")
    elif "temporal understanding" in qwen_caps_set and "temporal understanding" not in deepseek_caps_set:
        # If only Qwen has it, and Deepseek doesn't, we still take Deepseek's (which is to not have it)
        pass
    elif "temporal understanding" in deepseek_caps_set and "temporal understanding" in qwen_caps_set:
        merged_caps.add("temporal understanding") # Both have it, keep it

    if "spatial understanding" in qwen_caps_set and "spatial understanding" not in deepseek_caps_set:
        merged_caps.add("spatial understanding")
    elif "spatial understanding" in deepseek_caps_set and "spatial understanding" not in qwen_caps_set:
        # If only Deepseek has it, and Qwen doesn't, we still take Qwen's (which is to not have it)
        pass
    elif "spatial understanding" in deepseek_caps_set and "spatial understanding" in qwen_caps_set:
        merged_caps.add("spatial understanding") # Both have it, keep it

    if "reasoning" in qwen_caps_set and "reasoning" not in deepseek_caps_set:
        merged_caps.add("reasoning")
    elif "reasoning" in deepseek_caps_set and "reasoning" not in qwen_caps_set:
        pass
    elif "reasoning" in deepseek_caps_set and "reasoning" in qwen_caps_set:
        merged_caps.add("reasoning")

    if "attribute recognition" in qwen_caps_set and "attribute recognition" not in deepseek_caps_set:
        merged_caps.add("attribute recognition")
    elif "attribute recognition" in deepseek_caps_set and "attribute recognition" not in qwen_caps_set:
        pass
    elif "attribute recognition" in deepseek_caps_set and "attribute recognition" in qwen_caps_set:
        merged_caps.add("attribute recognition")
            
    return sorted(list(merged_caps))

def main():
    parser = argparse.ArgumentParser(description="Classify question capabilities with DeepSeek (3 calls) and Qwen (3 calls) models.")
    parser.add_argument('--deepseek_api_key', type=str, default=None, required=True,
                        help='DeepSeek API Key.')
    parser.add_argument('--qwen_api_key', type=str, default=None, required=True, 
                        help='DashScope API Key for Qwen.')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index for processing questions (inclusive).')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='Ending index for processing questions (exclusive). Use -1 to process all questions from start_idx.')
    args = parser.parse_args()

    print(f"Available capabilities: {[cap['name'] for cap in AVAILABLE_CAPABILITIES]}")

    output_dir = os.path.dirname(OUTPUT_FILE)
    os.makedirs(output_dir, exist_ok=True)

    with open(INPUT_FILE, 'r') as infile, \
         open(OUTPUT_FILE, 'a') as outfile:
        for i, line in enumerate(infile):
            if args.end_idx != -1 and i >= args.end_idx:
                break
            if i < args.start_idx:
                continue

            item = json.loads(line)
            question_id = item.get('question_id')
            question = item.get('question')
            options = item.get('options', [])

            if question_id and question:
                print(f"Processing question_id: {question_id} (index: {i})")
                # Collect 3 DeepSeek results
                deepseek_results = []
                for call_num in range(3):
                    print(f"DeepSeek call {call_num+1} for question {question_id}")
                    res = classify_question_deepseek(question, options, args.deepseek_api_key)
                    deepseek_results.append(res.get('capabilities', []))
                # Collect 3 Qwen results
                qwen_results = []
                for call_num in range(3):
                    print(f"Qwen call {call_num+1} for question {question_id}")
                    res = classify_question_qwen(question, options, args.qwen_api_key)
                    qwen_results.append(res.get('capabilities', []))
                # Save combined results
                merged_capabilities = merge_capabilities(deepseek_results, qwen_results)
                # item["deepseek_calls"] = deepseek_results
                # item["qwen_calls"] = qwen_results
                item["capabilities"] = merged_capabilities
                combined_item = item
                outfile.write(json.dumps(combined_item) + '\n')
                outfile.flush()
            else:
                print(f"Skipping item due to missing question_id or question: {item}")

    print(f"Combined classification complete. Results saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()