import json
import argparse

def sort_jsonl_by_question_id(file_path):
    """
    Reads a JSONL file, sorts its contents by "question_id" in ascending order,
    and writes the sorted data back to the same file.   
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file {file_path}: {e}")
        return

    # Sort the data by "question_id"
    # Assuming "question_id" is present in each JSON object and is comparable
    try:
        data.sort(key=lambda x: x["question_id"])
    except KeyError:
        print("Error: 'question_id' not found in all JSON objects.")
        return

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Successfully sorted {file_path} by 'question_id'.")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort JSONL by 'question_id' ascending")
    parser.add_argument("input", help="Path to JSONL file to sort")
    args = parser.parse_args()
    sort_jsonl_by_question_id(args.input)
