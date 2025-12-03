import argparse
import random

def sample_qas(input_file, output_file, num_samples):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    if len(lines) < num_samples:
        print(f"Warning: Not enough QA pairs in the input file ({len(lines)}) to sample {num_samples}. Sampling all available.")
        sampled_lines = lines
    else:
        sampled_lines = random.sample(lines, num_samples)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in sampled_lines:
            f_out.write(line)

    print(f"Successfully sampled {len(sampled_lines)} QA pairs from {input_file} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a specified number of QA pairs from a JSONL file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of QA pairs to sample.")

    args = parser.parse_args()

    sample_qas(args.input_file, args.output_file, args.num_samples)