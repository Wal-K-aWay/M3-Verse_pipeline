# M^3-Verse Pipeline

## Overview
- Build the M^3-Verse dataset from AI2-THOR ProcTHOR scenes via a modular, end-to-end pipeline:
  - Explore indoor environments and record RGB, depth, agent trajectories, and room analyses
  - Extract object assets and describe them with DAM-3B, then summarize attributes using a local LLM
  - Generate single-state and multi-state QA pairs that cover spatial, temporal, counting, and change-detection abilities
  - Post-process QAs with a VLM to filter, rewrite, and tag capabilities
- Due to randomness and LLM non-determinism, generated QAs may vary between runs

## Installation
```
conda create -n m3verse python=3.10
conda activate m3verse
pip install -r requirements.txt
```

## Data and Models
- DAM-3B: download `nvidia/DAM-3B` to `2_object_descriptions/DAM-3B`
- Local LLM: download a causal LLM (e.g., Qwen3) and place it under `LLMs/YourModel`

Directory Layout
```
M^3-Verse_pipeline/
├── 0_ai2-thor_data/
│   ├── 0_download_procthor-10k.py
│   ├── procthor-10k/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── Other files and directories
├── 1_explore/
│   ├── main.py
│   └── ...
├── 2_object_descriptions/
│   ├── assets/
│   ├── descriptions/
│   ├── scripts/
│   │   ├── 1_filter_assets.py
│   │   └── 2_main_DAM-3B_LLM_compare.py
│   └── DAM-3B/
├── 3_QA_generation/
│   ├── main.py
│   └── ...
├── 4_postprocess/
│   ├── 0_sample_QAs.py
│   ├── 1_VLM_filter_rewrite_qa.py
│   └── 2_get_capabilities.py
├── LLMs/
│   └── Qwen3-4B-Instruct/
├── M^3-Verse/
│   └── data/
│   └── general_data/
│   └── QAs/
├── scripts/
└── README.md
```

## Quick Start
```bash
# 0) Download ProcTHOR-10k dataset
python 0_ai2-thor_data/0_download_procthor-10k.py

# 1) Explore scenes and save RGB, depth, trajectories, and room analysis
python 1_explore/main.py

# 2) Process object assets
python 2_object_descriptions/scripts/1_filter_assets.py # Generate descriptions with DAM-3B
python 2_object_descriptions/scripts/2_main_DAM-3B_LLM_compare.py --local_llm_path /path/to/local_llm # Summarize via a local LLM

# 3) Generate single-state and multi-state QA pairs
python 3_QA_generation/main.py

# 4) Post-process QAs
python 4_postprocess/0_sample_QAs.py --input_file M^3-Verse/QAs/M^3-Verse.jsonl --output_file M^3-Verse/QAs/M^3-Verse_sample.jsonl --num_samples 1000 # Optional, randomly select a subset of QAs
python 4_postprocess/1_VLM_filter_rewrite_qa.py --api_key <YOUR_DASHSCOPE_API_KEY> # Filter and rewrite QAs using a VLM (DashScope Qwen models)
python 4_postprocess/2_get_capabilities.py # Tag capability types with Deepseek and Qwen API
```

## Outputs
- Intermediate assets under `2_object_descriptions/assets`, object descriptions under `2_object_descriptions/descriptions`, and exploration outputs under `M^3-Verse_pipeline/M^3-Verse/data`
- Consolidated data under `M^3-Verse/`
- Final VLM-filtered QA file typically saved to `M^3-Verse/QAs`

## Notes and Tips
- Determinism: results can vary due to random sampling and LLM/VLM variability
- Data integrity: verify `procthor-10k` JSONL files exist under the expected path before running generation steps