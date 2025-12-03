M^3-Verse Pipeline

Overview
- A modular pipeline to build the M^3-Verse dataset from AI2-THOR scenes:
  - Explore ProcTHOR indoor environments and record RGB, depth, trajectories, and room analyses
  - Extract object assets and generate descriptions with DAM-3B, then summarize with a local LLM
  - Produce single-state and multi-state QA pairs covering spatial, temporal, counting, and change detection abilities
  - Post-process with a VLM to filter, rewrite, and tag capabilities
- Due to randomness and LLM non-determinism, generated QA pairs can vary between runs

Installation
```
# Prerequisites
# - Python 3.10
# - CUDA-capable GPU recommended for DAM-3B and local LLMs

# Setup
conda create -n m3verse python=3.10
conda activate m3verse
pip install --upgrade pip
pip install -r requirements.txt

# For CUDA-specific PyTorch builds, follow the official PyTorch installation guide
```

Models and Data
- DAM-3B
  - You need to download `nvidia/DAM-3B` manually to `2_object_descriptions/DAM-3B` from HuggingFace if not present
- Local LLM
  - You need to download a local LLM (e.g., Qwen or LLaMA family) manually to `LLMs` from HuggingFace if not present
- ProcTHOR-10k scenes
  - JSONL files are expected under `0_ai2-thor_data/procthor-10k/{train,val,test}.jsonl`

Quick Start
```
# 1) Explore ProcTHOR scenes and save RGB, depth, trajectories, room analysis
python 1_explore/main.py

# 2) Filter extracted object asset patches (.pkl) from exploration outputs
python 2_object_descriptions/scripts/1_filter_assets.py
# Generate object descriptions with DAM-3B, then summarize attributes via a local LLM
#    --local_llm_path: path to your local Transformers causal LM (e.g., Qwen/LLaMA)
python 2_object_descriptions/scripts/2_main_DAM-3B_LLM_compare.py --local_llm_path /path/to/local_llm

# 3) Generate raw single-state and multi-state QA pairs from scene data
python 3_QA_generation/main.py

# 4) Sample QAs for downstream processing
python 4_postprocess/0_sample_QAs.py
# Filter and rewrite QAs using a VLM (DashScope Qwen models)
#     --api_key: your DashScope API key for VLM calls
python 4_postprocess/1_VLM_filter_rewrite_qa.py --api_key <YOUR_DASHSCOPE_API_KEY>
# Tag capability types for each QA (optional, uses a model API)
python 4_postprocess/2_get_capabilities.py
```

Outputs
- Generated dataset and assets are written under `M^3-Verse/` and `2_object_descriptions/`
- Final VLM-filtered QA file defaults to `M^3-Verse/QAs/M^3-Verse_VLM_filtered.jsonl`
