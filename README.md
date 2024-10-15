# reasoning-paths-optimization
Official repo of EMNLP 2024 paper Reasoning Paths Optimization: Learning to Reason and Explore From Diverse Paths

### Set up environment
```
conda create -n rpo python=3.10 -y
conda activate rpo
pip install -r requirements.txt
```

### Download models
```
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir models/mistral
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir models/Meta-Llama-3-8B
```

### Quick start

To run RPO end-to-end, here is an example using mistral for MMLU(STEM)
```
bash scripts/rpo_quick_start.sh python \
mmlu_stem_mistral_beta_03 \
models/mistral \
mmlu_stem \
0.3
```

Alternatively, to run reasoning generation:
```
python evaluation.py evaluate_sc \
"gsm8k" \
--demo_name "gsm8k" \
--path_model "models/mistral" \
--data_split "train"
```

To run reasoning exploration:
```
python branching.py generate_paths \
gsm8k \
gsm8k \
outputs/gsm8k_mistral.json \
--path_model models/mistral \
--data_split train \
--existing_preds_path outputs_sc/eval_data=gsm8k-demo=gsm8k-split=train-num_sample=10-path_model=models-mistral.jsonl
```

Or split the datasets into several parts to run on different GPUs in parallel:
```
python branching.py generate_paths \
gsm8k \
gsm8k \
outputs/gsm8k_mistral_part1.json \
--path_model models/mistral \
--data_split train \
--start_index START_INDEX --end_index END_INDEX \
--existing_preds_path outputs_sc/eval_data=gsm8k-demo=gsm8k-split=train-num_sample=10-path_model=models-mistral.jsonl

python branching.py merge_path_data outputs/gsm8k_mistral_part*.json --path_out outputs/gsm8k_mistral.json
```

To run reasoning optimization:
```
bash scripts/train_reason_paths.sh python \
outputs/gsm8k_mistral.json \
gsm8k_mistral_beta_03_rpo \
models/mistral \
gsm8k \
0.3
```
