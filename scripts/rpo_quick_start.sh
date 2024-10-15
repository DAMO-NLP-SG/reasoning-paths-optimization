set -e

PYTHON=${1:-"python"}
EXPERIMENT_NAME=$2
BASE_MODEL=$3
DATA_NAME=${4:-"gsm8k"}
ORPO_BETA=${5:-0.1}

$PYTHON evaluation.py evaluate_sc \
--data_name $DATA_NAME \
--demo_name $DATA_NAME \
--path_model $BASE_MODEL \
--data_split "train"

$PYTHON branching.py generate_paths \
--data_name $DATA_NAME \
--demo_name $DATA_NAME \
--path_out outputs/$EXPERIMENT_NAME.json \
--path_model $BASE_MODEL \
--data_split train

$PYTHON branching.py save_tuning_data outputs/$EXPERIMENT_NAME.json data/paths/$EXPERIMENT_NAME.json

$PYTHON run_orpo.py \
--stage orpo \
--do_train \
--output_dir outputs_paths/$EXPERIMENT_NAME \
--model_name_or_path $BASE_MODEL \
--dataset $EXPERIMENT_NAME \
--dataset_dir "" \
--template default \
--finetuning_type lora \
--lora_target q_proj,v_proj \
--overwrite_cache \
--overwrite_output_dir \
--cutoff_len 1024 \
--preprocessing_num_workers 16 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--warmup_steps 100 \
--logging_steps 10 \
--learning_rate 5e-5 \
--num_train_epochs 3.0 \
--orpo_beta $ORPO_BETA \
--fp16

$PYTHON merging.py \
--model_name_or_path $BASE_MODEL \
--adapter_name_or_path outputs_paths/$EXPERIMENT_NAME \
--template default \
--finetuning_type lora \
--export_dir outputs_paths/$EXPERIMENT_NAME/final \
--export_size 2 \
--export_legacy_format False

$PYTHON evaluation.py run_eval_many outputs_paths/$EXPERIMENT_NAME/final --data_name $DATA_NAME --demo_name $DATA_NAME
