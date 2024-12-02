ulimit -c unlimited
#module load cuda-11.1.1

export WANDB_WATCH=gradients
export PYTHONPATH=.

#MODEL_NAME='t5-base'
#MODEL_NAME='google/flan-t5-base'
#MODEL_NAME='google/long-t5-local-base'
#MODEL_NAME='facebook/opt-350m'
MODEL_NAME='facebook/opt-125m'
TASK='section_all'
CONTEXT='section_all'
DESCRIPTION=${MODEL_NAME}-${TASK}-${CONTEXT}

CUDA_VISIBLE_DEVICES=4,6 python language_modelling/run_generation.py \
    --dataset wikiweb2m \
    --neighbor_mode raw \
    --model_name_or_path ${MODEL_NAME} \
    --task ${TASK} \
    --context ${CONTEXT} \
    --peft_type prefix \
    --position_type none \
    --max_input_length 512 \
    --max_output_length 128 \
    --epochs 50 \
    --steps_per_epoch 10000 \
    --val_steps_per_epoch 400 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --per_device_val_batch_size 2 \
    --dataloader_num_workers 8 \
    --grad_accumulation_steps 16 \
    --fp16 \
    --wandb_project MMHG \
    --wandb_run ${DESCRIPTION}
