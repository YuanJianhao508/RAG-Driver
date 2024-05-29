DATA_ROOT="video_process"
CUDA_VISIBLE_DEVICES=0 deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./cache_dir/vicuna-7b-v1.5 \
    --version v1 \
    --train_data_path video_process/conversation_train.json \
    --eval_data_path video_process/conversation_eval.json \
    --video_folder ${DATA_ROOT} \
    --image_folder ${DATA_ROOT} \
    --X "Video" "Image" \
    --video_tower ./cache_dir/LanguageBind_Video_merge \
    --image_tower ./cache_dir/LanguageBind_Image \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/Video-LLaVA-Pretrain-7B_drive \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"


        