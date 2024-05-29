

MODEL_PATH='./checkpoints/Video-LLaVA-7B_DRIVE_BDDX_F_v1'
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.eval_custom_predsig --model-path ${MODEL_PATH} --input "./video_process/final_t13/conversation_bddx_eval.json"  --output "./video_process/result/final_b_h" 

