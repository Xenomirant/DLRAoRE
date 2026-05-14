python run_super_glue.py \
    --model_name_or_path roberta-large \
    --task_name copa \
    --enable_dykaf \
    --lora_all_modules \
    --max_length 512 \
    --seed=1234 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 30 \
    --low_rank_factors \
    --factors_rank 16 \
    --low_rank_proj psi \
    --weight_decay 0.001 \
    --tracking_backend comet \
    --run_name superglue-copa-dykaf


# python run_super_glue.py \
#     --model_name_or_path roberta-large \
#     --task_name copa \
#     --enable_dykaf \
#     --max_length 512 \
#     --seed=1234 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 3e-4 \
#     --num_train_epochs 30 \
#     --precondition_frequency 20 \
#     --power_iterations 3 \
#     --weight_decay 0.0001 \
#     --tracking_backend comet \
#     --run_name superglue-copa-dykaf-full