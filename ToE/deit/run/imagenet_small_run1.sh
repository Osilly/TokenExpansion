result_dir=[your_result_path]
dataset_dir=[your_dataset_path]
device=0,1,2,3
master_port=6666
CUDA_VISIBLE_DEVICES=$device torchrun --nproc_per_node=4 --master_port=$master_port main.py \
--patch-size 16 \
--model deit_small_patch16_224 \
--batch-size 256 \
--data-path $dataset_dir \
--output_dir $result_dir \
--num_workers 8 \
--seed 3407 \
--expansion-step 0 100 200 \
--keep-rate 0.5 0.75 1.0 \
--initialization-keep-rate 0.25 \
--expansion-multiple-stage 2 \