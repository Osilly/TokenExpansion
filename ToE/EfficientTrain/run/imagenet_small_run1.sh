result_dir=[your_result_path]
dataset_dir=[your_dataset_path]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ET_training.py \
--data_path $dataset_dir \
--output_dir $result_dir \
--model deit_small_patch16_224 \
--final_bs 256 --epochs 300 \
--num_gpus 8 --num_workers 8 \