result_dir=[your_result_path]
dataset_dir=[your_dataset_path]
pretrain_path=[your_pretrain_model_path]
device=0,1,2,3,4,5,6,7
master_port=6666
CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=8 --master_port=$master_port --use_env main.py \
    --coco_path $dataset_dir \
    --batch_size 2 \
    --lr 5e-5 \
    --epochs 300 \
    --backbone_name tiny \
    --pre_trained $pretrain_path \
    --eval_size 512 \
    --init_pe_size 800 1333 \
    --output_dir $result_dir \
    --expansion-step 5 100 200 \
    --keep-rate 0.5 0.75 1.0 \
    --initialization-keep-rate 0.25 \
    --expansion-multiple-stage 2 \