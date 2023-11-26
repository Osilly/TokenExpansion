#!/bin/bash
result_dir=[your_result_path]
dataset_dir=[your_dataset_path]
label_dir=[your_token_labeling_dataset_path]
device=4,5,6,7
master_port=6666
shift
CUDA_VISIBLE_DEVICES=$device python3 -m torch.distributed.launch --nproc_per_node=4 main.py "$@" $dataset_dir \
--output $result_dir \
--model lvvit_s \
-b 256 \
--img-size 224 \
--drop-path 0.1 \
--token-label \
--token-label-data $label_dir \
--token-label-size 14 \
--model-ema \
--apex-amp \
--expansion-step 0 100 200 \
--keep-rate 0.4 0.7 1.0 \
--initialization-keep-rate 0.2 \
--expansion-multiple-stage 2 \