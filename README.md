# TokenExpansion

The official pytorch implementation of TokenExpansion (ToE).

## Requirements

```
torch>=1.12.0
torchvision>=0.13.0
timm==0.9.2
```

## How to apply ToE

We provide the main code `token_select.py`. It can be seamlessly integrated into the training of ViTs.

You can find the examples of applying ToE to popular ViTs (e.g., DeiT in `ToE/deit `and LV-ViT in `ToE/lvvit `) and existing efficient training frameworks (e.g., EfficientTrain in `ToE/EfficientTrain `).

It is simple to change the existing codes, and the codes for the changes we make to the original model codes are wrapped in two `# ---------------------#`. 

For example (`ToE/deit/main.py `):

```
 model = create_model(
     args.model,
     pretrained=False,
     num_classes=args.nb_classes,
     drop_rate=args.drop,
     drop_path_rate=args.drop_path,
     drop_block_rate=None,
     img_size=args.input_size,
 )
 # ---------------------#
 model.token_select = TokenSelect(
     expansion_step=args.expansion_step,
     keep_rate=args.keep_rate,
     initialization_keep_rate=args.initialization_keep_rate,
     expansion_multiple_stage=args.expansion_multiple_stage,
     distance=args.distance,
 )
 # ---------------------#

```

## Training

The detailed training scripts are presents in the specific code paths (e.g., DeiT in in `ToE/deit/run/ `). You should prepare the environments and datasets.

We take a few simple training examples:

### DeiT-small

We train the DeiT-small on four GPUs, the ImageNet-1K dataset is  required.

```
cd ToE/deit
bash run/imagenet_small_run1.sh
```

**imagenet_small_run1.sh**:

```
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
```

### LV-ViT-S

We train the LV-ViT-S on four GPUs, the ImageNet-1K dataset and the label data (see [original LV-ViT repo](https://github.com/zihangJiang/TokenLabeling)) are required.

```
cd ToE/lvvit
bash run/imagenet_small_run1.sh
```

**imagenet_small_run1.sh**:

```
#!/bin/bash
result_dir=[your_result_path]
dataset_dir=[your_dataset_path]
label_dir=[your_token_labeling_dataset_path]
device=0,1,2,3
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
```

### EfficientTrain (DeiT-small)

We train the EfficientTrain (DeiT-small) on eight GPUs, the ImageNet-1K dataset is required.

```
cd ToE/EfficientTrain
bash run/imagenet_small_run1.sh
```

**imagenet_small_run1.sh**:

```
result_dir=[your_result_path]
dataset_dir=[your_dataset_path]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ET_training.py \
--data_path $dataset_dir \
--output_dir $result_dir \
--model deit_small_patch16_224 \
--final_bs 256 --epochs 300 \
--num_gpus 8 --num_workers 8 \
```

The codes of speedup factors of ToE are presents in `ToE/EfficientTrain/ET_training.py `. For example:

```
"deit_tiny_patch16_224": " --use_amp true --clip_grad 5.0 \
--expansion-step 0 100 200 --keep-rate 0.6 0.8 1.0 \
--initialization-keep-rate 0.3 --expansion-multiple-stage 2 "
```

