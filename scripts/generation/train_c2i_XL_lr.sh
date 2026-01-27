# !/bin/bash
set -x

torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
--rdzv_endpoint="${MASTER_ADDR}:${MASTER_PROT}" --rdzv_backend=c10d \
/your/code/LlamaGen/autoregressive/train/train_c2i_lr.py \
--code-path ./imagenet_codes_256_qbridge/imagenet_code_c2i_flip_ten_crop \
--image-size 256 \
--gpt-model GPT-XL \
--results-dir ./results/outdir_llmgen_qbridge_GPT-XL_lr_1 \
--global-batch-size 1024 \
--no-compile \
--epochs 350 \