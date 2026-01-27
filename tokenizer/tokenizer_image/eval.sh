#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

VQ_CKPT_LIST=(
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge/log/c2i/simvq/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize16384_codebookdim256_QTtypeQbridge-lin-1_lr1e-4_codebookl2f_dlr_wp4/0385000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge/log/c2i/simvq/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize16384_codebookdim256_QTtypeQbridge-lin-1_lr1e-4_codebookl2f_dlr_wp4/0390000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge/log/c2i/simvq/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize16384_codebookdim256_QTtypeQbridge-lin-1_lr1e-4_codebookl2f_dlr_wp4/0395000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge/log/c2i/simvq/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize16384_codebookdim256_QTtypeQbridge-lin-1_lr1e-4_codebookl2f_dlr_wp4/0400000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge/log/c2i/simvq/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize16384_codebookdim256_QTtypeQbridge-lin-1_lr1e-4_codebookl2f_dlr_wp0_1/0385000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge/log/c2i/simvq/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize16384_codebookdim256_QTtypeQbridge-lin-1_lr1e-4_codebookl2f_dlr_wp0_1/0390000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge/log/c2i/simvq/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize16384_codebookdim256_QTtypeQbridge-lin-1_lr1e-4_codebookl2f_dlr_wp0_1/0395000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge/log/c2i/simvq/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize16384_codebookdim256_QTtypeQbridge-lin-1_lr1e-4_codebookl2f_dlr_wp0_1/0400000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge-1/log/c2i/uqb/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize262144_codebookdim256_QTtypeQbridge-lin-5_lr1e-4_codebookl2f_dlr_wp0_vq_const/0340000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge-1/log/c2i/uqb/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize262144_codebookdim256_QTtypeQbridge-lin-5_lr1e-4_codebookl2f_dlr_wp0_vq_const/0350000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge-1/log/c2i/uqb/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize262144_codebookdim256_QTtypeQbridge-lin-5_lr1e-4_codebookl2f_dlr_wp0_vq_const/0360000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge-1/log/c2i/uqb/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize262144_codebookdim256_QTtypeQbridge-lin-5_lr1e-4_codebookl2f_dlr_wp0_vq_const/0370000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge-1/log/c2i/uqb/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize262144_codebookdim256_QTtypeQbridge-lin-5_lr1e-4_codebookl2f_dlr_wp0_vq_const/0380000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge-1/log/c2i/uqb/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize262144_codebookdim256_QTtypeQbridge-lin-5_lr1e-4_codebookl2f_dlr_wp0_vq_const/0390000.pt
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge-1/log/c2i/uqb/fulltraining_llmgen_gpus16_gbs128_epochs40_codesize262144_codebookdim256_QTtypeQbridge-lin-5_lr1e-4_codebookl2f_dlr_wp0_vq_const/0400000.pt
)
VQ_CKPT_LIST=($(printf "%s\n" "${VQ_CKPT_LIST[@]}" | sort))

for ckpt in "${VQ_CKPT_LIST[@]}"; do
  echo "$ckpt"
done

for VQ_CKPT in "${VQ_CKPT_LIST[@]}"
do
  torchrun --nnodes 1 --nproc_per_node 2 --node_rank 0 \
    --master-addr localhost --master-port 6234 \
    /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/code/LlamaGen/tokenizer/tokenizer_image/eval_fid.py \
    --dataset-path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/dataset/imagenet/val \
    --image-size 256 \
    --image-size-eval 256 \
    --crop-raw-path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/evaluator_gen/imagenet_val_5w_256x256 \
    --sample-dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/Qbridge-1/samples/debug/256/ \
    --vq-ckpt "$VQ_CKPT"
done
