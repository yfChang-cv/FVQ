export CUDA_VISIBLE_DEVICES=2,3

torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    --node_rank 0 \
    --master-addr localhost \
    --master-port 6234 \
    /code/FVQ/tokenizer/tokenizer_image/eval_fid_vqgan.py \
    --dataset-path ./data/dataset/imagenet/val \
    --image-size 256 \
    --image-size-eval 256 \
    --crop-raw-path ./data/evaluator_gen/imagenet_val_5w_256x256 \
    --sample-dir your/path/sample-dir \
    --vq-ckpt ./fullvq2vqgan_ds16_res2_16384_256.pt
