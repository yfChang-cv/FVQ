CUDA_VISIBLE_DEVICES="2,3" torchrun --nnodes 1 \
    --nproc_per_node 2 \
    --node_rank 0 \
    --master-addr localhost \
    --master-port 6235 \
    /FVQ/tokenizer/tokenizer_image/convert_fullvq2vq.py \
    --dataset-path ./data/dataset/imagenet/val \
    --image-size 256 \
    --crop-raw-path./data/evaluator_gen/imagenet_val_5w_256x256 \
    --sample-dir ./samples/ \
    --vq-ckpt fvq_ckpt/0400000.pt
