# !/bin/bash
set -x

ROOT_DIR="root/path/"

DATA_PATH="your/data/path/dataset/imagenet/train"

GLOBAL_BATCH_SIZE=128
EPOCHS=40
CODEBOOK_SIZE=65536
nproc_per_node=8
QB_type='QBridge-L/8'
CODEBOOK_EMBED_DIM=256

LEARNING_RATE=1e-4
nnodes=2
warmup_ep=4


# dir
QB_type_dir=${QB_type//\//-}
gpus=$((nproc_per_node * nnodes))
RESULTS_DIR="${ROOT_DIR}/gpus${gpus}_gbs${GLOBAL_BATCH_SIZE}_epochs${EPOCHS}_codesize${CODEBOOK_SIZE}_codebookdim${CODEBOOK_EMBED_DIM}_QTtype${QB_type_dir}_lr${LEARNING_RATE}_codebookl2f_dlr_wp${warmup_ep}_vq${vq_model}_const"

mkdir -p "${RESULTS_DIR}"

if [ -f "${RESULTS_DIR}/run.sh" ]; then
    cat "$0" > "${RESULTS_DIR}/run_1.sh"
else
    cat "$0" > "${RESULTS_DIR}/run.sh"
fi


torchrun --nnodes=${nnodes} --nproc_per_node=${nproc_per_node} --node_rank=0 \
--rdzv_endpoint="${MASTER_ADDR}:${MASTER_PROT}" --rdzv_backend=c10d \
/your/code/path/FVQ/tokenizer/tokenizer_image/vq_train_qbridge_lr.py \
--data-path ${DATA_PATH} \
--image-size 256 \
--results-dir ${RESULTS_DIR} \
--global-batch-size ${GLOBAL_BATCH_SIZE} \
--lr ${LEARNING_RATE} \
--epochs ${EPOCHS} \
--codebook-size ${CODEBOOK_SIZE} \
--QB_type ${QB_type} \
--codebook-embed-dim ${CODEBOOK_EMBED_DIM} \
--warmup_ep ${warmup_ep} \
--sche_type 'lin' \
--is_uncondition \


