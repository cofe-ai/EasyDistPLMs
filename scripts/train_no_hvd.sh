#!/bin/bash
WANDB_KEY=$WANDB_KEY # paste your key here if wandb is enabled
TIMESTAMP=$(date +'%Y-%m-%d-%H-%M')
HOSTFILE="./hostfile"
LOGDIR="logs"
TASK_NAME="imdb-cls"
mkdir -p $LOGDIR

NCCL_ENVS="NCCL_DEBUG=info"

OPTIONS="
--amp \
--task-name $TASK_NAME \
--timestamp $TIMESTAMP \
--label-path config/labelspace_imdb \
--pretrained ../pretrained_models/bert-base-uncased \
--pooler-type cls \
--batch-size 256 \
--save-cache \
--use-cache \
--train-path data/imdb/train.csv \
--valid-path data/imdb/test.csv \
--num-data-workers 4 \
--ckpt-path ckpts \
--max-epochs 3 \
--patience 3 \
--lr 4e-5 \
--no-scale-lr \
--dropout 0.1 \
--shuffle \
--max-seq-len 128 \
--log-interval 5
"

RUN_CMD="${NCCL_ENVS} \
python -u run.py \
$OPTIONS \
2>&1 | tee ${LOGDIR}/${TASK_NAME}-${TIMESTAMP}.log"

echo ${RUN_CMD}
eval ${RUN_CMD}

set -x
