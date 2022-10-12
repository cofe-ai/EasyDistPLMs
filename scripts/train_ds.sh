#!/bin/bash

WANDB_KEY=$WANDB_KEY # paste your key here if wandb is enabled
TIMESTAMP=$(date +'%Y-%m-%d-%H-%M')
HOSTFILE="./hostfile"
LOGDIR="logs"
NUM_NODES=2
NUM_GPUS_PER_NODE=2
TASK_NAME="imdb-cls"
mkdir -p $LOGDIR

OPTIONS="
--deepspeed \
--deepspeed_config ds_config.json \
--task-name $TASK_NAME \
--timestamp $TIMESTAMP \
--label-path config/labelspace_imdb \
--pretrained ../pretrained_models/bert-base-uncased \
--pooler-type cls \
--save-cache \
--use-cache \
--train-path data/imdb/train.csv \
--valid-path data/imdb/test.csv \
--num-data-workers 4 \
--ckpt-path ckpts \
--max-epochs 3 \
--patience 3 \
--warmup 0.1 \
--dropout 0.1 \
--shuffle \
--max-seq-len 128 \
--log-interval 5
"

RUN_CMD="${NCCL_ENVS} \
deepspeed --hostfile $HOSTFILE --num_nodes $NUM_NODES --num_gpus $NUM_GPUS_PER_NODE \
run.py \
$OPTIONS \
2>&1 | tee ${LOGDIR}/${TASK_NAME}-${TIMESTAMP}.log"

echo ${RUN_CMD}
eval ${RUN_CMD}

set -x
