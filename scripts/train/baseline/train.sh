#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

ROOT=$(pwd)
SRC_PTH=$ROOT/src
MODALITIES='video'
CONFORMER_PATH=$SRC_PTH/pretrained_models/conformer_encoder/pretrained_lrs3/vsr_trlrs3_base.pth
LLM_PATH=$SRC_PTH/pretrained_models/llm/Meta-Llama-3-8B
NGPUS=8

OUT_PATH=$SRC_PTH/exp/baseline/conformer_llm

export TOKENIZERS_PARALLELISM=false

PYTHONPATH=$ROOT/fairseq \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-hydra-train --config-dir ${SRC_PTH}/conf/baseline --config-name conformer_llm.yaml \
    task.data=$ROOT/voxlrs-sa/baseline/ \
    task.label_dir=$ROOT/voxlrs-sa/baseline/ \
    task.modalities=[${MODALITIES}] \
    hydra.run.dir=${OUT_PATH} \
    common.user_dir=${SRC_PTH} \
    task.llm_ckpt_path=${LLM_PATH} \
    model.llm_ckpt_path=${LLM_PATH} \
    model.conformer_ckpt_path=${CONFORMER_PATH} \
    optimization.update_freq=[8] \
    optimization.lr=[5e-5] \
    optimization.max_update=30000 \
    lr_scheduler._name=cosine\
    lr_scheduler.warmup_updates=500 \
    distributed_training.distributed_world_size=${NGPUS} \
    distributed_training.nprocs_per_node=${NGPUS}
    