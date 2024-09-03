#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

ROOT=$(pwd)
SRC_PTH=$ROOT/src
MODALITIES='video'
CONFORMER_PATH=$SRC_PTH/pretrained_models/conformer_encoder/pretrained_w_llm/checkpoint_best.pt
LLM_PATH=$SRC_PTH/pretrained_models/llm/Meta-Llama-3-8B
NGPUS=1

export TOKENIZERS_PARALLELISM=false
for i in $(seq -f "%05g" 1 1)
do

OUT_PATH=$SRC_PTH/exp/adaptation/vision_language/voxlrs-$i

PYTHONPATH=$ROOT/fairseq \
CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train --config-dir ${SRC_PTH}/conf/adaptation --config-name vision_language.yaml \
    task.data=$ROOT/voxlrs-sa/adaptation/45min/voxlrs-$i \
    task.label_dir=$ROOT/voxlrs-sa/adaptation/45min/voxlrs-$i \
    task.modalities=[${MODALITIES}] \
    hydra.run.dir=${OUT_PATH} \
    common.user_dir=${SRC_PTH} \
    task.llm_ckpt_path=${LLM_PATH} \
    model.llm_ckpt_path=${LLM_PATH} \
    model.conformer_ckpt_path=${CONFORMER_PATH} \
    model._name=vision_language_adaptation \
    model.target_speaker_padding_prompt_pth=$SRC_PTH/exp/adaptation/vision/voxlrs-$i/checkpoints/checkpoint_best.pt \
    model.prompt_length=10 \
    model.speaker_id=$i \
    optimization.update_freq=[64] \
    optimization.lr=[5e-3] \
    optimization.max_update=100 \
    lr_scheduler._name=cosine \
    lr_scheduler.warmup_updates=0 \
    lr_scheduler.min_lr=1e-4 \
    lr_scheduler.lr_period_updates=5000 \
    distributed_training.distributed_world_size=${NGPUS} \
    distributed_training.nprocs_per_node=${NGPUS}

done
