#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


ROOT=$(pwd)
SRC_PTH=$ROOT/src
MODALITIES='video'
MODEL_PATH=$SRC_PTH/pretrained_models/conformer_encoder/pretrained_w_llm/checkpoint_best.pt  # path to trained model
LLM_PATH=$SRC_PTH/pretrained_models/llm/Meta-Llama-3-8B

OUT_PATH=$SRC_PTH/results/baseline

PYTHONPATH=$ROOT/fairseq \
CUDA_VISIBLE_DEVICES=0 python -B $SRC_PTH/eval.py --config-dir ${SRC_PTH}/conf --config-name s2s_decode \
    dataset.gen_subset=test \
    common.user_dir=${SRC_PTH} \
    generation.beam=1 \
    generation.lenpen=0 \
    override.llm_ckpt_path=${LLM_PATH} \
    override.modalities=['video'] \
    common_eval.path=${MODEL_PATH} \
    common_eval.results_path=${OUT_PATH} \
    override.label_dir=$ROOT/voxlrs-sa/baseline/ \
    override.data=$ROOT/voxlrs-sa/baseline/


