#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# set paths

ROOT=$(pwd)
SRC_PTH=$ROOT/src
LLM_PATH=$SRC_PTH/pretrained_models/llm/Meta-Llama-3-8B

for i in $(seq -f "%05g" 1 20)
do
    MODEL_PATH=$SRC_PTH/pretrained_models/adapted_model/vision_language/voxlrs-$i/checkpoints/checkpoint_best.pt
    OUT_PATH=$SRC_PTH/results/adaptation/vision_language/voxlrs-$i   # output path to save

    # start decoding
    PYTHONPATH=$ROOT/fairseq \
    CUDA_VISIBLE_DEVICES=0 python -B ${SRC_PTH}/eval.py \
        --config-dir ${SRC_PTH}/conf \
        --config-name s2s_decode \
            common.user_dir=${SRC_PTH} \
            dataset.gen_subset=test \
            generation.beam=1 \
            override.data=$ROOT/voxlrs-sa/adaptation/45min/voxlrs-$i \
            override.label_dir=$ROOT/voxlrs-sa/adaptation/45min/voxlrs-$i \
            generation.lenpen=0 \
            override.llm_ckpt_path=${LLM_PATH} \
            override.modalities=['video'] \
            common_eval.path=${MODEL_PATH} \
            common_eval.results_path=${OUT_PATH}
            
done