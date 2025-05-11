# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:29200 ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--rotate \
--optimized_rotation_path $5 \
--guided \
--save_qmodel_path "checkpoint/checkpoint_quantized_70b_guided_wikitext2_g1/$2_$3_$4/ptq_model.pt" \
--input_tokens_path "../cache/tokens/(Llama-2-70b-hf)-wikitext2_s128_blk2048.pt" \
--num_groups 1 \
--saliency_path "../cache/saliency/(Llama-2-70b-hf)-wikitext2_s128_blk2048_g1"
