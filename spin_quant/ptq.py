# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger

import os
import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq

log: Logger = utils.get_logger("spinquant")


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        device_map="auto",
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    # model.cuda()

    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length
    if local_rank == 0:
        # log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False

    if os.path.exists(os.path.join("cache_input_tokens", f"wikitext2_nsamples{ptq_args.nsamples}_seqlen2048_test.pt")):
        testloader = torch.load(os.path.join("cache_input_tokens", f"wikitext2_nsamples{ptq_args.nsamples}_seqlen2048_test.pt"))
    else:
        testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )
        os.makedirs("cache_input_tokens", exist_ok=True)
        torch.save(testloader, os.path.join("cache_input_tokens", f"wikitext2_nsamples{ptq_args.nsamples}_seqlen2048_test.pt"))

    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))

    if ptq_args.load_qmodel_path:
        print("load_qmodel_path: {}".format(ptq_args.load_qmodel_path))
    print("wiki2 ppl is: {}".format(dataset_ppl))


    if ptq_args.lm_eval:
        lm_eval_batch_size = 4
        print("lm_eval_batch_size: {}".format(lm_eval_batch_size))
        model.to("cuda")
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=lm_eval_batch_size)

        # task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
        task_names = ['boolq', 'piqa', 'social_iqa', 'arc_easy', 'arc_challenge', 'hellaswag', 'winogrande', 'openbookqa', 'lambada']
        log.info(task_names)
        print(task_names)
        results = lm_eval.simple_evaluate(hflm, tasks=task_names, batch_size=lm_eval_batch_size, device="auto")['results']

        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        log.info(metric_vals)
        if ptq_args.load_qmodel_path:
            print("load_qmodel_path: {}".format(ptq_args.load_qmodel_path))
        print(metric_vals)

    dist.barrier()


if __name__ == "__main__":
    train()
