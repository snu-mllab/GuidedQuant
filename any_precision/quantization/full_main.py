"""
Fine-tune an LLM that was previously quantized with AQLM;
based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
import argparse
import os
import datetime
import sys
from contextlib import nullcontext
from functools import partial
from typing import Dict, Optional, Tuple, List

import datasets
import torch
import torch.distributed
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import transformers
import contextlib
import logging
import shutil
import time
from torch import nn as nn
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    StateDictType,
)
from tqdm.auto import tqdm
from any_precision.quantization.finetune_utils import (
    QuantizedWeightFSDP, 
    QuantizedLinearFSDP, 
    IntCodes,
)
from any_precision.quantization.full_datautils import get_loaders
from any_precision.quantization.full_utils_v1 import (
    ConfigurableAdamW, 
    auto_model_load,
    group_texts, 
    split_long_texts,
    infer_module_classes,
    is_model_for_causal_lm,
    compute_kl_divergence_loss_values,
)
from any_precision.quantization.full_utils_v2 import (
    create_dequantized_model, 
    StraightThroughAdamW,
    split_quantized_weights_between_ranks,
    YourQuantizedWeightIsInAnotherRank,
    get_original_named_parameters_from_fsdp_module,
)
from any_precision.quantization.pack import pack_single_weight, unpack_single_weight
from any_precision.modules import AnyPrecisionForCausalLM
from multiprocessing import Pool

has_wandb = False

@contextlib.contextmanager
def one_rank_at_a_time(local: bool = False, group_size: int = 1):
    """
    In distributed setting, let only group_size processes enter at a time
    :param local: if True, the limit is enforced within each host, i.e. distributed hosts can act concurrently
    :param group_size: if more than one is specified,
    """
    distributed = torch.distributed.is_initialized()
    rank = int(os.environ.get("LOCAL_RANK" if local else "RANK", 0)) if distributed else 0
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE" if local else "WORLD_SIZE", 0)) if distributed else 1
    if distributed:
        torch.distributed.barrier()
    for current_group_index in range(world_size // group_size):
        if current_group_index == rank // group_size:
            yield
        if distributed:
            torch.distributed.barrier()


@contextlib.contextmanager
def master_rank_first(local: bool, master_rank: int = 0):
    distributed = torch.distributed.is_initialized()
    rank = int(os.environ.get("LOCAL_RANK" if local else "RANK", 0)) if distributed else 0
    if distributed and rank != master_rank:
        torch.distributed.barrier()
    yield
    if distributed and rank == master_rank:
        torch.distributed.barrier()


def prepare_training_dataset(args: argparse.Namespace, tokenizer: transformers.PreTrainedTokenizer) -> datasets.Dataset:
    if os.path.exists(args.dataset_name):
        dataset = datasets.load_from_disk(args.dataset_name)
    else:
        dataset = datasets.load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=args.split,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
            num_proc=args.download_num_workers if args.download_num_workers is not None else args.num_workers,
            streaming=False,
        )

    def is_tokenized(dataset):
        return "input_ids" in dataset.column_names

    if is_tokenized(dataset):
        if torch.distributed.get_rank() == 0:
            logging.info("Dataset already tokenized")
            assert len(dataset[0]["input_ids"]) == args.model_seqlen
        return dataset

    text_column_name = "text" if "text" in dataset.column_names else next(iter(dataset.column_names))

    if args.preprocessing_chunk_length is not None:
        dataset = dataset.map(
            lambda examples: {
                text_column_name: split_long_texts(examples[text_column_name], args.preprocessing_chunk_length)
            },
            batched=True,
            num_proc=args.preprocessing_num_workers if args.preprocessing_num_workers is not None else args.num_workers,
            remove_columns=list(dataset.column_names),
            keep_in_memory=args.preprocessing_keep_in_memory,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Splitting dataset over newline into chunks of ~{args.preprocessing_chunk_length} characters",
        )

    tokenized_dataset = dataset.map(
        lambda example: tokenizer(example[text_column_name]),
        num_proc=args.preprocessing_num_workers if args.preprocessing_num_workers is not None else args.num_workers,
        remove_columns=list(dataset.column_names),
        keep_in_memory=args.preprocessing_keep_in_memory,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    lm_dataset = tokenized_dataset.map(
        partial(group_texts, block_size=args.model_seqlen, add_labels=False),
        batched=True,
        num_proc=args.preprocessing_num_workers if args.preprocessing_num_workers is not None else args.num_workers,
        keep_in_memory=args.preprocessing_keep_in_memory,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {args.model_seqlen}",
    )
    assert is_tokenized(lm_dataset)
    return lm_dataset


def load_teacher_model(args: argparse.Namespace, device: torch.device) -> FullyShardedDataParallel:
    """Load unquantized model with frozen parameters"""
    _, _, model = auto_model_load(args.base_model)
    model.train(False)
    for param in model.parameters():
        param.requires_grad = False

    model.config.use_cache = False
    transformer_block_types = infer_module_classes(model, args.block_type)

    return wrap_model_with_fsdp_(
        model,
        auto_wrap_policy=lambda module, recurse, **_etc: recurse or isinstance(module, transformer_block_types),
        cpu_offload=CPUOffload(offload_params=args.offload_teacher_params) if args.offload_teacher_params else None,
        limit_all_gathers=args.limit_all_gathers,
        forward_prefetch=args.forward_prefetch,
        device_id=device,
    )


def mp_unpack_single_weight(args):
    name, weight, parent_precision = args
    return name, unpack_single_weight(weight, parent_precision)


def mp_pack_single_weight(args):
    name, weight, parent_precision = args
    return name, pack_single_weight(weight, parent_precision)


def unpack_quantized_parameters(quantized_model_path: str, quantized_model: AnyPrecisionForCausalLM):
    model_path = os.path.join(quantized_model_path, "pytorch_model.bin")
    state_dict = torch.load(model_path)

    cpu_count = int(os.popen("nproc").read().strip())
    # Limit cpu_count to 8 as larger values use too much memory, without much speedup
    _max_cpu_count = 8
    if cpu_count > _max_cpu_count:
        logging.warning(f"cpu_count will be limited to 8 to avoid excessive memory usage. "
                        f"Original value: {cpu_count}")
        cpu_count = _max_cpu_count

    # Get first qweight tensor to determine number of bits
    qweight_key = next(key for key in state_dict if key.endswith('.qweight'))
    num_bits = state_dict[qweight_key].shape[0]

    codes_dict = {k: v for k, v in state_dict.items() if k.endswith('.qweight')}
    module_names = [k[:-len('.qweight')] for k in codes_dict.keys()]
    args_list = [(k, v, num_bits) for k, v in codes_dict.items()]

    with Pool(cpu_count) as pool:
        for name, out_tensor in tqdm(pool.imap(mp_unpack_single_weight, args_list), total=len(args_list), desc="Unpacking quantized weights"):
            state_dict[name] = out_tensor

    for module_name in module_names:
        parent_module = quantized_model.model
        for name in module_name.split('.')[:-1]:
            parent_module = getattr(parent_module, name)
        
        codes = state_dict[module_name + '.qweight']
        codebooks = state_dict[module_name + f'.lut{num_bits}'].unsqueeze(1) # add dim for group_count (supports only 1 for now)
        qlinear = QuantizedLinearFSDP(QuantizedWeightFSDP(codes=codes, codebooks=codebooks))
        setattr(parent_module, module_name.split('.')[-1], qlinear)

    return state_dict


def load_student_model(
    args: argparse.Namespace, device: torch.device, dequantize: bool
) -> Tuple[FullyShardedDataParallel, Optional[Dict[str, QuantizedWeightFSDP]]]:
    """
    load student model for fine-tuning. If dequantize is set, dequantize all quantized weights to accumulate full grads
    """
    _, _, quantized_model = auto_model_load(args.quantized_model)
    student_model = quantized_model.model

    unpack_quantized_parameters(args.quantized_model, quantized_model)

    if args.embed_dtype != args.master_dtype:
        student_model.set_output_embeddings(student_model.get_output_embeddings().to(args.embed_dtype))
        student_model.set_input_embeddings(student_model.get_input_embeddings().to(args.embed_dtype))

    student_model.config.use_cache = False
    student_model.train(True)  # note: HF gradient checkpoints do not work for some models without train(True); see
    # https://github.com/huggingface/transformers/blob/2d92db8/src/transformers/models/llama/modeling_llama.py#L1006
    if args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
        student_model.enable_input_require_grads()

    # convert QuantizedModel state dict to make it compatible with FSDP
    for name, module in student_model.named_modules():
        if isinstance(module, QuantizedWeightFSDP):
            assert module.codes is not None
            if args.code_dtype is not None:
                module.codes = nn.Parameter(module.codes.to(args.code_dtype), requires_grad=module.codes.requires_grad)
            module.wrap_codes_for_fsdp_()
            assert module.codes is None and isinstance(module.codes_storage, IntCodes)
    assert any(isinstance(module, IntCodes) for module in student_model.modules())

    if dequantize:
        student_model, named_quantized_params = create_dequantized_model(
            student_model, dequantized_dtype=args.amp_dtype, reuse_non_quantized=True
        )
    else:
        named_quantized_params = None

    transformer_block_types = list(infer_module_classes(student_model, args.block_type))
    layernorm_types = list(transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    extra_block_types = list()
    for extra_module_name in args.wrap_separately:
        extra_block_types.extend(infer_module_classes(student_model, extra_module_name))
    block_types_to_wrap = tuple(
        set(
            transformer_block_types
            + layernorm_types
            + extra_block_types
            + [
                IntCodes,
            ]
        )
    )
    if torch.distributed.get_rank() == 0:
        logging.info(f"Blocks to be wrapped separately: {block_types_to_wrap}\n")

    mixed_precision = None
    if args.use_fsdp_amp:
        assert args.amp_dtype is not None, "requested to use_fsdp_amp, but amp_dtype is not None"
        block_types_for_amp_to_ignore = tuple(set(layernorm_types + extra_block_types))
        if torch.distributed.get_rank() == 0:
            logging.info(f"Blocks excluded from AMP: {block_types_for_amp_to_ignore}\n")
        mixed_precision = MixedPrecision(
            param_dtype=args.amp_dtype,
            reduce_dtype=args.amp_dtype,
            _module_classes_to_ignore=block_types_for_amp_to_ignore,
        )
    else:
        if torch.distributed.get_rank() == 0:
            logging.info(f"Not using FSDP native MixedPrecision; Local amp_dtype={args.amp_dtype}.")

    student_model = wrap_model_with_fsdp_(
        student_model,
        use_orig_params=True,
        auto_wrap_policy=lambda module, recurse, **_etc: recurse or isinstance(module, block_types_to_wrap),
        cpu_offload=CPUOffload(offload_params=args.offload_student_params) if args.offload_student_params else None,
        limit_all_gathers=args.limit_all_gathers,
        forward_prefetch=args.forward_prefetch,
        mixed_precision=mixed_precision,
        device_id=device,
    )

    if named_quantized_params is not None:
        if torch.distributed.get_world_size() > 1:
            # distributed pv: each rank holds a subset of all quantized weights; the rest are replaced with pointers
            named_quantized_params = split_quantized_weights_between_ranks(
                named_quantized_params, verify_checksums=False
            )
        for quantized_weight in named_quantized_params.values():
            if isinstance(quantized_weight, QuantizedWeightFSDP):
                quantized_weight.to(device)
            else:
                assert isinstance(quantized_weight, YourQuantizedWeightIsInAnotherRank)

    return student_model, named_quantized_params


def wrap_model_with_fsdp_(
    model: transformers.PreTrainedModel, auto_wrap_policy: callable, **kwargs
) -> FullyShardedDataParallel:
    """Wrap a model *ForCausalLM components: transformer and lm_head are wrapped as FSDP instances"""
    assert isinstance(model, transformers.PreTrainedModel) and is_model_for_causal_lm(model)
    base_model, lm_head = model.base_model, model.get_output_embeddings()

    def _modified_auto_wrap_policy(module, recurse, **kwargs):
        return auto_wrap_policy(module, recurse, **kwargs) or (module in (base_model, lm_head))

    model = FullyShardedDataParallel(model, auto_wrap_policy=_modified_auto_wrap_policy, **kwargs)

    assert isinstance(model.module, transformers.PreTrainedModel)
    assert isinstance(model.base_model, FullyShardedDataParallel)
    assert isinstance(model.get_output_embeddings(), FullyShardedDataParallel)
    return model


def trigger_fsdp_lazy_init_(
    tokenizer: transformers.PreTrainedTokenizer,
    teacher_model: FullyShardedDataParallel,
    student_model: FullyShardedDataParallel,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
):
    """Trigger FullyShardedDataParallel lazy init in the correct order to allow both training and eval"""
    logging.info("Initializing FSDP root")
    dummy_batch = tokenizer("I am the monument to all your sins", return_tensors="pt")
    dummy_batch = {k: v.to(device) for k, v in dummy_batch.items()}
    with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
        with torch.no_grad():
            teacher_model(**dummy_batch)
        (student_model(**dummy_batch).logits * 0).sum().backward()


def create_pv_optimizer(
    args: argparse.Namespace,
    student_model: FullyShardedDataParallel,
    named_quantized_params: Dict[str, QuantizedWeightFSDP],
) -> torch.optim.Optimizer:
    """Create optimizer for PV-Tuning using a de-quantized student model and a dictionary of quantized weights"""
    named_dequantized_params = get_original_named_parameters_from_fsdp_module(student_model)
    opt_device = torch.device("cpu") if args.offload_optimizer else next(student_model.parameters()).device
    assert all(name in named_dequantized_params for name in named_quantized_params)
    return StraightThroughAdamW(
        named_dequantized_params=named_dequantized_params,
        named_quantized_params=named_quantized_params,
        update_codes=dict(
            lr=args.code_lr,
            betas=(args.code_beta1, args.code_beta2),
            lamb=args.lamb,
            debias=args.debias,
            amsgrad=args.amsgrad,
            compute_dtype=args.master_dtype,
            exp_avg_dtype=torch.float16 if args.code_adam_16bit else args.master_dtype,
            exp_avg_sq_dtype=torch.bfloat16 if args.code_adam_16bit else args.master_dtype,
            v_hat_max_dtype=torch.float16 if args.code_adam_16bit else args.master_dtype,
            exp_avg_device=opt_device,
            exp_avg_sq_device=opt_device,
            v_hat_max_device=opt_device,
        )
        if args.update_codes
        else None,
        update_codebooks_and_scales=dict(
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            lamb=args.lamb,
            debias=args.debias,
            amsgrad=args.amsgrad,
            compute_dtype=args.master_dtype,
            exp_avg_dtype=args.master_dtype,
            exp_avg_sq_dtype=args.master_dtype,
            v_hat_max_dtype=args.master_dtype,
            exp_avg_device=opt_device,
            exp_avg_sq_device=opt_device,
            v_hat_max_device=opt_device,
        )
        if args.update_codebooks_and_scales
        else None,
        update_non_quantized_parameters=dict(
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            lamb=args.lamb,
            debias=args.debias,
            amsgrad=args.amsgrad,
            compute_dtype=args.master_dtype,
            exp_avg_dtype=args.master_dtype,
            exp_avg_sq_dtype=args.master_dtype,
            v_hat_max_dtype=args.master_dtype,
            exp_avg_device=opt_device,
            exp_avg_sq_device=opt_device,
            v_hat_max_device=opt_device,
        )
        if args.update_non_quantized_parameters
        else None,
        delta_decay=args.delta_decay,
        max_code_change_per_step=args.max_code_change_per_step,
        force_code_update=args.force_code_update,
        code_trust_ratio=args.code_trust_ratio,
        beam_size=args.beam_size,
        straight_through_buffer_dtype=args.straight_through_buffer_dtype,
        verbose=args.verbose_optimizer,
    )


def create_p_optimizer(args: argparse.Namespace, student_model: FullyShardedDataParallel) -> torch.optim.Optimizer:
    """Create optimizer for training only continuous parameters of a quantized model"""
    quantized_weight_continuous_parameters = set()
    for module in student_model.modules():
        if isinstance(module, QuantizedWeightFSDP):
            for param in module.parameters():
                if torch.is_floating_point(param) and param.requires_grad:
                    quantized_weight_continuous_parameters.add(param)
    all_trainable_params = []
    if args.update_codebooks_and_scales:
        all_trainable_params.extend(
            param for param in student_model.parameters() if param in quantized_weight_continuous_parameters
        )  # use iteration instead of simply adding list(set) to ensure deterministic order of parameters
    if args.update_non_quantized_parameters:
        all_trainable_params.extend(
            param
            for param in student_model.parameters()
            if torch.is_floating_point(param)
            and param.requires_grad
            and param not in quantized_weight_continuous_parameters
        )
    if args.update_codes:
        raise RuntimeError("When asked to update_codes, one should create_pv_optimizer, but this is create_p_optimizer")
    assert len(all_trainable_params) > 0, (
        "found no trainable parameters. Did you specify update_codes, "
        "update_codebooks_and_scales or update_non_quantized_parameters?"
    )
    opt_device = torch.device("cpu") if args.offload_optimizer else next(student_model.parameters()).device
    return ConfigurableAdamW(
        params=list(all_trainable_params),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        lamb=args.lamb,
        debias=args.debias,
        amsgrad=args.amsgrad,
        compute_dtype=args.master_dtype,
        exp_avg_dtype=args.master_dtype,
        exp_avg_sq_dtype=args.master_dtype,
        v_hat_max_dtype=args.master_dtype,
        exp_avg_device=opt_device,
        exp_avg_sq_device=opt_device,
        v_hat_max_device=opt_device,
    )


def save_training_state(
    args: argparse.Namespace, metadata: dict, quantized_model: nn.Module, optimizer: torch.optim.Optimizer
):
    """Save model, optimizer state dict and training metadata to be loaded via load_training_state"""
    if args.save is None:
        return
    rank = torch.distributed.get_rank()
    os.makedirs(args.save, exist_ok=True)
    if rank == 0:
        logging.info(f"Saving snapshot to {args.save}")
        torch.save(metadata, os.path.join(args.save, "metadata.pt"))
    with FullyShardedDataParallel.state_dict_type(quantized_model, StateDictType.LOCAL_STATE_DICT):
        torch.save(quantized_model.state_dict(), os.path.join(args.save, f"quantized_model_state_dict_rank{rank}.pt"))
        # model saves non-quantized weights and dequantized versions of QuantizedWeight; the latter is not necessary
    torch.save(optimizer.state_dict(), os.path.join(args.save, f"optimizer_state_dict_rank{rank}.pt"))
    # optimizer state dict saves statistics QuantizedWeight instances and straight-through buffers
    if args.on_save:
        exec(args.on_save)


def load_training_state(
    args: argparse.Namespace, metadata: dict, quantized_model: nn.Module, optimizer: torch.optim.Optimizer
):
    """Load model, optimizer state dict and metadata saved via save_training_state; update parameters in-place"""
    rank = torch.distributed.get_rank()
    if args.save is None or not os.path.exists(args.save):
        if args.save is not None and rank == 0:
            logging.info(f"No checkpoint found at {args.save}")
    else:
        with FullyShardedDataParallel.state_dict_type(quantized_model, StateDictType.LOCAL_STATE_DICT):
            # this loads non-quantized weights and de-quantized versions of QuantizedWeight instances
            state_dict_ptr = quantized_model.state_dict()
            loaded_state_dict = torch.load(os.path.join(args.save, f"quantized_model_state_dict_rank{rank}.pt"))
            with torch.no_grad():
                for key in state_dict_ptr:
                    state_dict_ptr[key].copy_(loaded_state_dict.pop(key))
                assert len(loaded_state_dict) == 0, f"Unused keys:, {tuple(loaded_state_dict.keys())}"
            del state_dict_ptr, loaded_state_dict

        # v-- loading optimizer state dict also loads all QuantizedWeights and straight-through buffers
        optimizer.load_state_dict(
            torch.load(os.path.join(args.save, f"optimizer_state_dict_rank{rank}.pt"), map_location="cpu")
        )
        metadata.update(torch.load(os.path.join(args.save, "metadata.pt")))
        if args.eval_datasets is not None and metadata["early_stop_on"] not in args.eval_datasets:
            if rank == 0:
                logging.info(f"Stopping criterion {metadata['early_stop_on']} is not in eval_datasets; resetting best loss.")
            metadata["early_stop_on"] = next(iter(args.eval_datasets))
            metadata["best_eval_perplexity"] = float("inf")
            metadata["best_step"] = 0
        if rank == 0:
            logging.info(f"Loaded training state from {args.save}: {metadata}")


def save_model(args: argparse.Namespace, student_model: FullyShardedDataParallel, optimizer: torch.optim.Optimizer):
    """Save model for either P- or PV-Tuning using the appropriate saver"""
    save_pv_model(args, student_model, optimizer)


def save_pv_model(
    args: argparse.Namespace, dequantized_model: FullyShardedDataParallel, optimizer: StraightThroughAdamW
):
    """Save consolidated model from PV tuning, can be exported later via convert_legacy_model_format.py"""
    output_path = os.path.join(args.save, "best_model")
    os.makedirs(output_path, exist_ok=True)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    local_quantized_weight_dict = dict()
    for name, quantized_weight in optimizer.iterate_local_quantized_weights():
        assert isinstance(quantized_weight.get_codes(), torch.Tensor)
        layer_name = name[:-len('.weight')]
        local_quantized_weight_dict[f"{layer_name}.quantized_weight.codes"] = quantized_weight.get_codes().cpu()
        local_quantized_weight_dict[f"{layer_name}.quantized_weight.codebooks"] = quantized_weight.get_codebooks().cpu()


    if rank == 0:
        start_time = time.time()
    quantized_weight_dict_by_rank = [None for _ in range(world_size)] if rank == 0 else None
    torch.distributed.gather_object(local_quantized_weight_dict, quantized_weight_dict_by_rank, dst=0)

    if rank == 0:
        end_time = time.time()
        logging.info(f"Gather object communication took {end_time - start_time:.2f} seconds")

    with FullyShardedDataParallel.state_dict_type(
        dequantized_model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state_dict = dequantized_model.state_dict()
        if rank == 0:
            dequantized_weight_names = set()
            for quantized_weight_dict in quantized_weight_dict_by_rank:
                for name in quantized_weight_dict.keys():
                    if name.endswith('.quantized_weight.codes'):
                        dequantized_weight_name = name.replace('.quantized_weight.codes', '.weight')
                    elif name.endswith('.quantized_weight.codebooks'):
                        dequantized_weight_name = name.replace('.quantized_weight.codebooks', '.weight')
                    else:
                        raise ValueError(f"Unknown quantized weight name: {name}")
                    dequantized_weight_names.add(dequantized_weight_name)
            for key in dequantized_weight_names:
                del model_state_dict[key]
            for local_quantized_weight_dict in quantized_weight_dict_by_rank:
                for name, tensor in local_quantized_weight_dict.items():
                    model_state_dict[name] = tensor
            
            cpu_count = int(os.popen("nproc").read().strip())
            # Limit cpu_count to 8 as larger values use too much memory, without much speedup
            _max_cpu_count = 8
            if cpu_count > _max_cpu_count:
                logging.warning(f"cpu_count will be limited to 8 to avoid excessive memory usage. "
                                f"Original value: {cpu_count}")
                cpu_count = _max_cpu_count
            codebook_key = next(key for key in model_state_dict if key.endswith('.quantized_weight.codebooks'))
            num_bits = model_state_dict[codebook_key].shape[-1].bit_length() - 1

            codes_dict = {k: v for k, v in model_state_dict.items() if k.endswith('.quantized_weight.codes')}
            assert all(isinstance(v, torch.Tensor) for v in codes_dict.values()), "Codes are not tensors"

            codebooks_dict = {k: v for k, v in model_state_dict.items() if k.endswith('.quantized_weight.codebooks')}
            args_list = [(k, v, num_bits) for k, v in codes_dict.items()]

            for pack_args in tqdm(args_list, desc="Packing quantized weights back"):
                name, out_tensor = mp_pack_single_weight(pack_args)
                model_state_dict[name] = torch.from_numpy(out_tensor)

            for name, codebook in tqdm(codebooks_dict.items(), desc="Squeezing codebooks"):
                model_state_dict[name] = codebook.squeeze(dim=1)

            quant_param_names = [
                k for k in model_state_dict if k.endswith('.quantized_weight.codes') or k.endswith('.quantized_weight.codebooks')
            ]

            for name in quant_param_names:
                if name.endswith('.quantized_weight.codes'):
                    new_name = name[:-len('.quantized_weight.codes')] + '.qweight'
                else:
                    new_name = name[:-len('.quantized_weight.codebooks')] + f'.lut{num_bits}'
                model_state_dict[new_name] = model_state_dict.pop(name)

            import shutil, glob
            for json_file in glob.glob(os.path.join(args.quantized_model, '*.json')):
                shutil.copy2(json_file, args.save)
            torch.save(model_state_dict, os.path.join(args.save, f"pytorch_model.bin"))
    torch.distributed.barrier()
    if rank == 0:
        start_time = end_time
        end_time = time.time()
        logging.info(f"Saving model took {end_time - start_time:.2f} seconds")

    if rank == 0:
        logging.info(f"Saved best model shards to {output_path}")

def compute_loss_on_batch(
    batch: dict,
    teacher_model: FullyShardedDataParallel,
    student_model: FullyShardedDataParallel,
    *,
    amp_dtype: Optional[torch.dtype],
    max_tokens_per_chunk: Optional[int],
) -> torch.Tensor:
    if max_tokens_per_chunk is not None:  # chunked inference, transformer and lm head must be separate FSDP instances
        with torch.no_grad():
            teacher_hidden_states = teacher_model.base_model(**batch).last_hidden_state
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
            student_hidden_states = student_model.base_model(**batch).last_hidden_state
            return compute_kl_divergence_loss_values(
                student_hidden_states=student_hidden_states,
                student_lm_head=student_model.get_output_embeddings(),
                teacher_hidden_states=teacher_hidden_states,
                teacher_lm_head=teacher_model.get_output_embeddings(),
                max_tokens_per_chunk=max_tokens_per_chunk,
                checkpoint_last_chunk=False,
                use_reentrant=False,
                determinism_check="none",
            ).mean()

    else:  # combined inference without gradient checkpointing
        with torch.no_grad():
            teacher_logprobs = F.log_softmax(teacher_model(**batch).logits, dim=-1)
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
            student_logprobs = F.log_softmax(student_model(**batch).logits, dim=-1)
            loss = F.kl_div(
                input=student_logprobs.flatten(0, -2),
                target=teacher_logprobs.flatten(0, -2),
                log_target=True,
                reduction="batchmean",
            ).mean()
        return loss

@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module, data: torch.Tensor, seqlen: int, device: torch.device, amp_dtype: Optional[torch.dtype] = None
) -> float:
    """Perplexity evaluation as per https://github.com/IST-DASLab/gptq (standard among quantization research)"""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    inps = [
        data[:, start : start + seqlen] for start in range(0, data.shape[1], seqlen) if start + seqlen < data.shape[1]
    ]  # ignore last incomplete sequence as in the GPTQ paper
    num_sequences_without_padding = len(inps)

    # pad sequences to be divisible by world_size for DDP/FSDP compatibility
    num_padding_sequences = -len(inps) % world_size
    inps.extend([inps[-1]] * num_padding_sequences)

    total_nll_and_tokens = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    total_nll, total_tokens = total_nll_and_tokens[0], total_nll_and_tokens[1]

    for sequence_index, input_ids in enumerate(tqdm(inps, desc="Evaluating perplexity") if rank == 0 else inps):
        if sequence_index % world_size != rank:
            continue
        input_ids = input_ids.to(device)
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
            lm_logits = model(input_ids).logits

        if sequence_index < num_sequences_without_padding:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_nll += loss.float() * shift_labels.numel()
            total_tokens += shift_labels.numel()

    if world_size > 1:
        torch.distributed.all_reduce(total_nll_and_tokens, op=torch.distributed.ReduceOp.SUM)
    ppl = torch.exp(total_nll / total_tokens)
    return ppl.item()

def compute_validation_perplexities(args: argparse.Namespace, model: nn.Module, eval_datasets: dict):
    rank = torch.distributed.get_rank()
    perplexities = {}
    for dataset_name, eval_dataset in eval_datasets.items():
        if rank == 0:
            logging.info(f"Evaluating perplexity on {dataset_name} ...")
        device = next(model.parameters()).device
        original_dtype = args.load_dtype if args.load_dtype != "auto" else None
        amp_dtype = args.amp_dtype if args.amp_dtype is not None else original_dtype
        ppl = evaluate_perplexity(model, eval_dataset, args.model_seqlen, device=device, amp_dtype=amp_dtype)
        if rank == 0:
            logging.info(f"{dataset_name} perplexity: {ppl:.9f}")
        perplexities[dataset_name] = ppl
    return perplexities


def full_nuq(args):
    assert torch.cuda.is_available() and torch.distributed.is_available()
    torch.distributed.init_process_group()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    assert torch.distributed.is_initialized()
    assert args.batch_size is not None, "please specify batch size"
    assert args.batch_size % world_size == 0
    if args.microbatch_size is None:
        args.microbatch_size = args.batch_size // world_size
    assert args.batch_size % (world_size * args.microbatch_size) == 0
    grad_accumulation_steps = args.batch_size // (world_size * args.microbatch_size)

    args.master_dtype = getattr(torch, args.master_dtype)
    args.embed_dtype = getattr(torch, args.embed_dtype) if args.embed_dtype is not None else args.master_dtype
    args.load_dtype = getattr(torch, args.load_dtype) if args.load_dtype != "auto" else "auto"
    args.code_dtype = getattr(torch, args.code_dtype) if args.code_dtype is not None else None
    args.amp_dtype = getattr(torch, args.amp_dtype) if args.amp_dtype is not None else None

    if args.straight_through_buffer_dtype is not None:
        args.straight_through_buffer_dtype = getattr(torch, args.straight_through_buffer_dtype)
    else:
        args.straight_through_buffer_dtype = args.master_dtype

    if args.save_every_steps is not None:
        assert args.save is not None, f"save_every_steps={args.save_every_steps}, but --save path not specified"
    if args.keep_best_model:
        assert args.save is not None, f"--keep_best_model requires --save path"
        assert args.eval_every_steps is not None, f"--keep_best_model requires --eval_every_steps"
        assert args.eval_datasets is not None, f"--keep_best_model requires --eval_datasets"

    if args.wandb and rank == 0:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        # wandb.init(config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")})

    if rank == 0:
        # Logging with time sans date, level name, and message
        log_dir = "logs_full"
        log_file_name = os.path.basename(args.save)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        log_file_name = f"{log_file_name}_{timestamp}"

        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s | %(levelname)s] %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f"{log_dir}/{log_file_name}.txt"),
            ]
        )   
        logging.info(args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    assert tokenizer.eos_token_id is not None
    tokenizer.pad_token = tokenizer.eos_token

    with master_rank_first(local=True):
        dataset = prepare_training_dataset(args, tokenizer)
        if args.save_dataset_and_exit is not None:
            if rank == 0:
                dataset.save_to_disk(args.save_dataset_and_exit)

    if args.save_dataset_and_exit is not None:
        torch.distributed.barrier()
        return

    sampler = torch.utils.data.DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.seed
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.microbatch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=transformers.default_data_collator,
    )
    eval_datasets = {
        dataset_name: get_loaders(
            dataset_name,
            seed=args.seed,
            model_path=args.base_model,
            seqlen=args.model_seqlen,
            eval_mode=True,
        )
        for dataset_name in args.eval_datasets
    }

    if rank == 0:
        logging.info(f"Training with PV Tuning")

    with one_rank_at_a_time(local=True, group_size=args.limit_parallel_inits):
        teacher_model = load_teacher_model(args, device)
        student_model, named_quantized_params = load_student_model(args, device, dequantize=True)
        if rank == 0:
            logging.info("Wrapped model:")
            logging.info(student_model)
            for name, param in student_model.named_parameters():
                logging.info(f"{name}: {param.shape}, {param.dtype}")

    optimizer = create_pv_optimizer(args, student_model, named_quantized_params)

    metadata = dict(
        current_epoch=0,
        microbatches_since_epoch_start=0,
        total_microbatches=0,
        total_optimizer_steps=0,
        loss_numerator=0,
        loss_denominator=0,
        aggregated_loss=float("nan"),
        grad_steps_accumulated=0,
        early_stop_on=next(iter(args.eval_datasets)) if args.eval_datasets else None,
        best_eval_perplexity=float("inf"),
        best_step=0,
    )

    load_training_state(args, metadata, student_model, optimizer)
    torch.distributed.barrier()
    trigger_fsdp_lazy_init_(tokenizer, teacher_model, student_model, device, amp_dtype=args.amp_dtype)

    for current_epoch in range(args.max_epochs):
        if current_epoch < metadata["current_epoch"]:
            continue  # skip finished epochs
        sampler.set_epoch(current_epoch)

        batch_iter = tqdm(train_dataloader, desc=f"Training epoch #{current_epoch}") if rank == 0 else train_dataloader
        for batch_index, batch in enumerate(batch_iter):
            if args.max_steps is not None:
                if metadata["total_optimizer_steps"] >= args.max_steps:
                    break
            if batch_index <= metadata["microbatches_since_epoch_start"]:
                continue  # skip batches processed before checkpoint
            metadata["microbatches_since_epoch_start"] += 1
            metadata["total_microbatches"] += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss_on_batch(
                batch,
                teacher_model,
                student_model,
                amp_dtype=args.amp_dtype,
                max_tokens_per_chunk=args.loss_tokens_per_chunk,
            )

            metadata["loss_numerator"] += loss.item()
            metadata["loss_denominator"] += 1
            metadata["grad_steps_accumulated"] += 1
            if metadata["grad_steps_accumulated"] < grad_accumulation_steps:
                with student_model.no_sync() if args.minimize_sync else nullcontext():
                    (loss / grad_accumulation_steps).backward()
            else:
                (loss / grad_accumulation_steps).backward()
                optimizer.step()
                optimizer.zero_grad()
                metadata["grad_steps_accumulated"] = 0
                metadata["total_optimizer_steps"] += 1

                if args.print_every_steps and metadata["total_optimizer_steps"] % args.print_every_steps == 0:
                    loss_numerator_and_denominator = torch.tensor(
                        [metadata["loss_numerator"], metadata["loss_denominator"]], dtype=torch.float64, device=device
                    )

                    torch.distributed.all_reduce(loss_numerator_and_denominator, op=torch.distributed.ReduceOp.SUM)
                    loss_numerator, loss_denominator = loss_numerator_and_denominator.tolist()
                    metadata["aggregated_loss"] = loss_numerator / loss_denominator
                    metadata["loss_numerator"] = metadata["loss_denominator"] = 0
                    if rank == 0:
                        logging.info(
                            f"epoch {metadata['current_epoch']}\tbatch {batch_index}"
                            f"\t| total updates = {metadata['total_optimizer_steps']}"
                            f"\tloss = {metadata['aggregated_loss']:.9f}"
                        )

                if args.eval_every_steps and metadata["total_optimizer_steps"] % args.eval_every_steps == 0:
                    perplexity_scores = compute_validation_perplexities(args, student_model, eval_datasets)
                    for dataset_name, perplexity in perplexity_scores.items():
                        metadata[f"perplexity_{dataset_name}"] = perplexity
                    metric_name = metadata["early_stop_on"]
                    if perplexity_scores[metric_name] < metadata["best_eval_perplexity"]:
                        if rank == 0:
                            logging.info(f"New best perplexity ({metric_name}) = {perplexity_scores[metric_name]:.9f}")
                        metadata["best_eval_perplexity"] = perplexity_scores[args.eval_datasets[0]]
                        metadata["best_step"] = metadata["total_optimizer_steps"]
                        # if args.keep_best_model:
                        #     save_model(args, student_model, optimizer)
                    save_model(args, student_model, optimizer)
                # if args.wandb and rank == 0:
                #     wandb.log(metadata, step=metadata["total_microbatches"])
                if args.save_every_steps and metadata["total_optimizer_steps"] % args.save_every_steps == 0:
                    save_training_state(args, metadata, student_model, optimizer)

        metadata["microbatches_since_epoch_start"] = 0
        metadata["current_epoch"] += 1

    save_training_state(args, metadata, student_model, optimizer)
    