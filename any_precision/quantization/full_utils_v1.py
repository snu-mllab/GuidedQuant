from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import functools
import datetime
import math
import os
from typing import Sequence
from itertools import chain
from typing import Callable, TypeVar
from torch.utils.checkpoint import checkpoint

NO_DATA = torch.empty(0)

from transformers import AutoModelForCausalLM, AutoTokenizer
from ..modules import AnyPrecisionForCausalLM

T = TypeVar("T")


def compute_kl_divergence_loss_values(
    *,
    student_hidden_states: torch.Tensor,
    student_lm_head: nn.Module,
    teacher_hidden_states: torch.Tensor,
    teacher_lm_head: nn.Module,
    max_tokens_per_chunk: int = 256,
    checkpoint_last_chunk: bool = True,
    **checkpoint_kwargs,
) -> torch.Tensor:
    """
    Compute token-wise KL divergence loss without materializing all logits/logprobs simultaneously
    :param student_hidden_states: input hidden states for student head, [batch_size, sequence_length, student_dim]
    :param student_lm_head: a token-wise layer (e.g. nn.Linear) mapping from student_dim to logits [vocabulary_size]
    :param teacher_hidden_states: input hidden states for teacher head, [batch_size, sequence_length, teacher_dim]
    :param teacher_lm_head: a token-wise layer (e.g. nn.Linear) mapping from teacher_dim to logits [vocabulary_size]
    :note: teacher is applied to hidden states without no_grad. If required, set requires_grad=False on teacher manually
    :param max_tokens_per_chunk: materialize logits logprobs for at most this many tokens at a time
    :param checkpoint_kwargs: additional arguments passed to checkpoint (e.g. use_reentrant or determinism_check)
    :param checkpoint_last_chunk: if False, do not apply gradient checkpointing to the very last chunk of inputs
        since they are the first ones to be re-materialized anyway. Useful if loss is backpropagated immediately.
    :returns: token-wise KL loss values of shape [batch_size, sequence_length]
    """
    assert student_hidden_states.requires_grad or teacher_hidden_states.requires_grad or not torch.is_grad_enabled()
    assert teacher_hidden_states.shape[:-1] == student_hidden_states.shape[:-1]
    flat_student_hidden_states = student_hidden_states.flatten(0, -2)
    flat_teacher_hidden_states = teacher_hidden_states.flatten(0, -2)
    total_tokens = flat_teacher_hidden_states.shape[0]

    loss_values_by_chunk = []
    for chunk_start in range(0, total_tokens, max_tokens_per_chunk):
        is_last_chunk = chunk_start + max_tokens_per_chunk >= total_tokens
        loss_values_by_chunk.append(
            maybe_checkpoint(
                _compute_kl_div_from_flat_hidden_states,
                flat_student_hidden_states[chunk_start : chunk_start + max_tokens_per_chunk],
                student_lm_head,
                flat_teacher_hidden_states[chunk_start : chunk_start + max_tokens_per_chunk],
                teacher_lm_head,
                checkpoint_enabled=torch.is_grad_enabled() and (checkpoint_last_chunk or not is_last_chunk),
                **checkpoint_kwargs,
            )
        )
    return torch.cat(loss_values_by_chunk).reshape(*student_hidden_states.shape[:2])


def _compute_kl_div_from_flat_hidden_states(
    flat_student_hidden_states: torch.Tensor,
    student_lm_head: nn.Module,
    flat_teacher_hidden_states: torch.Tensor,
    teacher_lm_head: nn.Module,
) -> torch.Tensor:
    student_logprobs = F.log_softmax(student_lm_head(flat_student_hidden_states), dim=-1)
    teacher_logprobs = F.log_softmax(teacher_lm_head(flat_teacher_hidden_states), dim=-1)
    return F.kl_div(input=student_logprobs, target=teacher_logprobs, log_target=True, reduction="none").sum(-1)


def maybe_checkpoint(func, *inputs, checkpoint_enabled: bool, **checkpoint_kwargs) -> T:
    """Execute function normally or with checkpointing, depending on checkpoint_enabled. Forward **checkpoint_kwargs"""
    return func(*inputs) if checkpoint_enabled else checkpoint(func, *inputs, **checkpoint_kwargs)


def infer_module_classes(model: nn.Module, class_name: str) -> Tuple[type[nn.Module], ...]:
    """find transformer block classes that should be wrapped with inner FullyShardedDataParallel (auto_wrap_policy)"""
    found_module_types = []
    for module in model.modules():
        if module.__class__.__name__ == class_name:
            found_module_types.append(type(module))
    if not found_module_types:
        raise ValueError(f"Could not find {class_name} among submodules of {model}")
    found_module_types = tuple(found_module_types)
    assert any(isinstance(module, found_module_types) for module in model.modules())
    return found_module_types

def is_model_for_causal_lm(model: nn.Module):
    assert isinstance(model, transformers.PreTrainedModel)
    assert len(model.base_model_prefix) > 0 and hasattr(model, model.base_model_prefix)
    assert model.get_output_embeddings() is not None
    return True

def split_long_texts(inputs: Sequence[str], split_max_length: int):
    """Split examples that exceed split_max_length into multiple sub-examples"""
    outputs = []
    for index, input_str in enumerate(inputs):
        while True:
            truncation_index = input_str.find("\n", split_max_length)
            if truncation_index == -1:
                outputs.append(input_str)
                break
            outputs.append(input_str[:truncation_index])
            input_str = input_str[truncation_index + 1 :]  # continue after \n
    return outputs


def group_texts(examples: Sequence[Sequence[int]], block_size: int, add_labels: bool = True):
    """Group tokenized examples together and split them into blocks of up to block_size tokens"""
    # based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
    }
    if add_labels:
        result["labels"] = result["input_ids"].copy()
    return result


def get_tokenizer_type(model_path):
    if 'llama-2' in model_path.lower():
        tokenizer_type = 'llama-2'
    elif 'llama-3' in model_path.lower():
        tokenizer_type = 'llama-3'
    elif 'llama' in model_path.lower():
        tokenizer_type = 'llama'
    elif 'opt' in model_path.lower():
        tokenizer_type = 'opt'
    elif 'mistral' in model_path.lower():
        tokenizer_type = 'mistral'
    elif 'phi-2' in model_path.lower():
        tokenizer_type = 'phi-2'
    elif 'gemma' in model_path.lower():
        tokenizer_type = 'gemma'
    else:
        tokenizer_type = None

    return tokenizer_type

def get_timestamp():
    """ Get the current timestamp for prefixing log entries """
    return datetime.datetime.now().strftime("%H:%M:%S")

def logprint(verbose, *args, **kwargs):
    """ Print if verbose is True, and prefix with timestamp """
    assert isinstance(verbose, bool), "The first argument `verbose` must be a boolean."
    if verbose:
        print(f"[{get_timestamp()}]", end=" ")
        print(*args, **kwargs)


@torch.no_grad()
def auto_model_load(model_path, device='cuda', dtype=torch.float16, verbose=True):
    """
    Args:
        model_path: path of the model to evaluate
        device: the device to use for evaluation, either 'cuda' or 'cpu'
        dtype: the dtype to use for evaluation, either torch.float16 or torch.float32
        verbose: whether to print progress

    Returns:
        (tokenizer, model) tuple loaded from the given path, with the given device and dtype.
    """
    logprint(verbose, "Loading tokenizer and model...")

    if any(os.path.basename(model_path).startswith(prefix) for prefix in ["anyprec-", "layerwise-", "blockwise-"]):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AnyPrecisionForCausalLM.from_quantized(model_path).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype,
                                                     trust_remote_code=True).to(device)

    logprint(verbose, f"{model.__class__.__name__} model loaded to device: {model.device}")

    tokenizer_type = get_tokenizer_type(model_path)

    if tokenizer_type is None:
        logprint(verbose, f"Unknown tokenizer type for {model_path}. Cannot use cached input tokens.")

    return tokenizer_type, tokenizer, model


class ConfigurableAdamW(torch.optim.Optimizer):
    r"""
    A version of Adam optimizer that supports custom parameter dtypes, amsgrad, lamb or rmsprop on per-group basis.
    Adam and Amsgrad based on https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py
    Lamb flag based on https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
    This was tested to match Adam and Lamb exactly for torch 2.3.0 (when compute_dtypes are all None)
    :param exp_avg_dtype: dtype for storing first moments; only created if betas[0] != 0; defaults to param dtype
    :param exp_avg_sq_dtype: dtype for storing second moments; only created if betas[1] != 0; defaults to param dtype
    :param v_hat_max_dtype: dtype for storing maximum v_hat; only created if amsgrad=True; defaults to param dtype
    :param exp_avg_device: device for storing exp_avg buffers; only created if betas[0]!=0; defaults to param.device
    :param exp_avg_sq_device: device for storing exp_avg_sq only created if betas[1]!=0; defaults to param.device
    :param v_hat_max_device: device for storing v_hat buffers; only created if amsgrad=True; defaults to param.device
    :note: if any of these devices are CPU, they will be prefetched for optimizer step using pinned memory
    :param compute_dtype: dtype for optimizer step computation; defaults to param dtype
    """

    def __init__(
        self,
        params: Iterable[Union[torch.Tensor, dict]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        debias: Optional[bool] = None,
        amsgrad: bool = False,
        lamb: bool = False,
        clamp_value: Optional[float] = None,
        compute_dtype: Optional[torch.dtype] = None,
        exp_avg_dtype: Optional[torch.dtype] = None,
        exp_avg_sq_dtype: Optional[torch.dtype] = None,
        v_hat_max_dtype: Optional[torch.dtype] = None,
        exp_avg_device: torch.device = None,
        exp_avg_sq_device: torch.device = None,
        v_hat_max_device: torch.device = None,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            debias=debias,
            amsgrad=amsgrad,
            lamb=lamb,
            clamp_value=clamp_value,
            compute_dtype=compute_dtype,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=exp_avg_sq_dtype,
            v_hat_max_dtype=v_hat_max_dtype,
            exp_avg_device=exp_avg_device,
            exp_avg_sq_device=exp_avg_sq_device,
            v_hat_max_device=v_hat_max_device,
        )
        super().__init__(params, defaults)

    def _maybe_init_state(self, param: torch.Tensor, group: dict) -> dict:
        state = self.state[param]
        if "step" not in state:
            state["step"] = 0
        if group["betas"][0] != 0 and "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(
                param,
                dtype=group["exp_avg_dtype"],
                memory_format=torch.preserve_format,
                device=group["exp_avg_device"],
            )
        if group["betas"][1] not in (0, 1) and "exp_avg_sq" not in state:
            state["exp_avg_sq"] = torch.zeros_like(
                param,
                dtype=group["exp_avg_sq_dtype"],
                memory_format=torch.preserve_format,
                device=group["exp_avg_sq_device"],
            )
        if group["amsgrad"] and "v_hat_max" not in state:
            state["v_hat_max"] = torch.zeros_like(
                param,
                dtype=group["v_hat_max_dtype"],
                memory_format=torch.preserve_format,
                device=group["v_hat_max_device"],
            )
        return state

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, p, state in self.iterate_groups_with_prefetch():
            assert p.grad is not None
            assert not p.grad.is_sparse, f"{self} does not support sparse gradients"
            grad = p.grad.data

            state["step"] += 1
            beta1, beta2 = group["betas"]
            compute_dtype = group.get("compute_dtype") or p.dtype

            if not group["lamb"] and group["weight_decay"] != 0:
                p.data = p.data.mul_(1 - group["lr"] * group["weight_decay"])
                # adam weight decay is not scaled by bias correction

            # Decay the first and second moment running average coefficient
            update = _inner_adam_step_and_update_statistics(
                p,
                grad,
                state.get("exp_avg", p),
                state.get("exp_avg_sq", p),
                state.get("v_hat_max", p),
                beta1,
                beta2,
                group["eps"],
                group["amsgrad"],
                compute_dtype,
            )

            if group["lamb"] and group["weight_decay"] != 0:
                update = update.add(p, alpha=group["weight_decay"])
                # lamb weight decay is later multiplied by -lr * trust_ratio * bias_correction

            update_scale = -group["lr"]
            # below: to save compute, we update scalar coefficient to account for debias/lamb/.. and multiply once
            if group["debias"] if group["debias"] is not None else (not group["lamb"]):
                # if not specified, default to True for Adam, False for Lamb
                mt_debias = 1.0 / (1 - beta1 ** state["step"]) if beta1 != 0 else 1
                vt_debias = 1.0 / math.sqrt(1 - beta2 ** state["step"]) if beta2 != 0 else 1
                bias_correction = mt_debias / vt_debias
                update_scale *= bias_correction

            if group["lamb"]:
                weight_norm = torch.norm(p.data.to(compute_dtype))
                update_norm = torch.norm(update)
                # note: lamb does not count debiasing when computing trust ratio
                if group["clamp_value"] is not None:
                    weight_norm = torch.clamp_max_(weight_norm, group["clamp_value"])
                if weight_norm == 0 or update_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / update_norm
                update_scale *= trust_ratio

            p.data.add_(update, alpha=update_scale)
        return loss

    def iterate_groups_with_prefetch(self):
        """Iterate parameters and optimizer states; skip parameters that do not require grad"""
        flat_params = [
            (group, param) for group, param in _get_flat_param_groups(self.param_groups) if param.grad is not None
        ]

        active_group, active_param = flat_params[0]
        active_state = self._maybe_init_state(active_param, active_group)
        active_state_fetched = _fetch_state_to_device(active_state, active_param.device)

        for next_group, next_param in flat_params[1:] + [(active_group, active_param)]:
            next_state = self._maybe_init_state(next_param, next_group)
            next_state_fetched = _fetch_state_to_device(next_state, next_param.device)

            yield active_group, active_param, active_state_fetched

            _commit_state_updates(active_state, active_state_fetched)

            active_group, active_param, active_state, active_state_fetched = (
                next_group,
                next_param,
                next_state,
                next_state_fetched,
            )

@functools.lru_cache()
def maybe_script(fn: callable) -> callable:
    """Apply torch.jit.script to function unless one is using TPU. TPU does not support torch.jit.script."""
    using_tpu = bool(os.environ.get("TPU_NAME"))
    # this is a reserved variable that must be set to TPU address (e.g. grpc://11.22.33.44:1337) for TPU to function
    should_script = int(os.environ.get("AQ_USE_JIT", not using_tpu))
    return torch.jit.script(fn) if should_script else fn


@maybe_script
def _inner_adam_step_and_update_statistics(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    v_hat_max: torch.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    amsgrad: bool,
    compute_dtype: torch.dtype,
):
    grad = grad.to(compute_dtype, copy=True)
    stored_exp_avg, stored_exp_avg_sq, stored_v_hat_max = exp_avg, exp_avg_sq, v_hat_max
    if beta1 != 0:
        exp_avg = exp_avg.to(compute_dtype).lerp(grad, 1 - beta1)
        stored_exp_avg.copy_(exp_avg, non_blocking=True)
        update = exp_avg
    else:
        update = grad.clone()

    if beta2 == 1:
        pass
    else:
        if beta2 == 0:
            exp_avg_sq = grad.square()
        else:
            exp_avg_sq = exp_avg_sq.to(compute_dtype).lerp(grad.square(), (1 - beta2))
            stored_exp_avg_sq.copy_(exp_avg_sq, non_blocking=True)
        if amsgrad:
            exp_avg_sq = torch.maximum(exp_avg_sq, v_hat_max, out=exp_avg_sq)
            stored_v_hat_max.copy_(exp_avg_sq, non_blocking=True)

        update /= exp_avg_sq.sqrt().add(eps)

    return update


def _get_flat_param_groups(param_groups):
    return [(group, param) for group in param_groups for param in group["params"]]


def _fetch_state_to_device(state, device):
    fetchable_state_keys = {"exp_avg", "exp_avg_sq", "v_hat_max"}.intersection(state.keys())
    fetched_states = {state_key: state[state_key].to(device, non_blocking=True) for state_key in fetchable_state_keys}
    return state | fetched_states


def _commit_state_updates(offloaded_states, fetched_states):
    fetched_keys = {"exp_avg", "exp_avg_sq", "v_hat_max"}
    for state_key in offloaded_states:
        if state_key not in fetched_keys:
            offloaded_states[state_key] = fetched_states[state_key]
        elif offloaded_states[state_key] is not fetched_states[state_key]:
            offloaded_states[state_key].copy_(fetched_states[state_key], non_blocking=True)
