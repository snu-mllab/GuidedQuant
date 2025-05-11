from typing import Optional, Tuple, Union, Dict, List, Any, Iterator, Sequence
from itertools import chain
import torch
import torch.nn as nn
import transformers
import dataclasses
import random
import hashlib
import json
import time
import contextlib
from torch.optim.optimizer import StateDict
from copy import deepcopy
from collections import defaultdict
from enum import Enum, auto
from typing import Sequence

NO_DATA = torch.empty(0)

from any_precision.quantization.full_utils_v1 import ConfigurableAdamW
from any_precision.quantization.finetune_utils import QuantizedWeightFSDP, QuantizedLinearFSDP
from any_precision.quantization.utils import get_progress_bar

class ParameterRole(Enum):
    QUANTIZED_PARAMETER = auto()  # entire quantized weight, in a de-quantized form
    QUANTIZED_REPRESENTATION_PARAMETER = auto()  # part of quantized weight inner parameters, e.g. codebooks or scales
    NON_QUANTIZED_PARAMETER = auto()

@dataclasses.dataclass(init=True, frozen=True)
class YourQuantizedWeightIsInAnotherRank:
    """This replaces quantized weights that are not held on this rank"""

    rank: int

class StraightThroughAdamW(ConfigurableAdamW):
    """
    A wrapper for a PyTorch optimizer that can perform updates on quantized and/or de-quantized parameters
    :param update_non_quantized_params: how to update parameters that are not directly linked to a QuantizedWeight.
        This may include biases, embeddings/heads, normalization layers or parts of the model that were not quantized.
        This should be either None (do not update) or a dictionary of optimizer kwargs. In the latter case, these
        keyword arguments will be used when configuring optimizer for this specific parameter group.
    :param update_codebooks_and_scales: how to update continuous params of QuantizedWeight: codebooks and scales.
        This should be either None (do not update) or a dictionary of optimizer kwargs. In the latter case, these
        keyword arguments will be used when configuring optimizer for this specific parameter group.
    :param update_codes: how to update codes in each QuantizedWeight with beam search and straight-through grad.
        This should be either None (do not update codes) or a dictionary of hyperparameter, similarly to above.
    :param delta_decay: determines whether to use straight-through estimation, direct optimization or a mixture thereof
        - if delta_decay == 1, do not use straight-through estimation. In this regime, the optimizer first updates
         de-quantized weights as though they were continuous, then uses modified weights to update codes, codebooks and
         scales; at the end of each step, the optimizer overwrites de-quantized weights to a de-quantization of the
         possibly updated quantized representations (codes, codebooks, scales).
        - if delta_decay == 0, use standard straight-through estimation. In this regime, the optimizer creates
        an internal set of straight-through buffers in the shape of de-quantized weights. The optimizer trains these
        buffers as though they were continuous; the quantized weights are then updated to minimize the L2 distance to
        these straight-through buffers; finally, the optimizer updates de-quantized weights from the quantized versions.
        - if delta_decay is between 0 and 1, use penalized straight-through estimation. The optimizer acts as though
        using standard straight-through estimation (see delta_decay == 0), but after every step, the straight-through
        buffers are set to (1 - delta_decay) * straight_through_buffer + delta_decay * quantized_weight.

    :param max_code_change_per_step: max portion of discrete code groups that can be updated; only affects codes
    :param code_trust_ratio: the maximum relative change to quantized weights per step, as a fraction of weight norm;
        see details in src/beam_search_l2.py, and in particular, beam_search_optimal_codes docstring.
    :param code_selection_temperature: if max_code_change or code_trust_ratio is set, the optimizer will by default
        prioritize updating codes with the largest delta = ||dequantized_weight_after_sgd_step - quantized_weight||_2 .
        If code_selection_temperature is above 0, it will instead sample codes randomly in proportion to the same
        delta ^ (1 / temperature). If temperature is very high, the optimizer will choose codes uniformly at random.
    :param force_code_update: if True, beam search will force codes to change even if code is optimal in
        terms of mean squared error. By default, the algorithm forces *all* weights to update this way, which may change
        weights too much. To limit the numer of updated weights, set max_code_change and trust_ratio.
    :param stochastic_rounding_tau: if above 0, use stochastic rounding with this temperature. See aq.py

    :param beam_size: beam search width used only when updating codes. See beam_size in aq.py

    :param straight_through_buffer_dtype: use this dtype when accumulating updates to de-quantized weight matrices
        Used only if delta_decay != 1.

    """

    def __init__(
        self,
        named_dequantized_params: Dict[str, nn.Parameter],
        named_quantized_params: Dict[str, Union[QuantizedWeightFSDP, YourQuantizedWeightIsInAnotherRank]],
        *,
        update_non_quantized_parameters: Optional[dict] = None,
        update_codebooks_and_scales: Optional[dict] = None,
        update_codes: Optional[dict] = None,
        beam_size: int,
        delta_decay: float = 1,
        max_code_change_per_step: float,
        code_trust_ratio: Optional[float] = None,
        code_selection_temperature: float = 0,
        force_code_update: bool = False,
        stochastic_rounding_tau: float = 0,
        straight_through_buffer_dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
        **kwargs,
    ):
        assert 0 <= delta_decay <= 1
        assert all(
            isinstance(qw, (QuantizedWeightFSDP, YourQuantizedWeightIsInAnotherRank))
            for qw in named_quantized_params.values()
        )
        assert all(name in named_dequantized_params for name in named_quantized_params), "param names mismatch"

        self.sharded = not all(isinstance(qw, QuantizedWeightFSDP) for qw in named_quantized_params.values())
        self.is_straight_through = delta_decay != 1
        if verbose and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            print(end=f"PV optimizer init:\n\tAre quantized weights sharded? : {self.sharded}.\n")
            print(end=f"\tOptimizing {('without', 'with')[self.is_straight_through]} straight-through buffers\n")
        param_groups, all_optimized_params = self._select_optimized_parameters(
            named_dequantized_params=named_dequantized_params,
            named_quantized_params=named_quantized_params,
            update_non_quantized_parameters=update_non_quantized_parameters,
            update_codebooks_and_scales=update_codebooks_and_scales,
            update_codes=update_codes,
            straight_through_buffer_dtype=straight_through_buffer_dtype,
        )

        super().__init__(param_groups, **kwargs)
        self.ordered_quantized_weight_names = tuple(sorted(named_quantized_params.keys()))
        self.optimized_param_to_name = {param: name for name, param in all_optimized_params.items()}
        self.quantized_weights_by_name = {
            name: qw
            for name, qw in named_quantized_params.items()
            if isinstance(qw, (QuantizedWeightFSDP, YourQuantizedWeightIsInAnotherRank))
        }
        self.straight_through_buffer_by_name = (
            {
                name: all_optimized_params[name]
                for name in self.quantized_weights_by_name.keys()
                if name in all_optimized_params
            }
            if self.is_straight_through
            else {}
        )
        self.dequantized_weights_by_name = {
            name: param for name, param in named_dequantized_params.items() if name in named_quantized_params
        }
        if self.sharded:
            self.sharded_param_sizes_by_rank = _get_sharded_param_sizes_by_rank(named_dequantized_params)
            self.target_rank_by_name = {
                name: qw.rank if isinstance(qw, YourQuantizedWeightIsInAnotherRank) else torch.distributed.get_rank()
                for name, qw in self.quantized_weights_by_name.items()
            }

        self.should_update_non_quantized_parameters = update_non_quantized_parameters is not None
        self.should_update_codebooks_and_scales = update_codebooks_and_scales is not None
        self.should_update_codes = update_codes is not None

        self.delta_decay = delta_decay
        self.max_code_change_per_step = max_code_change_per_step
        self.verbose = verbose

    def _select_optimized_parameters(
        self,
        named_dequantized_params,
        named_quantized_params,
        straight_through_buffer_dtype,
        update_non_quantized_parameters: Optional[dict],
        update_codebooks_and_scales: Optional[dict],
        update_codes: Optional[dict],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, nn.Parameter]]:
        """Choose which version of parameter to optimize: the parameter itself or a straight-through buffer"""
        non_quantized_params, quantized_params, quantized_representation_params = dict(), dict(), dict()
        for name, param in named_dequantized_params.items():
            if name not in named_quantized_params or isinstance(named_quantized_params[name], torch.Tensor):
                non_quantized_params[name] = param
            elif isinstance(named_quantized_params[name], QuantizedWeightFSDP):
                quantized_weight = named_quantized_params[name]
                if self.is_straight_through:  # create an accumulator for optimizer updates; sharded alongside FSDP
                    with torch.no_grad():
                        dequantized_weight = quantized_weight()
                    dequantized_weight = nn.Parameter(
                        dequantized_weight.to(dtype=straight_through_buffer_dtype),
                        requires_grad=dequantized_weight.requires_grad,
                    )
                else:
                    dequantized_weight = param
                quantized_params[name] = dequantized_weight
                for subparam_name, subparam in quantized_weight.named_parameters():
                    full_name = f"{name}.{subparam_name}"
                    assert full_name not in quantized_representation_params, full_name
                    quantized_representation_params[full_name] = subparam
            elif isinstance(named_quantized_params[name], YourQuantizedWeightIsInAnotherRank):
                assert self.sharded  # running sharded optimizer, this weight should be optimized by another rank
            else:
                raise RuntimeError(f"Unxpected quantized param type {type(named_quantized_params[name])}")

        total_params = len(set(non_quantized_params) | set(quantized_params) | set(quantized_representation_params))
        assert total_params == len(non_quantized_params) + len(quantized_params) + len(quantized_representation_params)
        param_groups = []
        all_optimized_params = dict()
        if update_non_quantized_parameters is not None:
            all_optimized_params.update(non_quantized_params)
            param_groups.append(
                dict(
                    params=list(non_quantized_params.values()),
                    role=ParameterRole.NON_QUANTIZED_PARAMETER,
                    **update_non_quantized_parameters,
                )
            )
        if update_codebooks_and_scales is not None:
            all_optimized_params.update(quantized_representation_params)
            param_groups.append(
                dict(
                    params=list(quantized_representation_params.values()),
                    role=ParameterRole.QUANTIZED_REPRESENTATION_PARAMETER,
                    **update_codebooks_and_scales,
                )
            )
        if update_codes is not None:
            all_optimized_params.update(quantized_params)
            param_groups.append(
                dict(params=list(quantized_params.values()), role=ParameterRole.QUANTIZED_PARAMETER, **update_codes)
            )
        assert len(param_groups) > 0, (
            "Please set at least one of update_codes, update_codebooks_and_scales " "or update_non_quantized_parameters"
        )
        return param_groups, all_optimized_params

    def step(self, *args, **kwargs):
        with print_runtime_stats("_propagate_grads_to_optimized_parameters", enabled=self.verbose):
            self._propagate_grads_to_optimized_parameters()
        with print_runtime_stats("super().step", enabled=self.verbose):
            original_output = super().step(*args, **kwargs)
        with print_runtime_stats("_optimize_quantized_weights", enabled=self.verbose):
            self._optimize_quantized_weights()
        with print_runtime_stats("_update_dequantized_weights", enabled=self.verbose):
            self._update_dequantized_weights()
        return original_output

    def _aggregate_gradients_for_dequantized_weights(self):
        """collect full parameter gradients from fsdp-sharded parameters, return dict[name -> grad]"""
        grad_shards_by_name = dict()

        for name in self.ordered_quantized_weight_names:
            if self.dequantized_weights_by_name[name].grad is None:
                assert self.dequantized_weights_by_name[name].numel() == 0
                self.dequantized_weights_by_name[name].grad = torch.zeros_like(self.dequantized_weights_by_name[name])
            grad = self.dequantized_weights_by_name[name].grad
            assert grad is not None, name
            grad_shards_by_name[name] = grad

        if self.sharded:
            aggregated_grads_by_name = _aggregate_tensors_by_name(
                grad_shards_by_name,
                self.sharded_param_sizes_by_rank,
                self.target_rank_by_name,
                name_order=self.ordered_quantized_weight_names,
            )
        else:
            aggregated_grads_by_name = grad_shards_by_name

        aggregated_grads_by_name = {
            name: grad.view(self.quantized_weights_by_name[name].shape)
            for name, grad in aggregated_grads_by_name.items()
        }
        if self.verbose:
            for name, grad in aggregated_grads_by_name.items():
                print(end=f"aggregated grad norm for {name}: {grad.norm().item()}\n")
        return aggregated_grads_by_name

    def _aggregate_dequantized_weights(self):
        """collect full (possibly optimizer-updated) dequantized weights"""
        if not self.sharded:
            return self.dequantized_weights_by_name
        dequantized_flat_param_shards = {
            name: param.data.flatten() for name, param in self.dequantized_weights_by_name.items()
        }
        flat_aggregated_params_by_name = _aggregate_tensors_by_name(
            dequantized_flat_param_shards,
            self.sharded_param_sizes_by_rank,
            self.target_rank_by_name,
            name_order=self.ordered_quantized_weight_names,
        )
        aggregated_params_by_name = {
            name: param.view(self.quantized_weights_by_name[name].shape)
            for name, param in flat_aggregated_params_by_name.items()
        }
        return aggregated_params_by_name

    @torch.no_grad()
    def _propagate_grads_to_optimized_parameters(self):
        """Ensure that every optimized parameter receives gradient"""
        aggregated_grads_by_name = self._aggregate_gradients_for_dequantized_weights()
        for param_group in self.param_groups:
            for param in param_group["params"]:
                name = self.optimized_param_to_name[param]
                if param_group["role"] == ParameterRole.QUANTIZED_PARAMETER:
                    if self.is_straight_through:
                        assert param is self.straight_through_buffer_by_name[name]
                        # pass gradients to straight-through update buffer or (possibly offloaded) quantized parameter
                        grad_wrt_dequantized_parameter = aggregated_grads_by_name[name]
                        assert grad_wrt_dequantized_parameter.shape == param.shape
                        param.grad = grad_wrt_dequantized_parameter.to(dtype=param.dtype, device=param.device)
                    else:
                        assert len(self.straight_through_buffer_by_name) == 0, self.straight_through_buffer_by_name
                        assert param.grad is not None
                elif param_group["role"] == ParameterRole.NON_QUANTIZED_PARAMETER:
                    assert name not in self.dequantized_weights_by_name and name not in self.quantized_weights_by_name
                elif param_group["role"] == ParameterRole.QUANTIZED_REPRESENTATION_PARAMETER:
                    assert name not in self.dequantized_weights_by_name
                    assert self.should_update_codebooks_and_scales
                    # gradients w.r.t quantized representation parameters are computed below via backprop
                else:
                    raise RuntimeError(f"Unexpected param role: {param_group['role']}")

        if self.should_update_codebooks_and_scales:
            # propagate gradients from dequantized weights to quantization parameters so they can be updated in step;
            # if sharded, every rank propagates gradients only for the QuantizedWeight instances owned by this rank
            with torch.enable_grad():
                for name, quantized_weight in self.quantized_weights_by_name.items():
                    if isinstance(quantized_weight, QuantizedWeightFSDP):
                        quantized_weight.forward().backward(aggregated_grads_by_name[name])

    @torch.no_grad()
    def _optimize_quantized_weights(self):
        """Update discrete state representations to approximate straight through buffers"""
        # note: if sharded, this only updates the subset of quantized weights that are assigned to local rank
        remaining_quantized_weights = {
            name: qw for name, qw in self.quantized_weights_by_name.items() if isinstance(qw, QuantizedWeightFSDP)
        }
        if self.is_straight_through:
            reference_weights_by_name = self.straight_through_buffer_by_name
        else:
            reference_weights_by_name = self._aggregate_dequantized_weights()

        for param_group in self.param_groups:
            if param_group["role"] == ParameterRole.QUANTIZED_PARAMETER:
                for param in param_group["params"]:
                    # param is either a dequantized weight or a special straight-through buffer (if is_straight_through)
                    name = self.optimized_param_to_name[param]
                    quantized_weight = remaining_quantized_weights.pop(name)
                    reference_weight = reference_weights_by_name[name]
                    assert reference_weight.shape == quantized_weight.shape, (
                        reference_weight.shape,
                        quantized_weight.shape,
                    )
                    assert isinstance(quantized_weight, QuantizedWeightFSDP)

                    prev_codes = quantized_weight.get_codes().clone()  # [num_output_groups, num_input_groups]
                    new_codes = quantized_weight.update_codes_(
                        reference_weight=reference_weight,
                        max_update_fraction=self.max_code_change_per_step,
                        dim_rng=random.Random(None),
                    )  # note: this updates quantized_weight codes in-place
                    if self.delta_decay != 0 and self.is_straight_through:
                        self.straight_through_buffer_by_name[name][...] = (
                            self.delta_decay * quantized_weight() + (1 - self.delta_decay) * reference_weight
                        )
                        # if not is_straight_throuh, param will be properly updated in _update_dequantized_weights

                    if self.verbose:
                        code_change_rate = torch.not_equal(prev_codes, new_codes).float().mean().item()
                        maybe_distributed_msg = ""
                        if torch.distributed.is_initialized():
                            maybe_distributed_msg = f" (rank {torch.distributed.get_rank()})"
                        maybe_limit_msg = ""
                        if self.max_code_change_per_step is not None:
                            maybe_limit_msg = f"(limit {self.max_code_change_per_step})"
                        maybe_delta_msg = ""
                        if self.delta_decay != 1:
                            _dequantized_weight = quantized_weight()
                            delta_norm = (reference_weight - _dequantized_weight).norm().item()
                            relative_error = delta_norm / max(_dequantized_weight.norm().item(), 1e-9)
                            maybe_delta_msg = (
                                f"\t||quantized_weight - optimized_weight|| / ||quantized_weight||"
                                f" = {relative_error}\n"
                            )
                        print(
                            end=f"Updated codes for {name}{maybe_distributed_msg}:\n\tFraction of weights with at "
                            f"least one code change: {code_change_rate:.8f} "
                            f"{maybe_limit_msg}\n{maybe_delta_msg}\n"
                        )
        assert len(remaining_quantized_weights) == 0

    @torch.no_grad()
    def _update_dequantized_weights(self):
        """Assign dequantized weight buffers to latest quantized weights after codebook/scale/code updates"""
        own_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        async_ops = list()
        for name in self.ordered_quantized_weight_names:
            quantized_weight = self.quantized_weights_by_name[name]
            dequantized_weight_buffer = self.dequantized_weights_by_name[name]
            dequantized_weight_buffer.fill_(float("nan"))  # this is to ensure that the update reaches the buffer

            if not self.sharded:
                dequantized_weight_buffer[...] = quantized_weight()

            else:
                if isinstance(quantized_weight, QuantizedWeightFSDP):
                    new_dequantized_weight = quantized_weight().to(dequantized_weight_buffer.dtype)
                    shard_sizes: Sequence[int] = self.sharded_param_sizes_by_rank[name]
                    assert sum(shard_sizes) == new_dequantized_weight.numel()
                    new_dequantized_weight_parts = new_dequantized_weight.flatten().split_with_sizes(shard_sizes)
                    for i in range(world_size):
                        if i != own_rank:
                            async_ops.append(torch.distributed.isend(new_dequantized_weight_parts[i], dst=i))
                        else:
                            dequantized_weight_buffer.copy_(new_dequantized_weight_parts[i])

                else:
                    assert isinstance(quantized_weight, YourQuantizedWeightIsInAnotherRank)
                    source_rank = self.quantized_weights_by_name[name].rank
                    async_ops.append(torch.distributed.irecv(dequantized_weight_buffer, src=source_rank))
        for handle in async_ops:
            handle.wait()

    def zero_grad(self, set_to_none: bool = True, *args, **kwargs) -> None:
        super().zero_grad(set_to_none=set_to_none, *args, **kwargs)
        for param in self.dequantized_weights_by_name.values():
            # dequantized weights are not in param_groups, but they still accumulate grads; reset them manually
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def iterate_local_quantized_weights(self) -> Iterator[Tuple[str, QuantizedWeightFSDP]]:
        """Iterate over (name, QuantizedWeight) pairs for all quantized weights trained by this optimizer and rank"""
        for name, quantized_weight in self.quantized_weights_by_name.items():
            if isinstance(quantized_weight, QuantizedWeightFSDP):  # skip YourQuantizedWeightIsInAnotherRank if sharded
                yield name, quantized_weight

    def state_dict(self) -> StateDict:
        state_dict = super().state_dict()
        assert "quantized_weight_state_dicts" not in state_dict
        state_dict["quantized_weight_state_dicts"] = {
            name: quantized_weight.state_dict() for name, quantized_weight in self.iterate_local_quantized_weights()
        }
        state_dict["straight_through_buffers"] = dict(self.straight_through_buffer_by_name)  # may be empty
        # note: the de-quantized params are not saved here; instead, they are saved with model.state_dict
        return state_dict

    def load_state_dict(self, state_dict: StateDict) -> None:
        quantized_weight_state_dicts: Dict[str, StateDict] = dict(state_dict.pop("quantized_weight_state_dicts"))
        for name, quantized_weight in self.iterate_local_quantized_weights():
            quantized_weight.load_state_dict(quantized_weight_state_dicts.pop(name))
        assert len(quantized_weight_state_dicts) == 0, f"unused keys: {quantized_weight_state_dicts.keys()}"

        straight_through_buffers = state_dict.pop("straight_through_buffers")
        assert all(name in straight_through_buffers for name in self.straight_through_buffer_by_name)
        for name, loaded_values in straight_through_buffers.items():
            self.straight_through_buffer_by_name[name][...] = loaded_values
        super().load_state_dict(state_dict)


def _get_sharded_param_sizes_by_rank(named_dequantized_params: Dict[str, torch.Tensor]) -> Dict[str, Sequence[int]]:
    """For each parameter name, return a tuple of sizes (numbers of elements) this parameter across all FSDP ranks"""
    assert torch.distributed.is_initialized()
    own_dequantized_param_shard_size = {name: param.numel() for name, param in named_dequantized_params.items()}
    world_size = torch.distributed.get_world_size()
    gathered_list = [{} for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_list, own_dequantized_param_shard_size)
    assert all(name in sizes_dict for sizes_dict in gathered_list for name in own_dequantized_param_shard_size)
    dequantized_param_sizes_by_rank = dict()
    for name in named_dequantized_params.keys():
        dequantized_param_sizes_by_rank[name] = [gathered_list[rank][name] for rank in range(world_size)]
    return dequantized_param_sizes_by_rank


def _aggregate_tensors_by_name(
    sharded_tensors_by_name: Dict[str, torch.Tensor],
    shard_sizes_by_name: Dict[str, Sequence[int]],
    target_rank_by_name: Dict[str, int],
    name_order: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    :param sharded_tensors_by_name: a dictionary from string to flat (1d) tensors available on the current shard
    :note: the keys should be the same across ranks and go in the same order; if not, use ordered_names
    :param shard_sizes_by_name: a dictionary from name to a list of sizes (numel) for this key across ranks
    :param target_rank_by_name: a dictionary from name to a rank that this name should be aggregated to
    :param name_order: if specified, this defines the order in which devices go over named shards
    """
    assert torch.distributed.is_initialized()
    own_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    aggregated_tensors_by_name = dict()
    async_ops = list()

    for name in sorted(sharded_tensors_by_name.keys()) if name_order is None else name_order:
        shard = sharded_tensors_by_name[name]
        assert shard.ndim == 1
        destination_rank = target_rank_by_name[name]
        shard_sizes: Sequence[int] = shard_sizes_by_name[name]
        if destination_rank == own_rank:
            total_numel = sum(shard_sizes)
            combined_buffer = torch.full((total_numel,), fill_value=torch.nan, dtype=shard.dtype, device=shard.device)
            gather_buffers = list(combined_buffer.split_with_sizes(shard_sizes))
            assert all(
                part.untyped_storage().data_ptr() == combined_buffer.untyped_storage().data_ptr()
                for part in gather_buffers
            )
            for i in range(world_size):
                if shard_sizes[i] == 0:
                    continue  # optimization: this handles FSDP where some param/grad shards are empty
                elif i != own_rank:
                    async_ops.append(torch.distributed.irecv(gather_buffers[i], src=i))
                else:
                    gather_buffers[i].copy_(shard)
            aggregated_tensors_by_name[name] = combined_buffer
        else:
            if shard_sizes[own_rank] == 0:
                continue
            async_ops.append(torch.distributed.isend(shard, destination_rank))

    for handle in async_ops:
        handle.wait()
    return aggregated_tensors_by_name


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


def create_dequantized_model(
    model: transformers.PreTrainedModel, *, reuse_non_quantized: bool, dequantized_dtype: Optional[torch.dtype] = None
) -> transformers.PreTrainedModel:
    """
    Create a version of the model where all QuanizedWeight and derivative layers are de-quantized and cast to dtype.
    :param model: model to be dequantized (out-of-place)
    :param reuse_non_quantized: if True, any non-quantized parameters and buffers are reused for de-quantized model;
        otherwise (default) they are copied and linked in the returned dictionary
    :returns: a model (converted out-of-place) and a mapping (dict) from de-quantized to master parameters
    """
    memo = dict()  # for deepcopy with replacement
    master_parameters = dict()
    all_quantized_weight_parameters = set()

    num_quantized_linear = len([module for module in model.modules() if isinstance(module, QuantizedLinearFSDP)])
    pb = get_progress_bar(num_quantized_linear, "Creating dequantized model")

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinearFSDP):
            assert module not in master_parameters and id(module) not in memo, f"{name} is converted more than once"
            quantized_weight = module.quantized_weight

            dequantized_module = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                dtype=dequantized_dtype if dequantized_dtype is not None else quantized_weight.get_codebooks().dtype,
                device=next(quantized_weight.parameters()).device,
            )
            quantized_weight.to("cuda")
            with torch.no_grad():
                dequantized_module.weight[...] = quantized_weight().cpu()
                dequantized_module.weight.requires_grad = any(p.requires_grad for p in quantized_weight.parameters())

                if module.bias is not None and not reuse_non_quantized:
                    dequantized_module.bias[...] = module.bias
                    dequantized_module.bias.requires_grad = dequantized_module.bias.requires_grad
                elif module.bias is not None and reuse_non_quantized:
                    dequantized_module.bias = module.bias

            memo[id(module)] = dequantized_module
            master_parameters[f"{name}.weight"] = quantized_weight
            if dequantized_module.bias is not module.bias:
                master_parameters[f"{name}.bias"] = module.bias
            all_quantized_weight_parameters |= set(quantized_weight.parameters())
            del quantized_weight
            assert all(
                param in {dequantized_module.weight, dequantized_module.bias}
                for param in dequantized_module.parameters()
            )
            pb.update(1)
    pb.close()

    for name, param_or_buffer in chain(model.named_parameters(), model.named_buffers()):
        if name in master_parameters or param_or_buffer in all_quantized_weight_parameters:
            continue  # parameter already accounted for in the previous loop
        assert name not in master_parameters, name
        assert id(param_or_buffer) not in memo, name
        if reuse_non_quantized:
            new_param_or_buffer = param_or_buffer
        elif isinstance(param_or_buffer, nn.Parameter):
            new_param_or_buffer = nn.Parameter(param_or_buffer.data.clone(), param_or_buffer.requires_grad)
        else:
            new_param_or_buffer = param_or_buffer.detach().clone().requires_grad_(param_or_buffer.requires_grad)
        if new_param_or_buffer is not param_or_buffer:
            master_parameters[name] = new_param_or_buffer
        memo[id(param_or_buffer)] = new_param_or_buffer

    dequantized_model = deepcopy(model, memo=memo)

    for name, module in dequantized_model.named_modules():
        assert not isinstance(module, QuantizedWeightFSDP), (
            f"Dequantized model should not have quantized weights, " f"but found {name} that is {module}"
        )
    if reuse_non_quantized:
        assert all(isinstance(master, QuantizedWeightFSDP) for master in master_parameters.values())
    verify_dequantized_model(dequantized_model, master_parameters)
    return dequantized_model, master_parameters


def verify_dequantized_model(dequantized_model: nn.Module, master_parameters: dict):
    """Test that the dequantized model parameters still match the dequantized_to_master dictionary"""
    unmatched_master_parameters = set(master_parameters.keys())
    for name, param_or_buffer in chain(dequantized_model.named_parameters(), dequantized_model.named_buffers()):
        if name not in master_parameters:
            continue  # non-quantized weight
        master_param_or_buffer = master_parameters[name]
        assert param_or_buffer.shape == master_param_or_buffer.shape
        unmatched_master_parameters.remove(name)
    assert len(unmatched_master_parameters) == 0, f"Found unmatched tensors: {unmatched_master_parameters}"


def get_original_named_parameters_from_fsdp_module(dequantized_model) -> Dict[str, nn.Parameter]:
    return {name.replace("_fsdp_wrapped_module.", ""): param for name, param in dequantized_model.named_parameters()}


@contextlib.contextmanager
def print_runtime_stats(operation_name: str, enabled: bool = True, device: Optional[torch.device] = None):
    if not enabled:
        yield
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if device is None:
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    yield
    if torch.device.type == "cuda":
        torch.cuda.synchronize(device)
    maybe_distributed_msg = f"rank {rank} " if torch.distributed.is_initialized() else ""
    print(end=f"{maybe_distributed_msg}{operation_name} took {time.perf_counter() - start_time}\n")


def split_quantized_weights_between_ranks(quantized_weights: Dict[str, QuantizedWeightFSDP], verify_checksums: bool):
    """
    Split all quantized weights between ranks in a distributed setup; uses greedy knapsack heuristic.
    Note that unlike FSDP, this heuristic will always assign the entire quantized weight to one rank.

    :param quantized_weights: a dictionary [parameter_name] -> QuantizedWeight
    :returns: a dictionary similar to quantized weights or pointers to different ranks.
        If your rank stores this quantized weight for [name], then returned_dict[name] is quantized_weights[name]
        Otherwise, returned_dict[name] = YourQuantizedWeightIsInAnotherRank(rank=where_it_is_stored)
    :param verify_checksums: if True, synchronize with other ranks and verify that parameters are split consistently.
        If False, do not synchronize, but instead print a hash of checksum for each rank to be verified by the user.
    """
    assert torch.distributed.is_initialized()
    own_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    all_quantized_weights: Dict[QuantizedWeightFSDP, List[str]] = defaultdict(list)
    for name, quantized_weight in quantized_weights.items():
        all_quantized_weights[quantized_weight].append(name)

    # order quantized weights in a rank-agnostic way: order by (param size desc, linked param name asc)
    def _compute_size(qw: QuantizedWeightFSDP) -> float:
        return qw.out_features * qw.in_features * qw.estimate_nbits_per_parameter()

    ordered_quantized_weights = sorted(
        all_quantized_weights, key=lambda qw: (-_compute_size(qw), min(all_quantized_weights[qw]))
    )
    assert len(ordered_quantized_weights) > 0, "internal error: could not find any linked QuantizedWeight in state"

    # split between ranks
    quantized_weight_to_rank = dict()
    total_size_by_rank = [0 for _ in range(world_size)]
    for quantized_weight in ordered_quantized_weights:
        least_busy_rank = min(range(world_size), key=lambda rank: total_size_by_rank[rank])
        total_size_by_rank[least_busy_rank] += _compute_size(quantized_weight)
        quantized_weight_to_rank[quantized_weight] = least_busy_rank

    checksum = tuple(
        (min(all_quantized_weights[qw]), quantized_weight_to_rank[qw], _compute_size(qw))
        for qw in ordered_quantized_weights
    )
    if verify_checksums:
        checksums = [() for _ in range(world_size)]
        torch.distributed.all_gather_object(checksums, checksum)
        assert checksums[own_rank] == checksum, (checksums, own_rank, checksum)
        assert all(other_checksum == checksum for other_checksum in checksums), checksums
    else:
        hashing = hashlib.sha256()
        hashing.update(json.dumps(checksum).encode())
        print(end=f"Splitting quantized weights, rank {own_rank} checksum hash: {hashing.hexdigest()}\n")

    sharded_quantized_weights = dict()
    for name, quantized_weight in list(quantized_weights.items()):
        target_rank = quantized_weight_to_rank[quantized_weight]
        if target_rank == own_rank:
            sharded_quantized_weights[name] = quantized_weight
        else:
            sharded_quantized_weights[name] = YourQuantizedWeightIsInAnotherRank(target_rank)
    return sharded_quantized_weights
