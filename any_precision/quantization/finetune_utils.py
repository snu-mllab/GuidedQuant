""" Codes adopted from https://github.com/Vahe1994/AQLM """
from __future__ import annotations

from typing import List, Optional, Tuple, Union, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm.auto import trange

from numpy import ndarray
import numpy as np
from tqdm.auto import tqdm
import random


def _dequantize_weight(
    codes: torch.Tensor, # [num_rows, group_count, group_size]
    codebooks: torch.Tensor, # [num_rows, group_count, bit_size]
) -> torch.Tensor:

    num_rows, group_count, group_size = codes.shape
    num_rows, group_count, bit_size = codebooks.shape
    num_groups = num_rows *group_count

    codes = codes.reshape(num_groups, group_size)
    codebooks = codebooks.reshape(num_groups, bit_size)

    codes = codes.long()
    
    # Expand C and gather original weights
    C_expanded_org = codebooks.unsqueeze(1).expand(-1, group_size, -1)
    W_hat = torch.gather(C_expanded_org, dim=2, index=codes.unsqueeze(-1)).squeeze(-1) # Shape: (num_rows * group_count, group_size)
    W_hat = W_hat.reshape(num_rows, group_count * group_size)

    return W_hat


def minimize_weight_mse(
    reference_weight: torch.Tensor,
    codebooks: torch.Tensor,
    prev_codes: torch.Tensor,
    chunk_size_bytes: int = 2**32,
    dim_rng:  random.Random = None,
    max_update_fraction: float = 1.0,
) -> torch.Tensor:
    """
    Args:
        reference_weight: [num_rows, num_cols] float tensor of the desired weight values.
        codebooks:        [num_rows, 1, codebook_size] float tensor of possible values per row.
        prev_codes:       [num_rows, 1, num_cols] uint8/int tensor of old chosen code indices.
        chunk_size_bytes: process columns in chunks if needed to save memory.
        dim_rng:          optional PRNG to shuffle update order of positions.
        max_update_fraction: fraction of positions (row,col) allowed to update (e.g. 0.5 => only half).

    Returns:
        new_codes: [num_rows, 1, num_cols] integer tensor of updated code indices.
    """

    # 1) Dequantize the current (old) codes
    # codebooks[row, 0, :] => possible values for that row
    # prev_codes[row, 0, col] => chosen code index
    def _dequantize(codes: torch.Tensor) -> torch.Tensor:
        # codebooks_expanded: [num_rows, num_cols, codebook_size]
        # gather(...) => picks shape [num_rows, num_cols]
        num_rows, _, codebook_size = codebooks.shape
        _, _, num_cols = codes.shape

        codebooks_expanded = codebooks.expand(-1, num_cols, -1)
        codes = codes.long()
        # codes.unsqueeze(-1): [num_rows, num_cols, 1]
        # gather => [num_rows, num_cols, 1]
        gathered = torch.gather(
            codebooks_expanded, dim=2, index=codes.squeeze(1).unsqueeze(-1)
        )
        return gathered.squeeze(-1)  # [num_rows, num_cols]

    old_deq = _dequantize(prev_codes)

    # 2) Compute error for each position => used to pick which ones to update
    diff = reference_weight - old_deq  # [num_rows, num_cols]
    sq_err = diff.square()            # squared error per position
    flat_sq_err = sq_err.flatten()    # shape [num_rows * num_cols]

    total_positions = flat_sq_err.numel()
    max_updates = int(math.ceil(max_update_fraction * total_positions))

    # 3) Choose which positions to update: top-K or sample by error^(1/T)
    if max_updates < total_positions:
        # pick top-K largest errors
        indices_to_update = torch.topk(flat_sq_err, k=max_updates, largest=True).indices
    else:
        # update all positions
        indices_to_update = torch.arange(total_positions, device=flat_sq_err.device)

    # Optionally shuffle the update order if dim_rng is given
    if dim_rng is not None:
        indices_list = indices_to_update.tolist()
        dim_rng.shuffle(indices_list)
        indices_to_update = torch.tensor(indices_list, device=indices_to_update.device)

    # 4) Function to compute new codes for each position in pos_indices
    def _update_positions(pos_indices: torch.Tensor) -> torch.Tensor:
        """
        pos_indices: 1D array of positions, where each pos = row * num_cols + col
        returns new_codes for those positions (1D)
        """
        device = pos_indices.device
        num_cols = reference_weight.size(1)

        rows = pos_indices // num_cols
        cols = pos_indices % num_cols

        # Gather codebooks for these rows: shape [N, codebook_size]
        row_codebooks = codebooks[rows, 0, :]   # [N, codebook_size]
        ref_vals      = reference_weight[rows, cols]  # [N]

        # Distances: (ref - codebooks)^2 => [N, codebook_size]
        dist = (ref_vals.unsqueeze(1) - row_codebooks).square()
        best_dist, best_idx = dist.min(dim=1)

        # Otherwise always pick best_idx
        return best_idx.to(torch.uint8)

    # 5) Update in chunks to avoid OOM if big
    new_codes = prev_codes.clone()
    element_size = reference_weight.element_size()
    chunk_size_vals = chunk_size_bytes // element_size
    start = 0
    while start < indices_to_update.numel():
        end = min(start + chunk_size_vals, indices_to_update.numel())
        batch_idx = indices_to_update[start:end]
        updated_1d_codes = _update_positions(batch_idx)

        # Place those codes back
        num_cols = reference_weight.size(1)
        rows = batch_idx // num_cols
        cols = batch_idx % num_cols
        new_codes[rows, 0, cols] = updated_1d_codes
        start = end

    return new_codes


class IntCodes(nn.Module):
    """
    A storage for integer codes that makes them compatible with FullyShardedDataParallel,
    see https://github.com/pytorch/pytorch/issues/123528 for details
    """

    def __init__(self, codes: torch.tensor, storage_dtype: torch.dtype = torch.float64):
        super().__init__()
        assert torch.finfo(storage_dtype).bits % torch.iinfo(codes.dtype).bits == 0
        self.dtype, self.shape, self.numel = codes.dtype, codes.shape, codes.numel()
        size_ratio = torch.finfo(storage_dtype).bits // torch.iinfo(codes.dtype).bits
        codes = F.pad(codes.flatten().clone(), pad=[0, -codes.numel() % size_ratio])
        assert len(codes.untyped_storage()) == codes.nbytes  # no offset / stride / tail
        self.storage_dtype = storage_dtype
        self.data = nn.Parameter(
            torch.as_tensor(codes.untyped_storage(), device=codes.device, dtype=storage_dtype), requires_grad=False
        )

    def forward(self):
        assert self.data.is_contiguous() and self.data.dtype == self.storage_dtype
        byte_offset = self.data.storage_offset() * self.data.nbytes // self.data.numel()
        return torch.as_tensor(
            self.data.untyped_storage()[byte_offset : byte_offset + self.data.nbytes],
            device=self.data.device,
            dtype=self.dtype,
        )[: self.numel].view(*self.shape)


class QuantizedLinearFSDP(nn.Module):
    def __init__(self, quantized_weight: QuantizedWeightFSDP, bias: Optional[nn.Parameter]=None):
        super().__init__()
        self.out_features, self.in_features = quantized_weight.out_features, quantized_weight.in_features
        self.quantized_weight: QuantizedWeightFSDP = quantized_weight
        self.bias = bias
        self.use_checkpoint = False

    def _forward(self, input: torch.Tensor):
        return F.linear(input, self.quantized_weight(), self.bias)

    def forward(self, input: torch.Tensor):
        if getattr(self, "use_checkpoint", False) and torch.is_grad_enabled():
            return checkpoint(
                self._forward, input, use_reentrant=False, preserve_rng_state=False, determinism_check="none"
            )
        return self._forward(input)
    
    def project_weight(self, use_gradient: bool=True, num_iter: int=10):
        if use_gradient:
            gradient = self.quantized_weight.weight.grad
            assert gradient is not None
        else:
            gradient = None
        self.quantized_weight.project_weight(gradient, num_iter=num_iter)


class QuantizedWeightFSDP(nn.Module):
    """
    A quantized weight module that is compatible with FullyShardedDataParallel.
    """
    def __init__(
        self,
        codes: Union[ndarray, torch.Tensor],
        codebooks: Union[ndarray, torch.Tensor],
        code_dtype: torch.dtype = torch.uint8,
    ):
        super().__init__()

        assert isinstance(codes, torch.Tensor) == isinstance(codebooks, torch.Tensor), "Codes and codebooks must be of the same type"

        if isinstance(codes, torch.Tensor):
            tensor_codebooks = codebooks.detach().clone()
            tensor_codes = codes.detach().clone()
        else:
            tensor_codebooks = torch.from_numpy(codebooks)
            tensor_codes = torch.from_numpy(codes)

        num_rows, group_count, num_bits = tensor_codebooks.shape
        num_rows, group_count, group_size = tensor_codes.shape

        self.codebooks = nn.Parameter(tensor_codebooks, requires_grad=True)
        self.codes: Optional[nn.Parameter] = nn.Parameter(
            tensor_codes.to(code_dtype), requires_grad=False
        )
        self.codes_storage: Optional[IntCodes]= None # Storage for FSDP compatibility

        self.num_bits = num_bits.bit_length() - 1

        self.out_features = num_rows
        self.in_features = group_count * group_size

        self.avg_bits = self.estimate_nbits_per_parameter()

    @classmethod
    def from_quantized_weight(cls, codes:ndarray, codebooks:ndarray):
        return cls(codes, codebooks)

    def get_codes(self) -> torch.IntTensor:
        """Get a non view to codes, regardless of how codes are stored"""
        assert (self.codes is None) != (self.codes_storage is None), "must have either .codes or storage, but not both"
        codes = self.codes if self.codes is not None else self.codes_storage()
        return codes

    def set_codes(self, new_codes: torch.Tensor, selection: Union[slice, ellipsis, torch.Tensor] = ..., **kwargs):
        """Update codes[selection] to new_codes, regardless of their dtype and whether they are wrapped as storage"""
        assert (self.codes is None) != (self.codes_storage is None), "must have either .codes or storage, but not both"
        codes_ptr = self.codes if self.codes is not None else self.codes_storage()
        codes_ptr[selection].copy_(new_codes, **kwargs)

    def wrap_codes_for_fsdp_(self, **kwargs):
        """Make this module compatible with FullyShardedDataParallel; modifies state dict in-place"""
        assert self.codes is not None and self.codes_storage is None
        self.codes_storage, self.codes = IntCodes(self.codes, **kwargs), None

    def unwrap_codes_(self):
        """Undo the effect of wrap_codes_for_fsdp_; modifies state dict in-place"""
        assert self.codes is None and self.codes_storage is not None
        self.codes, self.codes_storage = nn.Parameter(self.codes_storage(), requires_grad=False), None

    def get_codebooks(self) -> torch.Tensor:
        """Get quantization codebooks or reconstruct them from second level quantization (see codebook_values_nbits)"""
        return self.codebooks

    @property
    def shape(self) -> Tuple[int, int]:
        return self.out_features, self.in_features

    def forward(self, selection: Union[slice, ellipsis, torch.Tensor] = ...):
        """
        Differentably reconstruct the weight (or parts thereof) from compressed components
        :param selection: By default, reconstruct the entire weight. If selection is specified, this method will instead
            reconstruct a portion of weight for the corresponding output dimensions (used for parallelism).
            The indices / slices must correspond to output channels (if out_group_size==1) or groups (if > 1).
            Formally, the indices must be in range [ 0 , self.out_features // self.out_group_size )

        """
        W_hat = _dequantize_weight(self.get_codes()[selection], self.get_codebooks()[selection])
        return W_hat

    @torch.no_grad()
    def update_codes_(
        self,
        *,
        reference_weight: torch.Tensor,
        selection: Union[slice, ellipsis, torch.LongTensor] = ...,
        **kwargs,
    ) -> torch:
        """
        Update own codes in-place via beam search so as to minimize squared errors. Return the updated codes.
        :param reference_weight: original weight matrix that is being quantized, shape: [out_features, in_features]
        :note: if selection is specified, reference_weight must instead be [num_selected_out_features, in_features]
        :param selection:  By default, this function updates all codes, If selection specified, it will instead
            update only the codes for a portion of output dimensions (used for parallelism).
            The indices / slices must correspond to output channels (if out_group_size==1) or groups (if > 1).
            Formally, the indices must be in range [ 0 , self.out_features // self.out_group_size )
        :param kwargs: any additional keyword arguments are forwarded to beam_search_optimal_codes function
        :returns: the updated codes, in the same shape as self.get_codes()[selection]
        """
        codebooks = self.get_codebooks()[selection]
        prev_codes = self.get_codes()[selection]

        new_codes = minimize_weight_mse(
            reference_weight=reference_weight, codebooks=codebooks, prev_codes=prev_codes, **kwargs
        )
        self.set_codes(new_codes, selection)
        return new_codes


    def estimate_nbits_per_parameter(self) -> float:
        """Calculate the effective number of bits per original matrix parameters"""

        num_parameters = self.out_features * self.in_features

        codes_bits = self.get_codes().numel() * self.num_bits
        codebooks_bits = self.get_codebooks().numel() * 16

        return (codes_bits + codebooks_bits) / num_parameters

    def extra_repr(self) -> str:
        return f"{self.out_features=}, {self.in_features=}, bits_per_parameter={self.avg_bits:.2f}"
