# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import transformers
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from plugin import *

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = None
    model_name: Optional[str] = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        assert name in transformer_configs, f"Unknown model name: {name}, available: {transformer_configs.keys()}"
        return cls(**transformer_configs[name])

transformer_configs = {
    "meta-llama/Meta-Llama-3-8B": dict(model_name="Meta-Llama-3-8B", block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "meta-llama/Meta-Llama-3-8B-Instruct": dict(model_name="Meta-Llama-3-8B-Instruct", block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "meta-llama/Meta-Llama-3.1-8B": dict(model_name="Meta-Llama-3.1-8B", block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": dict(model_name="Meta-Llama-3.1-8B-Instruct", block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "meta-llama/Llama-2-7b": dict(model_name="Llama-2-7b", block_size=4096, n_layer=32, n_head=32, n_local_heads=32, dim=4096, intermediate_size=11008, vocab_size=32000, rope_base=10000),
    "meta-llama/Llama-2-13b": dict(model_name="Llama-2-13b", block_size=4096, n_layer=40, n_head=40, n_local_heads=40, dim=5120, intermediate_size=13824, vocab_size=32000, rope_base=10000),
    "meta-llama/Llama-2-70b": dict(model_name="Llama-2-70b", block_size=4096, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=32000, rope_base=10000),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.half):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class Transformer(nn.Module):
    def __init__(self, dtype, config: ModelArgs, linear_class=nn.Linear, linear_kwargs=None, halve_layers=False, fuse_linears=True) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype

        # if halve_layers, halve the number of layers for testing purposes
        if halve_layers:
            config.n_layer = config.n_layer // 2

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, linear_class, linear_kwargs, fuse_linears) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.cache_initialized = False

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)

        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.cache_initialized = True

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.cache_initialized, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, dtype, name: str, linear_class=nn.Linear, linear_kwargs=None, halve_layers=False, fuse_linears=True) -> "Transformer":
        return cls(dtype, ModelArgs.from_name(name), linear_class=linear_class, linear_kwargs=linear_kwargs, halve_layers=halve_layers, fuse_linears=fuse_linears)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, linear_class=nn.Linear, linear_kwargs=None, fuse_linears=True) -> None:
        super().__init__()
        self.attention = Attention(config, linear_class, linear_kwargs, fuse_linears)
        self.feed_forward = FeedForward(config, linear_class, linear_kwargs, fuse_linears)

        if "llama" in config.model_name.lower():
            self.input_layernorm = RMSNorm(config.dim, config.norm_eps)
            self.post_attention_layernorm = RMSNorm(config.dim, config.norm_eps)
            self.pre_feedforward_layernorm = None
            self.post_feedforward_layernorm = None
        else:
            raise NotImplementedError

    def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(
                                self.input_layernorm(x), 
                                mask, input_pos
                                )

        if self.pre_feedforward_layernorm != None:
            h = self.pre_feedforward_layernorm(h)
        
        out = self.feed_forward(self.post_attention_layernorm(h))

        if self.post_feedforward_layernorm != None:
            out = self.post_feedforward_layernorm(out)

        out = h + out 

        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, linear_class=nn.Linear, linear_kwargs=None, fuse_linears=True) -> None:
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        if fuse_linears:
            self.wqkv = linear_class(config.dim, total_head_dim, bias=False, **(linear_kwargs or {}))
        else:
            self.wq = linear_class(config.dim, config.n_head*config.head_dim, bias=False, **(linear_kwargs or {}))
            self.wk = linear_class(config.dim, config.n_local_heads*config.head_dim, bias=False, **(linear_kwargs or {}))
            self.wv = linear_class(config.dim, config.n_local_heads*config.head_dim, bias=False, **(linear_kwargs or {}))

        self.wo = linear_class(config.dim, config.dim, bias=False, **(linear_kwargs or {}))

        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.config = config
        self.fuse_linears = fuse_linears

        self.scaling = 1/ math.sqrt(config.head_dim)

        if "llama" in config.model_name.lower():
            self.rotary_emb = LlamaRotaryEmbedding(
                    dim=self.head_dim,
                    max_position_embeddings=config.block_size,
                    base=config.rope_base
            )
            self.sdpa_scaling = None
        else:
            raise NotImplementedError

    def forward(self, x: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        if self.fuse_linears:
            q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        else:
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        
        cos, sin = self.rotary_emb(v, input_pos.unsqueeze(0))
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0) #, scale=self.sdpa_scaling)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        y = self.wo(y)

        return y



class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs, linear_class=nn.Linear, linear_kwargs=None, fuse_linears=True) -> None:
        super().__init__()
        self.config = config
        self.fuse_linears = fuse_linears

        if fuse_linears:
            self.w1w3 = linear_class(config.dim, config.intermediate_size*2, bias=False, **(linear_kwargs or {}))
        else:
            self.w1 = linear_class(config.dim, config.intermediate_size, bias=False, **(linear_kwargs or {}))
            self.w3 = linear_class(config.dim, config.intermediate_size, bias=False, **(linear_kwargs or {}))
        self.w2 = linear_class(config.intermediate_size, config.dim, bias=False, **(linear_kwargs or {}))

        self.act_fn = F.silu
        
        self.fuse_linears = fuse_linears

    def forward(self, x: Tensor) -> Tensor:
        if self.fuse_linears:
            w1_out, w3_out = self.w1w3(x).split([self.config.intermediate_size, self.config.intermediate_size], dim=-1)
        else:
            w1_out = self.w1(x)
            w3_out = self.w3(x)

        return self.w2(self.act_fn(w1_out) * w3_out)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaRotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs)
            self.register_buffer(
                "inv_freq", inv_freq,
                persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq",
                                 self.original_inv_freq,
                                 persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(
            device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float()
                     @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
