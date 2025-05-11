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

from plugin import *

import transformers

from tqdm import tqdm

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
        """
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
            
        return cls(**transformer_configs[config[0]])
        """


transformer_configs = {
    "Meta-Llama-3-8B-Instruct": dict(model_name="Meta-Llama-3-8B-Instruct", block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "Meta-Llama-2-7B": dict(model_name="Meta-Llama-2-7B", block_size=4096, n_layer=32, n_head=32, n_local_heads=32, dim=4096, intermediate_size=11008, vocab_size=32000, rope_base=10000),
    "Meta-Llama-2-13B": dict(model_name="Meta-Llama-2-13B", block_size=4096, n_layer=40, n_head=40, n_local_heads=40, dim=5120, intermediate_size=13824, vocab_size=32000, rope_base=10000),
    "Meta-Llama-2-70B": dict(model_name="Meta-Llama-2-70B", block_size=4096, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=32000, rope_base=10000),
    "Phi-3-medium-4k-instruct": dict(model_name='Phi-3-medium-4k-instruct', block_size=4096, n_layer=40, n_head=40, n_local_heads=10, dim=5120, intermediate_size=17920, vocab_size=32064, rope_base=10000),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.half):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        #cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)
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
    def __init__(self, dtype, config: ModelArgs, linear_class=nn.Linear, linear_kwargs=None, halve_layers=False) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype

        # if halve_layers, halve the number of layers for testing purposes
        if halve_layers:
            config.n_layer = config.n_layer // 2

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, linear_class, linear_kwargs) for _ in range(config.n_layer))
        if "phi" in config.model_name.lower():
            self.norm = Phi3RMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

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

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, head_dim, self.config.rope_base, dtype, self.config.rope_scaling)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, dtype, name: str, linear_class=nn.Linear, linear_kwargs=None, halve_layers=False) -> "Transformer":
        return cls(dtype, ModelArgs.from_name(name), linear_class=linear_class, linear_kwargs=linear_kwargs, halve_layers=halve_layers)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, linear_class=nn.Linear, linear_kwargs=None) -> None:
        super().__init__()
        self.attention = Attention(config, linear_class, linear_kwargs)
        self.feed_forward = FeedForward(config, linear_class, linear_kwargs)

        if "llama" in config.model_name.lower():
            self.input_layernorm = RMSNorm(config.dim, config.norm_eps)
            self.post_attention_layernorm = RMSNorm(config.dim, config.norm_eps)
            self.pre_feedforward_layernorm = None
            self.post_feedforward_layernorm = None
        elif "phi" in config.model_name.lower():
            self.input_layernorm = Phi3RMSNorm(config.dim, config.norm_eps)
            self.post_attention_layernorm = Phi3RMSNorm(config.dim, config.norm_eps)
            self.pre_feedforward_layernorm = None
            self.post_feedforward_layernorm = None

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(
                                self.input_layernorm(x), 
                                freqs_cis, mask, input_pos
                                )

        if self.pre_feedforward_layernorm != None:
            h = self.pre_feedforward_layernorm(h)
        
        out = self.feed_forward(self.post_attention_layernorm(h))

        if self.post_feedforward_layernorm != None:
            out = self.post_feedforward_layernorm(out)

        out = h + out 

        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, linear_class=nn.Linear, linear_kwargs=None) -> None:
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        
        self.wq = linear_class(config.dim, config.n_head*config.head_dim, bias=False, **(linear_kwargs or {}))
        self.wk = linear_class(config.dim, config.n_local_heads*config.head_dim, bias=False, **(linear_kwargs or {}))
        self.wv = linear_class(config.dim, config.n_local_heads*config.head_dim, bias=False, **(linear_kwargs or {}))
        self.wo = linear_class(config.dim, config.dim, bias=False, **(linear_kwargs or {}))

        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.config = config

        self.scaling = 1/ math.sqrt(config.head_dim)

        if "phi" in config.model_name.lower():
            self.rotary_emb = Phi3RotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=config.block_size,
                    base=config.rope_base
            )
            self.sdpa_scaling = None
        else:
            self.rotary_emb = None
            self.sdpa_scaling = None


    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])
    
    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        
        if self.rotary_emb == None:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)
        else:
            cos, sin = self.rotary_emb(v, input_pos.unsqueeze(0))
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0) #, scale=self.sdpa_scaling)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        y = self.wo(y)

        return y



class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs, linear_class=nn.Linear, linear_kwargs=None) -> None:
        super().__init__()
        self.config = config
        self.w1 = linear_class(config.dim, config.intermediate_size, bias=False, **(linear_kwargs or {}))
        self.w3 = linear_class(config.dim, config.intermediate_size, bias=False, **(linear_kwargs or {}))
        self.w2 = linear_class(config.intermediate_size, config.dim, bias=False, **(linear_kwargs or {}))

        self.act_fn = F.silu

    def forward(self, x: Tensor) -> Tensor:
        w1_out = self.w1(x)
        w3_out = self.w3(x)
        return self.w2(self.act_fn(w1_out) * w3_out)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class Phi3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Gemma2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Phi3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Phi3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

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


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

