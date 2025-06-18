# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch._dynamo.config
import torch._inductor.config

from transformers import AutoTokenizer
from APLinear import APLinear
from LUTGEMMLinear import LUTGEMMLinear
from lib.linear.quantized_linear import QuantizedLinear as QTIPLinear

import warnings

warnings.filterwarnings(
    "ignore", 
    category=FutureWarning
)

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
#torch._functorch.config.enable_autograd_cache = True

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

#from model_no_fuse import Transformer
from model import Transformer

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits=logits.float()
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_one_token_inplace(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, next_token: torch.Tensor, next_prob: torch.Tensor, **sampling_kwargs):
    next_token[...], next_prob[...] = decode_one_token(model, x, input_pos, **sampling_kwargs)


def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, use_graph=False, callback=lambda _: _, **sampling_kwargs):
    # WARNING: DO NOT pass use_graph=True for the first invocation after torch.compile, as torch.compile will only successfully compile with use_graph=False
    # if you're using torch.compile without CUDA graphs, you may set use_graph to True after the first invocation to speed up subsequent calls
    if use_graph:
        # Allocate static input tensors
        static_cur_token = cur_token.clone()
        static_input_pos = input_pos.clone()

        # Set requires_grad=False for inference
        static_cur_token.requires_grad = False
        static_input_pos.requires_grad = False

        # Pre-allocate static output tensors
        static_next_token = torch.empty_like(static_cur_token)
        static_next_prob = torch.empty((1, model.config.vocab_size), dtype=torch.float32, device='cuda')

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode_one_token_inplace(
                model, static_cur_token, static_input_pos, static_next_token, static_next_prob, **sampling_kwargs
            )
        torch.cuda.synchronize()

    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            if use_graph:
                # Update inputs in-place
                static_cur_token.copy_(cur_token)
                static_input_pos.copy_(input_pos)
                graph.replay()
                # Retrieve outputs from static output tensors
                next_token = static_next_token.clone()
                next_prob = static_next_prob.clone()
            else:
                next_token, next_prob = decode_one_token(
                    model, cur_token, input_pos, **sampling_kwargs
                )
            
            # Update position and tokens
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.clone()
    torch.cuda.synchronize()

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device, dtype=torch.int32)

    if T != 1:
        next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
        seq[:, T] = next_token.squeeze()
        input_pos = torch.tensor([T], device=device, dtype=torch.int).view(1)
        generated_tokens, new_probs = decode_n_tokens(model, next_token.view(batch_size, -1),
                          input_pos, max_new_tokens-1, callback=callback, use_graph=False, **sampling_kwargs)
        seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)

    else:
        generated_tokens, new_probs = decode_n_tokens(model, prompt.view(batch_size, -1),
                              input_pos, max_new_tokens, callback=callback, **sampling_kwargs)
        peak_memory = torch.cuda.max_memory_allocated()
        seq[:, 1:] = torch.cat(generated_tokens, dim=-1)

    return seq

def encode_tokens(tokenizer, string, device=default_device):
    tokens = tokenizer.encode(string)
    return torch.tensor(tokens, dtype=torch.int, device=device)

def encode_bos(tokenizer, device=default_device):
    return torch.tensor([tokenizer.bos_token_id], dtype=torch.int, device=device)

def load_model(model_name, device, backend,  
                bitwidth, random_init,
                checkpoint_path, 
                config_path=None, dtype=None, halve_layers=False):
    use_cuda = 'cuda' in device

    linear_kwargs = {}
    match backend:
        case "ap":
            linear_class = APLinear
            linear_kwargs["bitwidth"] = bitwidth
        case "lutgemm":
            linear_class = LUTGEMMLinear
            linear_kwargs["bitwidth"] = bitwidth
            linear_kwargs['group_size'] = -1
        case "qtip":
            import json
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
            linear_class = QTIPLinear
            linear_kwargs['td_x'] = config['quip_params']['td_x']
            linear_kwargs['td_y'] = config['quip_params']['td_y']
            linear_kwargs['L'] = config['quip_params']['L']
            linear_kwargs['K'] = config['quip_params']['K']
            linear_kwargs['V'] = config['quip_params']['V']
            linear_kwargs['tlut_bits'] = config['quip_params']['tlut_bits']
            linear_kwargs['decode_mode'] = config['quip_params']['decode_mode']
        case None:
            linear_class = nn.Linear
            assert (bitwidth == 16)

    print("Building model ...", flush=True)
    model = Transformer.from_name(
        name=model_name, dtype=dtype,
        linear_class=linear_class,
        linear_kwargs=linear_kwargs,
        halve_layers=halve_layers,
    )

    if not random_init:
        print("Loading weights ...", flush=True)
        checkpoint = torch.load(os.path.join(checkpoint_path, "converted_pytorch_model.bin"), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True, strict=False if (backend == "qtip") else True)

    print("Dispatching model to device ...", flush=True)
    model=model.to(device=device, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
   
    print("Model loaded.", flush=True)
    return model.eval(), tokenizer

def _get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            size = sum(p.numel() * p.dtype.itemsize for p in itertools.chain(child.parameters(), child.buffers()))

    return model_size, params

B_INST, E_INST = "[INST]", "[/INST]"

def main(
    prompt: str = None,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    compile: int = 2,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    device=default_device,
    model_name = None,
    backend = None,
    bitwidth = None,
    checkpoint_path = None,
    config_path = None,
    dtype = None,
    print_result = False,
    random_init = False,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    print(f"Using device={device}")
    if (dtype == "float16"): 
        dtype = torch.float16
    elif (dtype == "bfloat16"): 
        dtype = torch.bfloat16
    elif (dtype == "float32"): 
        dtype = torch.float32

    t0 = time.time()
    model, tokenizer = load_model(model_name, device, backend,
                                  bitwidth, random_init, 
                                  checkpoint_path, 
                                  config_path, dtype)

    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds", flush=True)

    # encode prompt (bos)
    if prompt != None:
        encoded = encode_tokens(tokenizer, prompt, device=device)
    else:
        encoded = encode_bos(tokenizer, device=device)
    prompt_length = encoded.size(-1)

    torch.manual_seed(1234)
    model_size, params = _get_model_size(model)

    if compile:
        global decode_one_token, prefill
        mode = 'max-autotune-no-cudagraphs' if compile == 1 else 'max-autotune'
        decode_one_token = torch.compile(decode_one_token, mode=mode, fullgraph=True, dynamic=False)
        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
        
    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        callback = lambda x : x
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        import contextlib
        if (i != num_samples - 1 or not profile):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y = generate(
                model,
                encoded,
                max_new_tokens,
                batch_size=batch_size,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
            )
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds", flush=True)
            continue

        torch.cuda.synchronize()
        time_elapsed = time.perf_counter() - t0

        if print_result:
            print(tokenizer.decode(y[0].tolist()))
        
        tokens_generated = y.size(-1) - prompt_length
        generated_tokens_sec = tokens_generated / time_elapsed
        aggregate_metrics['tokens_per_sec'].append(generated_tokens_sec)
        if i + 1 == num_samples:
            print(f"Time for inference {i + 1}: {time_elapsed:.02f} sec total, {generated_tokens_sec:.02f} tokens/sec")
            print(f"Bandwidth achieved: {model_size * generated_tokens_sec / 1e9:.02f} GB/s")
            total_tokens_sec = y.numel() / time_elapsed
            print(f"FLOPS achieved: {params * total_tokens_sec * 2 / 1e12:.02f} TF/s")
            print(flush=True)
    print("==========")

    print(f"Batch Size: {batch_size}")
    print(f"Prompt Length: {prompt_length}")
    print(f"Generated tokens: {max_new_tokens}")
    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print(f"Std of tokens/sec: {torch.std(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--prompt', type=str, default=None, help="Input prompt. Set to None to use only the bos token for benchmarking")
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of new tokens.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to benchmark with')
    parser.add_argument('--top_k', type=int, default=32, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for sampling.')
    parser.add_argument('--compile', type=int, default=2, help='Whether to compile the model')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--model_name', type=str, default=None, help='model_name', 
                        choices=['Meta-Llama-3-8B-Instruct', 'Meta-Llama-2-7B', 'Meta-Llama-2-13B', 'Meta-Llama-2-70B'])
    parser.add_argument('--bitwidth', type=int, default=None, help='bitwidth', choices=[2,3,4,16])
    parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint path')
    parser.add_argument('--config_path', type=str, default=None, help='QTIP config path')
    parser.add_argument('--dtype', type=str, default="float16", help='dtype', choices=["float16", "float32", "bfloat16"])
    parser.add_argument('--backend', type=str, default=None, help='quantization backend to use', choices=["ap", "lutgemm", "qtip", None])
    parser.add_argument('--print_result', action='store_true')
    parser.add_argument('--random_init', action='store_true')

    args = parser.parse_args()

    main(
        args.prompt, args.num_samples, args.max_new_tokens, args.batch_size, args.top_k,
        args.temperature, args.compile, args.compile_prefill, args.profile, 
        args.device, args.model_name, args.backend, args.bitwidth, 
        args.checkpoint_path, args.config_path, 
        args.dtype, args.print_result, args.random_init
    )

