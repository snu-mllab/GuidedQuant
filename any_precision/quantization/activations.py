import torch
import torch.nn as nn
from tqdm.auto import trange
import os
import logging
from .config import *
from .utils import get_progress_bar
from any_precision.analyzer.analyzer import ModelAnalyzer
from typing import List, Tuple, Literal, Optional, Sequence, Dict, Callable, Iterator, Any
from itertools import chain
from tqdm import tqdm


@torch.no_grad()
def get_inps(
    analyzer: ModelAnalyzer,
    data: Sequence,
    model_seqlen: int,
    devices: List[torch.device],
    offload_activations: bool,
) -> Tuple[List[torch.Tensor], Dict]:
    """
    mocks model launch to collect inputs to the first model layer
    :returns: a list of torch tensors with activations for each device in devices.
    Each tensor has shape [nsample_per_device, seq_len, hid_size]
    """
    logging.info("Catching layer inputs from data")
    layers = analyzer.get_layers()
    model = analyzer.model
    device = devices[0] if not offload_activations else torch.device("cpu")

    if isinstance(data, torch.Tensor) and data.shape[0] == 1:  # given a single long tensor, split it into sequences
        assert data.ndim == 2, "data must be either a single tensor with a long sequence or a list of pre-cut sequences"
        num_sequences, num_tokens_dropped = data.numel() // model_seqlen, data.numel() % model_seqlen
        data = [data[:, i * model_seqlen : (i + 1) * model_seqlen].to(device) for i in range(num_sequences)]
        print(f"Got {len(data)} sequences of {model_seqlen} tokens, dropped last {num_tokens_dropped} tokens")
        del num_sequences, num_tokens_dropped

    assert all(sequence.shape[1] == model_seqlen for sequence in data)
    emb = model.get_input_embeddings()
    emb_device = emb.weight.device
    if emb_device.type != "cuda":
        emb = emb.to(device)
    device = emb.weight.device  # now default device is the one where the embeddings are.
    layer_device = next(layers[0].parameters()).device
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    nsamples_per_device = (len(data) - 1) // len(devices) + 1
    inps = [
        torch.zeros(
            (min(nsamples_per_device, len(data) - i * nsamples_per_device), model_seqlen, model.config.hidden_size),
            dtype=dtype,
            device=devices[i] if not offload_activations else "cpu",
        )
        for i in range(len(devices))
    ]
    forward_arg_names = ["attention_mask", "position_ids", "position_embeddings"]

    cache = {"i": 0}

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"] // nsamples_per_device][cache["i"] % nsamples_per_device] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise CatcherExit()

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch_inps in tqdm(data, desc="Catching layer inputs from data"):
        try:
            if isinstance(batch_inps, (list, tuple)):
                batch_inps, *_ = batch_inps
            batch_inps = batch_inps.to(device)
            # call model.forward to trigger the Catcher
            model(batch_inps, attention_mask=torch.ones_like(batch_inps))
        except CatcherExit:
            pass  # exit after catcher finished without running the rest of the model layers

    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_device)
    model.get_input_embeddings().to(emb_device)
    torch.cuda.empty_cache()
    forward_args = {k: cache[k] for k in forward_arg_names}
    assert cache["i"] == sum(len(inp_tensor) for inp_tensor in inps), "internal error: found empty rows in inps"
    return inps, forward_args


@torch.no_grad()
def update_outs(
    layer: nn.Module, inps_tensor: torch.Tensor, outs_tensor: torch.Tensor, compute_mse: bool, is_after_quant: bool, **forward_args
) -> Sequence[float]:
    """
    Update outs_tensor with new activations and optionally compute sample-wise mse loss with previous activations
    :param layer: transformer layer with one or more linear layer to be quantized
    :param inps_tensor: a tensor of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs_tensor: a tensor to write output activations into, [nsamples_per_device, seq_len, hidden_size]
    :note: outs_tensor must contain previous activations with which to compute MSE loss
    :param compute_mse: if True, return a list of sample-wise mse losses; if False, return an empty sequence
    :param is_after_quant: if True, calculate outputs after quantization; if False, calculate outputs before quantization
    :param forward_args: additional keyword arguments, e.g. attention mask
    :returns: a list of mean squared errors for each sequence
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    out_losses = []
    description = "Calculating outputs for next layer"
    for j in trange(len(inps_tensor), desc=description, leave=False):
        outs_batch = layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0]
        outs_tensor[j].copy_(outs_batch.reshape_as(outs_tensor[j]), non_blocking=True)
    return out_losses


@torch.no_grad()
def update_outs_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    compute_mse: bool,
    is_after_quant: bool,
    **forward_args,
) -> Sequence[float]:
    """Parallel version of update_outs_and_compute_losses; works on lists of input/output tensors"""
    layer.to(devices[0])
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    funcs_by_device = [update_outs for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    for i in range(len(devices)):
        inputs_by_device.append((layer_replicas[i], inps[i], outs[i], compute_mse, is_after_quant))
        processed_args = {}
        for k, v in forward_args.items():
            if isinstance(v, torch.Tensor):
                processed_args[k] = v.to(devices[i], non_blocking=True)
            elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
                processed_args[k] = tuple(x.to(devices[i], non_blocking=True) for x in v)
            else:
                processed_args[k] = v
        kwargs_by_device.append(processed_args)
    out_losses_by_device: Sequence[Sequence[float]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    return list(chain(*out_losses_by_device))


import os
import logging
import torch
import torch.nn as nn
from typing import Sequence, Dict, Any, Optional, Tuple, List
from tqdm import trange


##############################################################################
# 1) SaliencyEngine that stores entire (N, seq_len, G) for that sub-layer
##############################################################################

class SaliencyEngine(nn.Module):
    """
    Holds a (N, seq_len, G) saliency buffer for one sub-layer's entire dataset,
    accumulates X^T X in shape (D, D, G).
    """
    def __init__(
        self,
        in_features: int,
        saliency: torch.Tensor,   # shape (N, seq_len, G)
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.device = device
        self.index = 0  # track how many samples we've consumed

        # in_features = layer.weight.shape[1], assuming it's nn.Linear
        num_groups = saliency.shape[-1]

        # store saliency in a buffer
        self.register_buffer("saliencies", saliency.to(dtype))
        self.nsamples = saliency.shape[0]

        # Hessian buffer (D, D, G)
        self.register_buffer(
            "XTX",
            torch.zeros(in_features, in_features, num_groups, dtype=dtype, device=self.device),
        )

    @torch.no_grad()
    def add_batch(self, X: torch.Tensor):
        """
        X: shape [batch_size, seq_len, in_features].

        We'll slice self.saliencies[index : index + batch_size],
        do the einsum => accumulate into self.XTX,
        then index += batch_size.
        """

        bsz = X.shape[0]
        # slice out shape => (bsz, seq_len, G)
        sal_batch = self.saliencies[self.index : self.index + bsz].to(self.device)
        self.index += bsz

        # Flatten
        if X.dim() == 3:
            X = X.reshape(-1, X.shape[-1])  # => [bsz*seq_len, D]
            sal_batch = sal_batch.reshape(-1, sal_batch.shape[-1])  # => [bsz*seq_len, G]

        X = X.to(self.XTX.dtype)
        S = sal_batch.to(self.XTX.dtype)

        sal_weighted_X = torch.einsum("nj,ng->njg", X, S)
        block = torch.einsum("ni,njg->ijg", X, sal_weighted_X)
        self.XTX.add_(block)

    def __repr__(self):
        return f"SaliencyEngine(XTX.shape={tuple(self.XTX.shape)}, index/nsamples={self.index}/{self.nsamples})"


class _LayerWrapperThatAccumulatesSaliency(nn.Module):
    """
    Intercepts the sub-layer's forward. On each call:
      1) reads the current batch size from 'input'
      2) calls engine.add_batch(input)
      3) calls real sub-layer forward
    """
    def __init__(self, real_layer: nn.Module, engine: SaliencyEngine):
        super().__init__()
        self.wrapped_layer = real_layer
        self.engine = engine

    def forward(self, input, *args, **kwargs):
        self.engine.add_batch(input)
        return self.wrapped_layer(input, *args, **kwargs)

def print_gpu_usage(message: str):
    import torch
    allocated_memory = torch.cuda.memory_allocated()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"{message}: {allocated_memory / 1024 ** 3:.1f} GB / {total_memory / 1024 ** 3:.1f} GB")


def print_cpu_memory_usage(message: str):
    import psutil
    memory_info = psutil.virtual_memory()
    used_memory = memory_info.used
    total_memory = memory_info.total
    print(f"{message}: {used_memory / 1024 ** 3:.1f} GB / {total_memory / 1024 ** 3:.1f} GB")

##############################################################################
# 2) Single-Device version: init_saliency_engines_single_wrapper
##############################################################################

def init_saliency_engines_single_wrapper(
    layer: nn.Module,
    sublayer_names: List[str],
    inp: torch.Tensor,
    layer_saliencies: Dict[str, torch.Tensor],  # { module_name -> (N, seq_len, G) }
    **forward_args,
) -> Dict[str, SaliencyEngine]:
    """
    Single-device version:
      - For each sublayer_name in sublayer_names, create SaliencyEngine with layer_saliencies[name].
      - Wrap the sub-layer with _LayerWrapperThatAccumulatesSaliency(engine).
      - Forward pass over 'inp' => fill 'out' => accumulate X^T X in each engine's .XTX.
      - Unwrap.
      - Return { sublayer_name -> SaliencyEngine }.
    """

    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    layer = layer.to(device)

    # 1) find the actual sub-layers
    found_sublayers = _find_sublayers(layer)
    sublayers = {nm: found_sublayers[nm] for nm in sublayer_names if nm in found_sublayers}

    # 2) Build an engine for each sub-layer
    engines = {}
    for nm, submodule in sublayers.items():
        if nm not in layer_saliencies:
            raise ValueError(f"No saliency found for sublayer '{nm}' in layer_saliencies keys = {list(layer_saliencies.keys())}")
        engine = SaliencyEngine(submodule.weight.shape[1], layer_saliencies[nm], dtype=torch.float32, device=device)
        engines[nm] = engine

    # 3) Wrap sub-layers
    _wrap_sublayers(layer, engines)

    processed_args = {}
    for k, v in forward_args.items():
        if isinstance(v, torch.Tensor):
            processed_args[k] = v.to(device, non_blocking=True)
        elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
            processed_args[k] = tuple(x.to(device, non_blocking=True) for x in v)
        else:
            processed_args[k] = v
    forward_args = processed_args

    with torch.no_grad():
        # 4) Forward pass over 'inp'
        for i in trange(len(inp), desc="capturing saliency weighted XTX", leave=False):
            local_inp = inp[i].to(device).unsqueeze(0)   # => [1, seq_len, D]
            out_batch = layer(local_inp, **forward_args)[0]

    # 5) Unwrap
    _unwrap_sublayers(layer)

    return engines


def init_saliency_engines_parallel_wrapper(
    devices: Sequence[torch.device],
    layer: nn.Module,
    sublayer_names: List[str],
    inps: Sequence[torch.Tensor],
    layer_saliencies_by_device: Sequence[Dict[str, torch.Tensor]],
    **forward_args,
) -> Dict[str, SaliencyEngine]:
    """
    Parallel version. Each device i calls init_saliency_engines_single_wrapper(...) on a replica,
    using sublayer_names + inps[i] + outs[i] + layer_saliencies_by_device[i].
    Then we combine the partial XTX in the main engine (device[0]).
    """
    from torch.nn.parallel import replicate, parallel_apply

    layer.to(devices[0])
    # replicate the layer
    layer_replicas = replicate(layer, devices=devices, detach=True)
    layer_replicas[0] = layer

    funcs = [init_saliency_engines_single_wrapper for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []

    for i, dev in enumerate(devices):
        # We'll pass (layer_replicas[i], sublayer_names, inps[i], layer_saliencies_by_device[i])
        inputs_by_device.append((layer_replicas[i], sublayer_names, inps[i], layer_saliencies_by_device[i]))
        # forward_args -> dev
        dev_kwargs = {}
        for k,v in forward_args.items():
            if isinstance(v, torch.Tensor):
                dev_kwargs[k] = v.to(dev, non_blocking=True)
            elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
                dev_kwargs[k] = tuple(x.to(dev, non_blocking=True) for x in v)
            else:
                dev_kwargs[k] = v
        kwargs_by_device.append(dev_kwargs)

    partial_results: List[Dict[str, SaliencyEngine]] = parallel_apply(
        funcs, inputs_by_device, kwargs_by_device, devices=devices
    )
    # partial_results[i] is { sublayer_name -> SaliencyEngine } from device i

    # Merge them on device[0]
    main_engines = partial_results[0]
    for nm, main_engine in main_engines.items():
        total_nsamples = main_engine.nsamples
        for i in range(1, len(devices)):
            eng_i = partial_results[i][nm]
            main_engine.XTX.add_(eng_i.XTX.to(main_engine.device))
            total_nsamples += eng_i.nsamples
        main_engine.nsamples = total_nsamples

    return main_engines

##############################################################################
# 4) Updated "accumulate_saliency_weighted_hessians" using saliency_path
##############################################################################

def accumulate_saliency_weighted_hessians(
    analyzer,
    data: List[torch.Tensor],
    saliency_path: str,             # Path containing l{L}.pt => { sublayer_name -> saliency Tensor }
    output_folder: str,
    num_groups: int
):
    """
    1) get_inps(...) => inps, forward_args
    2) Prepare outs
    3) For each layer L in [0..num_layers-1]:
       - update_outs_parallel(...) so 'outs' = layer L's outputs
       - load the saliencies for layer L => a dict { sublayer_name -> (N, seq_len, G) }
         or for multi-GPU, a list of per-device dicts
       - call init_saliency_engines_parallel_wrapper(...) or single_wrapper(...) to accumulate
         the Hessians in .XTX
       - save the final Hessians to output_folder/l{L}.pt
       - swap inps, outs
    """

    if output_folder and os.path.exists(output_folder):
        if all(os.path.exists(os.path.join(output_folder, f"l{i}.pt")) 
                for i in range(len(analyzer.get_layers()))):
            logging.info(f"Cached hessians found in {output_folder}")
            return output_folder

    # 0) Setup devices
    if torch.cuda.is_available():
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device("cpu")]

    model_seqlen = data[0].shape[-1]
    if data[0].dim() == 1:
        data = [d.unsqueeze(0) for d in data]

    os.makedirs(output_folder, exist_ok=True)

    inps, forward_args = get_inps(
        analyzer=analyzer,
        data=data,
        model_seqlen=model_seqlen,
        devices=devices,
        offload_activations=True,
    )

    outs = [torch.zeros_like(inp_tensor) for inp_tensor in inps]

    # 2) Layers
    layers = analyzer.get_layers()
    num_layers = len(layers)

    processed_layers = []
    for l in range(num_layers):
        if os.path.exists(os.path.join(output_folder, f"l{l}.pt")):
            processed_layers.append(l)
        
    logging.info(f"Processed layers: {processed_layers}")

    module_names = analyzer.module_names  # e.g. sub-layer names
    from .utils import get_progress_bar     # adapt if needed
    pb = get_progress_bar(num_layers, "Accumulating saliency Hessians blockwise")

    # 3) Blockwise
    for l in range(num_layers):

        layer = layers[l]
        layer.to(torch.device("cpu"))

        if l in processed_layers:
            logging.info(f"Skipping layer {l} because it has already been processed")
            torch.cuda.empty_cache()

            # # (A) update_outs_parallel => outs
            update_outs_parallel(
                devices=devices,
                layer=layer,
                inps=inps,
                outs=outs,
                compute_mse=False,
                is_after_quant=False,
                **forward_args
            )

            layer.to(torch.device("cpu"))

            # (E) Swap inps, outs
            inps, outs = outs, inps

            torch.cuda.empty_cache()
            layers[l] = None

            pb.update(1)
            continue


        # (B) Load saliencies for layer l
        #     We expect e.g. saliency_path/l{l}.pt => either:
        #       { sublayer_name -> (N, seq_len, G) } in single-GPU
        #     or a list of length len(devices), each is { sublayer_name -> partial saliency } for multi-GPU.
        file_path = os.path.join(saliency_path, f"l{l}.pt")
        # This might be either a dict or a list-of-dicts
        loaded = torch.load(file_path)

        orig_num_groups = list(loaded.values())[0].shape[-1]
        assert orig_num_groups % num_groups == 0, f"orig_num_groups {orig_num_groups} must be divisible by num_groups {num_groups}"
        group_subchannels = orig_num_groups // num_groups
        loaded = {k: v.view(v.shape[0], v.shape[1], num_groups, group_subchannels).mean(dim=-1) for k, v in loaded.items()}

        nsamples = list(loaded.values())[0].shape[0]
        nsamples_per_device = (nsamples - 1) // len(devices) + 1
        assert nsamples_per_device == inps[0].shape[0], f"nsamples_per_device {nsamples_per_device} must match inps[0].shape[0] {inps[0].shape[0]}"

        splitted_loaded = []
        for i in range(len(devices)):
            start = i * nsamples_per_device
            end = min(start + nsamples_per_device, nsamples)
            splitted_loaded.append({k: v[start:end] for k, v in loaded.items()})

        assert list(splitted_loaded[0].values())[0].shape[0] == nsamples_per_device

        # (C) Accumulate Hessians
        if len(devices) == 1:
            # single device
            # loaded must be { sublayer_name -> (N, seq_len, G) }
            saliency_handlers = init_saliency_engines_single_wrapper(
                layer,
                module_names,    # or the sub-layers you actually want
                inps[0],
                splitted_loaded[0],         # => layer_saliencies
                **forward_args
            )
        else:
            # multi-gpu => loaded presumably is a list of length len(devices),
            # each element is { sublayer_name -> partial saliency } for that device
            if not isinstance(splitted_loaded, list) or len(splitted_loaded) != len(devices):
                raise ValueError(f"Expected saliencies for layer {l} to be a list of length {len(devices)}")
            # loaded[i]: { sublayer_name -> (partialN, seq_len, G) }
            saliency_handlers = init_saliency_engines_parallel_wrapper(
                devices,
                layer,
                module_names,
                inps,
                splitted_loaded,  # layer_saliencies_by_device
                **forward_args
            )

        # (D) Save final Hessians => output_folder/l{l}.pt
        result_dict = {}
        for nm, engine in saliency_handlers.items():
            result_dict[nm] = engine.XTX.detach().cpu().float()

        out_file = os.path.join(output_folder, f"l{l}.pt")
        torch.save(result_dict, out_file)
        logging.info(f"[Layer {l}] Saved saliency-weighted Hessians to {out_file}")

        del result_dict


        import gc;gc.collect()
        

        del saliency_handlers
        torch.cuda.empty_cache()

        # # (A) update_outs_parallel => outs
        update_outs_parallel(
            devices=devices,
            layer=layer,
            inps=inps,
            outs=outs,
            compute_mse=False,
            is_after_quant=False,
            **forward_args
        )

        layer.to(torch.device("cpu"))

        # (E) Swap inps, outs
        inps, outs = outs, inps

        torch.cuda.empty_cache()

        layers[l] = None

        pb.update(1)

    pb.close()
    logging.info("Done accumulating saliency-weighted Hessians for all layers.")


##############################################################################
# 5) Minimal Helper Stubs
##############################################################################

def _find_sublayers(layer: nn.Module) -> Dict[str, nn.Module]:
    """
    Recursively gather { full_name: submodule } for all sub-layers (e.g. nn.Linear).
    Adapt as needed.
    """
    result = {}
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear) and name:
            result[name] = module
    return result

def _wrap_sublayers(layer: nn.Module, engines: Dict[str, SaliencyEngine]):

    for name, submodule in layer.named_modules():
        if name in engines.keys():
            wrapper = _LayerWrapperThatAccumulatesSaliency(submodule, engines[name])
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent_module = getattr(layer, parent_name)
            setattr(parent_module, child_name, wrapper)

def _unwrap_sublayers(layer: nn.Module):

    for name, submodule in layer.named_modules():
        if isinstance(submodule, _LayerWrapperThatAccumulatesSaliency):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent_module = getattr(layer, parent_name)
            setattr(parent_module, child_name, submodule.wrapped_layer)
