import copy
import logging
import math
import pprint
import time

import torch
import torch.nn as nn
import tqdm
import os

from utils import quant_utils, utils
from .gptq_utils import GPTQ


class GPTQGuided:
    def __init__(self, 
        layer, 
        saliency: torch.Tensor, # shape (N, seq_len, G)
        num_groups: int
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        # Instead of passing H in, we allocate a 3D Hessian buffer
        # that will hold sub-channel Hessians along the last dim.
        self.num_groups = num_groups
        assert self.num_groups == saliency.shape[2], "Number of groups for GuidedQuant must match saliency shape!"

        self.saliencies = saliency.float()
        self.H = torch.zeros(
            (self.columns, self.columns, self.num_groups),
            device=self.dev
        )
        self.nsamples = saliency.shape[0]
        self.index = 0

        # Assert row partition is valid:
        # we do the same partition as before:
        assert self.rows % self.num_groups == 0, (
            f"Number of rows ({self.rows}) must be divisible "
            f"by num_groups ({self.num_groups})"
        )

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """
        inp: shape [batch_size, seq_len, in_features]

        We'll slice self.saliencies[index: index + batch_size]
        do the einsum => accumulate into self.H
        then index += batch_size.
        """
        # If input is 2D or 1D, reshape to [batch, seq_len, dim] for consistency
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)  # => [1, seq_len, dim]
        else:
            assert inp.dim() == 3, "Input must be 2D or 3D. Got %dD." % inp.dim()

        bsz = inp.shape[0]
        # slice out shape => (bsz, seq_len, G)
        sal_batch = self.saliencies[self.index: self.index + bsz].to(self.dev)
        self.index += bsz

        # Flatten
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
            sal_batch = sal_batch.reshape(-1, sal_batch.shape[-1])
            
        inp = inp.float()
        sal_batch = sal_batch.float()

        sal_weighted_inp = torch.einsum("nj,ng->njg", inp, sal_batch)
        block = torch.einsum("ni,njg->ijg", inp, sal_weighted_inp)
        self.H.add_(block)

    def __repr__(self):
        return f"GPTQGuided(H.shape={tuple(self.H.shape)}, index/nsamples={self.index}/{self.nsamples})"


    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        export_to_et=False,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()
        Scale = self.layer.weight.data.clone()
        Scale = Scale.float()
        W_int = self.layer.weight.data.clone()
        W_int = W_int.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)


        # We will partition the rows into num_groups slices
        rows_per_sub = self.rows // self.num_groups

        # Prepare final Q, W_int, Scale
        Q_final = torch.zeros_like(W)
        W_int_final = torch.zeros_like(W)
        Scale_final = torch.zeros_like(W)


        # Loop over each row partition, using H[..., sub_idx]
        for sub_idx in range(self.num_groups):
            row_start = sub_idx * rows_per_sub
            row_end = (sub_idx + 1) * rows_per_sub

            # Sub-slice of W
            W_sub = W[row_start:row_end, :]

            # Hessian sub-part
            H_sub = self.H[:, :, sub_idx].clone()

            # Apply the same "dead columns" logic
            dead = torch.diag(H_sub) == 0
            H_sub[dead, dead] = 1
            W_sub[:, dead] = 0


            # Static grouping for columns
            if static_groups:
                groups = []
                for i in range(0, self.columns, groupsize):
                    quantizer = copy.deepcopy(self.quantizer)
                    quantizer.find_params(W[:, i : (i + groupsize)])
                    groups.append(quantizer)

            # Possibly reorder columns by diag(H_sub)
            if actorder:
                perm = torch.argsort(torch.diag(H_sub), descending=True)
                W_sub = W_sub[:, perm]
                H_sub = H_sub[perm][:, perm]
                invperm = torch.argsort(perm)

            # Create local buffers
            Losses = torch.zeros_like(W_sub)
            Q = torch.zeros_like(W_sub)

            damp = percdamp * torch.mean(torch.diag(H_sub))
            diag = torch.arange(self.columns, device=self.dev)
            H_sub[diag, diag] += damp
            H_sub = torch.linalg.cholesky(H_sub)
            H_sub = torch.cholesky_inverse(H_sub)
            H_sub = torch.linalg.cholesky(H_sub, upper=True)
            Hinv = H_sub

            W_int_sub = torch.zeros_like(W_sub)
            Scale_sub = torch.zeros_like(W_sub)

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W_sub[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                W_int1 = torch.zeros_like(W1)
                Scale1 = torch.zeros_like(W1).to(Scale_sub.dtype)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(
                                    W[:, (i1 + i) : (i1 + i + groupsize)]
                                )
                        else:
                            idx = i1 + i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // groupsize]

                    q, int_weight, scale = self.quantizer.fake_quantize(
                        w.unsqueeze(1), st_idx=row_start, end_idx=row_end
                    )
                    Q1[:, i] = q.flatten()
                    q = q.flatten()
                    W_int1[:, i] = int_weight.flatten()
                    Scale1[:, i] = scale.flatten()

                    Losses1[:, i] = (w - q) ** 2 / d**2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                W_int_sub[:, i1:i2] = W_int1
                Scale_sub[:, i1:i2] = Scale1
                Losses[:, i1:i2] = Losses1 / 2

                # Propagate error across the rest
                W_sub[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # If we permuted columns, we un-permute the result
            if actorder:
                Q = Q[:, invperm]
                W_int_sub = W_int_sub[:, invperm]
                Scale_sub = Scale_sub[:, invperm]

            # Write the subchannel results back
            Q_final[row_start:row_end, :] = Q
            W_int_final[row_start:row_end, :] = W_int_sub
            Scale_final[row_start:row_end, :] = Scale_sub


        torch.cuda.synchronize()

        if export_to_et:
            self.layer.register_buffer(
                "int_weight", W_int_final.reshape(self.layer.weight.shape)
            )
            self.layer.register_buffer("scale", Scale_final)
        self.layer.weight.data = Q_final.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in weights")

            pprint.pprint(
                self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point
            )
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    """
    From GPTQ repo
    Now optionally supports GPTQGuided if args.guided is True.
    """
    logging.info("-----GPTQ Quantization-----")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    quantizers = {}
    sequential = [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
        ],
        ["self_attn.o_proj.module"],
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
        ["mlp.down_proj.module"],
    ]
    from tqdm import tqdm

    for i in tqdm(range(len(layers)), ncols=80, desc="Quantizing Layers"):

        if args.guided:
            saliency_dict = torch.load(os.path.join(args.saliency_path, f"l{i}.pt"))
            print("Loaded saliency for layer", i)
        else:
            print("Using original GPTQ")

        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                if "lm_head" in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and "down_proj" in name:
                    layer_weight_bits = 8

                if args.guided:
                    # Use GPTQGuided
                    key_name = name[:-len(".module")]
                    gptq[name] = GPTQGuided(
                        subset[name], 
                        saliency=saliency_dict[key_name], 
                        num_groups=args.num_groups,
                    )
                else:
                    # Use the original GPTQ
                    gptq[name] = GPTQ(subset[name])

                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.w_clip,
                )

            # Now gather data for each sub-layer
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                    export_to_et=args.export_to_et,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info("-----GPTQ Guided Quantization Done-----\n")
    return quantizers


@torch.no_grad()
def rtn_fwrd(model, dev, args, custom_layers=None):
    """
    From GPTQ repo
    """
    # assert args.w_groupsize == -1, "Groupsize not supported in RTN!"
    if custom_layers:
        layers = custom_layers
    else:
        layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(
            layer, layers=[torch.nn.Linear, torch.nn.Embedding]
        )

        for name in subset:
            layer_weight_bits = args.w_bits
            w_groupsize = args.w_groupsize
            if "lm_head" in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and "down_proj" in name:
                layer_weight_bits = 8
            if args.export_to_et:
                layer_weight_bits = 8  # all per channel 8 bits for executorch export
                w_groupsize = -1
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.w_clip,
                weight_groupsize=w_groupsize,
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            q, int_weight, scale = quantizer.fake_quantize(W)
            subset[name].weight.data = q.to(next(iter(layer.parameters())).dtype)
            if args.export_to_et:
                subset[name].register_buffer("int_weight", int_weight)
                subset[name].register_buffer("scale", scale)
            quantizers["model.layers.%d.%s" % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory(verbos=True)
    return quantizers
