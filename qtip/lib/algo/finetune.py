"""
Utilities for fine tuning
"""
import os
import copy
import math
from contextlib import contextmanager
from operator import attrgetter

import glog
import torch
from torch import multiprocessing as mp
from torch import nn
from transformers import AutoModelForCausalLM

from lib import codebook, utils
from lib.linear import QuantizedLinear

from lib.algo import ldlq
import time

@contextmanager
def use_tf32():
    fp32_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(fp32_matmul_precision)


def finetune_decoder_layer(layer, name, device, train_dl, valid_dl, orig_dtype,
                           args):
    with use_tf32():
        layer = layer.to(device)

        source = next(iter(train_dl))[0]
        position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
        # manifest tensor parallel attributes in layer
        output = layer(source.to(device),
                       position_ids=position_ids)[0]
        
        best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
        utils.clean()

        optim = torch.optim.Adam(layer.parameters(), lr=args.ft_lr)
        if args.ft_epochs > 0:
            best_loss = utils.calculate_weighted_mse_loss(layer, valid_dl, device)
            glog.info(f'layer {name} initial loss {best_loss}')
            scaler = torch.cuda.amp.GradScaler(enabled=(orig_dtype==torch.float16))
            worse_ct = 0

            for epoch in range(args.ft_epochs):
                for bidx, data in enumerate(train_dl):
                    if len(data) == 2:
                        source, targets = data
                        block_hess = None
                    elif len(data) == 3:
                        source, targets, block_hess = data
                        block_hess = block_hess.to(device)
                    else:
                        raise ValueError("data should be either 2 or 3 elements")

                    targets = targets.to(device, non_blocking=True)
                    with torch.autocast(device_type='cuda',
                                        dtype=orig_dtype,
                                        enabled=True):
                        output = layer(source.to(device),
                                    position_ids=position_ids)[0]
                        loss = utils.weighted_mse(output, targets, block_hess)
                    scaler.scale(loss).backward()
                    if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                            train_dl) - 1:
                        scaler.step(optim)
                        scaler.update()
                        optim.zero_grad()

                if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
                    test_loss = utils.calculate_weighted_mse_loss(layer, valid_dl, device)
                    if test_loss < best_loss:
                        glog.info(
                            f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                        )
                        best_loss = test_loss
                        best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
                        utils.clean()
                        worse_ct = 0
                    else:
                        glog.info(
                            f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                        )
                        worse_ct += 1
                        if worse_ct >= args.ft_early_stop:
                            break

    del optim, train_dl, valid_dl

    layer = layer.cpu()
    layer.load_state_dict(best_sd)
    utils.clean()

def preprocess_single(HR, W, cb, SU, SV, scale_override, sigma_reg, device):
    '''
        Args:
            HR : n x n
            W : m x n
        output:
            HRr : n x n
            Wr : m x n
            Wscale : float
    '''
    # original implementation for non-grouped HR
    HR = utils.regularize_H(HR, sigma_reg)
    Wr = utils.matmul_hadUt(
        utils.matmul_hadUt(W.T.to(device) * SV).T * SU)
    HRr = utils.matmul_hadUt(
        utils.matmul_hadUt(HR.to(device) * SU).T * SU)
    
    Wscale = Wr.square().mean().sqrt() / (
        cb.lut.to(torch.float64).square().mean().sqrt().float() *
        scale_override)
    Wr /= Wscale

    return HRr, Wr, Wscale

def preprocess_group(HR, W, cb, SU, SV, scale_override, sigma_reg, device):
    # modified implementation to support grouped HR 
    '''
        Args:
            HR : g x n x n
            W : m x n
        output:
            HRr : g x n x n
            Wr : m x n
            Wscale : float
    '''
    Wr = utils.matmul_hadUt(
            utils.matmul_hadUt(W.T.to(device) * SV).T * SU)
    Wscale = Wr.square().mean().sqrt() / (
            cb.lut.to(torch.float64).square().mean().sqrt().float() *
            scale_override)
    Wr /= Wscale

    HRr = torch.zeros_like(HR).to(device)
    for i in range(HR.shape[0]):
        HR[i] = utils.regularize_H(HR[i], sigma_reg)
        HRr[i] = utils.matmul_hadUt(
                utils.matmul_hadUt(HR[i].to(device) * SU).T * SU)
    
    return HRr, Wr, Wscale

def test_new_preprocess():
    from lib.codebook import bitshift

    g, m, n = 1, 512, 512
    L, K, V = 16, 2, 2
    tlut_bits, decode_mode = 9, 'quantlut_sym'
    HR = torch.randn(g, n, n).to("cuda")
    W = torch.randn(m, n).to("cuda")
    SU = (torch.randn(n).sign() + 1e-5).sign().to("cuda")
    SV = (torch.randn(m).sign() + 1e-5).sign().to("cuda")
    cb = bitshift.bitshift_codebook(L=L,
                                    K=K,
                                    V=V,
                                    tlut_bits=tlut_bits,
                                    decode_mode=decode_mode)
    scale_override = 1.0
    sigma_reg = 1e-5

    HRr, Wr, Wscale = preprocess_group(HR, W, cb, SU, SV, scale_override, sigma_reg, "cuda")
    HRr2, Wr2, Wscale2 = preprocess_single(HR[0], W, cb, SU, SV, scale_override, sigma_reg, "cuda")

    try:
        assert torch.allclose(HRr[0], HRr2, atol=1e-6)
        assert torch.allclose(Wr, Wr2, atol=1e-6)
        assert torch.allclose(Wscale, Wscale2, atol=1e-6)
        print("test_new_preprocess passed")
    except:
        import ipdb;ipdb.set_trace()
    return 

def get_HR(use_saliency, in_hess_path, idx, in_hess_name, linear_attr, n, device, dtype_, layer_data):
    if not use_saliency:
        in_hess_path = f'{in_hess_path}/{idx}_{in_hess_name}.pt'
        H_data = torch.load(in_hess_path, map_location=torch.device('cpu'))
        HR = utils.flat_to_sym(H_data['flatH'], H_data['n'])
        if 'mu' in H_data:
            mu = H_data['mu']
            HR += mu[None, :] * mu[:, None]
            del mu
        del H_data
        HR = HR.unsqueeze(0) # 1 x n x n
    elif not os.path.exists(os.path.join(in_hess_path, f'l{idx}.pt')):
        path = os.path.join(in_hess_path, f'l{idx}_{in_hess_name}.pt')
        HR = torch.load(path, map_location=f'cuda:{device}').to(dtype_)
        if HR.shape[-1] != n:
            HR = HR.permute(2, 0, 1).contiguous()
    else:
        HR = layer_data[linear_attr].to(device).to(dtype_)
        if HR.shape[-1] != n:
            HR = HR.permute(2, 0, 1).contiguous()
    return HR

def quantize_finetune_decoder_layer(mixed_layer, quant_order, idx, cb, args,
                                    device, pre_orig_emb, orig_emb):
    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)
    torch.set_grad_enabled(False)

    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    orig_dtype = None
    for p in mixed_layer.parameters():
        orig_dtype = p.dtype
        break
    mixed_layer = mixed_layer.float()

    if args.block_hess_path is not None and args.block_hess_path != 'None':
        block_hess_path = os.path.join(args.block_hess_path, f'l{idx}.pt')
        block_hess_data = torch.load(block_hess_path, map_location=torch.device('cpu'))
        assert args.devset_size <= block_hess_data.shape[0], f'{args.devset_size} should be less than {block_hess_data.shape[0]}'
        block_hess_data = block_hess_data[:args.devset_size]
    else:
        print("block_hess_path is None")
        block_hess_data = None
    train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, block_hess_data, args)

    has_kernel = utils.has_kernel(args.decode_mode, args.L, args.K, args.V,
                                  args.tlut_bits, args.td_x, args.td_y)

    if args.use_saliency and os.path.exists(os.path.join(args.in_hess_path, f'l{idx}.pt')):
        layerpath = os.path.join(args.in_hess_path, f'l{idx}.pt')
        layer_data = torch.load(layerpath, map_location=torch.device('cpu'))
    else:
        layer_data = None
    
    if args.sub_use_saliency and os.path.exists(os.path.join(args.sub_in_hess_path, f'l{idx}.pt')):
        sublayerpath = os.path.join(args.sub_in_hess_path, f'l{idx}.pt')
        sublayer_data = torch.load(sublayerpath, map_location=torch.device('cpu'))
    else:
        sublayer_data = None

    time_start = time.time()
    for quant_i, (linear_attr, name, in_hess_name, out_hess_name,
                  rcp) in enumerate(quant_order):
        utils.clean()
        cb = cb.to(device).to(orig_dtype)
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        W = orig_linear.weight.to(dtype_)
        del orig_linear
        (m, n) = W.shape
        SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
        SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)

        if quant_i == 0 or (args.sub_in_hess_path is None or args.sub_in_hess_path == 'None'):
        # if in_hess_name == 'qkv' or (args.sub_in_hess_path is None or args.sub_in_hess_path == 'None'):
            HR = get_HR(args.use_saliency, args.in_hess_path, idx, in_hess_name, linear_attr, n, device, dtype_, layer_data)
            g_sal = HR.shape[0] # group size for saliency
            assert g_sal == args.g_sal or not args.use_saliency, f'{g_sal} should be equal to {args.g_sal}'
            assert m % g_sal == 0, f'{m} should be divisible by {g_sal}'
        else:
            HR = get_HR(args.sub_use_saliency, args.sub_in_hess_path, idx, in_hess_name, linear_attr, n, device, dtype_, sublayer_data)
            g_sal = HR.shape[0] # group size for saliency
            assert g_sal == args.sub_g_sal or not args.sub_use_saliency, f'{g_sal} should be equal to {args.sub_g_sal}'
            assert m % g_sal == 0, f'{m} should be divisible by {g_sal}'

        # pre-process W and HR
        HRr, Wr, Wscale = preprocess_group(HR, W, cb, SU, SV, args.scale_override, args.sigma_reg, device)
        Wr_gpd = Wr.reshape(g_sal, m // g_sal, n).contiguous() # g x m/g x n
        
        hatWr = torch.zeros(m, n, dtype=dtype_, device=device)
        Qidxs = torch.zeros(m, n // args.V, dtype=cb.idx_dtype, device=device)

        for i in range(g_sal):
            t0 = time.time()
            LRr, _ = utils.block_LDL(HRr[i], args.td_y)
            diag = torch.arange(n, device=LRr.device)
            LRr[diag, diag] = 0
            t1 = time.time()
            hatWr_i, Qidxs_i = ldlq.LDLQ(Wr_gpd[i], LRr, cb, args, for_kernel=has_kernel)
            hatWr[i * m // g_sal : (i + 1) * m // g_sal] = hatWr_i
            Qidxs[i * m // g_sal : (i + 1) * m // g_sal] = Qidxs_i
            del LRr, diag, hatWr_i, Qidxs_i
            t2 = time.time()
            print(quant_i, linear_attr, f'group {i} time {t2 - t0}')
        del Wr_gpd

        Qidxs = Qidxs.cpu()
        packed = cb.pack_trellis(
            Qidxs.reshape(m // args.td_x, args.td_x, n // args.td_y,
                          args.td_y // args.V).transpose(1, 2).reshape(
                              -1, args.td_x * args.td_y // args.V))

        if has_kernel:
            packed = packed.view(torch.uint8).view(-1, 2).flip(
                (-1, )).reshape(m // 16 // 2, 2, n // 16 // 2, 2, 16 * 16 // 8,
                                args.K).permute(0, 2, 4, 3, 1, 5).flip(
                                    (-1, )).contiguous().flatten().view(
                                        torch.int16).reshape(packed.shape)
        else:
            packed = packed.view(torch.int16)

        if rcp == 'col':
            Wr = (Wr.reshape(args.tp_rank, m * n // args.tp_rank) *
                  Wscale.unsqueeze(-1)).reshape(m, n)
            hatWr = (hatWr.reshape(args.tp_rank, m * n // args.tp_rank) *
                     Wscale.unsqueeze(-1)).reshape(m, n)
        elif rcp == 'row':
            Wr = Wr.reshape(m, args.tp_rank, n // args.tp_rank).transpose(
                0, 1).reshape(args.tp_rank, -1) * Wscale.unsqueeze(-1)
            Wr = Wr.reshape(args.tp_rank, m,
                            n // args.tp_rank).transpose(0, 1).reshape(m, n)
            hatWr = hatWr.reshape(m, args.tp_rank,
                                  n // args.tp_rank).transpose(0, 1).reshape(
                                      args.tp_rank, -1) * Wscale.unsqueeze(-1)
            hatWr = hatWr.reshape(args.tp_rank, m,
                                  n // args.tp_rank).transpose(0, 1).reshape(
                                      m, n)
        else:
            Wr *= Wscale
            hatWr *= Wscale

        with torch.no_grad():
            Wr_gpd = Wr.reshape(g_sal, m // g_sal, n).contiguous()
            hatWr_gpd = hatWr.reshape(g_sal, m // g_sal, n).contiguous()
            diff = Wr_gpd - hatWr_gpd
            numerator = torch.diagonal(
                torch.bmm(torch.bmm(diff, HRr), diff.transpose(1, 2)),
                dim1=-2, dim2=-1
            ).sum()
            denominator = torch.diagonal(
                torch.bmm(torch.bmm(Wr_gpd, HRr), Wr_gpd.transpose(1, 2)),
                dim1=-2, dim2=-1
            ).sum()
            err = numerator / denominator
        print(
            f'{idx}_{name} proxy err {err.item()} tr(WHW.T) {denominator.item()}'
        )

        save_path = f'{args.save_path}/{idx}_{name}.pt'

        # 0 = no tensor parallelism, 1 = row parallel, 2 = column parallel
        rcp_int = 0
        if args.split_for_tp:
            rcp_int = 1 if rcp == 'row' else 2

        torch.save(
            {
                'trellis':
                packed.cpu(),
                'SU':
                SU.to(orig_dtype).cpu(),
                'SV':
                SV.to(orig_dtype).cpu(),
                'Wscale':
                Wscale,
                'proxy_err':
                err.item(),
                'tlut':
                cb.tlut.data.to(orig_dtype).cpu()
                if hasattr(cb, 'tlut') else None,
                'rcp':
                rcp_int,
                'tp_rank':
                args.tp_rank
            }, save_path)


        del HR, HRr, Wr, hatWr, Qidxs, Wr_gpd, hatWr_gpd, diff, numerator, denominator, err
        utils.clean()
        
        q_linear = QuantizedLinear(
            n,
            m,
            args.td_x,
            args.td_y,
            args.L,
            args.K,
            args.V,
            args.tlut_bits,
            args.decode_mode,
            mode='train-recons' if args.ft_train_recons else 'train-fixW',
            use_prev_kernel=not args.ft_train_lut,
            dtype=orig_dtype,
            grad_ckpt=args.ft_grad_ckpt)
        q_linear.trellis.copy_(packed)
        q_linear.SU.copy_(SU)
        q_linear.SV.copy_(SV)
        q_linear.rcp.copy_(rcp_int)
        q_linear.tp_rank.copy_(args.tp_rank)
        q_linear = q_linear.to(device).float()

        del packed, SU, SV
        utils.clean()
        
        if rcp == 'row':
            q_linear.SU = nn.Parameter(
                (q_linear.SU.reshape(args.tp_rank, -1) *
                 Wscale.unsqueeze(-1)).reshape(q_linear.SU.shape),
                requires_grad=True)
            q_linear.SV = nn.Parameter(q_linear.SV, requires_grad=True)
        elif rcp == 'col':
            q_linear.SU = nn.Parameter(q_linear.SU, requires_grad=True)
            q_linear.SV = nn.Parameter(
                (q_linear.SV.reshape(args.tp_rank, -1) *
                 Wscale.unsqueeze(-1)).reshape(q_linear.SV.shape),
                requires_grad=True)
        else:
            q_linear.SU = nn.Parameter(q_linear.SU, requires_grad=True)
            q_linear.SV = nn.Parameter(q_linear.SV * Wscale,
                                       requires_grad=True)

        if q_linear.tlut is not None:
            q_linear.tlut.copy_(cb.tlut.data)
            q_linear.tlut.requires_grad = args.ft_train_lut

        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            q_linear)

        with torch.enable_grad():
            finetune_decoder_layer(mixed_layer, f'{idx}_{name}', device,
                                   train_dl, valid_dl, orig_dtype, args)

        cb = cb.cpu()
        utils.clean()

    for quant_i, (linear_attr, name, in_hess_name, out_hess_name,
                  rcp) in enumerate(quant_order):
        quant_linear = attrgetter(linear_attr)(mixed_layer)
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        data = torch.load(save_path)
        if rcp == 'row':
            data['SU'] = (
                ((quant_linear.SU.data).reshape(args.tp_rank, -1) /
                 data['Wscale'].to(quant_linear.SU.device).unsqueeze(-1)
                 ).reshape(quant_linear.SU.data.shape)).to(orig_dtype).cpu()
            data['SV'] = quant_linear.SV.data.to(orig_dtype).cpu()
        elif rcp == 'col':
            data['SU'] = quant_linear.SU.data.to(orig_dtype).cpu()
            data['SV'] = (
                ((quant_linear.SV.data).reshape(args.tp_rank, -1) /
                 data['Wscale'].to(quant_linear.SV.device).unsqueeze(-1)
                 ).reshape(quant_linear.SV.data.shape)).to(orig_dtype).cpu()
        else:
            data['SU'] = quant_linear.SU.data.to(orig_dtype).cpu()
            data['SV'] = (quant_linear.SV.data / data['Wscale'].to(
                quant_linear.SV.device)).to(orig_dtype).cpu()

        if quant_linear.tlut is not None:
            data['tlut'] = quant_linear.tlut.data.to(orig_dtype).cpu()
        torch.save(data, save_path)

    mixed_layer = mixed_layer.to(orig_dtype).cpu()

    del block_hess_data
    utils.clean()
    torch.set_grad_enabled(False)
    time_end = time.time()
    print(f"####TIME_ELAPSED{idx}=", time_end - time_start)


def infer(args, end_dev, n_layers, in_q, out_q):
    with torch.no_grad():
        fake_dev_map = {
            'model.embed_tokens': 0,
            'model.rotary_emb': 0,
            'model.norm': end_dev - 1,
            'lm_head': end_dev - 1
        }
        per_dev = math.ceil(n_layers / end_dev)
        for i in range(n_layers):
            fake_dev_map[f'model.layers.{i}'] = (i + 1) // per_dev

        model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                     torch_dtype='auto',
                                                     device_map=fake_dev_map,
                                                     low_cpu_mem_usage=True)
        while True:
            data = in_q.get()
            if data is None:
                return
            out_q.put(
                model(data.to(0))['logits'][:, :-1].contiguous().softmax(
                    dim=-1).cpu())


def finetune_susv_e2e(quant_model, start_dev, devset, orig_dtype, args):

    in_q = mp.Queue()
    out_q = mp.Queue()
    p = mp.Process(target=infer,
                   args=(args, start_dev, len(quant_model.model.layers), in_q,
                         out_q))
    p.start()

    train_dl, valid_dl = utils.split_data(devset, devset, None, args)

    optim = torch.optim.Adam(quant_model.parameters(), lr=args.ft_lr)

    best_loss = utils.calculate_ce_loss_model(quant_model, valid_dl, start_dev,
                                              in_q, out_q)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_sd = copy.deepcopy(quant_model.state_dict())
    glog.info(f'initial loss {best_loss}')
    worse_ct = 0
    for epoch in range(args.ft_epochs):
        for bidx, (source, _) in enumerate(train_dl):
            if bidx % 256 == 0:
                glog.info(f'epoch {epoch} batch {bidx}')
            in_q.put(source)
            with torch.autocast(device_type='cuda',
                                dtype=orig_dtype,
                                enabled=True):
                output = quant_model(
                    source.to(start_dev))['logits'][:, :-1].contiguous()
                target = out_q.get().to(output.device)
                target = target.view(-1, target.shape[-1])
                loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
                                             target)
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_ce_loss_model(quant_model, valid_dl,
                                                      start_dev, in_q, out_q)
            if test_loss < best_loss:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(quant_model.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    in_q.put(None)
    p.join()
    with torch.no_grad():
        quant_model.load_state_dict(best_sd)


if __name__ == '__main__':
    test_new_preprocess()