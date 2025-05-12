import numpy as np
from tqdm import tqdm
import os
import torch
import logging
from multiprocessing import Pool
import numba

_bytes_per_thread = 4


@numba.njit(cache=True)
def _permute_bitmaps(bitmaps, inverse=False):
    """
    If inverse=False (default), permute forward.
    If inverse=True, apply the inverse permutation.
    """
    _, _, total_bytes = bitmaps.shape
    assert total_bytes % 4 == 0, "Number of bytes must be a multiple of 4"

    threads_per_warp = 32
    bytes_per_warp = threads_per_warp * _bytes_per_thread

    # Calculate the number of full warps and the starting index of remaining bytes
    full_warps_bytes = (total_bytes // bytes_per_warp) * bytes_per_warp
    remaining_bytes_start_idx = full_warps_bytes

    # Create an array of byte indices for full warps
    full_warp_byte_indices = np.arange(full_warps_bytes)
    # Calculate new indices for full warp bytes
    new_full_warp_byte_indices = _calculate_new_indices(full_warp_byte_indices, threads_per_warp)

    remaining_bytes = total_bytes - full_warps_bytes
    # Handle remaining bytes
    if remaining_bytes:
        remaining_byte_indices = np.arange(remaining_bytes)
        # Adjust the calculation for remaining bytes, which might not fill a complete warp
        adjusted_threads_per_warp = remaining_byte_indices.size // _bytes_per_thread
        new_remaining_byte_indices = _calculate_new_indices(remaining_byte_indices,
                                                            adjusted_threads_per_warp,
                                                            offset=remaining_bytes_start_idx)

        # Combine indices - the choice to not use np.concatenate is for numba compatibility
        new_byte_indices = np.empty(total_bytes, dtype=np.int64)
        new_byte_indices[:full_warps_bytes] = new_full_warp_byte_indices
        new_byte_indices[full_warps_bytes:] = new_remaining_byte_indices
    else:
        new_byte_indices = new_full_warp_byte_indices

    if not inverse:
        permuted_bitmaps = bitmaps[:, :, np.argsort(new_byte_indices)]
    else:
        permuted_bitmaps = bitmaps[:, :, np.argsort(np.argsort(new_byte_indices))]

    return permuted_bitmaps


@numba.njit(cache=True)
def _calculate_new_indices(byte_indices, threads_per_warp, offset=0):
    """
    Calculate new byte indices for a given array of byte indices.
    """
    bytes_per_warp = threads_per_warp * _bytes_per_thread

    warp_idx, byte_offsets_within_warp = np.divmod(byte_indices, bytes_per_warp)

    warp_offsets = warp_idx * bytes_per_warp
    thread_indices = byte_indices % threads_per_warp

    # Change endianness within each thread and calculate new byte positions
    byte_offsets_within_thread = byte_offsets_within_warp // threads_per_warp
    byte_offsets_within_thread ^= 3  # Change endianness
    new_byte_indices = warp_offsets + thread_indices * _bytes_per_thread + byte_offsets_within_thread + offset

    return new_byte_indices


@numba.njit(cache=True)
def _permute_bitmaps_int32(bitmaps):
    """Return a permuted version of the input bitmaps, reshaped to int32."""
    w_bits, N, total_bytes = bitmaps.shape
    bitmaps = _permute_bitmaps(bitmaps)
    return bitmaps.reshape(-1, 4).view(np.int32).reshape(w_bits, N, total_bytes // 4)


def _process_layer_data(args):
    layer_idx, lut_path, model_name, layers_name, module_names, parent_precision, seed_precision = args
    layer_data = {}

    weightpath = os.path.join(lut_path, 'weights', f'l{layer_idx}.pt')
    layer_weights = torch.load(weightpath)

    misc_weightpath = os.path.join(lut_path, 'misc_weights', f'l{layer_idx}.pt')
    if os.path.exists(misc_weightpath):
        layer_misc_weights = torch.load(misc_weightpath)

    for i, name in enumerate(module_names):
        N, group_count, group_size = layer_weights[name].shape
        K = group_count * group_size

        qweight_flattened = layer_weights[name].flatten()
        bitarray = np.empty((parent_precision, len(qweight_flattened) // 8), dtype=np.uint8)
        mask = 1 << (parent_precision - 1)  # MSB first
        for bit in range(parent_precision):
            curbitpack = np.packbits((qweight_flattened & mask).astype(bool))
            bitarray[bit] = curbitpack
            mask >>= 1

        bitarray = bitarray.reshape((parent_precision, N, K // 8))
        weighttensor = _permute_bitmaps_int32(bitarray)

        param_name = f'{model_name}.{layers_name}.{layer_idx}.{name}'
        layer_data[param_name + '.qweight'] = weighttensor

        for bit in range(seed_precision, parent_precision + 1):
            layer_lut_path = os.path.join(lut_path, f'lut_{bit}', f'l{layer_idx}.pt')
            layer_lut = torch.load(layer_lut_path)

            curLUT = np.empty((N, 2 ** bit), dtype=np.float16)
            for r_idx in range(N):
                curLUT[r_idx] = layer_lut[name][r_idx][0]  # the 0 here assumes group_count == 1

            layer_data[param_name + '.lut' + str(bit)] = curLUT

    if os.path.exists(misc_weightpath):
        block_name = f'{model_name}.{layers_name}.{layer_idx}'
        for misc_name, misc_weighttensor in layer_misc_weights.items():
            layer_data[block_name + '.' + misc_name] = misc_weighttensor
        
    return layer_idx, layer_data


def pack(
        analyzer,
        lut_path,
        output_model_path,
        seed_precision,
        parent_precision,
        dns=False,
        cpu_count=None
):
    group_count=1

    if group_count != 1:
        raise NotImplementedError("Group counts other than 1 are not supported yet for packing")

    if dns:
        raise NotImplementedError("D&S packing is not supported yet")

    if cpu_count is None:
        cpu_count = int(os.popen("nproc").read().strip())

    # Limit cpu_count to 8 as larger values use too much memory, without much speedup
    _max_cpu_count = 8
    if cpu_count > _max_cpu_count:
        logging.warning(f"cpu_count will be limited to 8 to avoid excessive memory usage. "
                        f"Original value: {cpu_count}")
        cpu_count = _max_cpu_count

    tokenizer = analyzer.tokenizer

    num_layers = analyzer.num_layers

    model_name = analyzer.model_name
    layers_name = analyzer.layers_name
    module_names = analyzer.module_names
    config = analyzer.config  # original model config
    arch_config = analyzer.get_arch_config()

    state_dict = analyzer.state_dict

    args_list = [(layer_idx, lut_path, model_name, layers_name, module_names, parent_precision, seed_precision) for
                 layer_idx in range(num_layers)]

    with Pool(cpu_count) as pool:
        for layer_idx, layer_data in tqdm(pool.imap(_process_layer_data, args_list), total=num_layers, desc="Packing"):
            for key, value in layer_data.items():
                if not "qweight" in key and not "lut" in key:
                    # Misc weights are stored in state_dict
                    assert key in state_dict
                    assert state_dict[key].shape == value.shape
                if isinstance(value, np.ndarray):
                    state_dict[key] = torch.from_numpy(value)  # Update with modified weights
                else:
                    # Misc weights are stored as float16 tensors
                    assert isinstance(value, torch.Tensor) and value.dtype == torch.float16
                    state_dict[key] = value

    # add new config parameters
    anyprec_configs = {
        'seed_precision': seed_precision,
        'parent_precision': parent_precision,
        'group_count': group_count,
        'arch_config': arch_config
    }
    config.anyprec = anyprec_configs

    logging.info(f"Writing model to disk...")
    os.makedirs(output_model_path, exist_ok=True)
    torch.save(state_dict, os.path.join(output_model_path, 'pytorch_model.bin'))
    tokenizer.save_pretrained(output_model_path)
    config.save_pretrained(output_model_path)
    logging.info(f"Model saved to {output_model_path}")



def _process_mixed_layer_data(args):
    layer_idx, lut_paths, model_name, layers_name, module_names, layerwise_config = args
    layer_data = {}

    for i, name in enumerate(module_names):
        lut_path = lut_paths[layerwise_config[f"{layer_idx:02}={name}"]["lut_idx"]]
        precision = layerwise_config[f"{layer_idx:02}={name}"]["precision"]

        weightpath = os.path.join(lut_path, 'weights', f'l{layer_idx}.pt')
        layer_weights = torch.load(weightpath)

        N, group_count, group_size = layer_weights[name].shape
        K = group_count * group_size

        qweight_flattened = layer_weights[name].flatten()
        bitarray = np.empty((precision, len(qweight_flattened) // 8), dtype=np.uint8)
        mask = 1 << (precision - 1)  # MSB first
        for bit in range(precision):
            curbitpack = np.packbits((qweight_flattened & mask).astype(bool))
            bitarray[bit] = curbitpack
            mask >>= 1

        bitarray = bitarray.reshape((precision, N, K // 8))
        weighttensor = _permute_bitmaps_int32(bitarray)

        param_name = f'{model_name}.{layers_name}.{layer_idx}.{name}'
        layer_data[param_name + '.qweight'] = weighttensor

        for bit in range(precision, precision + 1):
            layer_lut_path = os.path.join(lut_path, f'lut_{bit}', f'l{layer_idx}.pt')
            layer_lut = torch.load(layer_lut_path)

            curLUT = np.empty((N, 2 ** bit), dtype=np.float16)
            for r_idx in range(N):
                curLUT[r_idx] = layer_lut[name][r_idx][0]  # the 0 here assumes group_count == 1

            layer_data[param_name + '.lut' + str(bit)] = curLUT

    return layer_idx, layer_data


def mixed_pack(
        analyzer,
        lut_paths,
        output_model_path,
        ground_precisions,
        layerwise_config,
        cpu_count=None,
        group_count=1,
):

    if cpu_count is None:
        cpu_count = int(os.popen("nproc").read().strip())

    # Limit cpu_count to 8 as larger values use too much memory, without much speedup
    _max_cpu_count = 8
    if cpu_count > _max_cpu_count:
        logging.warning(f"cpu_count will be limited to 8 to avoid excessive memory usage. "
                        f"Original value: {cpu_count}")
        cpu_count = _max_cpu_count

    tokenizer = analyzer.tokenizer

    num_layers = analyzer.num_layers

    model_name = analyzer.model_name
    layers_name = analyzer.layers_name
    module_names = analyzer.module_names
    config = analyzer.config  # original model config
    arch_config = analyzer.get_arch_config()

    state_dict = analyzer.state_dict

    args_list = [(layer_idx, lut_paths, model_name, layers_name, module_names, layerwise_config) for
                 layer_idx in range(num_layers)]

    with Pool(cpu_count) as pool:
        for layer_idx, layer_data in tqdm(pool.imap(_process_mixed_layer_data, args_list), total=num_layers, desc="Packing"):
            for key, value in layer_data.items():
                state_dict[key] = torch.from_numpy(value)  # Update with modified weights

    # add new config parameters
    anyprec_configs = {
        'lut_paths': lut_paths,
        'layerwise_config': layerwise_config,
        'arch_config': arch_config
    }
    config.anyprec = anyprec_configs

    logging.info(f"Writing model to disk...")
    os.makedirs(output_model_path, exist_ok=True)
    torch.save(state_dict, os.path.join(output_model_path, 'pytorch_model.bin'))
    tokenizer.save_pretrained(output_model_path)
    config.save_pretrained(output_model_path)
    logging.info(f"Model saved to {output_model_path}")


def pack_single_weight(weight: torch.Tensor, parent_precision: int):
    weight = weight.detach().cpu().numpy()

    N, group_count, group_size = weight.shape
    K = group_count * group_size

    qweight_flattened = weight.flatten()
    bitarray = np.empty((parent_precision, len(qweight_flattened) // 8), dtype=np.uint8)
    mask = 1 << (parent_precision - 1)  # MSB first
    for bit in range(parent_precision):
        curbitpack = np.packbits((qweight_flattened & mask).astype(bool))
        bitarray[bit] = curbitpack
        mask >>= 1

    bitarray = bitarray.reshape((parent_precision, N, K // 8))
    weighttensor = _permute_bitmaps_int32(bitarray)

    return weighttensor


def unpack_single_weight(packed_weight: torch.Tensor, parent_precision: int):
    """
    Reverse the bit-packing and permutation done by pack_single_weight,
    returning an 8-bit tensor with shape = original_shape.
    """
    # 1) Convert int32 -> uint8 bytes
    w_bits, N, ints_per_row = packed_weight.shape
    total_bytes = ints_per_row * 4
    bitmaps_uint8 = packed_weight.view(torch.uint8).reshape(w_bits, N, total_bytes)

    # 2) Unpermute bytes (apply _permute_bitmaps again; it's symmetrical)
    bitmaps_unpermuted = _permute_bitmaps(bitmaps_uint8.detach().cpu().numpy(), inverse=True)

    # 3) Flatten per-bit data and unpack bits
    bitmaps_unpermuted_2d = bitmaps_unpermuted.reshape(parent_precision, -1)
    original_shape = (N, 1, ints_per_row * 32)
    out_flat = np.zeros(np.prod(original_shape), dtype=np.uint8)
    for bit in range(parent_precision):
        bools = np.unpackbits(bitmaps_unpermuted_2d[bit])
        out_flat |= (bools.astype(np.uint8) << (parent_precision - bit - 1))

    # 4) Reshape to the original 8-bit tensor
    out_tensor = torch.from_numpy(out_flat.reshape(original_shape))
    return out_tensor
