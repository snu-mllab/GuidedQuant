import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from any_precision.analyzer.analyzer import ModelAnalyzer
from typing import List, Tuple, Literal, Optional
import time

from .utils import get_progress_bar

@torch.no_grad()
def objective_function(
    W: torch.Tensor, 
    H: torch.Tensor, 
    labels: torch.Tensor, 
    C: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the quantization error (objective value).
    
    Args:
    W: Weight matrix (output_dim, input_dim)
    H: Hessian matrix (num_groups, input_dim, input_dim)
    labels: Assignment matrix (output_dim, input_dim)
    C: Centroid matrix (output_dim, n_cluster)
    
    Returns:
    Objective value (scalar)
    """

    device = torch.device("cuda")
    labels, C = labels.to(device), C.to(device)
    W_hat = torch.gather(
        C.unsqueeze(1).expand(-1, labels.shape[1], -1), 
        dim=2, 
        index=labels.unsqueeze(-1).long()
    ).squeeze(-1)
    delta_w = W_hat - W

    num_groups = H.shape[0]
    group_size = W.shape[0] // num_groups

    delta_w = delta_w.reshape(num_groups, group_size, delta_w.shape[-1])
    objective_value = torch.einsum('nij,njk,nik->i', delta_w, H, delta_w) 
    total_error = objective_value.mean()

    return total_error

def update_P(
    W: torch.Tensor,  # Shape: (output_dim, input_dim)
    H: torch.Tensor,  # Shape: (num_groups, input_dim, input_dim)
    labels: torch.Tensor,  # Shape: (output_dim, input_dim)
    C: torch.Tensor,  # Shape: (output_dim, n_cluster)
    cd_cycles: int,
    verbose: bool = True,
):
    device = torch.device("cuda")
    C = C.to(device)
    assignments_prev = labels.to(device).long()  # Shape: (output_dim, input_dim)
    b, d = assignments_prev.shape
    n_cluster = C.size(1)
    num_groups = H.shape[0]
    group_size = W.shape[0] // num_groups


    assignments = assignments_prev.clone()

    update_size = cd_cycles * d

    W_hat = torch.gather(C.unsqueeze(1).expand(-1, d, -1), dim=2, index=assignments.unsqueeze(-1)).squeeze(-1) # Shape: (output_dim, input_dim)


    assert W.shape[0] % num_groups == 0

    pb = get_progress_bar(update_size, f"Updating P inside")

    W_grp = W.reshape(num_groups, group_size, W.shape[-1]) # Shape: (num_groups, group_size, input_dim)
    C_grp = C.reshape(num_groups, group_size, C.shape[-1]) # Shape: (num_groups, group_size, n_cluster)
    W_hat_grp = W_hat.reshape(num_groups, group_size, W_hat.shape[-1]) # Shape: (num_groups, group_size, input_dim)
    H_grp = H.clone().to(device)
    B_grp = torch.zeros_like(W_grp).to(device)

    for i in range(num_groups):
        H_grp_diag = H_grp[i, torch.arange(d), torch.arange(d)]
        H_grp_diag = H_grp_diag.reshape(1, 1, -1)
        H_grp[i, :, :] = H_grp[i, :, :] / H_grp_diag

    cd_block_size = 128

    for k in range(cd_cycles):

        B_grp = torch.bmm(W_hat_grp - W_grp, torch.tril(H_grp, diagonal=-1))

        for start_idx in range(0, d, cd_block_size):

            end_idx = min(start_idx + cd_block_size, d)

            for update_idx in range(start_idx, end_idx):

                index = torch.arange(update_idx, update_idx + 1, device=device)
                sol = W_grp[:, :, index] - B_grp[:, :, index]

                assert sol.shape == (num_groups, group_size, 1)

                sol_dist = torch.abs(sol - C_grp) # Shape: (num_groups, group_size, n_cluster)
                min_dist, argmin_dist = sol_dist.min(dim=-1) # Shape: (num_groups, group_size)

                assignments[:, index] = argmin_dist.reshape(-1, 1)
                W_hat_grp[:, :, index] = torch.gather(C_grp, dim=-1, index=argmin_dist.unsqueeze(-1))

                if update_idx < end_idx - 1:
                    B_grp[:, :, update_idx + 1:end_idx] += torch.bmm(W_hat_grp[:, :, index] - W_grp[:, :, index], H_grp[:, index, update_idx + 1:end_idx])
                pb.update(1)
            
            B_grp[:, :, end_idx:] += torch.bmm(W_hat_grp[:, :, start_idx:end_idx] - W_grp[:, :, start_idx:end_idx], H_grp[:, start_idx:end_idx, end_idx:])
    pb.close()
    
    num_changed = (assignments_prev != assignments).sum().item()
    total_assignments = assignments_prev.numel()
    percentage_changed = num_changed / total_assignments * 100
    if verbose:
        logging.info(f"Percentage of assignments changed: {percentage_changed:.2f}%")

    return assignments

def update_C(
    W: torch.Tensor, # Shape: (output_dim, input_dim)
    H: torch.Tensor, # Shape: (num_groups, input_dim, input_dim)
    labels: torch.Tensor, # Shape: (output_dim, input_dim)
    C: torch.Tensor, # Shape: (output_dim, n_cluster)
    iteration: int
):
    device = torch.device("cuda")
    channel_size = W.shape[0]
    input_size = H.shape[1]
    sub_channel_size = 64
    sub_input_size = 2 ** 16

    num_groups = H.shape[0]
    group_size = W.shape[0] // num_groups
    L = torch.empty_like(H)
    for i in range(num_groups):
        L[i] = torch.linalg.cholesky(H[i])

    reduced_X = L.transpose(-2, -1)

    assert channel_size // sub_channel_size >= num_groups
    assert channel_size % (sub_channel_size * num_groups) == 0

    C_hat_list = []
    pb = get_progress_bar(channel_size // sub_channel_size, "Updating centroids")
    for st_idx in range(0, channel_size, sub_channel_size):

        group_idx = st_idx // group_size
        reduced_X_blk = reduced_X[group_idx] # Shape: (input_dim, input_dim)

        end_idx = min(st_idx + sub_channel_size, channel_size)
        
        A_batch_list, b_batch_list = [], []
        labels_batch = labels[st_idx:end_idx].to(device)
        for st_idx_inp in range(0, input_size, sub_input_size):
            end_idx_inp = min(st_idx_inp + sub_input_size, input_size)
            X_batch = reduced_X_blk[st_idx_inp:end_idx_inp].to(device)
            P_batch = torch.nn.functional.one_hot(labels_batch.long(), num_classes=C.shape[-1]).float()  # (i, j, c)
            A_batch_tmp = torch.einsum('bj,ijc->ibc', X_batch, P_batch) # Shape: (output_dim, num_samples, n_cluster)
            b_batch_tmp = torch.einsum('bj,ij->ib', X_batch, W[st_idx:end_idx]).unsqueeze(-1) # Shape: (output_dim, num_samples, 1)
            A_batch_list.append(A_batch_tmp)
            b_batch_list.append(b_batch_tmp)

        A_batch = torch.cat(A_batch_list, dim=1)
        b_batch = torch.cat(b_batch_list, dim=1)

        ######### REGULARIZATION #########
        lambda_reg = 1e-7
        # Get dimensions
        batch_size, num_samples, n_cluster = A_batch.shape
        dtype, device = A_batch.dtype, A_batch.device

        # Create sqrt(lambda) * I matrix for regularization
        sqrt_lambda = torch.sqrt(torch.tensor(lambda_reg, dtype=dtype, device=device))
        I = sqrt_lambda * torch.eye(n_cluster, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        # Augment A_batch and b_batch for regularization
        A_batch = torch.cat([A_batch.transpose(1, 2), I], dim=2).transpose(1, 2)
        zeros = torch.zeros((batch_size, n_cluster, 1), dtype=dtype, device=device)
        b_batch = torch.cat([b_batch, zeros], dim=1)

        ##################################

        # Compute the least squares solution for this batch
        C_hat_batch = torch.linalg.lstsq(A_batch, b_batch).solution  # Shape: (output_dim, num_samples, n_cluster)
        # Check if C_hat_batch has nan values
        if torch.isnan(C_hat_batch).any():
            logging.error(f"NaN values detected in C_hat_batch for indices {st_idx} to {end_idx}")
            exit()

        C_hat_batch = C_hat_batch.squeeze(-1)  # Shape: (output_dim, num_samples, n_cluster)
        
        C_hat_list.append(C_hat_batch)
        pb.update(1)
    pb.close()

    C = torch.cat(C_hat_list, dim=0).cpu()

    return C

def train_least_squares(
    W: np.ndarray, # Shape: (output_dim, input_dim)
    init_labels: np.ndarray, # Shape: (output_dim, input_dim)
    init_centroids: np.ndarray, # Shape: (output_dim, n_cluster)
    H: np.ndarray, # Shape: (num_groups, input_dim, input_dim)
    num_iterations: int = 3,
    cd_cycles: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda")

    labels = torch.tensor(init_labels, dtype=torch.int8, device="cpu")
    C = torch.tensor(init_centroids, dtype=torch.float32, device="cpu")
    W = torch.tensor(W, dtype=torch.float32).to(device)
    H = torch.tensor(H, dtype=torch.float32).to(device)

    diag = torch.arange(H.shape[1], device=device)
    for i in range(H.shape[0]):
        avg_diag = torch.mean(torch.diag(H[i]))
        damp, prev_damp = 1e-5, 0.
        while True:
            try:
                torch.linalg.cholesky(H[i])
                logging.info(f"{i+1}-th H is PD, dampening factor={prev_damp:.2e}")
                break
            except Exception as e:
                print(e)
                logging.info(f"{i+1}-th H is not PD, try dampening with factor={damp:.2e}")
                H[i, diag, diag] += (damp - prev_damp) * avg_diag
                prev_damp = damp
                damp *= 10
                if damp > 1e0:
                    exit()

    best_obj_value = objective_function(W, H, labels, C).item()
    best_labels, best_C = labels.detach().cpu().clone(), C.detach().cpu().clone()
    logging.info(f"Initial objective: {best_obj_value:.6f}")

    log_dict = {"objective": [], "iteration": []}
    log_dict["objective"].append(best_obj_value)
    log_dict["iteration"].append(0)

    for iteration in range(num_iterations):
        start_time = time.time()

        ######### Update P #########
        if iteration > 0:
            labels = update_P(W, H, labels, C, cd_cycles=cd_cycles)

        # Compute objective value for logging
        obj_value = objective_function(W, H, labels, C).item()
        logging.info(f"Iteration {iteration + 1} (P update): Objective: {obj_value:.4f}")
        log_dict["objective"].append(obj_value)
        log_dict["iteration"].append(iteration + 1)


        ######### Update C #########
        C = update_C(W, H, labels, C, iteration)

        # Check if the objective value improved
        current_obj_value = objective_function(W, H, labels, C).item()
        log_dict["objective"].append(current_obj_value)
        log_dict["iteration"].append(iteration + 1)
        if current_obj_value < best_obj_value:
            best_obj_value = current_obj_value
            best_labels, best_C = labels.detach().cpu().clone(), C.detach().cpu().clone()
            logging.info(f"Iteration {iteration + 1} (C update): Objective: {current_obj_value:.4f} | Improved and using this one.")
        else:
            logging.info(f"Iteration {iteration + 1} (C update): Objective: {current_obj_value:.4f} | Not improved. Using previous best values.")
            labels, C = best_labels, best_C
            break  # Early stopping

        end_time = time.time()

        logging.info(f"Iteration {iteration + 1} / {num_iterations} completed. "
                     f"Update time: {end_time - start_time:.2f} sec")

    end_time = time.time()
    logging.info(f"Least squares training time: {end_time - start_time:.2f} seconds")

    labels = labels.detach().cpu().numpy()
    C = C.detach().cpu().numpy().astype(np.float32)

    return labels, C, log_dict

def free_and_log_memory():
    torch.cuda.empty_cache()
    import gc; gc.collect()
    logging.info(f"Left memory: {torch.cuda.mem_get_info()[0] / (10 ** 9)} GB")

def seed_layer(
    l: int,
    module_names: List[str],
    layer_modules: List[np.ndarray],
    layer_init_labels: List[np.ndarray],
    layer_init_centroids: List[np.ndarray],
    layer_hessian: List[np.ndarray],
    seed_bit: int,
    group_count: int,
    num_iterations: int = 3,
    cd_cycles: int = 4,
) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
    lut_by_bit_by_module = []
    parent_weights_by_modules = []
    log_dict_by_module = []

    n_cluster = 2**seed_bit

    for m_idx in range(len(layer_modules)):
        module_name = module_names[m_idx]
        logging.info(f"Quantizing Layer [{l}], Module [{module_name}] ({m_idx + 1}/{len(layer_modules)})")

        module_weight = layer_modules[m_idx]
        module_init_labels = layer_init_labels[m_idx]
        module_init_centroids = layer_init_centroids[m_idx]
        module_hessian = layer_hessian[m_idx]

        assert group_count == 1, "Group-wise quantization is not supported yet"

        output_dim = module_weight.shape[0]
        input_dim = module_weight.shape[1]

        lut_by_bit = []
        for bit in range(seed_bit, seed_bit + 1):
            lut_by_bit.append(
                np.empty((output_dim, 1, 2**bit), dtype=np.float32)
            )

        init_labels = module_init_labels.reshape(output_dim, input_dim)
        init_centroids = module_init_centroids.reshape(output_dim, n_cluster) # Shape: (output_dim, n_cluster)
        reshaped_module_weight = module_weight.reshape(output_dim, input_dim) # Shape: (output_dim, input_dim)

        labels, C, log_dict = train_least_squares(reshaped_module_weight, init_labels, init_centroids, module_hessian, num_iterations=num_iterations, cd_cycles=cd_cycles)

        labels = labels.astype(np.uint8) # Shape: (output_dim, input_dim)
        labels = labels.reshape(output_dim, 1, input_dim) # Shape: (output_dim, 1, input_dim)
        C = C.reshape(output_dim, 1, n_cluster) # Shape: (output_dim, 1, n_cluster)

        for k, bit in enumerate(range(seed_bit, seed_bit + 1)):
            lut_by_bit[k] = C

        parent_weights_by_modules.append(labels)
        lut_by_bit_by_module.append(lut_by_bit)
        log_dict_by_module.append(log_dict)

    return lut_by_bit_by_module, parent_weights_by_modules, log_dict_by_module


# Minimal inline function to ensure shape is (num_groups, input_dim, input_dim)
def fix_hessian_shape(H: torch.Tensor) -> torch.Tensor:
    if H.shape[1] == H.shape[2]:
        # Already (num_groups, input_dim, input_dim)
        return H
    elif H.shape[0] == H.shape[1]:
        # Then it's (input_dim, input_dim, num_groups), so permute
        return H.permute(2, 0, 1)
    else:
        raise ValueError(f"Invalid Hessian shape: {H.shape}")


def get_layer_loader(analyzer, module_names, initialization_path, hessians_path, seed_precision):
    def layer_loader(l):
        # Load the initialization data (labels and centroids)
        init_labels_file_name = os.path.join(initialization_path, "weights", f"l{l}.pt")
        init_labels = torch.load(init_labels_file_name)
        init_centroids_file_name = os.path.join(initialization_path, f"lut_{seed_precision}", f"l{l}.pt")
        init_centroids = torch.load(init_centroids_file_name)
        hessian_file_name = os.path.join(hessians_path, f"l{l}.pt")
        hessian = torch.load(hessian_file_name)

        # Organize the data by module
        init_labels_layer = [
            init_labels[name] for name in module_names
        ]
        init_centroids_layer = [
            init_centroids[name].astype(np.float32) for name in module_names
        ]
        hessian_layer = [
            fix_hessian_shape(hessian[name]).float().numpy() for name in module_names
        ]
        model_layer = [
            analyzer.get_layer_weights(l)[name].float().numpy()
            for name in module_names
        ]
        return module_names, model_layer, init_labels_layer, init_centroids_layer, hessian_layer

    return layer_loader


def _save_results(
    parent_parameters_path,
    seed_precision,
    parent_precision,
    module_names,
    luts_by_bit_by_module,
    parent_weights,
    log_dict,
    l,
):
    # Note that it is important to cast the luts to fp16 before saving them.
    for i, bit in enumerate(range(seed_precision, parent_precision + 1)):
        output_lut_file_name = f"{parent_parameters_path}/lut_{bit}/l{l}.pt"
        output_log_dict_file_name = f"{parent_parameters_path}/lut_{bit}/log_dict{l}.pt"
        os.makedirs(os.path.dirname(output_lut_file_name), exist_ok=True)
        lut_dict = {}
        module_name_to_log_dict = {}
        for j in range(len(module_names)):
            lut_dict[module_names[j]] = luts_by_bit_by_module[j][i].astype(np.float16)
            module_name_to_log_dict[module_names[j]] = log_dict[j]
        torch.save(lut_dict, output_lut_file_name)
        torch.save(module_name_to_log_dict, output_log_dict_file_name)

    parent_weight_dict = {
        module_names[j]: parent_weights[j].astype(np.uint8)
        for j in range(len(module_names))
    }

    output_weights_layer_file_name = f"{parent_parameters_path}/weights/l{l}.pt"
    os.makedirs(os.path.dirname(output_weights_layer_file_name), exist_ok=True)
    torch.save(parent_weight_dict, output_weights_layer_file_name)


def get_saver(parent_parameters_path, seed_precision, parent_precision, module_names):
    """Returns a function that saves the results for a given layer"""

    def save_results(luts_by_bit_by_module, parent_weights, log_dict, l):
        return _save_results(
            parent_parameters_path,
            seed_precision,
            parent_precision,
            module_names,
            luts_by_bit_by_module,
            parent_weights,
            log_dict,
            l,
        )

    return save_results


def load_progress(
    parent_parameters_path, seed_precision, parent_precision, layer_count
):
    # Check if the layer has already been processed
    todo_ran = []
    processed_ran = []
    for l in range(layer_count):
        if all(
            [
                os.path.exists(f"{parent_parameters_path}/lut_{bit}/l{l}.pt")
                for bit in range(seed_precision, parent_precision + 1)
            ]
        ) and os.path.exists(f"{parent_parameters_path}/weights/l{l}.pt"):
            processed_ran.append(l)
        else:
            todo_ran.append(l)
    return todo_ran, processed_ran


def seed(
    analyzer: ModelAnalyzer,
    module_names: List[str],
    initialization_path: str,
    hessians_path: str,
    output_folder: str,
    seed_precision: int,
    cpu_count: int = None,
    num_iterations: int = 3,
    cd_cycles: int = 4,
    sub_qlayer: Tuple[int, int] = None,
):
    group_count = 1

    if cpu_count is None:
        cpu_count = int(os.popen("nproc").read().strip())
    # Determine IO and threading settings based on the number of cores
    if cpu_count >= 8:
        pipelined_io = True
        io_workers = 2 if cpu_count >= 64 else 1
    else:
        pipelined_io = False
        io_workers = 0  # No separate IO workers needed for non-pipelined IO

    logging.info(f"Using {cpu_count} cores for parallelization")

    logging.info(f"Seeding for {seed_precision}-bit")

    layers_to_process, completed_layers = load_progress(
        output_folder, seed_precision, seed_precision, analyzer.num_layers
    )

    if sub_qlayer:
        layers_to_process = [i for i in layers_to_process if i in range(sub_qlayer[0], sub_qlayer[1])]

    if completed_layers:
        logging.info(
            f"The following layers will be skipped as they have already been processed:\n{completed_layers}"
        )
        logging.info(
            f"To reprocess these layers, delete the corresponding files in {output_folder}"
        )

    if not layers_to_process:
        logging.info("All layers have already been processed. Exiting...")
        return

    logging.info(f"Quantizing layers {layers_to_process}")

    layer_loader = get_layer_loader(
        analyzer, module_names, initialization_path, hessians_path, seed_precision
    )
    layer_saver = get_saver(
        output_folder, seed_precision, seed_precision, module_names
    )

    if pipelined_io:
        with ThreadPoolExecutor(max_workers=io_workers) as io_executor:
            pb = get_progress_bar(len(layers_to_process), "Quantizing layers...")
            for l in layers_to_process:
                if l == layers_to_process[0]:
                    future_load = io_executor.submit(layer_loader, l)

                module_names, model_layer, init_labels_layer, init_centroids_layer, hessian_layer = future_load.result()

                if l != layers_to_process[-1]:
                    future_load = io_executor.submit(layer_loader, l + 1)

                luts_by_bit_by_module, parent_weights, log_dict = seed_layer(
                    l,
                    module_names,
                    model_layer,
                    init_labels_layer,
                    init_centroids_layer,
                    hessian_layer,
                    seed_precision,
                    group_count,
                    num_iterations=num_iterations,
                    cd_cycles=cd_cycles,
                )

                io_executor.submit(
                    layer_saver, luts_by_bit_by_module, parent_weights, log_dict, l
                )
                pb.update(1)
            pb.close()
            logging.info("Waiting for IO to finish...")
    else:
        pb = get_progress_bar(len(layers_to_process), "Quantizing layers...")
        for l in layers_to_process:
            module_names, model_layer, init_labels_layer, init_centroids_layer, hessian_layer = layer_loader(l)

            luts_by_bit_by_module, parent_weights, log_dict = seed_layer(
                l,
                module_names,
                model_layer,
                init_labels_layer,
                init_centroids_layer,
                hessian_layer,
                seed_precision,
                group_count,
                num_iterations=num_iterations,
                cd_cycles=cd_cycles,
            )

            layer_saver(luts_by_bit_by_module, parent_weights, log_dict, l)
            pb.update(1)
        pb.close()
