import os
import torch
from tqdm import tqdm
import logging
from typing import Optional, Tuple
from .config import *
from any_precision.analyzer import dispatch_model


def get_gradients(
        analyzer,
        input_tokens,
        save_path: Optional[str] = None,
        saliency_path: Optional[str] = None,
        num_groups: Optional[int] = None,
        sub_saliency: Optional[Tuple[int, int]] = None,
        skip_save_gradients: bool = False,
):
    """
    Calculates weight gradients for the given input tokens. Optionally also calculates
    'saliency' (mean absolute gradient w.r.t. each module's output activations, grouped
    by channel) if 'saliency_path' is provided. In that case, we save one file per layer
    under 'saliency_path' directory (e.g., l0.pt, l1.pt, ...).

    If 'sub_saliency' is provided (e.g. (start_layer, end_layer)), we only attach saliency
    hooks (and save files) for layers in [start_layer, end_layer). Layers outside that
    range won't generate saliency data or files.

    Args:
        analyzer:        Analyzer object with `.model`, `.get_layers()`, `.get_modules(layer)`.
        input_tokens:    Collection of token tensors, each shape [seq_len].
        save_path:       Path to save the final weight gradients (list of dicts).
                         If the file already exists, user is prompted before overwriting.
        saliency_path:   Directory in which to save the saliency files (one file per layer).
                         If None, no saliency is computed/saved.
        num_groups:      Number of groups to chunk the channel dimension for saliency.
                         E.g. if hidden_dim=4096 and num_groups=4, each group has 1024 channels.
        sub_saliency:    (start_layer, end_layer). If provided, only layers in that range
                         will collect saliency. Otherwise, collect for all layers.

    Returns:
        gradients (list of dict): The list of per-layer, per-module weight gradients.
    """

    # ----------------------------------------------------------------
    # 1) Possibly load from cache (gradients only)
    # ----------------------------------------------------------------
    if save_path is not None and os.path.isfile(save_path):
        logging.info(f"Gradients already calculated and saved at {save_path}.")
        logging.info(f"Loading cached gradients...")
        return torch.load(save_path)

    logging.info(f"Calculating gradients on {len(input_tokens)} tokens...")

    # ----------------------------------------------------------------
    # 2) Prepare model
    # ----------------------------------------------------------------
    model = analyzer.model
    if torch.cuda.device_count() > 1:
        model = dispatch_model(model)

    model = model.bfloat16()
    model.eval()

    if model.device.type != 'cuda' and torch.cuda.device_count() == 1:
        model.cuda()

    layers = analyzer.get_layers()

    # If sub_saliency is given, parse it
    # We'll use these to decide whether to hook/save a given layer
    if sub_saliency is not None:
        start_layer, end_layer = sub_saliency
    else:
        start_layer, end_layer = (None, None)

    # ----------------------------------------------------------------
    # 3) If we want saliency, set up forward hooks
    # ----------------------------------------------------------------
    # We'll store a list-of-dicts parallel to `layers`:
    #   saliency_data[i_layer][module_name] = list of [bsz, seq_len, num_groups]
    saliency_data = None
    saliency_hooks = []

    if saliency_path is not None:
        # We'll store chunk-lists for all layers, but only fill them
        # for the sub_saliency range
        saliency_data = [
            {module_name: [] for module_name in analyzer.get_modules(layer).keys()}
            for layer in layers
        ]

        def make_forward_hook(layer_idx, module_name):
            def forward_hook(module, inp, out):
                # We'll store gradient on 'out', so we must retain it
                out.retain_grad()

                def grad_hook(grad):
                    """
                    grad shape typically [bsz, seq_len, hidden_dim].
                    We group the channels, take abs, then average.
                    """
                    bsz, seq_len, hidden_dim = grad.shape
                    group_size = hidden_dim // num_groups

                    grad_squared = (grad.float() * 1e3).pow(2).view(bsz, seq_len, num_groups, group_size)
                    mean_squared_grad = grad_squared.mean(dim=-1)  # -> [bsz, seq_len, num_groups]

                    # Move to CPU and store
                    saliency_data[layer_idx][module_name].append(mean_squared_grad.bfloat16().cpu())

                # Attach the gradient hook to 'out'
                out.register_hook(grad_hook)
            return forward_hook

        # Attach hooks only for layers in [start_layer, end_layer) if set
        for layer_idx, layer in enumerate(layers):
            if (start_layer is not None) and (end_layer is not None):
                if not (start_layer <= layer_idx < end_layer):
                    # skip hooking this layer
                    continue

            # Register forward hooks for each module
            for module_name, module in analyzer.get_modules(layer).items():
                h = module.register_forward_hook(make_forward_hook(layer_idx, module_name))
                saliency_hooks.append(h)

    # ----------------------------------------------------------------
    # 4) Weight-gradient hook (square_grad_hook)
    # ----------------------------------------------------------------
    def square_grad_hook(grad):
        return grad.pow(2)

    weight_hooks = []
    for layer_idx in layers:
        for module in analyzer.get_modules(layer_idx).values():
            weight_hooks.append(module.weight.register_hook(square_grad_hook))

    # ----------------------------------------------------------------
    # 5) Forward/backward pass over data
    # ----------------------------------------------------------------
    for tokens in tqdm(input_tokens, desc="Calculating gradients"):
        tokens = tokens.to(model.device).unsqueeze(0)
        outputs = model(input_ids=tokens, labels=tokens)
        loss = outputs.loss
        loss.backward()

    # ----------------------------------------------------------------
    # 6) Remove hooks
    # ----------------------------------------------------------------
    for h in weight_hooks:
        h.remove()

    for h in saliency_hooks:
        h.remove()

    # ----------------------------------------------------------------
    # 7) Move model back to CPU
    # ----------------------------------------------------------------
    model.cpu()

    # ----------------------------------------------------------------
    # 8) Harvest the weight gradients
    # ----------------------------------------------------------------
    gradients = []
    for layer_idx in layers:
        gradients_per_layer = {}
        for module_name, module in analyzer.get_modules(layer_idx).items():
            gradients_per_layer[module_name] = module.weight.grad
        gradients.append(gradients_per_layer)

    # ----------------------------------------------------------------
    # 9) Save saliency per layer, if computed
    # ----------------------------------------------------------------
    if saliency_path is not None:
        logging.info(f"Saving saliency files to {saliency_path}...")

        # Ensure directory exists
        os.makedirs(saliency_path, exist_ok=True)

        # For each layer, gather module data -> single dictionary, then save
        for layer_idx, layer in enumerate(layers):
            # If sub_saliency is set, only save if layer_idx in range
            if (start_layer is not None) and (end_layer is not None):
                if not (start_layer <= layer_idx < end_layer):
                    continue

            # Build dict of { module_name -> cat_tensor or None }
            layer_dict = {}
            for module_name, chunk_list in saliency_data[layer_idx].items():
                if len(chunk_list) > 0:
                    cat_tensor = torch.cat(chunk_list, dim=0)  # shape: [N, seq_len, num_groups]
                else:
                    cat_tensor = None
                layer_dict[module_name] = cat_tensor

            # If there's no data at all (empty?), you can choose to skip saving
            # But we'll save anyway for consistency
            filename = os.path.join(saliency_path, f"l{layer_idx}.pt")

            if os.path.exists(filename):
                input(f"[WARNING] File {filename} already exists. "
                      "Press Enter to overwrite or Ctrl+C to cancel.")

            # Save each layer's dictionary to l{layer_idx}.pt
            torch.save(layer_dict, filename)

    # ----------------------------------------------------------------
    # 10) Save the gradients (if needed)
    # ----------------------------------------------------------------
    if save_path is not None and not skip_save_gradients:
        logging.info(f"Saving gradients to {save_path}...")
        if not save_path.endswith('.pt'):
            save_path = save_path + '.pt'
        if os.path.exists(save_path):
            input(f"[WARNING] File {save_path} already exists. "
                  "Press Enter to overwrite or Ctrl+C to cancel.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(gradients, save_path)

    # ----------------------------------------------------------------
    # 11) Return the gradients
    # ----------------------------------------------------------------
    return gradients
