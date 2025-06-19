import torch
import os
import numpy as np
import re
import argparse
from safetensors.torch import load_file

### llama
replacements = {
    'embed_tokens': 'tok_embeddings',
    'self_attn': 'attention',
    'q_proj': 'wq',
    'k_proj': 'wk',
    'v_proj': 'wv',
    'o_proj': 'wo',
    'mlp': 'feed_forward',
    'gate_proj': 'w1',
    'up_proj': 'w3',
    'down_proj': 'w2',
    #'input_layernorm': 'attention_norm',
    #'post_attention_layernorm': 'ffn_norm',
    'lm_head': 'output'
}

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str)
args = parser.parse_args()

ckpt_dir = args.ckpt_dir
ckpt = load_file(os.path.join(ckpt_dir, "model.safetensors"))

new_dict = {}
for key, value in ckpt.items():
    # Remove 'model.' from the key
    new_key = key.replace('model.', '')

    # Perform the replacements as specified
    for old, new in replacements.items():
        new_key = new_key.replace(old, new)

    # Update the new dictionary
    new_dict[new_key] = value

torch.save(new_dict, os.path.join(ckpt_dir, "converted_pytorch_model.bin"))


